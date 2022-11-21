import os

import drjit as dr
import mitsuba as mi
import numpy as np

from ..problem import MitsubaProblem
from ..utils import to_float

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CAUSTICS_SCENE_DIR = os.path.join(ROOT_DIR, "double-sided-caustics")
DEFAULT_CAUSTICS_CONFIG = {
    "render_resolution": (128, 128),
    "heightmap_resolution": (512, 512),
    "reference": os.path.join(CAUSTICS_SCENE_DIR, "references/sunday-128.exr"),
    "emitter": "bayer",
}


def create_flat_lens_mesh(resolution):
    # Generate UV coordinates
    U, V = dr.meshgrid(
        dr.linspace(mi.Float, 0, 1, resolution[0]),
        dr.linspace(mi.Float, 0, 1, resolution[1]),
        indexing="ij",
    )
    texcoords = mi.Vector2f(U, V)

    # Generate vertex coordinates
    X = 2.0 * (U - 0.5)
    Y = 2.0 * (V - 0.5)
    vertices = mi.Vector3f(X, Y, 0.0)

    # Create two triangles per grid cell
    faces_x, faces_y, faces_z = [], [], []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            v00 = i * resolution[1] + j
            v01 = v00 + 1
            v10 = (i + 1) * resolution[1] + j
            v11 = v10 + 1
            faces_x.extend([v00, v01])
            faces_y.extend([v10, v10])
            faces_z.extend([v01, v11])

    # Assemble face buffer
    faces = mi.Vector3u(faces_x, faces_y, faces_z)

    # Instantiate the mesh object
    mesh = mi.Mesh(
        "lens-mesh",
        resolution[0] * resolution[1],
        len(faces_x),
        has_vertex_texcoords=True,
    )

    # Set its buffers
    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = dr.ravel(vertices)
    mesh_params["vertex_texcoords"] = dr.ravel(texcoords)
    mesh_params["faces"] = dr.ravel(faces)
    mesh_params.update()

    return mesh


class SingleCausticProblem(MitsubaProblem):
    def __init__(self, config=DEFAULT_CAUSTICS_CONFIG):
        super().__init__(
            n_var=np.prod(config["heightmap_resolution"]), xl=-1.0, xu=1.0
        )
        self.config = config

        _, params = self.initialize_scene()
        self.positions_initial = dr.unravel(
            mi.Vector3f, params["lens.vertex_positions"]
        )
        self.normals_initial = dr.unravel(
            mi.Vector3f, params["lens.vertex_normals"]
        )
        self.lens_si = dr.zeros(
            mi.SurfaceInteraction3f, dr.width(self.positions_initial)
        )
        self.lens_si.uv = dr.unravel(
            type(self.lens_si.uv), params["lens.vertex_texcoords"]
        )

    def set_params_from_vector(self, params, vector):
        # reshape and convert vector to mi.Bitmap
        # texture = mi.Bitmap(
        #     dr.enable_grad(
        #         mi.TensorXf(
        #             vector.reshape(self.config["heightmap_resolution"])
        #         )
        #     )
        # )
        texture = mi.Bitmap(
            dr.zeros(mi.TensorXf, self.config["heightmap_resolution"])
        )
        heightmap_texture = mi.load_dict(
            {
                "type": "bitmap",
                "id": "heightmap_texture",
                "bitmap": texture,
                "raw": True,
            }
        )
        # params["heightmap_texture.data"] = mi.traverse(heightmap_texture)[
        params["data"] = mi.traverse(heightmap_texture)["data"]
        return heightmap_texture

    def set_vector_from_params(self, params, vector):
        vector[:] = params["heightmap_texture.data"].numpy()

    def apply_transformations(self, scene_params, params, **kwargs):
        # Enforce reasonable range. For reference, the receiving plane
        # is 7 scene units away from the lens.
        vmax = 1 / 100.0
        # key = "heightmap_texture.data"
        key = "data"
        params[key] = dr.clamp(params[key], -vmax, vmax)
        dr.enable_grad(params[key])
        scene_params.update(params)

        height_values = kwargs["heightmap_texture"].eval_1(self.lens_si)
        new_positions = (
            height_values * self.normals_initial + self.positions_initial
        )
        scene_params["lens.vertex_positions"] = dr.ravel(new_positions)
        scene_params.update()

    def initialize_scene(self):
        scene = self.reset_scene()
        params = mi.traverse(scene)
        return scene, params

    def _get_emitter(self):
        emitter = None
        if self.config["emitter"] == "gray":
            emitter = {
                "type": "directionalarea",
                "radiance": {"type": "spectrum", "value": 0.8},
            }
        elif self.config["emitter"] == "bayer":
            bayer = dr.zeros(mi.TensorXf, (32, 32, 3))
            bayer[::2, ::2, 2] = 2.2
            bayer[::2, 1::2, 1] = 2.2
            bayer[1::2, 1::2, 0] = 2.2

            emitter = {
                "type": "directionalarea",
                "radiance": {
                    "type": "bitmap",
                    "bitmap": mi.Bitmap(bayer),
                    "raw": True,
                    "filter_type": "nearest",
                },
            }

        return emitter

    def reset_scene(self):
        emitter = self._get_emitter()

        integrator = {
            "type": "ptracer",
            "samples_per_pass": 256,
            "max_depth": 4,
            "hide_emitters": False,
        }

        sensor_to_world = mi.ScalarTransform4f.look_at(
            target=[0, -20, 0], origin=[0, -4.65, 0], up=[0, 0, 1]
        )
        resx, resy = self.config["render_resolution"]
        sensor = {
            "type": "perspective",
            "near_clip": 1,
            "far_clip": 1000,
            "fov": 45,
            "to_world": sensor_to_world,
            "sampler": {
                "type": "independent",
                "sample_count": 512,  # Not really used
            },
            "film": {
                "type": "hdrfilm",
                "width": resx,
                "height": resy,
                "pixel_format": "rgb",
                "rfilter": {
                    # Important: smooth reconstruction filter with a footprint larger than 1 pixel.
                    "type": "gaussian"
                },
            },
        }

        lens_res = self.config.get(
            "lens_res", self.config["heightmap_resolution"]
        )
        lens_fname = os.path.join(
            CAUSTICS_SCENE_DIR, "outputs", "lens_{}_{}.ply".format(*lens_res)
        )
        m = create_flat_lens_mesh(lens_res)
        m.write_ply(lens_fname)

        scene = {
            "type": "scene",
            "sensor": sensor,
            "integrator": integrator,
            # Glass BSDF
            "simple-glass": {
                "type": "dielectric",
                "id": "simple-glass-bsdf",
                "ext_ior": "air",
                "int_ior": 1.5,
                "specular_reflectance": {"type": "spectrum", "value": 0},
            },
            "white-bsdf": {
                "type": "diffuse",
                "id": "white-bsdf",
                "reflectance": {"type": "rgb", "value": (1, 1, 1)},
            },
            "black-bsdf": {
                "type": "diffuse",
                "id": "black-bsdf",
                "reflectance": {"type": "spectrum", "value": 0},
            },
            # Receiving plane
            "receiving-plane": {
                "type": "obj",
                "id": "receiving-plane",
                "filename": os.path.join(
                    CAUSTICS_SCENE_DIR, "meshes/rectangle.obj"
                ),
                "to_world": mi.ScalarTransform4f.look_at(
                    target=[0, 1, 0], origin=[0, -7, 0], up=[0, 0, 1]
                ).scale((5, 5, 5)),
                "bsdf": {"type": "ref", "id": "white-bsdf"},
            },
            # Directional area emitter placed behind the glass slab
            "focused-emitter-shape": {
                "type": "obj",
                "filename": os.path.join(
                    CAUSTICS_SCENE_DIR, "meshes/rectangle.obj"
                ),
                "to_world": mi.ScalarTransform4f.look_at(
                    target=[0, 0, 0], origin=[0, 5, 0], up=[0, 0, 1]
                ),
                "bsdf": {"type": "ref", "id": "black-bsdf"},
                "focused-emitter": emitter,
            },
            # Glass slab, excluding the 'exit' face (added separately below)
            "slab": {
                "type": "obj",
                "id": "slab",
                "filename": os.path.join(
                    CAUSTICS_SCENE_DIR, "meshes/slab.obj"
                ),
                "to_world": mi.ScalarTransform4f.rotate(
                    axis=(1, 0, 0), angle=90
                ),
                "bsdf": {"type": "ref", "id": "simple-glass"},
            },
            # Glass rectangle, to be optimized
            "lens": {
                "type": "ply",
                "id": "lens",
                "filename": lens_fname,
                "to_world": mi.ScalarTransform4f.rotate(
                    axis=(1, 0, 0), angle=90
                ),
                "bsdf": {"type": "ref", "id": "simple-glass"},
            },
            "heightmap_texture": {
                "type": "bitmap",
                "id": "heightmap_texture",
                "bitmap": mi.Bitmap(
                    dr.zeros(mi.TensorXf, self.config["heightmap_resolution"])
                ),
                "raw": True,
            },
        }

        return mi.load_dict(scene)


class DoubleCausticProblem(MitsubaProblem):
    def __init__(self, config=DEFAULT_CAUSTICS_CONFIG):
        super().__init__(
            n_var=np.prod(self.config["heightmap_resolution"]), xl=-1.0, xu=1.0
        )
        self.config = config
        _, params = self.initialize_scene()
        self.initial_vertex_positions = [
            dr.unravel(mi.Point3f, params[f"bunny{i}.vertex_positions"])
            for i in range(self.nb_bunnies)
        ]

    def set_params_from_vector(self, params, vector):
        raise NotImplementedError
        # reshape and convert vector to mi.Bitmap
        # texture = mi.Bitmap(
        #     dr.enable_grad(
        #         mi.TensorXf(
        #             vector.reshape(self.config["heightmap_resolution"])
        #         )
        #     )
        # )
        lenses = ("lens1", "lens2")
        heightmap_textures = mi.load_dict(
            dict(
                {"type": "scene"},
                **{
                    k: {
                        "type": "bitmap",
                        "id": f"heightmap_texture_{k}",
                        "bitmap": (
                            mi.Bitmap(
                                dr.zeros(
                                    mi.TensorXf,
                                    self.config["heightmap_resolution"],
                                )
                            )
                        ),
                        "raw": True,
                    }
                    for k in lenses
                },
            )
        )
        # # params["heightmap_texture.data"] = mi.traverse(heightmap_texture)[
        # params["data"] = mi.traverse(heightmap_texture)["data"]
        # return heightmap_texture

    def set_vector_from_params(self, params, vector):
        raise NotImplementedError
        # vector[:] = params["heightmap_texture.data"].numpy()

    def apply_transformations(self, scene_params, params, **kwargs):
        raise NotImplementedError
        # # Enforce reasonable range. For reference, the receiving plane
        # # is 7 scene units away from the lens.
        # vmax = 1 / 100.0
        # # key = "heightmap_texture.data"
        # key = "data"
        # params[key] = dr.clamp(params[key], -vmax, vmax)
        # dr.enable_grad(params[key])
        # scene_params.update(params)

        # height_values = kwargs["heightmap_texture"].eval_1(self.lens_si)
        # new_positions = (
        #     height_values * self.normals_initial + self.positions_initial
        # )
        # scene_params["lens.vertex_positions"] = dr.ravel(new_positions)
        # scene_params.update()

    def initialize_scene(self):
        scene = self.reset_scene()
        params = mi.traverse(scene)
        return scene, params

    def _get_emitter(self):
        emitter = None
        if self.config["emitter"] == "gray":
            emitter = {
                "type": "directionalarea",
                "radiance": {"type": "spectrum", "value": 0.8},
            }
        elif self.config["emitter"] == "bayer":
            bayer = dr.zeros(mi.TensorXf, (32, 32, 3))
            bayer[::2, ::2, 2] = 2.2
            bayer[::2, 1::2, 1] = 2.2
            bayer[1::2, 1::2, 0] = 2.2

            emitter = {
                "type": "directionalarea",
                "radiance": {
                    "type": "bitmap",
                    "bitmap": mi.Bitmap(bayer),
                    "raw": True,
                    "filter_type": "nearest",
                },
            }

        return emitter

    def reset_scene(self):
        emitter = self._get_emitter()

        integrator = {
            "type": "ptracer",
            "samples_per_pass": 256,
            "max_depth": 4,
            "hide_emitters": False,
        }

        sensor_to_world = mi.ScalarTransform4f.look_at(
            target=[0, -20, 0], origin=[0, -4.65, 0], up=[0, 0, 1]
        )
        resx, resy = self.config["render_resolution"]
        sensor = {
            "type": "perspective",
            "near_clip": 1,
            "far_clip": 1000,
            "fov": 45,
            "to_world": sensor_to_world,
            "sampler": {
                "type": "independent",
                "sample_count": 512,  # Not really used
            },
            "film": {
                "type": "hdrfilm",
                "width": resx,
                "height": resy,
                "pixel_format": "rgb",
                "rfilter": {
                    # Important: smooth reconstruction filter with a footprint larger than 1 pixel.
                    "type": "gaussian"
                },
            },
        }

        lens_res = self.config.get(
            "lens_res", self.config["heightmap_resolution"]
        )
        lens_fname = os.path.join(
            CAUSTICS_SCENE_DIR, "outputs", "lens_{}_{}.ply".format(*lens_res)
        )
        m = create_flat_lens_mesh(lens_res)
        m.write_ply(lens_fname)

        scene = {
            "type": "scene",
            "sensor": sensor,
            "integrator": integrator,
            # Glass BSDF
            "simple-glass": {
                "type": "dielectric",
                "id": "simple-glass-bsdf",
                "ext_ior": "air",
                "int_ior": 1.5,
                "specular_reflectance": {"type": "spectrum", "value": 0},
            },
            "white-bsdf": {
                "type": "diffuse",
                "id": "white-bsdf",
                "reflectance": {"type": "rgb", "value": (1, 1, 1)},
            },
            "black-bsdf": {
                "type": "diffuse",
                "id": "black-bsdf",
                "reflectance": {"type": "spectrum", "value": 0},
            },
            # Receiving plane
            "receiving-plane": {
                "type": "obj",
                "id": "receiving-plane",
                "filename": os.path.join(
                    CAUSTICS_SCENE_DIR, "meshes/rectangle.obj"
                ),
                "to_world": mi.ScalarTransform4f.look_at(
                    target=[0, 1, 0], origin=[0, -7, 0], up=[0, 0, 1]
                ).scale((5, 5, 5)),
                "bsdf": {"type": "ref", "id": "white-bsdf"},
            },
            # Directional area emitter placed behind the glass slab
            "focused-emitter-shape": {
                "type": "obj",
                "filename": os.path.join(
                    CAUSTICS_SCENE_DIR, "meshes/rectangle.obj"
                ),
                "to_world": mi.ScalarTransform4f.look_at(
                    target=[0, 0, 0], origin=[0, 5, 0], up=[0, 0, 1]
                ),
                "bsdf": {"type": "ref", "id": "black-bsdf"},
                "focused-emitter": emitter,
            },
            # Glass slab, excluding the 'exit' face (added separately below)
            "slab": {
                "type": "obj",
                "id": "slab",
                "filename": os.path.join(
                    CAUSTICS_SCENE_DIR, "meshes/slab-contour.obj"
                ),
                "to_world": mi.ScalarTransform4f.rotate(
                    axis=(1, 0, 0), angle=90
                ),
                "bsdf": {"type": "ref", "id": "simple-glass"},
            },
            # Glass rectangle, to be optimized
            "lens1": {
                "type": "ply",
                "id": "lens1",
                "filename": lens_fname,
                "to_world": mi.ScalarTransform4f.rotate(
                    axis=(1, 0, 0), angle=90
                ),
                "bsdf": {"type": "ref", "id": "simple-glass"},
            },
            "lens2": {
                "type": "ply",
                "id": "lens2",
                "filename": lens_fname,
                "to_world": (
                    mi.ScalarTransform4f.translate([0, 0.086984, 0])
                    @ mi.ScalarTransform4f.rotate(axis=(0, 0, 1), angle=180)
                    @ mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=90)
                ),
                "bsdf": {"type": "ref", "id": "simple-glass"},
            },
        }

        return mi.load_dict(scene)
