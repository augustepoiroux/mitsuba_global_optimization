import os

import drjit as dr
import mitsuba as mi
import numpy as np
from mitsuba.scalar_rgb import Transform4f as T

from ..problem import MitsubaProblem
from ..utils import to_float

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


class BunniesProblem(MitsubaProblem):
    def __init__(self, nb_bunnies=1, colored=False, scene_seed=0, range_translation=(-1.0,1.0)):
        self.range_translation = range_translation
        n_var = 3 * nb_bunnies
        xl = -np.ones(n_var)
        xl[::3] = range_translation[0]
        xl[1::3] = range_translation[0]
        xu = np.ones(n_var)
        xu[::3] = range_translation[1]
        xu[1::3] = range_translation[1]
        super().__init__(n_var=n_var, xl=xl, xu=xu)
        self.nb_bunnies = nb_bunnies
        self.colored = colored
        self.scene_seed = scene_seed
        _, params = self.initialize_scene()
        self.initial_vertex_positions = [
            dr.unravel(mi.Point3f, params[f"bunny{i}.vertex_positions"])
            for i in range(self.nb_bunnies)
        ]

    def set_params_from_vector(self, params, vector):
        for i in range(self.nb_bunnies):
            params[f"trans{i}"] = mi.Point2f(vector[3 * i], vector[3 * i + 1])
            params[f"angle{i}"] = mi.Float(vector[3 * i + 2])

    def set_vector_from_params(self, params, vector):
        for j in range(self.nb_bunnies):
            vector[3 * j] = to_float(params[f"trans{j}"].x)
            vector[3 * j + 1] = to_float(params[f"trans{j}"].y)
            vector[3 * j + 2] = to_float(params[f"angle{j}"])

    def apply_transformations(self, scene_params, params):
        for i in range(self.nb_bunnies):
            params[f"trans{i}"] = dr.clamp(params[f"trans{i}"], self.range_translation[0], self.range_translation[1])
            params[f"angle{i}"] = dr.clamp(params[f"angle{i}"], -1.0, 1.0)
            trafo = mi.Transform4f.translate(
                [params[f"trans{i}"].x, params[f"trans{i}"].y, 0.0]
            ).rotate([0, 1, 0], params[f"angle{i}"] * 180.0)
            scene_params[f"bunny{i}.vertex_positions"] = dr.ravel(
                trafo @ self.initial_vertex_positions[i]
            )
        scene_params.update()

    def initialize_scene(self):
        scene = self.reset_scene()
        params = mi.traverse(scene)
        return scene, params

    def reset_scene(self):
        scene_dict = {
            "type": "scene",
            "integrator": {
                "type": "direct_reparam",
            },
            "sensor": {
                "type": "perspective",
                "to_world": T.look_at(
                    origin=(0, 0, 2), target=(0, 0, 0), up=(0, 1, 0)
                ),
                "fov": 60,
                "film": {
                    "type": "hdrfilm",
                    "width": 256,
                    "height": 256,
                    "rfilter": {"type": "gaussian"},
                    "sample_border": True,
                },
            },
            "wall": {
                "type": "obj",
                "filename": f"{ROOT_DIR}/scenes/meshes/rectangle.obj",
                "to_world": T.translate([0, 0, -2]).scale(2.0),
                "face_normals": True,
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": (0.5, 0.5, 0.5)},
                },
            },
            "light": {
                "type": "obj",
                "filename": f"{ROOT_DIR}/scenes/meshes/sphere.obj",
                "emitter": {
                    "type": "area",
                    "radiance": {"type": "rgb", "value": [1e3, 1e3, 1e3]},
                },
                "to_world": T.translate([2.5, 2.5, 7.0]).scale(0.25),
            },
        }

        if self.nb_bunnies == 1:
            scene_dict["bunny0"] = {
                "type": "ply",
                "filename": f"{ROOT_DIR}/scenes/meshes/bunny.ply",
                "to_world": T.scale(6.5),
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": (0.3, 0.3, 0.75)},
                },
            }
        else:
            # set numpy seed
            np.random.seed(self.scene_seed)

            for i in range(self.nb_bunnies):
                pos_x = np.random.uniform(self.range_translation[0], self.range_translation[1])
                pos_y = np.random.uniform(self.range_translation[0], self.range_translation[1])
                rot = np.random.uniform(0.0, 360.0)
                if self.colored:
                    color = tuple(np.random.uniform(0.0, 1.0, size=3))
                else:
                    color = (0.3, 0.3, 0.75)
                scene_dict[f"bunny{i}"] = {
                    "type": "ply",
                    "filename": f"{ROOT_DIR}/scenes/meshes/bunny.ply",
                    "to_world": T.translate([pos_x, pos_y, 0])
                    .rotate([0, 1, 0], rot)
                    .scale(5.0 / np.sqrt(self.nb_bunnies)),
                    "bsdf": {
                        "type": "diffuse",
                        "reflectance": {"type": "rgb", "value": color},
                    },
                }

        return mi.load_dict(scene_dict)
