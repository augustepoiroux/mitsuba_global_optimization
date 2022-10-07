from enum import Enum

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mi.set_variant("scalar_rgb")
TENT_RFILTER = mi.load_dict({"type": "tent"})
mi.set_variant("cuda_ad_rgb")


class Scene(Enum):
    COND1 = 1
    COND2 = 2


ROUGH_KEY = "plane.bsdf.alpha.data"
ENVLIGHT_KEY = "emitter.data"


def reset_scene(scene: Scene):
    return mi.load_file(
        os.path.join(
            ROOT_DIR, f"scenes/roughness_optimization_{scene.value}.xml"
        )
    )


def plot_rough_envlight(
    images_bm, rough_bm, envlight_bm, titles, size_factor=3
):
    fig, ax = plt.subplots(
        ncols=3,
        nrows=len(images_bm),
        figsize=(8, size_factor * len(images_bm)),
        squeeze=False,
    )
    for i, (img, tex, envlight, title) in enumerate(
        zip(images_bm, rough_bm, envlight_bm, titles)
    ):
        ax[i, 0].imshow(img)
        ax[i, 1].imshow(tex)
        ax[i, 2].imshow(envlight)
        ax[i, 0].set_ylabel(title, size=14)
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
        ax[i, 2].set_xticks([])
        ax[i, 2].set_yticks([])
    ax[0, 0].set_title("Rendering", weight="bold", size=14)
    ax[0, 1].set_title("Roughness texture", weight="bold", size=14)
    ax[0, 2].set_title("Environment light", weight="bold", size=14)
    plt.tight_layout()
    plt.show()


def plot_rough_envlight2(res_dict, size_factor=3):
    titles = list(res_dict.keys())
    images_bm = [res_dict[title][0] for title in titles]
    textures_bm = [res_dict[title][1] for title in titles]
    envlights_bm = [res_dict[title][2] for title in titles]
    plot_rough_envlight(
        images_bm, textures_bm, envlights_bm, titles, size_factor
    )


def generate_rand_rough_tex(
    scene_name: Scene, seed=0, init_res=32, opt_res=512
):
    scale = 0.5 if scene_name == Scene.COND1 else 0.3

    # Initialize textured roughness with random values
    np.random.seed(seed)
    initial_values = mi.Bitmap(
        (scale * np.random.uniform(size=init_res**2)).reshape(
            init_res, init_res, -1
        )
    )

    # Upsample texture to full resolution
    return mi.TensorXf(
        np.array(initial_values.resample([opt_res, opt_res], TENT_RFILTER))[
            ..., np.newaxis
        ]
    )


def generate_rand_envlight(scene_name: Scene, seed=0, init_res=32):
    scale = 0.5 if scene_name == Scene.COND1 else 0.3

    # Initialize textured roughness with random values
    np.random.seed(seed)
    initial_values = mi.Bitmap(
        (scale * np.random.uniform(size=2 * init_res**2 * 4)).reshape(
            2 * init_res, init_res, -1
        )
    )

    # Upsample texture to full resolution
    return mi.TensorXf(
        np.array(initial_values.resample([513, 256], TENT_RFILTER))
    )


def generate_rand_init_values(scene_name: Scene, seed=0):
    rough_values = generate_rand_rough_tex(scene_name, seed)
    envlight_values = generate_rand_envlight(scene_name, seed)
    return {ROUGH_KEY: rough_values, ENVLIGHT_KEY: envlight_values}


def get_full_zeros_params():
    rough_values = mi.TensorXf(np.zeros((512, 512, 1)))
    envlight_values = mi.TensorXf(np.zeros((256, 513, 4)))
    return {ROUGH_KEY: rough_values, ENVLIGHT_KEY: envlight_values}


def get_full_ones_params():
    rough_values = mi.TensorXf(np.ones((512, 512, 1)))
    envlight_values = mi.TensorXf(np.ones((256, 513, 4)))
    return {ROUGH_KEY: rough_values, ENVLIGHT_KEY: envlight_values}
