import os
from enum import Enum

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

from utils.utils import image_to_bm, rel_l1_loss, unidim_to_bm

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


reset_func = {_k: (lambda _k=_k: reset_scene(_k)) for _k in Scene}


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
        # TODO set vmin vmax
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


def upsample(x, final_res):
    return mi.TensorXf(
        np.array(x.resample([final_res, final_res], TENT_RFILTER))[
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


def run_opt_set_init(
    img_ref,
    init_values,
    reset_scene_func,
    optimizer_name="sgd",
    lr=120.0,
    loss_fn=rel_l1_loss,
    n_iterations=100,
    spp_primal=32,
    spp_grad=4,
):
    losses = []
    image_bm_init = []
    image_bm_end = []
    tex_bm_init = []
    tex_bm_end = []
    envlight_bm_init = []
    envlight_bm_end = []
    params_end = []

    nb_opt_samples = len(init_values)

    for (opt_sample, dict_init_values) in enumerate(init_values):
        # Load scene
        scene = reset_scene_func()

        params = mi.traverse(scene)
        params[ROUGH_KEY] = dict_init_values[ROUGH_KEY]
        params[ENVLIGHT_KEY] = dict_init_values[ENVLIGHT_KEY]
        params.update()

        # Standard stochastic gradient descent optimizer
        if optimizer_name.lower() == "sgd":
            opt = mi.ad.optimizers.SGD(lr=lr)
        elif optimizer_name.lower() == "adam":
            opt = mi.ad.optimizers.Adam(lr=lr)
        else:
            raise ValueError(f"Unknown optimizer {optimizer_name}")

        opt[ROUGH_KEY] = params[ROUGH_KEY]
        opt[ENVLIGHT_KEY] = params[ENVLIGHT_KEY]
        params.update(opt)

        losses.append([])
        for it in range(n_iterations):
            image = mi.render(
                scene,
                params,
                seed=it * nb_opt_samples + opt_sample,
                spp=spp_primal,
                spp_grad=spp_grad,
            )

            if it == 0:
                image_bm_init.append(image_to_bm(image))
                tex_bm_init.append(unidim_to_bm(params[ROUGH_KEY]))
                envlight_bm_init.append(image_to_bm(params[ENVLIGHT_KEY]))

            # Apply loss function
            loss = loss_fn(image, img_ref)

            # Backpropagate
            dr.backward(loss)

            # Optimizer: take a gradient step
            opt.step()
            opt[ROUGH_KEY] = dr.clamp(opt[ROUGH_KEY], 1e-2, 1.0)
            opt[ENVLIGHT_KEY] = dr.clamp(opt[ENVLIGHT_KEY], 0.0, 1.0)

            # Optimizer: Update the scene parameters
            params.update(opt)

            if it == n_iterations - 1:
                image_bm_end.append(image_to_bm(image))
                tex_bm_end.append(unidim_to_bm(params[ROUGH_KEY]))
                envlight_bm_end.append(image_to_bm(params[ENVLIGHT_KEY]))
                params_end.append(
                    {
                        ROUGH_KEY: mi.TensorXf(params[ROUGH_KEY]),
                        ENVLIGHT_KEY: params[ENVLIGHT_KEY],
                    }
                )

            print(
                f"[Sample {opt_sample+1}/{nb_opt_samples}]  Iteration {it:03d}: loss={loss[0]:.5f}",
                end="\r",
            )
            losses[opt_sample].append(loss)
    return {
        "losses": losses,
        "image_bm_init": image_bm_init,
        "image_bm_end": image_bm_end,
        "tex_bm_init": tex_bm_init,
        "tex_bm_end": tex_bm_end,
        "envlight_bm_init": envlight_bm_init,
        "envlight_bm_end": envlight_bm_end,
        "params_end": params_end,
    }


def plot_opt_results(results, refs, nb_results=2, size_factor=3):
    titles = (
        [f"Initial {i}" for i in range(nb_results)]
        + [f"Final {i}" for i in range(nb_results)]
        + ["Reference"]
    )
    images_bm = (
        [results["image_bm_init"][i] for i in range(nb_results)]
        + [results["image_bm_end"][i] for i in range(nb_results)]
        + [refs["img"]]
    )
    textures_bm = (
        [results["tex_bm_init"][i] for i in range(nb_results)]
        + [results["tex_bm_end"][i] for i in range(nb_results)]
        + [refs["params_rough"]]
    )
    envlights_bm = (
        [results["envlight_bm_init"][i] for i in range(nb_results)]
        + [results["envlight_bm_end"][i] for i in range(nb_results)]
        + [refs["params_envlight"]]
    )
    plot_rough_envlight2(
        {
            titles[i]: (images_bm[i], textures_bm[i], envlights_bm[i])
            for i in range(len(titles))
        },
        size_factor=size_factor,
    )
