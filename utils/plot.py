import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

from .utils import mse, rel_l1_loss, to_float


def plot_loss_linear_interp_1d(
    img_ref,
    params_1,
    params_2,
    reset_scene_func,
    loss_fn=rel_l1_loss,
    n_samples=50,
    title=None,
    spp=32,
):
    alphas = np.linspace(0, 1, n_samples)
    losses = []
    for i, alpha in enumerate(alphas):
        scene = reset_scene_func()
        params = mi.traverse(scene)
        for key in params_1.keys():
            params[key] = (1 - alpha) * params_1[key] + alpha * params_2[key]
        params.update()
        image = mi.render(scene, params, seed=i, spp=spp)
        losses.append(loss_fn(image, img_ref))
    if title is not None:
        plt.title(title)
    plt.plot(alphas, losses)


def plot_loss_bilinear_interp_2d(
    img_ref,
    params_1,
    params_2,
    params_3,
    params_4,
    reset_scene_func,
    loss_fn=rel_l1_loss,
    n_samples=7,
    title=None,
    spp=32,
):
    alphas = np.linspace(0, 1, n_samples)
    losses = np.zeros((n_samples, n_samples))
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(alphas):
            scene = reset_scene_func()
            params = mi.traverse(scene)
            for key in params_1.keys():
                params[key] = (
                    (1 - alpha) * (1 - beta) * params_1[key]
                    + alpha * (1 - beta) * params_2[key]
                    + (1 - alpha) * beta * params_3[key]
                    + alpha * beta * params_4[key]
                )
            params.update()
            image = mi.render(scene, params, seed=i * len(alphas) + j, spp=spp)
            losses[i, j] = to_float(loss_fn(image, img_ref))
    if title is not None:
        plt.title(title)
    plt.contourf(alphas, alphas, losses)
    plt.colorbar()


def contour_plot_1d(
    img_ref,
    params,
    dir,
    reset_scene_func,
    loss_fn=rel_l1_loss,
    n_samples=100,
    title=None,
    spp=8,
):
    alphas = np.linspace(0, 1, n_samples)
    losses = []
    for i, alpha in enumerate(alphas):
        scene = reset_scene_func()
        params_loc = mi.traverse(scene)
        for key in params.keys():
            params_loc[key] = dr.clamp(
                params[key] + alpha * dir[key], 0.0, 1.0
            )
        params_loc.update()
        image = mi.render(scene, params_loc, seed=i, spp=spp)
        losses.append(loss_fn(image, img_ref))
    if title is not None:
        plt.title(title)
    plt.plot(alphas, losses)


def contour_plot_2d(
    img_ref,
    params,
    dir1,
    dir2,
    reset_scene_func,
    loss_fn=rel_l1_loss,
    n_samples=10,
    title=None,
    spp=8,
):
    # TODO check that the labels are correct
    alphas = np.linspace(0, 1, n_samples)
    losses = np.zeros((n_samples, n_samples))
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(alphas):
            scene = reset_scene_func()
            params_loc = mi.traverse(scene)
            for key in params.keys():
                params_loc[key] = dr.clamp(
                    params[key] + alpha * dir1[key] + beta * dir2[key],
                    0.0,
                    1.0,
                )
            params_loc.update()
            image = mi.render(
                scene, params_loc, seed=i * len(alphas) + j, spp=spp
            )
            losses[i, j] = to_float(loss_fn(image, img_ref))
    if title is not None:
        plt.title(title)
    plt.xlabel("dir1")
    plt.ylabel("dir2")
    plt.contourf(alphas, alphas, losses)
    plt.colorbar()


def plot_losses(losses, baseline=None):
    plt.figure()
    for loss in losses:
        plt.semilogy(loss)
    if baseline is not None:
        plt.semilogy(baseline * np.ones(len(losses[0])))
    plt.title("Loss convergence", weight="bold", size=14)
    plt.show()
