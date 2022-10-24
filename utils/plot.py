from typing import Dict, List

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

from .utils import LOSS_FNS, to_float


def plot_loss_linear_interp_1d(
    img_ref,
    params_1,
    params_2,
    reset_scene_func,
    loss_fns=LOSS_FNS,
    n_samples=50,
    title=None,
    spp=32,
):
    alphas = np.linspace(0, 1, n_samples)
    losses = {key: [] for key in loss_fns.keys()}
    for i, alpha in enumerate(alphas):
        scene = reset_scene_func()
        params = mi.traverse(scene)
        for key in params_1.keys():
            params[key] = (1 - alpha) * params_1[key] + alpha * params_2[key]
        params.update()
        image = mi.render(scene, params, seed=i, spp=spp)
        for key in loss_fns.keys():
            losses[key].append(loss_fns[key](image, img_ref))

    fig, axs = plt.subplots(len(losses), 1, sharex=True)
    for i, key in enumerate(losses.keys()):
        axs[i].plot(alphas, losses[key])
        axs[i].set_ylabel(key)
    if title is not None:
        axs[0].set_title(title)
    plt.show()


def plot_loss_bilinear_interp_2d(
    img_ref,
    params_1,
    params_2,
    params_3,
    params_4,
    reset_scene_func,
    loss_fns=LOSS_FNS,
    n_samples=7,
    title=None,
    spp=32,
):
    alphas = np.linspace(0, 1, n_samples)
    losses = {key: np.zeros((n_samples, n_samples)) for key in loss_fns.keys()}
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
            for key in loss_fns.keys():
                losses[key][i, j] = to_float(loss_fns[key](image, img_ref))

    fig, axs = plt.subplots(1, len(losses), figsize=(len(losses) * 5, 5))
    for i, key in enumerate(losses.keys()):
        axs[i].contourf(alphas, alphas, losses[key])
        axs[i].set_title(key)
    if title is not None:
        fig.suptitle(title)
    plt.show()


def contour_plot_1d(
    img_ref,
    params,
    dir,
    reset_scene_func,
    loss_fns=LOSS_FNS,
    n_samples=100,
    title=None,
    spp=8,
):
    alphas = np.linspace(0, 1, n_samples)
    losses = {key: [] for key in loss_fns.keys()}
    for i, alpha in enumerate(alphas):
        scene = reset_scene_func()
        params_loc = mi.traverse(scene)
        for key in params.keys():
            params_loc[key] = dr.clamp(
                params[key] + alpha * dir[key], 0.0, 1.0
            )
        params_loc.update()
        image = mi.render(scene, params_loc, seed=i, spp=spp)
        for key in loss_fns.keys():
            losses[key].append(loss_fns[key](image, img_ref))

    fig, axs = plt.subplots(len(losses), 1, sharex=True)
    for i, key in enumerate(losses.keys()):
        axs[i].plot(alphas, losses[key])
        axs[i].set_ylabel(key)
    if title is not None:
        axs[0].set_title(title)
    plt.show()


def contour_plot_2d(
    img_ref,
    params,
    dir1,
    dir2,
    reset_scene_func,
    loss_fns=LOSS_FNS,
    n_samples=10,
    title=None,
    spp=8,
):
    # TODO check that the labels are correct
    alphas = np.linspace(0, 1, n_samples)
    losses = {key: np.zeros((n_samples, n_samples)) for key in loss_fns.keys()}
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
            for key in loss_fns.keys():
                losses[key][i, j] = to_float(loss_fns[key](image, img_ref))

    fig, axs = plt.subplots(1, len(losses))
    for i, key in enumerate(losses.keys()):
        axs[i].contourf(alphas, alphas, losses[key])
        axs[i].set_title(key)
    if title is not None:
        axs[0].set_title(title)
    plt.show()


def plot_losses(losses, baseline=None, title=None):
    plt.figure()
    if title is None:
        title = "Loss convergence"
    plt.title(title, weight="bold", size=14)
    for loss in losses:
        plt.semilogy(loss)
    if baseline is not None:
        plt.axhline(y=baseline, color="r", linestyle="-")
    plt.show()
