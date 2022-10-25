from operator import itemgetter

import drjit as dr
import mitsuba as mi
import numpy as np

mi.set_variant("scalar_rgb")
TENT_RFILTER = mi.load_dict({"type": "tent"})
mi.set_variant("cuda_ad_rgb")


def subdict(d, ks):
    vals = []
    if len(ks) >= 1:
        vals = itemgetter(*ks)(d)
        if len(ks) == 1:
            vals = [vals]
    return dict(zip(ks, vals))


def to_float(x):
    # hacky way to convert Mitsuba float to Python float
    return float(np.array(x))


def image_to_bm(image):
    return mi.util.convert_to_bitmap(image)


def unidim_to_bm(params):
    return mi.util.convert_to_bitmap(params).convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, False
    )


def rel_l1_loss(img, img_ref):
    # Relative L1 loss
    return dr.sum(
        dr.abs(img - img_ref) / (dr.maximum(dr.abs(img_ref), 0.001))
    ) / len(img)


def mse(img, img_ref):
    return dr.mean(dr.sqr(img - img_ref))


LOSS_FNS = {"Relative L1": rel_l1_loss, "MSE": mse}


def upsample(x, final_res):
    if isinstance(final_res, int):
        final_res = (final_res, final_res)

    if not isinstance(x, mi.Bitmap):
        x = mi.Bitmap(x)

    res = mi.TensorXf(
        np.array(x.resample([final_res[0], final_res[1]], TENT_RFILTER))
    )
    if x.channel_count() == 1:
        res = res[..., np.newaxis]
    return res


def generate_rand_tex(
    init_res=(32, 32), opt_res=(512, 512), n_channels=1, seed=0
):
    if isinstance(init_res, int):
        init_res = (init_res, init_res)
    if isinstance(opt_res, int):
        opt_res = (opt_res, opt_res)
    np.random.seed(seed)
    initial_values = mi.Bitmap(
        np.random.uniform(size=(init_res[0], init_res[1], n_channels))
    )

    # Upsample texture to full resolution
    res = mi.TensorXf(
        np.array(
            initial_values.resample([opt_res[0], opt_res[1]], TENT_RFILTER)
        )
    )
    if n_channels == 1:
        res = res[..., np.newaxis]
    return res
