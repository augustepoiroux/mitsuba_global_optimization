from operator import itemgetter

import drjit as dr
import mitsuba as mi
import numpy as np


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
