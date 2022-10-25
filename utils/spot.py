import os
from enum import Enum

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

from utils.utils import image_to_bm, rel_l1_loss, unidim_to_bm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mi.set_variant("cuda_ad_rgb")

TEX_KEY = "cow_tex.data"

def reset_scene():
    return mi.load_file(os.path.join(ROOT_DIR, f"scenes/spot.xml"))
