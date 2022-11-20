from abc import ABC, abstractmethod
import numpy as np
import mitsuba as mi
from pymoo.core.problem import Problem
from .utils import image_to_bm


class MitsubaProblem(ABC):
    def __init__(self, n_var, xl, xu):
        self.problem = Problem(n_var=n_var, n_obj=1, xl=xl, xu=xu)

    @abstractmethod
    def set_params_from_vector(self, params, vector):
        raise NotImplementedError

    @abstractmethod
    def set_vector_from_params(self, params, vector):
        raise NotImplementedError

    @abstractmethod
    def apply_transformations(self, scene_params, params):
        raise NotImplementedError

    @abstractmethod
    def initialize_scene(self):
        raise NotImplementedError

    def _clip_vector(self, x):
        return np.clip(x, self.problem.xl, self.problem.xu)

    def render_individual(self, x, spp, seed=0) -> mi.Bitmap:
        scene, params = self.initialize_scene()
        opt = {}
        self.set_params_from_vector(opt, x)
        self.apply_transformations(params, opt)
        image = mi.render(scene, params, seed=seed, spp=spp)
        return image, image_to_bm(image)

    def render(self, spp, seed=0):
        scene, params = self.initialize_scene()
        image = mi.render(scene, params, seed=seed, spp=spp)
        return image, image_to_bm(image)
