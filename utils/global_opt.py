from abc import ABC, abstractmethod

import drjit as dr
import mitsuba as mi
import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.duplicate import (
    DefaultDuplicateElimination,
    NoDuplicateElimination,
)
from pymoo.core.evaluator import Evaluator
from pymoo.core.initialization import Initialization
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from tqdm import tqdm

from .utils import image_to_bm, to_float


class StoppingCriteria(ABC):
    @abstractmethod
    def __call__(self, loss: float) -> bool:
        pass


class IterationStoppingCriteria(StoppingCriteria):
    def __init__(self, max_iter: int):
        self.max_iter = max_iter
        self.iter = 0

    def __call__(self, loss: float) -> bool:
        self.iter += 1
        return self.iter >= self.max_iter


class NotImprovingStoppingCriteria(StoppingCriteria):
    def __init__(self, max_iter: int, eps: float = 0.0):
        self.max_iter = max_iter
        self.eps = eps
        self.iter = 0
        self.best_loss = None

    def __call__(self, loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = loss
            return False

        self.best_loss = min(self.best_loss, loss)

        if loss - self.best_loss > self.eps:
            self.iter += 1
        else:
            self.iter = 0

        return self.iter >= self.max_iter


class GlobalOptGrad(ABC):
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

    def render_individual(self, x, spp, seed=0) -> mi.Bitmap:
        scene, params = self.initialize_scene()
        opt = {}
        self.set_params_from_vector(opt, x)
        self.apply_transformations(params, opt)
        image = mi.render(scene, params, seed=seed, spp=spp)
        return image_to_bm(image)

    def run(
        self,
        algorithm: Algorithm,
        loss_fn,
        n_gen,
        grad_descent_stopping_criteria: StoppingCriteria,
        lr,
        spp,
        seed=0,
        verbose=False,
    ):
        # prepare the algorithm to solve the problem
        algorithm.setup(
            self.problem,
            termination=NoTermination(),
            seed=seed,
            verbose=False,
        )

        np.random.seed(seed)

        losses = []
        best_loss = np.inf
        nb_renders = 0

        progress_bar = tqdm(range(1, n_gen + 1), disable=not verbose)
        for gen in progress_bar:
            # ask the algorithm for the next solution to be evaluated
            pop = algorithm.ask()
            pop_size = len(pop)
            F = np.zeros((pop_size, 1))

            for i, ind in enumerate(pop):
                vector = ind.get("X")

                opt = mi.ad.Adam(lr=lr)
                self.set_params_from_vector(opt, vector)
                scene, params = self.initialize_scene()

                # optimize the individual using gradient descent
                it = 0
                while True:
                    progress_bar.set_description(
                        f"[Gen. {gen}/{n_gen}"
                        f" Ind. {i+1}/{pop_size}"
                        f"\tBest loss: {best_loss:.6f}"
                        f"\tNb renderings: {nb_renders}"
                    )
                    self.apply_transformations(params, opt)
                    img = mi.render(
                        scene,
                        params,
                        seed=seed + it * pop_size + i,
                        spp=spp,
                    )
                    nb_renders += 1
                    loss = loss_fn(img)
                    dr.backward(loss)
                    opt.step()
                    it += 1
                    if grad_descent_stopping_criteria(to_float(loss)):
                        break

                # compute the objective function value
                progress_bar.set_description(
                    f"[Gen. {gen}/{n_gen}"
                    f" Ind. {i+1}/{pop_size}"
                    f"\tBest loss: {best_loss:.6f}"
                    f"\tNb renderings: {nb_renders}"
                )
                self.apply_transformations(params, opt)
                final_img = mi.render(
                    scene,
                    params,
                    seed=seed,
                    spp=spp,
                )
                nb_renders += 1
                loss = to_float(loss_fn(final_img))
                best_loss = min(best_loss, loss)
                F[i, 0] = loss

                self.set_vector_from_params(opt, vector)
                ind.set("X", vector)

            # store the objective function value
            static = StaticProblem(self.problem, F=F)
            Evaluator().eval(static, pop)
            losses.append(F)

            # returned the evaluated individuals which have been evaluated or even modified
            algorithm.tell(infills=pop)

        # obtain the result objective from the algorithm
        res = algorithm.result()

        return res, losses


class RandomAlgorithm(Algorithm):
    def __init__(
        self,
        pop_size=None,
        sampling=None,
        eliminate_duplicates=DefaultDuplicateElimination(),
        **kwargs,
    ):

        super().__init__(**kwargs)

        # the population size used
        self.pop_size = pop_size

        # set the duplicate detection class - a boolean value chooses the default duplicate detection
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        self.initialization = Initialization(
            sampling,
            eliminate_duplicates=self.eliminate_duplicates,
        )

        # other run specific data updated whenever solve is called - to share them in all algorithms
        self.n_gen = None
        self.pop = None
        self.off = None

    def _initialize_infill(self):
        pop = self.initialization.do(
            self.problem, self.pop_size, algorithm=self
        )
        return pop
