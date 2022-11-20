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
from pymoo.core.termination import NoTermination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.problems.static import StaticProblem
from tqdm import tqdm

from ..problem import MitsubaProblem
from ..utils import to_float
from .utils import StoppingCriteria


class GlobalOptGrad:
    def __init__(self, problem: MitsubaProblem):
        self.mi_problem = problem
        self.nb_renderings = 0

    def _render_individual(self, scene, params, opt, spp, seed=0) -> mi.Bitmap:
        self.nb_renderings += 1
        self.mi_problem.apply_transformations(params, opt)
        return mi.render(
            scene,
            params,
            seed=seed,
            spp=spp,
        )

    def run(
        self,
        algorithm: Algorithm,
        loss_fn,
        n_gen,
        grad_descent_stopping_criteria: StoppingCriteria,
        lr,
        spp,
        use_grad_opt_as_fitness_measure_only=False,
        seed=0,
        verbose=False,
    ):
        # prepare the algorithm to solve the problem
        algorithm.setup(
            self.mi_problem.problem,
            termination=NoTermination(),
            seed=seed,
            verbose=False,
        )

        np.random.seed(seed)

        losses = []
        best_loss = np.inf
        self.nb_renderings = 0

        progress_bar = tqdm(range(1, n_gen + 1), disable=not verbose)
        for gen in progress_bar:
            # ask the algorithm for the next solution to be evaluated
            pop = algorithm.ask()
            pop_size = len(pop)
            F = np.zeros((pop_size, 1))

            for i, ind in enumerate(pop):
                vector = ind.get("X")

                opt = mi.ad.Adam(lr=lr)
                self.mi_problem.set_params_from_vector(opt, vector)
                scene, params = self.mi_problem.initialize_scene()
                grad_descent_stopping_criteria.reset()

                # optimize the individual using gradient descent
                it = 0
                while True:
                    progress_bar.set_description(
                        f"[Gen. {gen}/{n_gen}]"
                        f" Ind. {i+1}/{pop_size}"
                        f"\tBest loss: {best_loss:.6f}"
                        f"\tNb renderings: {self.nb_renderings}"
                    )
                    img = self._render_individual(
                        scene, params, opt, spp, seed=seed + it * pop_size + i
                    )
                    loss = loss_fn(img)
                    best_loss = min(best_loss, to_float(loss))
                    if grad_descent_stopping_criteria(to_float(loss)):
                        break

                    dr.backward(loss)
                    opt.step()
                    it += 1

                F[i, 0] = to_float(loss)

                if not use_grad_opt_as_fitness_measure_only:
                    self.mi_problem.set_vector_from_params(opt, vector)
                    ind.set("X", vector)

            # store the objective function value
            static = StaticProblem(self.mi_problem.problem, F=F)
            Evaluator().eval(static, pop)
            losses.append(F)

            # returned the evaluated individuals which have been evaluated or even modified
            algorithm.tell(infills=pop)

        # obtain the result objective from the algorithm
        res = algorithm.result()

        return res, losses, self.nb_renderings


class RandomStartGrad:
    def __init__(self, problem: MitsubaProblem):
        self.mi_problem = problem
        self.nb_renderings = 0

    def _render_individual(self, scene, params, opt, spp, seed=0) -> mi.Bitmap:
        self.nb_renderings += 1
        self.mi_problem.apply_transformations(params, opt)
        return mi.render(
            scene,
            params,
            seed=seed,
            spp=spp,
        )

    def run(
        self,
        loss_fn,
        pop_size,
        grad_descent_stopping_criteria: StoppingCriteria,
        lr,
        spp,
        seed=0,
        verbose=False,
    ):
        np.random.seed(seed)

        # prepare the algorithm to solve the problem
        algorithm = RandomAlgorithm(pop_size=pop_size)
        algorithm.setup(
            self.mi_problem.problem,
            termination=NoTermination(),
            seed=seed,
            verbose=False,
        )

        losses = []
        best_loss = np.inf
        self.nb_renderings = 0

        pop = algorithm.ask()
        F = np.zeros((pop_size, 1))

        progress_bar = tqdm(enumerate(pop), disable=not verbose)
        for i, ind in progress_bar:
            vector = ind.get("X")

            opt = mi.ad.Adam(lr=lr)
            self.mi_problem.set_params_from_vector(opt, vector)
            scene, params = self.mi_problem.initialize_scene()
            grad_descent_stopping_criteria.reset()
            losses.append([])

            # optimize the individual using gradient descent
            it = 0
            while True:
                progress_bar.set_description(
                    f"[Ind. {i+1}/{pop_size}]"
                    f" Grad step {it}"
                    f"\tBest loss: {best_loss:.6f}"
                    f"\tNb renderings: {self.nb_renderings}"
                )
                img = self._render_individual(
                    scene, params, opt, spp, seed=seed + it * pop_size + i
                )
                loss = loss_fn(img)

                loss_float = to_float(loss)
                losses[-1].append(loss_float)
                best_loss = min(best_loss, loss_float)

                if grad_descent_stopping_criteria(loss_float):
                    break

                dr.backward(loss)
                opt.step()
                it += 1

            F[i, 0] = loss_float

            self.mi_problem.set_vector_from_params(opt, vector)
            ind.set("X", vector)

        # store the objective function value
        static = StaticProblem(self.mi_problem.problem, F=F)
        Evaluator().eval(static, pop)

        # returned the evaluated individuals which have been evaluated or even modified
        algorithm.tell(infills=pop)

        # obtain the result objective from the algorithm
        res = algorithm.result()

        return res, losses, self.nb_renderings


class RandomAlgorithm(Algorithm):
    def __init__(
        self,
        pop_size=None,
        sampling=FloatRandomSampling(),
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
