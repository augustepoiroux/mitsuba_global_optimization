import drjit as dr
import mitsuba as mi
import numpy as np
from tqdm import tqdm

from cmaes import CMA
from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization


from ..problem import MitsubaProblem
from ..utils import to_float


class BOLeap:
    # Bayesian Optimization with Semi-local Leaps
    # https://arxiv.org/pdf/2207.00167.pdf

    def __init__(self, problem: MitsubaProblem):
        self.mi_problem = problem

    def _bo_params_bounds(self):
        return {
            f"x{i}": (
                self.mi_problem.problem.xl[i],
                self.mi_problem.problem.xu[i],
            )
            for i in range(self.mi_problem.problem.n_var)
        }

    def _vector_to_bo_params(self, x):
        return {f"x{i}": x[i] for i in range(self.mi_problem.problem.n_var)}

    def _bo_params_to_vector(self, bo_params):
        return np.array(
            [bo_params[f"x{i}"] for i in range(self.mi_problem.problem.n_var)]
        )

    def run(
        self,
        loss_fn,
        max_steps,
        local_steps,
        nb_grad_steps,
        lr,
        spp,
        pop_size=10,
        num_select_best=5,
        lcb_kappa=2.5,
        lcb_xi=0.0,
        seed=0,
        verbose=False,
    ):
        np.random.seed(seed)
        nb_renderings = 0

        def func_opt(x, opt=None, seed=0):
            nonlocal nb_renderings
            scene, params = self.mi_problem.initialize_scene()

            backward = opt is not None
            if not backward:
                opt = {}

            self.mi_problem.set_params_from_vector(opt, x)
            self.mi_problem.apply_transformations(params, opt)

            img = mi.render(
                scene,
                params,
                seed=seed,
                spp=spp,
            )
            nb_renderings += 1
            loss = loss_fn(img)

            if backward:
                dr.backward(loss)
                opt.step()

            return to_float(loss)

        bo_optimizer = BayesianOptimization(
            f=None,
            pbounds=self._bo_params_bounds(),
            verbose=2,
            random_state=seed,
        )
        seen_points = set()

        utility = UtilityFunction(kind="ucb", kappa=lcb_kappa, xi=lcb_xi)

        best_loss = np.inf
        best_ind = None
        losses = []

        progress_bar = tqdm(range(1, max_steps + 1), disable=not verbose)
        for step in progress_bar:
            x_step_bo = bo_optimizer.suggest(utility)
            x_step = self._bo_params_to_vector(x_step_bo)
            cma_optimizer = CMA(
                mean=x_step,
                sigma=1.0,
                bounds=np.stack(
                    [self.mi_problem.problem.xl, self.mi_problem.problem.xu]
                ).T,
                population_size=pop_size,
                seed=seed + step,
            )

            for local_step in range(local_steps):
                x_local_step = np.zeros(
                    (cma_optimizer.population_size, x_step.shape[0])
                )
                loss_local_step = np.zeros(pop_size)
                for i in range(cma_optimizer.population_size):
                    x = cma_optimizer.ask()
                    x = self.mi_problem._clip_vector(x)
                    x_local_step[i, :] = x
                    loss_local_step[i] = func_opt(
                        x,
                        seed=seed + step * local_steps + local_step + i,
                    )

                    if loss_local_step[i] < best_loss:
                        best_loss = loss_local_step[i]
                        best_ind = x
                    progress_bar.set_description(
                        f"[Step. {step}/{max_steps}"
                        f" Local step. {local_step+1}/{local_steps}"
                        f" Ind. {i+1}/{pop_size}"
                        f"\tBest loss: {best_loss:.6f}"
                        f"\tNb renderings: {nb_renderings}"
                    )

                    # add the new observation to the BO optimizer
                    if not x.tobytes() in seen_points:
                        seen_points.add(x.tobytes())
                        bo_optimizer.register(
                            params=self._vector_to_bo_params(x),
                            target=-loss_local_step[i],
                        )

                best_idx = np.argsort(loss_local_step)[:num_select_best]
                x_mean = np.mean(x_local_step[best_idx], axis=0)

                opt = mi.ad.Adam(lr=lr)
                for i in range(1, nb_grad_steps + 1):
                    loss = func_opt(x_mean, opt=opt)
                    losses.append(loss)
                    if loss < best_loss:
                        best_loss = loss
                        best_ind = x_mean
                    self.mi_problem.set_vector_from_params(opt, x_mean)
                    x_mean = self.mi_problem._clip_vector(x_mean)
                    if not x_mean.tobytes() in seen_points:
                        seen_points.add(x_mean.tobytes())
                        bo_optimizer.register(
                            params=self._vector_to_bo_params(x_mean),
                            target=-loss,
                        )
                    progress_bar.set_description(
                        f"[Step. {step}/{max_steps}"
                        f" Local step. {local_step+1}/{local_steps}"
                        f" Grad step. {i}/{nb_grad_steps}"
                        f"\tBest loss: {best_loss:.6f}"
                        f"\tNb renderings: {nb_renderings}"
                    )

                # update CMA-ES parameters
                # this operation is not clear in the paper
                # use strategy in paper: "Combining Evolution Strategy and Gradient Descent Method for Discriminative Learning of Bayesian Classifiers"
                cma_optimizer._mean = x_mean
                cma_optimizer.tell(
                    [(x, l) for x, l in zip(x_local_step, loss_local_step)]
                )

        return best_ind, losses, nb_renderings
