import numpy as np


class CMAESHeavyTailMutation:
    def __init__(self, objective_func, dim, k, pareto_a=2.5, q_max=2.0, pop_size=None,
                 x0=None, sigma0=1.0, max_iter=5000, tol=1e-8):
        np.random.seed(42)
        self.f = objective_func
        self.dim = dim
        self.k = k
        self.a = pareto_a
        self.q_max = q_max
        self.max_iter = max_iter
        self.tol = tol

        self.pop_size = pop_size if pop_size else int(4 + np.floor(3 * np.log(dim)))
        self.mu = self.pop_size // 2

        self.m = np.array(x0) if x0 is not None else np.random.randn(dim)
        self.sigma = sigma0

        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.C = np.eye(dim)
        self.invsqrtC = np.eye(dim)

        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights**2)

        self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.c1 = 2 / ((dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((dim + 2)**2 + self.mueff))
        self.damps = 1 + 0.5 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs

        self.iter = 0
        self.last_fitness = float('inf')
        self.best_fitness = float('inf')
        self.best_individual = self.m.copy()

        self.logs = {
            'iter': [],
            'sigma': [],
            'min_fitness': [],
            'best_solution': [],
            'mutation_sizes': [],
            'eigvals': [],
            'diversity': [],
            'mean_fitness': [],
            'best_fitness_ever': []
        }

    def ask(self):
        z = np.random.randn(self.pop_size, self.dim)
        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            epsilons = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
            for eps in epsilons:
                # print(f"Warning: Covariance matrix not PD at iter {self.iter}, adding {eps:.1e}*I")
                self.C += eps * np.eye(self.dim)
                try:
                    L = np.linalg.cholesky(self.C)
                    break
                except np.linalg.LinAlgError:
                    continue
            else:
                # print(f"Cholesky failed at iter {self.iter}, falling back to identity.")
                L = np.eye(self.dim)
        self.offspring = self.m + self.sigma * (z @ L.T)
        return self.offspring

    def _heavy_tail_mutation(self, individuals):
        mutated = individuals.copy()
        mutation_sizes = []
        for x in mutated:
            total_jump = 0.0
            for i in range(self.dim):
                if np.random.rand() < 1.0 / self.dim:
                    q = min(np.random.pareto(self.a) + 1e-10, self.q_max)
                    delta = q if np.random.rand() < 0.5 else -q
                    x[i] += delta
                    total_jump += abs(delta)
            mutation_sizes.append(total_jump)
        return mutated, mutation_sizes

    def tell(self, population, fitness):
        idx = np.argsort(fitness)
        sorted_pop = population[idx]
        sorted_fit = fitness[idx]

        m_old = self.m.copy()

        if sorted_fit[0] < self.best_fitness:
            self.best_fitness = sorted_fit[0]
            self.best_individual = sorted_pop[0].copy()

        # Compute recombined mean as usual
        recombined_mean = np.sum(self.weights[:, np.newaxis] * sorted_pop[:self.mu], axis=0)

        # Blend with best-so-far individual using elitism weight
        elite_weight = 0.3  # You can tune this (e.g., 0.1 to 0.4)

        if sorted_fit[0] < self.best_fitness:
            self.best_fitness = sorted_fit[0]
            self.best_individual = sorted_pop[0].copy()

        # Always blend the mean with the elite solution
        self.m = (1 - elite_weight) * recombined_mean + elite_weight * self.best_individual

        worst_indices = idx[self.mu:]
        mutation_sizes = []
        if len(worst_indices) > 0:
            selected_indices = np.random.choice(worst_indices, min(self.k, len(worst_indices)), replace=False)
            mutated, mutation_sizes = self._heavy_tail_mutation(population[selected_indices])
            for i, ind in enumerate(selected_indices):
                mutated_fit = self.f(mutated[i])
                if mutated_fit < fitness[ind]:
                    population[ind] = mutated[i]
                    fitness[ind] = mutated_fit
                    if mutated_fit < self.best_fitness:
                        self.best_fitness = mutated_fit
                        self.best_individual = mutated[i].copy()

        idx = np.argsort(fitness)
        sorted_pop = population[idx]
        sorted_fit = fitness[idx]

        y = (self.m - m_old) / (self.sigma + 1e-20)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (y @ self.invsqrtC)

        hsig = np.linalg.norm(self.ps) < (1.4 + 2 / (self.dim + 1)) * np.sqrt(self.dim)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y

        artmp = (sorted_pop[:self.mu] - m_old) / (self.sigma + 1e-20)
        C_mu = sum(w * np.outer(ar, ar) for w, ar in zip(self.weights, artmp))
        self.C = ((1 - self.c1 - self.cmu) * self.C +
                  self.c1 * np.outer(self.pc, self.pc) +
                  self.cmu * C_mu)

        self._update_invsqrtC()

        ps_norm = np.linalg.norm(self.ps)
        self.sigma = get_capped_sigma_update(ps_norm, self.sigma, self.cs, self.damps, self.dim)

        self.last_fitness = sorted_fit[0]

        diversity = np.mean([np.linalg.norm(x - self.m) for x in population])
        eigvals = np.linalg.eigvalsh(self.C)

        self.logs['iter'].append(self.iter)
        self.logs['sigma'].append(self.sigma)
        self.logs['min_fitness'].append(sorted_fit[0])
        self.logs['best_solution'].append(self.m.copy())
        self.logs['mutation_sizes'].append(mutation_sizes)
        self.logs['eigvals'].append(eigvals)
        self.logs['diversity'].append(diversity)
        self.logs['mean_fitness'].append(np.mean(fitness))
        self.logs['best_fitness_ever'].append(self.best_fitness)

        self.iter += 1

    def _update_invsqrtC(self):
        self.C = (self.C + self.C.T) / 2
        max_tries = 5
        epsilon = 1e-10

        for i in range(max_tries):
            try:
                eigval, eigvec = np.linalg.eigh(self.C)
                break
            except np.linalg.LinAlgError:
                # print(f"Warning: Eigen decomposition failed at iter {self.iter}, attempt {i+1}. Adding regularization.")
                self.C += epsilon * np.eye(self.dim)
                epsilon *= 10
        else:
            raise np.linalg.LinAlgError("Failed to compute eigendecomposition after regularization attempts.")

        eigval = np.clip(eigval, 1e-20, None)
        self.invsqrtC = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
        self.C = eigvec @ np.diag(eigval) @ eigvec.T

    def optimize(self, problem):
        best_solution = None
        best_fitness = float('inf')
        print(problem.initial_solution, self.f(problem.initial_solution))
        while self.iter < self.max_iter:
            population = self.ask()
            fitness = np.array([self.f(x) for x in population])
            self.tell(population, fitness)

            if self.last_fitness < best_fitness:
                best_fitness = self.last_fitness
                best_solution = self.m.copy()
            if  problem.final_target_hit:
                print(f"Converged at iteration {self.iter}, fitness: {best_fitness:.3e} and problem function {self.f(best_solution)}")
                break
        
        return best_solution, best_fitness, {}, self.logs


def get_capped_sigma_update(ps_norm, sigma, cs, damps, dim, max_exp=5, sigma_max=100):
    """Stable sigma update inspired by PyCMA."""
    exp_arg = min(cs / damps * (ps_norm / np.sqrt(dim) - 1), max_exp)
    sigma *= np.exp(exp_arg)
    return min(sigma, sigma_max)