import numpy as np
import random
from cocoex import Suite, Observer

class BeamSearchES:
    def __init__(self, dim, lambda_, initial_sigma, bounds, max_evals, 
                 p_heavy_jump=0.1, p_beam=0.2, pareto_alpha=1.5):
        """
        (1 + λ)-ES with Heavy-Tailed Jumps and Beam Search (always long jump).
        """
        self.dim = dim
        self.lambda_ = lambda_
        self.initial_sigma = initial_sigma
        self.bounds = bounds
        self.max_evals = max_evals
        self.p_heavy_jump = p_heavy_jump
        self.p_beam = p_beam
        self.pareto_alpha = pareto_alpha
        self.history_fitness = []
        self.history_candidates = []
        self.valley_centers = []
        self.evaluations = 0

    def random_unit_vector(self):
        v = np.random.randn(self.dim)
        return v / np.linalg.norm(v)

    def determine_valley(self, x):
        valley_center = np.zeros(self.dim)
        distance = np.linalg.norm(x - valley_center)
        return valley_center, distance

    def optimize(self, problem):
        x_parent = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        sigma = np.full(self.dim, self.initial_sigma)
        
        best_fitness = problem(x_parent)
        self.history_fitness.append(best_fitness)
        self.evaluations += 1
        
        valley_center, distance = self.determine_valley(x_parent)
        self.history_candidates.append(distance)
        self.valley_centers.append(valley_center)

        while self.evaluations < self.max_evals and not problem.final_target_hit:
            offspring = []
            offspring_fitness = []

            for _ in range(self.lambda_):
                r = random.random()
                if r < self.p_beam:  # Beam search always does a long jump
                    dir = self.random_unit_vector()
                    jump_size = np.random.pareto(self.pareto_alpha)
                    x_off = x_parent + jump_size * dir
                elif r < self.p_beam + self.p_heavy_jump:  # Heavy-tailed jump
                    sigma_heavy = np.random.pareto(self.pareto_alpha)
                    z = np.random.standard_cauchy(self.dim)
                    x_off = x_parent + sigma_heavy * z
                else:  # Regular Gaussian step
                    z = np.random.randn(self.dim)
                    x_off = x_parent + sigma * z

                x_off = np.clip(x_off, self.bounds[0], self.bounds[1])
                offspring.append(x_off)
                offspring_fitness.append(problem(x_off))
                self.evaluations += 1

            parent_fitness = problem(x_parent)
            self.evaluations += 1

            all_candidates = offspring + [x_parent]
            all_fitness = offspring_fitness + [parent_fitness]
            best_idx = np.argmin(all_fitness)
            best_child = all_candidates[best_idx]

            if all_fitness[best_idx] < self.history_fitness[-1]:
                x_parent = best_child.copy()
                self.history_fitness.append(all_fitness[best_idx])
                sigma *= 1.5  # Original step-size scaling
            else:
                self.history_fitness.append(self.history_fitness[-1])
                sigma *= 0.95

            if self.dim == 2:
                self.history_candidates.append(x_parent)
            else:
                _, distance = self.determine_valley(x_parent)
                self.history_candidates.append(distance)
            self.valley_centers.append(self.determine_valley(x_parent)[0])

            if problem.final_target_hit:
                print('Reached the target')
                break

        problem(x_parent)
        return x_parent, self.history_fitness[-1]

def run_benchmark():
    """
    Run the algorithm on all BBOB functions using COCO.
    """
    suite_name = "bbob"
    suite_options = "dimensions:20 function: 1-24, instance_indices:1-15"  # All 24 functions, 15 instances
    suite = Suite(suite_name, "", suite_options)
    observer_options = "result_folder:ESwBeam"
    observer = Observer(suite_name, observer_options)  
    
    for problem in suite:
        print(f"Running {problem.id}")
        problem.observe_with(observer)
        budget = problem.dimension*1e5  # Standard BBOB budget: 10^4 * n for n=2
        solver = BeamSearchES(
            dim=problem.dimension,
            lambda_=2*problem.dimension,
            initial_sigma=5.0,
            bounds=(-5, 5),
            max_evals=budget,
            p_heavy_jump=0.1,
            p_beam=0.2,
            pareto_alpha=2.5
        )
        
        sol, fit = solver.optimize(problem)
        
        print(f"  Iters: {len(solver.history_fitness)}, Fitness: {fit}, Evals: {problem.evaluations}")
    
    print("Benchmark complete. Results in '1LambdaES'.")
    print("Post-process: python3 -m cocopp StepAdaptationES_experiment  # or python2 cocopp.py StepAdaptationES_experiment")

if __name__ == "__main__":
    np.random.seed(42)
    run_benchmark()