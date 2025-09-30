import numpy as np
from cocoex import Suite, Observer
import random

class StepAdaptationES:
    def __init__(self, dim, lambda_, initial_sigma, bounds, max_evals, p_heavy_jump=0.3, pareto_alpha=3.0):
        """
        (1 + λ)-ES with Hybrid (Global and Individual) Step Size Adaptation.

        Parameters:
            dim: Dimensionality of the search space
            lambda_: Number of offspring
            initial_sigma: Initial individual step size
            bounds: Tuple, (lower, upper) bounds for the search space
            max_evals: Maximum number of function evaluations (budget)
            target: Target fitness precision (e.g., 1e-2 for BBOB)
            p_heavy_jump: Probability of heavy-tailed (Cauchy) jump
            pareto_alpha: Shape parameter for Pareto distribution in heavy jumps
        """
        self.dim = dim
        self.lambda_ = lambda_
        self.initial_sigma = initial_sigma
        self.bounds = bounds
        self.max_evals = max_evals
        self.p_heavy_jump = p_heavy_jump
        self.pareto_alpha = pareto_alpha
        self.history_fitness = []  # Track best fitness over time
        self.history_candidates = []  # Track distances to valley or candidate positions
        self.valley_centers = []  # Track valley centers
        self.evaluations = 0  # Track total evaluations

    def random_unit_vector(self):
        """Generate a random unit vector (not used here, but kept for consistency)."""
        v = np.random.randn(self.dim)
        return v / np.linalg.norm(v)

    def determine_valley(self, x):
        """
        Determine the valley center (global minimum at [0, 0, ..., 0] for simplicity)
        and Euclidean distance to it.
        """
        valley_center = np.zeros(self.dim)  # Global minimum at origin for this example
        distance = np.linalg.norm(x - valley_center)
        return valley_center, distance

    def optimize(self, problem):
        """
        Optimize the given problem (COCO problem or function) using (1 + λ)-ES.

        Parameters:
            problem: COCO Problem object or callable fitness function
        """
        # Initialize parent solution
        x_parent = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        individual_sigmas = [self.initial_sigma for _ in range(self.lambda_)]
        
        # Initial evaluation
        best_fitness = problem(x_parent)
        self.history_fitness.append(best_fitness)
        self.evaluations += 1
        
        # Initial monitoring
        valley_center, distance = self.determine_valley(x_parent)
        self.history_candidates.append(distance)
        self.valley_centers.append(valley_center)

        iter = 0
        while self.evaluations < self.max_evals and not problem.final_target_hit:
            offspring = []
            offspring_fitness = []

            # Generate offspring
            for j in range(self.lambda_):
                new = []
                for i in range(self.dim):
                    if random.random() > self.p_heavy_jump:  # 70% chance for normal mutation
                        sigma = max(individual_sigmas[j], 1e-100)  # Prevent too small sigma
                        new.append(x_parent[i] + sigma * np.random.normal())
                    else:  # 30% chance for heavy-tailed (Cauchy) jump
                        sigma = np.random.pareto(self.pareto_alpha)
                        new.append(x_parent[i] + sigma * np.random.standard_cauchy())
                # Enforce bounds
                child = np.clip(new, self.bounds[0], self.bounds[1])
                offspring.append(child)
                offspring_fitness.append(problem(child))
                self.evaluations += 1

            # Evaluate parent
            parent_fitness = problem(x_parent)
            self.evaluations += 1

            # Select best (1+λ)
            all_candidates = offspring + [x_parent]
            all_fitness = offspring_fitness + [parent_fitness]
            best_idx = np.argmin(all_fitness)
            best_child = all_candidates[best_idx]

            # Update parent if better
            if all_fitness[best_idx] < self.history_fitness[-1]:
                x_parent = best_child.copy()
                self.history_fitness.append(all_fitness[best_idx])
            else:
                self.history_fitness.append(self.history_fitness[-1])

            # Update monitoring
            if self.dim == 2:
                self.history_candidates.append(x_parent)  # Store position for 2D
            else:
                _, distance = self.determine_valley(x_parent)
                self.history_candidates.append(distance)
            self.valley_centers.append(self.determine_valley(x_parent)[0])

            # Update individual step sizes
            for k in range(self.lambda_):
                if offspring_fitness[k] < self.history_fitness[-1]:
                    individual_sigmas[k] *= 1.15  # Increase for successful
                else:
                    individual_sigmas[k] *= 0.95  # Decrease for unsuccessful

            if problem.final_target_hit:
                print('reach the target')
                break
            iter += 1

            # Optional early stopping (heuristic, not used for COCO logging)
            # if iter > 10 and abs(self.history_fitness[-1] - self.history_fitness[-2]) < self.target:
            #     print(f"Precision {self.target} reached at iter {iter}")
            #     break


        # Final evaluation for logging
        problem(x_parent)
        return x_parent, self.history_fitness[-1]

def run_benchmark():
    """
    Run the algorithm on all BBOB functions using COCO.
    """
    suite_name = "bbob"
    suite_options = "dimensions:40 function_indices:13-24 instance_indices:1-15"  # All 24 functions, 15 instances
    suite = Suite(suite_name, "", suite_options)
    observer_options = "result_folder:1LambdaES"
    observer = Observer(suite_name, observer_options)  
    
    for problem in suite:
        print(f"Running {problem.id}")
        problem.observe_with(observer)
        budget = problem.dimension*1e5  # Standard BBOB budget: 10^4 * n for n=2
        es = StepAdaptationES(
            dim=problem.dimension,
            lambda_=int(2*problem.dimension),  # Adjusted for better exploration
            initial_sigma=5.0,  # Larger initial step size
            bounds=(-5, 5),  # BBOB domain
            max_evals=budget,
            p_heavy_jump=0.1,
            pareto_alpha=2.5
        )
        
        sol, fit = es.optimize(problem)
        
        print(f"  Iters: {len(es.history_fitness)}, Fitness: {fit}, Evals: {problem.evaluations}")
    
    print("Benchmark complete. Results in '1LambdaES'.")
    print("Post-process: python3 -m cocopp StepAdaptationES_experiment  # or python2 cocopp.py StepAdaptationES_experiment")

if __name__ == "__main__":
    np.random.seed(42)
    run_benchmark()