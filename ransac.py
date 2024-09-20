import random
from math import ceil


class RANSAC:
    def __init__(self, solver, is_inlier_checker):
        self.solver = solver
        self.is_inlier_checker = is_inlier_checker

    def run(
        self,
        iterations: int,
        samples: list,
        needed_sample_count: int,
        wanted_inlier_proportion: float,
    ):
        def print_inlier_proportion(inlier_set):
            print(f"Inlier proportion: {len(inlier_set) / len(samples):.2f}")

        largest_inlier_set = None
        wanted_inliers = ceil(len(samples) * wanted_inlier_proportion)
        print(wanted_inliers)

        for i in range(iterations):
            print(f"RANSAC {i+1}/{iterations}")
            current_samples = random.sample(samples, needed_sample_count)
            solution = self.solver(current_samples)
            if not solution:
                continue
            inliers = [x for x in samples if self.is_inlier_checker(x, solution)]

            if len(inliers) >= wanted_inliers:
                print("RANSAC early stop")
                print_inlier_proportion(inliers)
                return self.solver(inliers), inliers

            if largest_inlier_set is None or len(inliers) > len(largest_inlier_set):
                largest_inlier_set = inliers

        print(f"RANSAC completed {iterations} iterations.")
        print_inlier_proportion(largest_inlier_set)
        return self.solver(largest_inlier_set), largest_inlier_set
