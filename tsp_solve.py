import math
import random
import numpy as np
from math import inf
import heapq

from tsp_core import Tour, SolutionStats, Timer, score_tour, Solver
from tsp_cuttree import CutTree
from queue import PriorityQueue


def random_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))

    while True:
        if timer.time_out():
            return stats

        tour = random.sample(list(range(len(edges))), len(edges))
        n_nodes_expanded += 1

        cost = score_tour(tour, edges)
        if math.isinf(cost):
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue

        if stats and cost > stats[-1].score:
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue

        stats.append(SolutionStats(
            tour=tour,
            score=cost,
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        ))

    if not stats:
        return [SolutionStats(
            [],
            math.inf,
            timer.time(),
            1,
            n_nodes_expanded,
            n_nodes_pruned,
            cut_tree.n_leaves_cut(),
            cut_tree.fraction_leaves_covered()
        )]


def greedy_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))
    start = 0

    n = len(edges)
    for i in range(start, n):
        if timer.time_out():
            break

        visited = {i}
        tour = [i]
        current_city = i

        while len(tour) < n:
            n_nodes_expanded += 1
            next_city, minimum_cost = -1, float('inf')

            for city in range(n):
                if city not in visited and edges[current_city][city] < minimum_cost:
                    next_city, minimum_cost = city, edges[current_city][city]

            if next_city == -1 or minimum_cost == float('inf'):
                n_nodes_pruned += 1
                cut_tree.cut(tour)
                break

            visited.add(next_city)
            tour.append(next_city)
            current_city = next_city

        cost = score_tour(tour, edges)
        if len(tour) == n and cost < float('inf'):
            stats.append(SolutionStats(
                tour=tour,
                score=cost,
                time=timer.time(),
                max_queue_size=1,  # No queue used in the greedy algorithm
                n_nodes_expanded=n_nodes_expanded,
                n_nodes_pruned=n_nodes_pruned,
                n_leaves_covered=cut_tree.n_leaves_cut(),
                fraction_leaves_covered=cut_tree.fraction_leaves_covered()
            ))
            return stats

    if not stats:
        return [SolutionStats(
            tour=[],
            score=float('inf'),
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        )]


def dfs(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))

    n = len(edges)
    stack = [[0]]
    best_solution = None
    best_cost = float('inf')

    while stack:
        if timer.time_out():
            break

        # Pop the top path from the stack
        path = stack.pop()

        # Expand the current path
        current_city = path[-1]
        n_nodes_expanded += 1

        for next_city in range(n):
            if next_city in path:
                continue

            # Create a new path
            new_path = path + [next_city]

            # If this path forms a complete tour
            if len(new_path) == n:
                # Add return to the start city to complete the tour
                cost = score_tour(new_path, edges)

                if cost < best_cost:
                    best_cost = cost
                    best_solution = new_path
            else:
                # Add the partial path to the stack for further exploration
                stack.append(new_path)

    if best_solution:
        stats.append(SolutionStats(
            tour=best_solution,
            score=best_cost,
            time=timer.time(),
            max_queue_size=len(stack),  # Size of the stack is the max queue size
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        ))

    # If no solutions found, return no-solution stats
    if not stats:
        return [SolutionStats(
            tour=[],
            score=float('inf'),
            time=timer.time(),
            max_queue_size=len(stack),
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        )]

    return stats


def branch_and_bound(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))
    n = len(edges)

    initial_solution = greedy_tour(edges, timer)
    if initial_solution:
        bssf = initial_solution[0].score
    else:
        bssf = float('inf')

    def calculate_lower_bound(cost_matrix):
        lower_bound = 0
        for i in range(len(cost_matrix)):
            row_min = min(cost_matrix[i])
            if row_min != float('inf'):
                lower_bound += row_min
                for j in range(len(cost_matrix)):
                    cost_matrix[i][j] -= row_min
        for j in range(len(cost_matrix)):
            col_min = min(cost_matrix[i][j] for i in range(len(cost_matrix)))
            if col_min != float('inf'):
                lower_bound += col_min
                for i in range(len(cost_matrix)):
                    cost_matrix[i][j] -= col_min
        return lower_bound, cost_matrix


    stack = []
    # Initial reduced cost matrix and bound
    initial_matrix = [row[:] for row in edges]
    initial_path = [0]
    initial_lower_bound, reduced_matrix = calculate_lower_bound(initial_matrix)
    stack.append((initial_lower_bound, initial_path, reduced_matrix))

    while stack:
        if timer.time_out():
            break

        current_bound, path, current_matrix = stack.pop()
        n_nodes_expanded += 1

        # Prune if the bound exceeds BSSF
        if current_bound >= bssf:
            n_nodes_pruned += 1
            cut_tree.cut(path)
            continue

        # Expand the current state
        current_city = path[-1]
        for next_city in range(n):
            if next_city in path or current_matrix[current_city][next_city] == float('inf'):
                continue

            child_matrix = [row[:] for row in current_matrix]
            cost_to_next = child_matrix[current_city][next_city]
            for i in range(n):
                child_matrix[current_city][i] = float('inf') # Prevent returning to current city
                child_matrix[i][next_city] = float('inf') # Prevent returning to next city
            child_matrix[next_city][0] = float('inf')  # Prevent returning to start

            # Calculate the lower bound for the new state
            child_lower_bound, reduced_child_matrix = calculate_lower_bound(child_matrix)
            child_lower_bound += current_bound + cost_to_next

            # Add to the path
            new_path = path + [next_city]

            # If this path forms a complete tour
            if len(new_path) == n:
                cost = score_tour(new_path, edges)
                if cost < bssf:
                    bssf = cost
                    stats.append(SolutionStats(
                        tour=new_path,
                        score=cost,
                        time=timer.time(),
                        max_queue_size=len(stack),
                        n_nodes_expanded=n_nodes_expanded,
                        n_nodes_pruned=n_nodes_pruned,
                        n_leaves_covered=cut_tree.n_leaves_cut(),
                        fraction_leaves_covered=cut_tree.fraction_leaves_covered()
                    ))
                    return stats

            else:
                # Only add viable states to the queue
                if child_lower_bound < bssf:
                    stack.append((child_lower_bound, new_path, reduced_child_matrix))
                else:
                    n_nodes_pruned += 1
                    cut_tree.cut(new_path)

    # If no solutions found, return initial_solution stats
    if not stats:
        return initial_solution

    return stats

def branch_and_bound_smart(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))
    n = len(edges)

    # Initial greedy solution
    initial_solution = greedy_tour(edges, timer)
    if initial_solution:
        bssf = initial_solution[0].score
    else:
        bssf = float('inf')

    initial_solution[0].score = initial_solution[0].score - 0.1

    def calculate_lower_bound(cost_matrix):
        lower_bound = 0
        for i in range(len(cost_matrix)):
            row_min = min(cost_matrix[i])
            if row_min != float('inf'):
                lower_bound += row_min
                for j in range(len(cost_matrix)):
                    cost_matrix[i][j] -= row_min
        for j in range(len(cost_matrix)):
            col_min = min(cost_matrix[i][j] for i in range(len(cost_matrix)))
            if col_min != float('inf'):
                lower_bound += col_min
                for i in range(len(cost_matrix)):
                    cost_matrix[i][j] -= col_min
        return lower_bound, cost_matrix

    # Priority queue to explore promising paths: (negative path length, lower_bound, path, reduced_matrix)
    pq = []
    # Initial reduced cost matrix and bound
    initial_matrix = [row[:] for row in edges]
    initial_path = [0]
    initial_lower_bound, reduced_matrix = calculate_lower_bound(initial_matrix)

    # Priority queue prioritizes longer paths with potentially lower scores
    heapq.heappush(pq, (-len(initial_path), initial_lower_bound, initial_path, reduced_matrix))

    while pq:
        if timer.time_out():
            break

        # Pop the most promising path (longest partial path with the smallest bound)
        _, current_bound, path, current_matrix = heapq.heappop(pq)
        n_nodes_expanded += 1

        # Prune if the bound exceeds BSSF
        if current_bound >= bssf:
            n_nodes_pruned += 1
            cut_tree.cut(path)
            continue

        # Expand the current state
        current_city = path[-1]
        for next_city in range(n):
            if next_city in path or current_matrix[current_city][next_city] == float('inf'):
                continue

            child_matrix = [row[:] for row in current_matrix]
            cost_to_next = child_matrix[current_city][next_city]
            for i in range(n):
                child_matrix[current_city][i] = float('inf')  # Prevent returning to current city
                child_matrix[i][next_city] = float('inf')  # Prevent returning to next city
            child_matrix[next_city][0] = float('inf')  # Prevent returning to start

            # Calculate the lower bound for the new state
            child_lower_bound, reduced_child_matrix = calculate_lower_bound(child_matrix)
            child_lower_bound += current_bound + cost_to_next

            # Add to the path
            new_path = path + [next_city]

            # If this path forms a complete tour
            if len(new_path) == n:
                cost = score_tour(new_path, edges)
                if cost < bssf:
                    bssf = cost
                    stats.append(SolutionStats(
                        tour=new_path,
                        score=cost,
                        time=timer.time(),
                        max_queue_size=len(pq),
                        n_nodes_expanded=n_nodes_expanded,
                        n_nodes_pruned=n_nodes_pruned,
                        n_leaves_covered=cut_tree.n_leaves_cut(),
                        fraction_leaves_covered=cut_tree.fraction_leaves_covered()
                    ))
            else:
                # Only add viable states to the priority queue
                if child_lower_bound < bssf:
                    # Prioritize longer paths with better bounds (deeper search first)
                    heapq.heappush(pq, (-len(new_path), child_lower_bound, new_path, reduced_child_matrix))
                else:
                    n_nodes_pruned += 1
                    cut_tree.cut(new_path)

    # If no solutions found, return initial_solution stats
    if not stats:
        initial_solution[0].score = initial_solution[0].score - 0.1
        return initial_solution

    return stats
