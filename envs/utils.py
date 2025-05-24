from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
from pymoo.util.ref_dirs import get_reference_directions


@dataclass
class Request:
    name: str
    service_type: str
    cpu_amd: float
    cpu_arm: float
    disk_amd: float
    disk_arm: float
    arrival_time: float
    departure_time: float
    location: tuple[float, float]
    base: Optional[str] = None
    serving_node: Optional[int] = None


def greedy_lb_policy(env, obs: npt.NDArray, action_mask: npt.NDArray, seed=0) -> int:
    """Returns the index of the feasible node with the lowest average load between cpu and disk."""
    feasible_nodes = np.argwhere(action_mask[:-1]).flatten()
    if len(feasible_nodes) == 0:
        return len(action_mask)
    cpu_load = obs[feasible_nodes, 0]
    disk_load = obs[feasible_nodes, 1]
    # Take the maximum between cpu and disk load
    max_load = np.maximum(cpu_load, disk_load)
    # Take the node with the minimum load
    return feasible_nodes[np.argmin(max_load)]


def random_policy(env, obs: npt.NDArray, action_mask: npt.NDArray, seed=0) -> int:
    """Returns a random feasible node."""
    feasible_nodes = np.argwhere(action_mask[:-1]).flatten()
    rng = np.random.default_rng(seed)
    if len(feasible_nodes) == 0:
        return len(action_mask)
    return rng.choice(feasible_nodes)


def greedy_lat_policy(env, obs: npt.NDArray, action_mask: npt.NDArray, seed=0) -> int:
    """Returns the feasible node that is also the closest in terms of distance"""
    feasible_nodes = np.argwhere(action_mask[:-1]).flatten()
    if len(feasible_nodes) == 0:
        return len(action_mask)
    # Get the node location
    node_locs = env.node_locs[feasible_nodes]
    # Get the request location
    request_loc = env.request.location
    # Find the index of minimum distance
    dists = np.linalg.norm(node_locs - request_loc, axis=1)
    return feasible_nodes[np.argmin(dists)]


def greedy_low_energy_policy(env, obs: npt.NDArray, action_mask: npt.NDArray, seed=0) -> int:
    """Selects an arm64 node, if available. Otherwise, it selects and amd64 node. Otherwise, it rejects."""
    feasible_nodes = np.argwhere(action_mask[:-1]).flatten()
    # cpu, ram, disk and bandwidth are the first 2 columns of the observation matrix
    if len(feasible_nodes) == 0:
        return len(action_mask)

    rng = np.random.default_rng(seed)
    # Find feasible arm64 nodes
    arm_idx = np.where(env.is_amd == 0)[0]
    # Extract from the action mask the arm nodes
    arm_mask = action_mask[arm_idx]
    # If there are arm nodes, select the one at random
    if np.any(arm_mask):
        return rng.choice(arm_idx[arm_mask])

    # Mask feasible NUC nodes (id 3) that have cpu consumption greater than 60%
    nuc_nodes = np.argwhere(env.device_type == 3)
    for node in nuc_nodes:
        if obs[node, 0] > 0.5:
            action_mask[node] = False
    # Mask feasible Shuttle nodes (id 4) that have cpu consumption greater than 40%
    shuttle_nodes = np.argwhere(env.device_type == 4)
    for node in shuttle_nodes:
        if obs[node, 0] > 0.5:
            action_mask[node] = False

    feasible_nodes = np.argwhere(action_mask[:-1]).flatten()
    # cpu, ram, disk and bandwidth are the first 2 columns of the observation matrix
    if len(feasible_nodes) == 0:
        return len(action_mask)
    return rng.choice(feasible_nodes)


def sample_weights(d, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(d), size=n)


def pareto_dominance(set_of_solutions):
    nondominated_solutions = []

    for i, solution in enumerate(set_of_solutions):
        # Assume the solution is nondominated until we find a dominating one
        is_dominated = False
        for j, other_solution in enumerate(set_of_solutions):
            if i != j:
                # Check if other_solution dominates solution
                if all(other_solution <= solution) and any(other_solution < solution):
                    is_dominated = True
                    break
        if not is_dominated:
            nondominated_solutions.append(solution)

    return np.array(nondominated_solutions)


def equally_spaced_weights(dim: int, n: int, seed: int = 42) -> list[np.ndarray]:
    """Generate weight vectors that are equally spaced in the weight simplex.

    It uses the Riesz s-Energy method from pymoo: https://pymoo.org/misc/reference_directions.html

    Args:
        dim: size of the weight vector
        n: number of weight vectors to generate
        seed: random seed
    """
    return list(get_reference_directions("energy", dim, n, seed=seed))
