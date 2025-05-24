import heapq
import logging

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pandas as pd
from gymnasium import spaces
from scipy.spatial import cKDTree

from envs.utils import Request

logging.basicConfig(level=logging.DEBUG)


class EdgeOrchestrationEnv(gym.Env):
    def __init__(
        self,
        n_nodes: int,
        arrival_rate_r: float,
        call_duration_r: float,
        area_size: tuple[float, float] = (1, 1),
        bs_density: float = 50,
        episode_length: int = 1000,
        reward_weights: tuple = (1.0, 0.0),
        seed: int = 42,
    ) -> None:
        # Set logger
        self.logger = logging.getLogger(__name__)

        # Set randomness
        self._np_random = np.random.default_rng(seed)

        ## RL setup

        # We define the state as a matrix having as rows the nodes and columns their associated metrics
        self.observation_space = spaces.Box(
            low=0, high=1, shape=((n_nodes + 1), 18), dtype=np.float32
        )
        # deploy the service on the 1, 2, ..., n-th node or reject it
        self.action_space = spaces.Discrete(n_nodes + 1)
        # The reward is a vector of three objectives: blocking, latency, and energy consumption

        self.reward_weights = reward_weights

        ## Simulation setup

        self.area_size = area_size
        # Generation of base station x and y locations according to a Poisson Point Process
        num_points = self.np_random.poisson(bs_density * self.area_size[0] * self.area_size[1])

        # Generate uniform random points
        x_coords = self.np_random.uniform(0, self.area_size[0], num_points)
        y_coords = self.np_random.uniform(0, self.area_size[0], num_points)

        self.bs_coords = np.stack((x_coords, y_coords), axis=1)
        # Precompute all pairwise distances between base stations
        self.bs_dist = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                self.bs_dist[i, j] = np.linalg.norm(self.bs_coords[i] - self.bs_coords[j], ord=2)

        # Generate Voronoi cells
        self.tree = cKDTree(self.bs_coords)

        self.episode_length = episode_length

        self.device_list = ["raspi2", "raspi3", "raspi4", "nuc", "shuttle"]
        # self.scaling_factors = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.scaling_factors = [1 / 0.45, 1.0, 1 / 1.42, 1 / 0.98, 1.0]
        self.energy_consumption = {
            device: pd.read_csv(f"./measurements/series_{device}_W.csv")["mean"].squeeze()
            for device in self.device_list
        }

        self.n_nodes = n_nodes

        self.arrival_rate_r = arrival_rate_r
        self.call_duration_r = call_duration_r

        # Initialize the event queue as an empty heap
        self.running_requests: list[tuple[float, Request]] = []

        # Uniformly generate nodes types among the possible different devices
        self.device_type = self._np_random.integers(low=0, high=len(self.device_list), size=n_nodes)
        self.device_type_oh = np.eye(len(self.device_list))[self.device_type]
        self.is_amd = np.array(
            [True if self.device_type[i] in [3, 4] else False for i in range(n_nodes)]
        )
        self.device_scaling = np.array(
            [self.scaling_factors[self.device_type[i]] for i in range(n_nodes)]
        )

        self.logger.debug(
            f"Device types: {[self.device_list[self.device_type[i]] for i in range(n_nodes)]}"
        )
        self.logger.debug(f"Is AMD: {self.is_amd}")

        # Assign each node to a random location in the area
        self.node_locs = self._np_random.uniform(0, self.area_size[0], (n_nodes, 2))

        service_list_df_amd = pd.read_csv("./envs/service_list_amd64.csv")
        service_df_arm = pd.read_csv("./envs/service_list_arm64.csv")

        cpu_load_rest_os_amd64 = (
            service_list_df_amd[service_list_df_amd["Name"] == "OS 22.04.2 server"][
                "CPU_rest"
            ].iloc[0]
            / 100
        )
        disk_load_os_amd64 = (
            service_list_df_amd[service_list_df_amd["Name"] == "OS 22.04.2 server"]["Disk"].iloc[0]
            / 500
        )

        cpu_load_rest_arm = service_df_arm[service_df_arm["Name"] == "OS"]["CPU_rest"].iloc[0] / 100
        disk_load_os_arm = service_df_arm[service_df_arm["Name"] == "OS"]["Disk"].iloc[0] / 16

        # Drop the service OS 22.04.2 server from the amd dataframe
        service_list_df_amd = service_list_df_amd[
            service_list_df_amd["Name"] != "OS 22.04.2 server"
        ]
        # Drop OS from the arm dataframe
        service_df_arm = service_df_arm[service_df_arm["Name"] != "OS"]

        # All nodes start with CPU occupation equal to the OS load at rest
        self.cpu = np.zeros(n_nodes)
        self.disk = np.zeros(n_nodes)

        self.cpu[self.is_amd] += cpu_load_rest_os_amd64 * self.device_scaling[self.is_amd]
        self.disk[self.is_amd] += disk_load_os_amd64

        self.cpu[~self.is_amd] += cpu_load_rest_arm * self.device_scaling[~self.is_amd]
        self.disk[~self.is_amd] += disk_load_os_arm

        self.service_list_df_amd = service_list_df_amd
        self.service_list_df_arm = service_df_arm

        self.service_type_codes = {"IaaS": 0, "PaaS": 1, "SaaS": 2, "FaaS": 3}

        self.logger.debug(f"Initial CPU load: {self.cpu}")
        self.logger.debug(f"Initial Disk load: {self.disk}")

        self.current_time = 0.0

        self.next_request()

        self.reset()

    def step(self, action):
        # Request is not blocked unless it is assigned to the dummy reject node
        block = False
        if action < self.n_nodes:
            self.logger.debug(
                f"Assigning request to node {action}, which is {self.device_list[self.device_type[action]]}"
            )
            if self.node_is_full(action):
                self.logger.debug(f"Blocking: node {action} is full")
                block = True
                raise ValueError(
                    "Action mask is not working properly. Full nodes should be always masked."
                )
            if not self.is_amd[action] and self.request.cpu_arm == -1:
                self.logger.debug(f"Blocking: node {action} is not AMD and service is AMD")
                block = True
                raise ValueError(
                    "Action mask is not working properly. ARM nodes should not be able to serve AMD services."
                )
            else:
                self.ep_accepted_requests += 1
                self.request.serving_node = action
                if self.is_amd[action]:
                    prior_cpu = self.cpu[action]
                    self.cpu[action] += (
                        self.request.cpu_amd * self.scaling_factors[self.device_type[action]]
                    )
                    self.disk[action] += self.request.disk_amd
                else:
                    prior_cpu = self.cpu[action]
                    self.cpu[action] += (
                        self.request.cpu_arm * self.scaling_factors[self.device_type[action]]
                    )
                    self.disk[action] += self.request.disk_arm
                self.enqueue_request(self.request)
        else:
            self.logger.debug("Rejecting request")
            block = True

        if block:
            block_reward = 0.0
        else:
            block_reward = 1.0

        if not block:
            # compute 2D distance between the request and the closest BS
            dist_req_bs, idx_req = self.tree.query(self.request.location)
            dist_node_bs, idx_bs = self.tree.query(self.node_locs[action])
            self.logger.debug(
                f"Request is connected to base station {idx_req} at a distance of {dist_req_bs}"
            )
            self.logger.debug(
                f"Chosen node {action} is connected to base station {idx_bs} at a distance of {dist_node_bs}"
            )
            self.logger.debug(
                f"Distance between base stations {idx_req} and {idx_bs} is {self.bs_dist[idx_req, idx_bs]}"
            )
            # Compute 2D distance between the request source and the BS that serves the node
            dist = dist_req_bs + dist_node_bs + self.bs_dist[idx_req, idx_bs]
            self.logger.debug(f"Request is assigned to node station {action} at distance {dist}")
            lat_reward = -dist
        else:
            lat_reward = 0.0

        if not block:
            node_type = self.device_list[self.device_type[action]]
            current_cpu = self.cpu[action]
            prior_node_power = np.interp(
                prior_cpu, np.arange(0, 1.1, 0.1), self.energy_consumption[node_type]
            )
            current_node_power = np.interp(
                current_cpu, np.arange(0, 1.1, 0.1), self.energy_consumption[node_type]
            )
            power_reward = prior_node_power - current_node_power
        else:
            power_reward = 0.0

        energy_reward = self.next_request()
        self.ep_energy_consumption += energy_reward

        self.logger.debug(
            f"Block reward: {block_reward}, Latency reward: {lat_reward}, Energy reward: {energy_reward}"
        )

        observation = self.observation()

        # Check episode termination
        if self.t_ == self.episode_length:
            terminated = True
        else:
            terminated = False

        self.t_ += 1

        vec_reward = np.array([block_reward, lat_reward, power_reward])
        scalar_reward = np.dot(vec_reward, self.reward_weights)

        info = {
            "block_reward": block_reward,
            "latency_reward": lat_reward,
            "energy_reward": energy_reward,
        }

        return observation, scalar_reward, terminated, False, info

    def reset(self, *, seed=None, options=None):
        self.ep_accepted_requests = 0
        self.ep_energy_consumption = 0.0
        self.t_ = 1
        return self.observation(), {}

    def observation(self) -> npt.NDArray:
        cloud_node = np.full((1, 10), -1)
        observation = np.stack(
            [self.cpu, self.disk, self.node_locs[:, 0], self.node_locs[:, 1]], axis=1
        )
        observation = np.concatenate([observation, self.device_type_oh], axis=1)
        node_pow_cons = []
        for i in range(self.n_nodes):
            node_type = self.device_list[self.device_type[i]]
            node_cpu = self.cpu[i]
            node_pow_cons.append(
                np.interp(node_cpu, np.arange(0, 1.1, 0.1), self.energy_consumption[node_type])
            )
        node_pow_cons = np.array(node_pow_cons)
        observation = np.concatenate([observation, node_pow_cons[:, np.newaxis]], axis=1)
        # Condition the elements in the set with the current node request
        node_demand = np.tile(
            np.array(
                [
                    self.request.cpu_amd,
                    self.request.cpu_arm,
                    self.request.disk_amd,
                    self.request.disk_arm,
                    self.request.location[0],
                    self.request.location[1],
                    self.service_type_codes[self.request.service_type],
                    self.dt,
                ]
            ),
            (self.n_nodes + 1, 1),
        )
        observation = np.concatenate([observation, cloud_node], axis=0)
        observation = np.concatenate([observation, node_demand], axis=1)
        return observation

    def enqueue_request(self, request: Request) -> None:
        heapq.heappush(self.running_requests, (request.departure_time, request))

    def action_masks(self):
        valid_actions = np.ones(self.n_nodes + 1, dtype=bool)
        for i in range(self.n_nodes):
            if self.node_is_full(i):
                valid_actions[i] = False
            else:
                valid_actions[i] = True
        if self.request.cpu_arm < 0:
            valid_actions[:-1][~self.is_amd] = False
        valid_actions[self.n_nodes] = True
        self.logger.debug(f"Valid actions: {valid_actions}")
        return valid_actions

    def node_is_full(self, action) -> bool:
        if self.is_amd[action]:
            if (
                self.cpu[action]
                + self.request.cpu_amd * self.scaling_factors[self.device_type[action]]
                > 1.0
                or self.disk[action] + self.request.disk_amd > 1
            ):
                return True
        else:
            if (
                self.cpu[action]
                + self.request.cpu_arm * self.scaling_factors[self.device_type[action]]
                > 1.0
                or self.disk[action] + self.request.disk_arm > 1
            ):
                return True
        return False

    def power_consumption(self) -> float:
        power_curr = 0.0
        for i in range(self.n_nodes):
            node_type = self.device_list[self.device_type[i]]
            node_cpu = self.cpu[i]
            node_power = np.interp(
                node_cpu, np.arange(0, 1.1, 0.1), self.energy_consumption[node_type]
            )
            self.logger.debug(f"Node {i} ({node_type}) CPU: {node_cpu}, Power: {node_power}")
            power_curr += node_power
        return power_curr

    def dequeue_request(self):
        _, request = heapq.heappop(self.running_requests)
        if self.is_amd[request.serving_node]:
            self.cpu[request.serving_node] -= (
                request.cpu_amd * self.scaling_factors[self.device_type[request.serving_node]]
            )
            self.disk[request.serving_node] -= request.disk_amd
        else:
            self.cpu[request.serving_node] -= (
                request.cpu_arm * self.scaling_factors[self.device_type[request.serving_node]]
            )
            self.disk[request.serving_node] -= request.disk_arm

    def next_request(self) -> float:
        arrival_time = self.current_time + self.np_random.exponential(scale=1 / self.arrival_rate_r)

        # We also compute the energy consumption of the nodes in this step
        exp_energy = 0.0
        # We get the current power consumption...
        power_curr = self.power_consumption()
        self.logger.debug(f"Current power consumption: {power_curr}")
        current_time = self.current_time
        while True:
            if self.running_requests:
                next_departure_time, _ = self.running_requests[0]
                # ... if a request departs...
                if next_departure_time < arrival_time:
                    self.logger.debug(f"Next departure: {next_departure_time}")
                    self.dequeue_request()
                    # ... we update the energy consumption up until the departure...
                    exp_energy += power_curr * (next_departure_time - current_time)
                    # ... and we update the power consumption with the reduced load
                    power_curr = self.power_consumption()
                    self.logger.debug(f"Power consumption after departure: {power_curr}")
                    current_time = next_departure_time
                    continue
            break
        exp_energy = power_curr * (arrival_time - current_time)

        self.current_time = arrival_time

        # Generate new request
        new_location = self.np_random.uniform(0, self.area_size[0], 2)
        next_service_name = self.np_random.choice(self.service_list_df_amd["Name"])
        next_service_idx_amd = self.service_list_df_amd["Name"] == next_service_name
        base_amd = self.service_list_df_amd[next_service_idx_amd]["Base"].iloc[0]
        if base_amd == "Bare":
            cpu_amd = self.service_list_df_amd[next_service_idx_amd]["CPUxUser"].iloc[0] / 100
            disk_amd = self.service_list_df_amd[next_service_idx_amd]["Disk"].iloc[0] / 500
        else:
            base_idx = self.service_list_df_amd["Name"] == base_amd
            cpu_base = self.service_list_df_amd[base_idx]["CPUxUser"].iloc[0] / 100
            disk_base = self.service_list_df_amd[base_idx]["Disk"].iloc[0] / 500
            cpu_service = self.service_list_df_amd[next_service_idx_amd]["CPUxUser"].iloc[0] / 100
            disk_service = self.service_list_df_amd[next_service_idx_amd]["Disk"].iloc[0] / 500
            cpu_amd = cpu_base + cpu_service
            disk_amd = disk_base + disk_service

        if next_service_name in self.service_list_df_arm["Name"].to_list():
            next_service_idx_arm = self.service_list_df_arm["Name"] == next_service_name
            base_arm = self.service_list_df_arm[next_service_idx_arm]["Base"].iloc[0]
            if base_arm == "Bare":
                cpu_arm = self.service_list_df_arm[next_service_idx_arm]["CPUxUser"].iloc[0] / 100
                disk_arm = self.service_list_df_arm[next_service_idx_arm]["Disk"].iloc[0] / 16
            else:
                base_idx = self.service_list_df_arm["Name"] == base_arm
                cpu_base = self.service_list_df_arm[base_idx]["CPUxUser"].iloc[0] / 100
                disk_base = self.service_list_df_arm[base_idx]["Disk"].iloc[0] / 16
                cpu_service = (
                    self.service_list_df_arm[next_service_idx_arm]["CPUxUser"].iloc[0] / 100
                )
                disk_service = self.service_list_df_arm[next_service_idx_arm]["Disk"].iloc[0] / 16
                cpu_arm = cpu_base + cpu_service
                disk_arm = disk_base + disk_service
        else:
            cpu_arm = -1
            disk_arm = -1

        service_type = self.service_list_df_amd[next_service_idx_amd]["Service Type"].iloc[0]
        if service_type == "IaaS":
            departure_time = arrival_time + self.np_random.exponential(
                scale=self.call_duration_r * 4
            )
        elif service_type == "PaaS":
            departure_time = arrival_time + self.np_random.exponential(
                scale=self.call_duration_r * 3
            )
        elif service_type == "SaaS":
            departure_time = arrival_time + self.np_random.exponential(
                scale=self.call_duration_r * 2
            )
        else:
            departure_time = arrival_time + self.np_random.exponential(scale=self.call_duration_r)
        self.dt = departure_time - arrival_time
        self.request = Request(
            name=next_service_name,
            service_type=service_type,
            cpu_amd=cpu_amd,
            disk_amd=disk_amd,
            cpu_arm=cpu_arm,
            disk_arm=disk_arm,
            base=None,
            arrival_time=self.current_time,
            departure_time=departure_time,
            location=new_location,
        )
        self.logger.debug(f"Next request: {self.request}")

        return exp_energy
