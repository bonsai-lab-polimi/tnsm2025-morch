# %%

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from envs.orch_env import EdgeOrchestrationEnv
from envs.utils import greedy_lat_policy, greedy_lb_policy, greedy_low_energy_policy, random_policy

SEED = 42
# %%
baselines = {
    "random": random_policy,
    "greedy_lb": greedy_lb_policy,
    "greedy_lat": greedy_lat_policy,
    "greedy_low_energy": greedy_low_energy_policy,
}

# %%
n_episodes = 100
vec_rewards = []
for baseline_name, baseline_policy in tqdm(baselines.items()):
    env = DummyVecEnv(
        [lambda: EdgeOrchestrationEnv(n_nodes=20, arrival_rate_r=30, call_duration_r=1)]
    )
    obs = env.reset()
    action_mask = np.array(env.env_method("action_masks"))
    ep_vec_rewards = []
    for _ in range(n_episodes):
        done = False
        ep_vec_rew = []
        while not done:
            action = baseline_policy(env, obs, action_mask, SEED)
            obs, rew, done, info = env.step(action)
            action_mask = np.array(env.env_method("action_masks"))
            vec_rew = [info[0]["block_reward"], info[0]["energy_reward"], info[0]["latency_reward"]]
            ep_vec_rew.append(vec_rew)
        ep_vec_rewards.append(np.sum(ep_vec_rew, axis=0))
    vec_rewards.append(np.mean(ep_vec_rewards, axis=0))

# %%
np.savetxt("./results_csv/baselines.csv", vec_rewards, delimiter=",")
