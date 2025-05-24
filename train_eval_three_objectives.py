# %%
import os

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from tqdm import tqdm

from drl_algos.ppo.ppo import PPO
from envs.orch_env import EdgeOrchestrationEnv
from envs.utils import equally_spaced_weights

SEED = 42
MONITOR_PATH = f"./results_3obj/ppo_deepset_{SEED}"

# %%
reward_weights = equally_spaced_weights(2, 100, seed=SEED)

# %%
for i, weights in tqdm(enumerate(reward_weights)):
    monitor_path = MONITOR_PATH + f"{weights[0]}_{weights[1]}.monitor.csv"
    model_path = f"./results_3obj/ppo_deepset_{SEED}_{weights[0]}_{weights[1]}_{weights[2]}.zip"
    if os.path.exists(model_path):
        print(f"Model with weights {weights} already trained, skipping")
        continue
    print(f"Training model with weights {weights}")
    envs = SubprocVecEnv(
        [
            lambda: EdgeOrchestrationEnv(
                n_nodes=20, arrival_rate_r=30, call_duration_r=1, reward_weights=weights
            )
            for _ in range(8)
        ]
    )
    monitor_path = MONITOR_PATH + f"{weights[0]}_{weights[1]}_{weights[2]}.monitor.csv"
    envs = VecMonitor(envs, monitor_path)
    agent = PPO(
        envs, num_steps=100, n_minibatches=8, ent_coef=0.001, tensorboard_log=None, seed=SEED
    )
    if i > 0:
        agent.load(
            f"./results_3obj/ppo_deepset_{SEED}_{reward_weights[i - 1][0]}_{reward_weights[i - 1][1]}_{reward_weights[i - 1][2]}.zip"
        )
    agent.learn(total_timesteps=1000000)
    agent.save(f"./results_3obj/ppo_deepset_{SEED}_{weights[0]}_{weights[1]}_{weights[2]}.zip")

# %%
n_episodes = 100

# %%
# eval
agent_files = [f for f in os.listdir("./results") if f.endswith(".zip")]

# %%
vec_rewards = []
for file in tqdm(agent_files):
    env = DummyVecEnv(
        [lambda: EdgeOrchestrationEnv(n_nodes=20, arrival_rate_r=30, call_duration_r=1)]
    )
    agent = PPO(
        env, num_steps=100, n_minibatches=8, ent_coef=0.001, tensorboard_log=None, seed=SEED
    )
    agent.load("./results_3obj/" + file)
    obs = env.reset()
    action_mask = np.array(env.env_method("action_masks"))
    ep_vec_rewards = []
    for _ in range(n_episodes):
        done = False
        ep_vec_rew = []
        while not done:
            action = agent.predict(obs, action_mask)
            obs, rew, done, info = env.step(action)
            action_mask = np.array(env.env_method("action_masks"))
            vec_rew = [info[0]["block_reward"], info[0]["energy_reward"], info[0]["latency_reward"]]
            ep_vec_rew.append(vec_rew)
        ep_vec_rewards.append(np.sum(ep_vec_rew, axis=0))
    vec_rewards.append(np.mean(ep_vec_rewards, axis=0))

# %%
np.savetxt(f"./results_csv/ppo_deepset_{SEED}_rewards_3obj.csv", vec_rewards, delimiter=",")
