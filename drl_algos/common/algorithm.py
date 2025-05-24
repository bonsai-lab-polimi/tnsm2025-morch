from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy.typing as npt
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from torch.utils.tensorboard.writer import SummaryWriter


class Algorithm(ABC):
    def __init__(
        self,
        env: Union[SubprocVecEnv, DummyVecEnv],
        learning_rate: float,
        seed: int,
        device: str,
        tensorboard_log: str,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = device
        self.writer = SummaryWriter(log_dir=tensorboard_log)

    @abstractmethod
    def learn(self, total_timesteps: int) -> None:
        """
        Executes the training loop for total_timesteps iterations.
        """

    @abstractmethod
    def predict(self, obs: npt.NDArray, masks: Optional[npt.NDArray] = None) -> npt.NDArray:
        """
        Predict the policy action from an observation
        """
