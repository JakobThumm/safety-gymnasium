# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper for limiting the time steps of an environment."""


from gymnasium.core import Wrapper
import numpy as np


INTERVENTION_START = 0.7
INTERVENTION_END = 0.9
INT_A = 2.0/(INTERVENTION_START-INTERVENTION_END)
INT_B = 1.0 - INT_A * INTERVENTION_START
INT_B_BACK = -1.0 + INT_A * INTERVENTION_START
ACTION_RANGE_MIN = np.array([-1.0, -1.0])
ACTION_RANGE_MAX = np.array([1.0, 1.0])

class SimpleActionMask(Wrapper):
    """This wrapper will limit the maximal forward action based on the current lidar signal.
    """

    def step(self, action):
        """Steps through the environment and adapts the action.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)``
        """
        obs = self.env.unwrapped.task.obs(force_unflattened=True)
        if self.env.unwrapped.task.debug:
            action = np.array(self.env.unwrapped.task.agent.get_debug_action(), dtype=np.float32)
        hazard_lidar = obs["hazards_lidar"]
        max_forward_lidar = max(hazard_lidar[0], hazard_lidar[15])
        max_backward_lidar = max(hazard_lidar[7], hazard_lidar[8])
        max_action = np.array([min(max(INT_A * max_forward_lidar + INT_B, -1), 1), 1.0])
        min_action = np.array([min(max(-INT_A * max_backward_lidar + INT_B_BACK, -1), 1), -1.0])
        scaled_action = (action - ACTION_RANGE_MIN) * (max_action-min_action)/(ACTION_RANGE_MAX-ACTION_RANGE_MIN) + min_action
        print(f"Hazard 0/15 = {hazard_lidar[0]:.2}, {hazard_lidar[15]:.2} | Action = [{scaled_action[0]:.2}, {scaled_action[1]:.2}]")
        return self.env.step(scaled_action)
