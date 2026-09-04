"""Bai 23. Mot sweep cua Policy Evaluation voi uniform random policy."""

import gymnasium as gym
import numpy as np

from mdp_utils import policy_evaluation_sweep


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy = np.ones((n_states, n_actions)) / n_actions
    V = np.zeros(n_states)

    new_V = policy_evaluation_sweep(env, policy, V, gamma=0.99)

    print("V sau 1 sweep:")
    print(new_V.reshape(4, 4))

    env.close()


if __name__ == "__main__":
    main()
