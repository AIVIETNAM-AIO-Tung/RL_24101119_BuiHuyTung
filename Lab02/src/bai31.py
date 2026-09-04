"""Bai 31. Mot sweep cua Value Iteration."""

import gymnasium as gym
import numpy as np

from mdp_utils import value_iteration_sweep


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    V = np.zeros(env.observation_space.n)
    new_V = value_iteration_sweep(env, V, gamma=0.99)

    print("V sau 1 sweep cua Value Iteration:")
    print(new_V.reshape(4, 4))

    env.close()


if __name__ == "__main__":
    main()
