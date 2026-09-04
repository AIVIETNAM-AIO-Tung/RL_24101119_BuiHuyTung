"""Bai 22. Tinh vector Q(s, .) cho tat ca action tai mot state."""

import gymnasium as gym
import numpy as np

from mdp_utils import ACTION_NAMES, action_values


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    V = np.zeros(env.observation_space.n)
    q = action_values(env, V, state=0, gamma=0.99)

    print("Q(0, .) voi V=0:")
    for a, val in enumerate(q):
        print(f"  {ACTION_NAMES[a]}: {val:.4f}")

    env.close()


if __name__ == "__main__":
    main()
