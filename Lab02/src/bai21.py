"""Bai 21. Mot Bellman backup: tinh Q(s,a) tu V."""

import gymnasium as gym
import numpy as np

from mdp_utils import q_from_v


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    V = np.zeros(env.observation_space.n)
    q = q_from_v(env, V, state=0, action=2, gamma=0.99)
    print(f"Q(0, RIGHT) voi V=0: {q}")

    # Thu voi V khac 0 de kiem tra
    V2 = np.arange(env.observation_space.n, dtype=float)
    q2 = q_from_v(env, V2, state=14, action=1, gamma=0.99)
    print(f"Q(14, DOWN) voi V=arange(16): {q2:.4f}")

    env.close()


if __name__ == "__main__":
    main()
