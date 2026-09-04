"""Bai 24. Iterative Policy Evaluation day du cho uniform random policy."""

import gymnasium as gym
import numpy as np

from mdp_utils import policy_evaluation


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = np.ones((n_states, n_actions)) / n_actions

    V, n_iterations = policy_evaluation(env, policy, gamma=0.99, theta=1e-8)

    print(f"Hoi tu sau {n_iterations} iterations.")
    print("V:")
    print(V.reshape(4, 4))

    env.close()


if __name__ == "__main__":
    main()
