"""Bai 27. Hien thi policy tren luoi 4x4 bang mui ten, H va G."""

import gymnasium as gym
import numpy as np

from mdp_utils import greedy_policy_from_value, policy_evaluation, print_frozenlake_policy


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    uniform_policy = np.ones((n_states, n_actions)) / n_actions

    V, _ = policy_evaluation(env, uniform_policy, gamma=0.99, theta=1e-8)
    greedy_policy = greedy_policy_from_value(env, V, gamma=0.99)

    print_frozenlake_policy(env, greedy_policy)

    env.close()


if __name__ == "__main__":
    main()
