"""Bai 28. Mot buoc Policy Improvement: Evaluation -> Improvement -> so sanh."""

import gymnasium as gym
import numpy as np

from mdp_utils import greedy_policy_from_value, policy_evaluation


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    n_states = env.observation_space.n

    # Policy khoi tao: luon chon action 0 (LEFT)
    old_policy = np.zeros(n_states, dtype=int)

    # 1. Policy Evaluation
    V, n_iter = policy_evaluation(env, old_policy, gamma=0.99, theta=1e-8)

    # 2. Greedy Policy Improvement
    new_policy = greedy_policy_from_value(env, V, gamma=0.99)

    # 3 & 4. So sanh va dem so state doi action
    n_changed = np.sum(old_policy != new_policy)

    print("Old policy:", old_policy)
    print("New policy:", new_policy)
    print(f"So state doi action: {n_changed}/{n_states}")

    env.close()


if __name__ == "__main__":
    main()
