"""Bai 30. Kiem tra policy stability tuong minh (khong goi thang policy_iteration)."""

import gymnasium as gym
import numpy as np

from mdp_utils import greedy_policy_from_value, policy_evaluation


def my_policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    n_states = env.observation_space.n
    policy = np.zeros(n_states, dtype=int)

    for i in range(1, max_iterations + 1):
        V, _ = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        new_policy = greedy_policy_from_value(env, V, gamma=gamma)

        # Tu lap trinh kiem tra policy_stable
        policy_stable = True
        for s in range(n_states):
            if new_policy[s] != policy[s]:
                policy_stable = False
                break

        policy = new_policy

        if policy_stable:
            print(f"Policy Iteration converged after {i} iterations.")
            return policy, V, i

    print(f"Reached max_iterations={max_iterations} without convergence.")
    return policy, V, max_iterations


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    policy, V, n_iterations = my_policy_iteration(env, gamma=0.99, theta=1e-8)
    print("Policy cuoi cung:", policy)

    env.close()


if __name__ == "__main__":
    main()
