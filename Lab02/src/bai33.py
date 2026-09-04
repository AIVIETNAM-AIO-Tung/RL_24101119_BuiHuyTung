"""Bai 33. Trich xuat optimal policy sau Value Iteration."""

import gymnasium as gym

from mdp_utils import greedy_policy_from_value, print_frozenlake_policy, value_iteration


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    V, n_iterations, deltas = value_iteration(env, gamma=0.99, theta=1e-8)
    optimal_policy = greedy_policy_from_value(env, V, gamma=0.99)

    print("Optimal state values:")
    print(V.reshape(4, 4))

    print("\nOptimal policy (dang so):")
    print(optimal_policy.reshape(4, 4))

    print("\nOptimal policy (dang luoi 4x4):")
    print_frozenlake_policy(env, optimal_policy)

    env.close()


if __name__ == "__main__":
    main()
