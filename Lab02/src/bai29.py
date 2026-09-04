"""Bai 29. Cai dat Policy Iteration hoan chinh."""

import gymnasium as gym

from mdp_utils import policy_iteration, print_frozenlake_policy


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    policy, V, n_iterations = policy_iteration(env, gamma=0.99, theta=1e-8)

    print(f"\nSo policy iterations: {n_iterations}")
    print("V:")
    print(V.reshape(4, 4))
    print("\nPolicy:")
    print_frozenlake_policy(env, policy)

    env.close()


if __name__ == "__main__":
    main()
