"""Bai 32. Cai dat Value Iteration hoan chinh."""

import gymnasium as gym

from mdp_utils import value_iteration


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    V, n_iterations, deltas = value_iteration(env, gamma=0.99, theta=1e-8)

    print(f"Hoi tu sau {n_iterations} iterations.")
    print("V:")
    print(V.reshape(4, 4))
    print(f"Delta cuoi cung: {deltas[-1]:.2e}")

    env.close()


if __name__ == "__main__":
    main()
