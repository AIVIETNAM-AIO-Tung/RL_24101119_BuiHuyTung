"""Bai 25. Theo doi hoi tu cua Policy Evaluation, ve delta theo iteration."""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from mdp_utils import policy_evaluation_with_history

FIGURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "figures", "policy_evaluation_convergence.png"
)


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = np.ones((n_states, n_actions)) / n_actions

    V, n_iterations, deltas = policy_evaluation_with_history(
        env, policy, gamma=0.99, theta=1e-8
    )

    print(f"Hoi tu sau {n_iterations} iterations.")

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(deltas) + 1), deltas, color="tab:orange")
    plt.yscale("log")
    plt.title("Hoi tu cua Iterative Policy Evaluation (uniform random policy)")
    plt.xlabel("Iteration")
    plt.ylabel("Delta (log scale)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    print(f"Da luu bieu do tai: {FIGURE_PATH}")

    env.close()


if __name__ == "__main__":
    main()
