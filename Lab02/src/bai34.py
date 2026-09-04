"""Bai 34. Danh gia policy bang simulation: random / VI / PI."""

import gymnasium as gym
import numpy as np

from mdp_utils import (
    evaluate_policy_by_simulation,
    greedy_policy_from_value,
    policy_iteration,
    value_iteration,
)


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    rng = np.random.default_rng(42)
    random_policy = rng.integers(0, n_actions, size=n_states)

    V_vi, _, _ = value_iteration(env, gamma=0.99, theta=1e-8)
    policy_vi = greedy_policy_from_value(env, V_vi, gamma=0.99)

    policy_pi, V_pi, _ = policy_iteration(env, gamma=0.99, theta=1e-8)

    for name, policy in [
        ("Random policy", random_policy),
        ("Value Iteration policy", policy_vi),
        ("Policy Iteration policy", policy_pi),
    ]:
        stats = evaluate_policy_by_simulation(env, policy, n_episodes=1000, seed=42)
        print(f"\n{name}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    env.close()


if __name__ == "__main__":
    main()
