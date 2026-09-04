"""
Lab02/src/main.py
==================
Bai 36 - Mini-project: Dynamic Programming Solver hoan chinh cho
FrozenLake-v1.

Chuong trinh:
  1. Tao environment (ho tro is_slippery True/False).
  2. Lay transition model.
  3. Chay Value Iteration va Policy Iteration (tu lap trinh, khong dung
     thuat toan DP co san tu thu vien RL).
  4. Hien thi value table va policy dang luoi.
  5. Danh gia bang simulation (>= 1000 episode).
  6. Do runtime, luu du lieu hoi tu, ve convergence curve.
  7. So sanh hai thuat toan.

Chay:
    python src/main.py
"""

import os
from time import perf_counter

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from mdp_utils import (
    evaluate_policy_by_simulation,
    greedy_policy_from_value,
    policy_evaluation_with_history,
    policy_iteration,
    print_frozenlake_policy,
    value_iteration,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def create_environment(map_name="4x4", is_slippery=True):
    """Tao FrozenLake-v1 environment."""
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    env.reset()
    return env


def get_transition_model(env):
    """Lay transition model P[state][action] = [(prob, next_state,
    reward, terminated), ...]."""
    return env.unwrapped.P


def run_value_iteration(env, gamma, theta, max_iterations):
    t0 = perf_counter()
    V, n_iterations, deltas = value_iteration(
        env, gamma=gamma, theta=theta, max_iterations=max_iterations
    )
    elapsed = perf_counter() - t0
    policy = greedy_policy_from_value(env, V, gamma=gamma)
    return {
        "name": "Value Iteration",
        "V": V,
        "policy": policy,
        "n_iterations": n_iterations,
        "deltas": deltas,
        "time": elapsed,
    }


def run_policy_iteration(env, gamma, theta, max_iterations):
    t0 = perf_counter()
    policy, V, n_iterations = policy_iteration(
        env, gamma=gamma, theta=theta, max_iterations=max_iterations
    )
    elapsed = perf_counter() - t0

    # Chay lai policy evaluation cuoi cung de lay deltas phuc vu ve do thi
    _, _, deltas = policy_evaluation_with_history(env, policy, gamma=gamma, theta=theta)

    return {
        "name": "Policy Iteration",
        "V": V,
        "policy": policy,
        "n_iterations": n_iterations,
        "deltas": deltas,
        "time": elapsed,
    }


def plot_convergence(deltas, title, filename):
    path = os.path.join(FIGURES_DIR, filename)
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(deltas) + 1), deltas)
    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Delta (log scale)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Da luu bieu do hoi tu tai: {path}")


def print_value_table(V, shape=(4, 4)):
    print(np.round(V.reshape(shape), 4))


def summarize(env, result, n_episodes=1000, seed=42):
    stats = evaluate_policy_by_simulation(env, result["policy"], n_episodes=n_episodes, seed=seed)
    result["stats"] = stats
    return result


def main(gamma=0.99, theta=1e-8, max_iterations=10000, map_name="4x4",
         is_slippery=True, n_eval_episodes=1000):
    print(f"=== Dynamic Programming Solver - FrozenLake-{map_name} "
          f"(is_slippery={is_slippery}) ===\n")

    env = create_environment(map_name=map_name, is_slippery=is_slippery)
    get_transition_model(env)  # xac nhan model duoc doc

    # --- Value Iteration ---
    vi_result = run_value_iteration(env, gamma, theta, max_iterations)
    print(f"[Value Iteration] hoi tu sau {vi_result['n_iterations']} iterations "
          f"trong {vi_result['time']:.4f}s")
    print("Value table:")
    print_value_table(vi_result["V"])
    print("Policy:")
    print_frozenlake_policy(env, vi_result["policy"])
    plot_convergence(
        vi_result["deltas"],
        "Value Iteration - Hoi tu",
        "value_iteration_convergence.png",
    )

    # --- Policy Iteration ---
    print()
    pi_result = run_policy_iteration(env, gamma, theta, max_iterations)
    print(f"[Policy Iteration] hoi tu sau {pi_result['n_iterations']} policy "
          f"iterations trong {pi_result['time']:.4f}s")
    print("Value table:")
    print_value_table(pi_result["V"])
    print("Policy:")
    print_frozenlake_policy(env, pi_result["policy"])
    plot_convergence(
        pi_result["deltas"],
        "Policy Iteration - Hoi tu cua Policy Evaluation cuoi cung",
        "policy_iteration_convergence.png",
    )

    # --- Danh gia bang simulation ---
    print(f"\n=== Danh gia bang simulation ({n_eval_episodes} episodes) ===")
    vi_result = summarize(env, vi_result, n_episodes=n_eval_episodes)
    pi_result = summarize(env, pi_result, n_episodes=n_eval_episodes)

    for result in [vi_result, pi_result]:
        print(f"\n{result['name']}:")
        for k, v in result["stats"].items():
            print(f"  {k}: {v}")

    # --- So sanh hai thuat toan ---
    print("\n=== So sanh Value Iteration vs Policy Iteration ===")
    print("| Thuat toan | So vong lap | Thoi gian (s) | Success rate | Mean reward |")
    print("|---|---:|---:|---:|---:|")
    for result in [vi_result, pi_result]:
        print(f"| {result['name']} | {result['n_iterations']} | "
              f"{result['time']:.4f} | {result['stats']['success_rate']:.3f} | "
              f"{result['stats']['mean_reward']:.3f} |")

    env.close()
    return vi_result, pi_result


if __name__ == "__main__":
    main()
