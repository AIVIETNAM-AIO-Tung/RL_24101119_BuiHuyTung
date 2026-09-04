"""Bai 35. So sanh Value Iteration va Policy Iteration: so vong lap, thoi
gian, success rate, mean reward."""

import os
from time import perf_counter

import gymnasium as gym
import matplotlib.pyplot as plt

from mdp_utils import (
    evaluate_policy_by_simulation,
    greedy_policy_from_value,
    policy_iteration,
    value_iteration,
)

FIGURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "figures", "algorithm_comparison.png"
)


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    # --- Value Iteration ---
    t0 = perf_counter()
    V_vi, n_iter_vi, _ = value_iteration(env, gamma=0.99, theta=1e-8)
    time_vi = perf_counter() - t0
    policy_vi = greedy_policy_from_value(env, V_vi, gamma=0.99)
    stats_vi = evaluate_policy_by_simulation(env, policy_vi, n_episodes=1000, seed=42)

    # --- Policy Iteration ---
    t0 = perf_counter()
    policy_pi, V_pi, n_iter_pi = policy_iteration(env, gamma=0.99, theta=1e-8)
    time_pi = perf_counter() - t0
    stats_pi = evaluate_policy_by_simulation(env, policy_pi, n_episodes=1000, seed=42)

    print("| Thuat toan | So vong lap | Thoi gian (s) | Success rate | Mean reward |")
    print("|---|---:|---:|---:|---:|")
    print(f"| Value Iteration | {n_iter_vi} | {time_vi:.4f} | "
          f"{stats_vi['success_rate']:.3f} | {stats_vi['mean_reward']:.3f} |")
    print(f"| Policy Iteration | {n_iter_pi} | {time_pi:.4f} | "
          f"{stats_pi['success_rate']:.3f} | {stats_pi['mean_reward']:.3f} |")

    # --- Bieu do so sanh ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    algos = ["Value\nIteration", "Policy\nIteration"]
    iters = [n_iter_vi, n_iter_pi]
    times = [time_vi, time_pi]

    axes[0].bar(algos, iters, color=["tab:blue", "tab:orange"])
    axes[0].set_title("So vong lap")
    axes[0].set_ylabel("So iterations")
    axes[0].grid(True, axis="y")

    axes[1].bar(algos, times, color=["tab:blue", "tab:orange"])
    axes[1].set_title("Thoi gian chay")
    axes[1].set_ylabel("Thoi gian (giay)")
    axes[1].grid(True, axis="y")

    fig.suptitle("So sanh Value Iteration va Policy Iteration - FrozenLake 4x4")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"\nDa luu bieu do tai: {FIGURE_PATH}")

    print("\nNhan xet:")
    print("1. Ca hai thuat toan hoi tu ve CUNG MOT optimal policy va optimal")
    print("   value function (co the kiem tra V_vi ~ V_pi).")
    print("2. Value Iteration can nhieu vong lap (sweep) hon vi moi vong lap")
    print("   chi thuc hien MOT buoc Bellman optimality backup.")
    print("3. Policy Iteration can it 'policy iterations' hon, nhung MOI vong")
    print("   lap ben trong lai chua ca mot qua trinh Policy Evaluation lap")
    print("   nhieu sweep cho toi khi hoi tu, nen tong chi phi tinh toan cua")
    print("   moi vong lap Policy Iteration lon hon nhieu so voi mot sweep VI.")
    print("4. Ve mat thoi gian thuc te, hai thuat toan thuong cho ket qua")
    print("   tuong duong nhau tren bai toan nho nhu FrozenLake 4x4; su khac")
    print("   biet ro ret hon xuat hien khi khong gian trang thai lon hon.")
    print("5. Success rate va mean reward cua hai policy la nhu nhau vi ca")
    print("   hai deu hoi tu ve chinh sach toi uu cho cung MDP.")
    print("6. Value Iteration thuong don gian hon de cai dat vi khong can")
    print("   vong lap con Policy Evaluation rieng.")
    print("7. Policy Iteration co the hoi tu nhanh hon ve so 'vong lap ngoai'")
    print("   khi policy ban dau da gan voi optimal policy.")
    print("8. Trong thuc te, viec chon thuat toan phu thuoc vao kich thuoc")
    print("   khong gian trang thai/hanh dong va yeu cau ve toc do hoi tu.")

    env.close()


if __name__ == "__main__":
    main()
