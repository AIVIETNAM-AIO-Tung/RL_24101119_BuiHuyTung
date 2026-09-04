"""
mdp_utils.py
============
Các hàm dùng chung cho Lab02 - MDP, Value Function, Dynamic Programming.

Toàn bộ thuật toán (Policy Evaluation, Policy Improvement, Policy Iteration,
Value Iteration) được tự lập trình, KHÔNG dùng thư viện RL nào gọi sẵn.
"""

from time import perf_counter

import numpy as np


ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP",
}

ACTION_SYMBOLS = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑",
}


# ---------------------------------------------------------------------------
# PHẦN A - Markov Chain
# ---------------------------------------------------------------------------

def validate_transition_matrix(P, tol=1e-10):
    """Kiểm tra P là transition matrix hợp lệ: vuông, phần tử trong [0,1],
    tổng mỗi hàng xấp xỉ 1."""
    P = np.asarray(P, dtype=float)

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False

    if np.any(P < -tol) or np.any(P > 1 + tol):
        return False

    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        return False

    return True


def state_distribution(p0, P, n_steps):
    """Tính phân phối trạng thái sau n_steps bước (p_t = p0 @ P^t)."""
    p0 = np.asarray(p0, dtype=float)
    P = np.asarray(P, dtype=float)

    p = p0.copy()
    for _ in range(n_steps):
        p = p @ P
    return p


def sample_next_state(current_state, P, rng):
    """Lấy mẫu trạng thái kế tiếp theo phân phối hàng current_state của P."""
    n_states = P.shape[0]
    probs = P[current_state]
    return rng.choice(n_states, p=probs)


# ---------------------------------------------------------------------------
# PHẦN B - Reward, Return, Discount factor
# ---------------------------------------------------------------------------

def compute_return(rewards, gamma):
    """Tính discounted return G_0 = sum_k gamma^k * r_k cho một chuỗi reward."""
    G = 0.0
    for k, r in enumerate(rewards):
        G += (gamma ** k) * r
    return G


def discounted_returns(rewards, gamma):
    """Tính G_t cho mọi t trong chuỗi rewards, đi từ cuối episode về đầu.

    G_t = R_(t+1) + gamma * G_(t+1),  với G_T = 0 tại cuối episode.
    """
    rewards = list(rewards)
    n = len(rewards)
    G = np.zeros(n, dtype=float)

    running = 0.0
    for t in reversed(range(n)):
        running = rewards[t] + gamma * running
        G[t] = running

    return G


# ---------------------------------------------------------------------------
# PHẦN C - Kiểm tra model MDP / Policy
# ---------------------------------------------------------------------------

def validate_mdp(P, n_states, n_actions, tol=1e-8):
    """Kiểm tra tổng xác suất transition của mỗi (state, action) xấp xỉ 1."""
    for s in range(n_states):
        for a in range(n_actions):
            transitions = P[s][a]
            total_prob = sum(t[0] for t in transitions)
            if not np.isclose(total_prob, 1.0, atol=tol):
                raise ValueError(
                    f"Invalid transition at state={s}, action={a} "
                    f"(tong xac suat = {total_prob})"
                )
    return True


def print_policy(policy, action_names=None):
    """In deterministic policy dạng: state -> action."""
    if action_names is None:
        action_names = ACTION_NAMES
    for s, a in enumerate(policy):
        name = action_names.get(int(a), str(a))
        print(f"State {s:>2d} -> Action {int(a)} ({name})")


# ---------------------------------------------------------------------------
# PHẦN D - Khám phá model FrozenLake
# ---------------------------------------------------------------------------

def describe_state(env, state):
    """In toàn bộ transition (action, probability, next_state, reward,
    terminated) của một state trong model P của env."""
    P = env.unwrapped.P
    n_actions = env.action_space.n

    print(f"--- State {state} ---")
    for a in range(n_actions):
        name = ACTION_NAMES.get(a, str(a))
        print(f"  Action {a} ({name}):")
        for prob, next_state, reward, terminated in P[state][a]:
            print(
                f"    prob={prob:.3f}  next_state={next_state:>2d}  "
                f"reward={reward}  terminated={terminated}"
            )


def print_frozenlake_policy(env, policy, action_symbols=None):
    """In policy lên lưới NxN của FrozenLake, dùng mũi tên; Hole -> H,
    Goal -> G."""
    if action_symbols is None:
        action_symbols = ACTION_SYMBOLS

    desc = env.unwrapped.desc  # ma trận ký tự bản đồ (bytes)
    nrow, ncol = desc.shape
    P = env.unwrapped.P

    lines = []
    for r in range(nrow):
        row_symbols = []
        for c in range(ncol):
            s = r * ncol + c
            tile = desc[r, c].decode("utf-8")
            if tile == "H":
                row_symbols.append("H")
            elif tile == "G":
                row_symbols.append("G")
            else:
                a = int(policy[s])
                row_symbols.append(action_symbols.get(a, "?"))
        lines.append(" ".join(row_symbols))

    print("\n".join(lines))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PHẦN E - Bellman backup & Policy Evaluation
# ---------------------------------------------------------------------------

def q_from_v(env, V, state, action, gamma):
    """Tính Q(s,a) = sum_(s',r) p(s',r|s,a) * [r + gamma * V(s') * (1-terminated)]

    Lưu ý: khi transition terminated=True, không có state kế tiếp thực sự để
    tiếp tục tích lũy reward, nhưng theo model của Gymnasium, V(next_state)
    tại các state terminal (Hole/Goal) sẽ tự nhiên hội tụ về giá trị đúng
    trong quá trình lặp (V=0 tại Hole vì không action nào rời khỏi Hole được
    mô phỏng, V tại Goal cũng không được cập nhật thêm reward sau đó).
    Ta vẫn cộng gamma*V(next_state) bình thường; điều này đúng theo Bellman
    equation chuẩn vì P(.|s,a) đã bao gồm đầy đủ next_state.
    """
    P = env.unwrapped.P
    q = 0.0
    for prob, next_state, reward, terminated in P[state][action]:
        q += prob * (reward + gamma * V[next_state])
    return q


def action_values(env, V, state, gamma):
    """Trả về vector Q(s, a) cho mọi action tại state."""
    n_actions = env.action_space.n
    q = np.zeros(n_actions)
    for a in range(n_actions):
        q[a] = q_from_v(env, V, state, a, gamma)
    return q


def policy_evaluation_sweep(env, policy, V, gamma):
    """Thực hiện MỘT sweep của Iterative Policy Evaluation, trả về new_V.

    policy có thể là:
      - deterministic: mảng 1D shape (n_states,) chứa action index
      - stochastic:   mảng 2D shape (n_states, n_actions) chứa pi(a|s)
    """
    n_states = env.observation_space.n
    new_V = np.zeros(n_states)

    policy = np.asarray(policy)

    for s in range(n_states):
        if policy.ndim == 1:
            a = int(policy[s])
            new_V[s] = q_from_v(env, V, s, a, gamma)
        else:
            v = 0.0
            for a, pi_a_s in enumerate(policy[s]):
                if pi_a_s == 0:
                    continue
                v += pi_a_s * q_from_v(env, V, s, a, gamma)
            new_V[s] = v

    return new_V


def policy_evaluation(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    """Iterative Policy Evaluation đầy đủ. Trả về (V, n_iterations)."""
    n_states = env.observation_space.n
    V = np.zeros(n_states)

    for i in range(1, max_iterations + 1):
        new_V = policy_evaluation_sweep(env, policy, V, gamma)
        delta = np.max(np.abs(new_V - V))
        V = new_V
        if delta < theta:
            return V, i

    return V, max_iterations


def policy_evaluation_with_history(env, policy, gamma=0.99, theta=1e-8,
                                    max_iterations=10000):
    """Giống policy_evaluation nhưng lưu lại lịch sử delta mỗi iteration."""
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    deltas = []

    for i in range(1, max_iterations + 1):
        new_V = policy_evaluation_sweep(env, policy, V, gamma)
        delta = np.max(np.abs(new_V - V))
        deltas.append(delta)
        V = new_V
        if delta < theta:
            break

    return V, len(deltas), deltas


# ---------------------------------------------------------------------------
# PHẦN F - Policy Improvement & Policy Iteration
# ---------------------------------------------------------------------------

def greedy_policy_from_value(env, V, gamma=0.99):
    """Trích policy tham lam (deterministic) từ hàm giá trị V."""
    n_states = env.observation_space.n
    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        q_values = action_values(env, V, s, gamma)
        policy[s] = np.argmax(q_values)

    return policy


def policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    """Policy Iteration: xen kẽ Policy Evaluation và Policy Improvement.

    Trả về (policy, V, n_policy_iterations).
    """
    n_states = env.observation_space.n
    policy = np.zeros(n_states, dtype=int)  # khởi tạo: luôn chọn action 0

    for i in range(1, max_iterations + 1):
        V, _ = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        new_policy = greedy_policy_from_value(env, V, gamma=gamma)

        policy_stable = np.array_equal(new_policy, policy)
        policy = new_policy

        if policy_stable:
            print(f"Policy Iteration converged after {i} iterations.")
            return policy, V, i

    print(f"Policy Iteration reached max_iterations={max_iterations}.")
    return policy, V, max_iterations


# ---------------------------------------------------------------------------
# PHẦN G - Value Iteration
# ---------------------------------------------------------------------------

def value_iteration_sweep(env, V, gamma):
    """Một sweep của Value Iteration: new_V[s] = max_a Q(s,a)."""
    n_states = env.observation_space.n
    new_V = np.zeros(n_states)

    for s in range(n_states):
        q_values = action_values(env, V, s, gamma)
        new_V[s] = np.max(q_values)

    return new_V


def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):
    """Value Iteration đầy đủ. Trả về (V, n_iterations, deltas)."""
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    deltas = []

    for i in range(1, max_iterations + 1):
        new_V = value_iteration_sweep(env, V, gamma)
        delta = np.max(np.abs(new_V - V))
        deltas.append(delta)
        V = new_V

        if delta < theta:
            return V, i, deltas

    return V, max_iterations, deltas


# ---------------------------------------------------------------------------
# PHẦN H - Đánh giá bằng simulation
# ---------------------------------------------------------------------------

def evaluate_policy_by_simulation(env, policy, n_episodes=1000, seed=42):
    """Chạy policy (deterministic, mảng action theo state) trên env
    n_episodes lần, thu thập success rate, mean reward, mean/min/max
    episode length."""
    rewards = []
    lengths = []
    successes = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        terminated = False
        truncated = False
        ep_reward = 0.0
        ep_length = 0

        while not (terminated or truncated):
            action = int(policy[obs])
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1

        rewards.append(ep_reward)
        lengths.append(ep_length)
        if ep_reward > 0:
            successes += 1

    rewards = np.array(rewards)
    lengths = np.array(lengths)

    return {
        "success_rate": successes / n_episodes,
        "mean_reward": float(rewards.mean()),
        "mean_length": float(lengths.mean()),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
    }


def timed_run(fn, *args, **kwargs):
    """Chạy fn(*args, **kwargs), trả về (result, elapsed_seconds)."""
    start = perf_counter()
    result = fn(*args, **kwargs)
    elapsed = perf_counter() - start
    return result, elapsed
