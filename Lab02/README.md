# Lab02 - Markov Decision Process và Dynamic Programming

## Thông tin sinh viên

- Họ tên: Bùi Huy Tùng
- MSSV: 24101119
- Lớp: EEE.AI-24106.1
- GitHub username: https://github.com/AIVIETNAM-AIO-Tung



## Mục tiêu

Lab02 chuyển từ việc tương tác với môi trường (Lab01) sang mô hình hóa và
giải bài toán học tăng cường bằng Markov Decision Process (MDP). Sau khi
hoàn thành, sinh viên có khả năng:

- Biểu diễn và mô phỏng Markov chain bằng NumPy.
- Biểu diễn MDP rời rạc, phân biệt state/action/transition/reward.
- Tính return, phân tích ảnh hưởng của discount factor `gamma`.
- Tính `V(s)`, `Q(s,a)`, cài đặt Bellman backup.
- Tự cài đặt Policy Evaluation, Policy Improvement, Policy Iteration và
  Value Iteration (không dùng thuật toán DP có sẵn từ thư viện RL).
- Giải `FrozenLake-v1` bằng Dynamic Programming, trích xuất và đánh giá
  optimal policy bằng simulation.

## Cấu trúc thư mục

```text
Lab02/
├── README.md
├── requirements.txt
├── src/
│   ├── bai01.py ... bai36.py     # 36 bài tập
│   ├── mdp_utils.py               # các hàm dùng chung (thuật toán DP)
│   └── main.py                    # mini-project / DP solver hoàn chỉnh
├── notebooks/
│   └── Lab02_MSSV_HoTen.ipynb     # notebook tổng hợp + trả lời câu hỏi lý thuyết
├── figures/
│   ├── markov_distribution.png
│   ├── gamma_comparison.png
│   ├── policy_evaluation_convergence.png
│   ├── value_iteration_convergence.png
│   ├── policy_iteration_convergence.png
│   └── algorithm_comparison.png
└── data/
    └── README.md
```

## Cài đặt

```bash
cd Lab02
pip install -r requirements.txt
```

Kiểm tra:

```bash
python --version
pip show gymnasium
pip show numpy
```

## Cách chạy

```bash
cd Lab02
pip install -r requirements.txt

# chạy từng bài
python src/bai01.py
...
python src/bai36.py    # (không tồn tại riêng, xem main.py)

# các bài mốc quan trọng
python src/bai24.py    # Iterative Policy Evaluation
python src/bai29.py    # Policy Iteration
python src/bai32.py    # Value Iteration

# mini-project tổng hợp
python src/main.py
```

Notebook:

```bash
jupyter notebook notebooks/Lab02_MSSV_HoTen.ipynb
```

## Thuật toán đã cài đặt

Toàn bộ thuật toán bên dưới được tự lập trình trong `src/mdp_utils.py`,
không gọi hàm `value_iteration()` / `policy_iteration()` có sẵn từ bất kỳ
thư viện RL nào.

### Policy Evaluation

- `policy_evaluation_sweep`: một sweep Bellman expectation backup qua toàn
  bộ state.
- `policy_evaluation`: lặp sweep cho đến khi `delta = max|V_new - V| < theta`.
  Với uniform random policy trên `FrozenLake-v1 (4x4, is_slippery=True)`,
  thuật toán hội tụ sau **71 iterations** (gamma=0.99, theta=1e-8).

### Policy Iteration

- `greedy_policy_from_value`: trích policy tham lam từ `V` bằng
  `argmax_a Q(s,a)`.
- `policy_iteration`: lặp (Policy Evaluation → Policy Improvement) cho đến
  khi policy không đổi. Hội tụ sau **7 policy iterations**.

### Value Iteration

- `value_iteration_sweep`: một sweep Bellman optimality backup
  (`V_new[s] = max_a Q(s,a)`).
- `value_iteration`: lặp sweep cho đến khi hội tụ. Hội tụ sau
  **438 iterations** (gamma=0.99, theta=1e-8) — nhiều hơn Policy Iteration
  về số vòng lặp "ngoài", nhưng mỗi vòng lặp rẻ hơn nhiều.

## Kết quả FrozenLake

Với `FrozenLake-v1`, `map_name="4x4"`, `is_slippery=True`, `gamma=0.99`:

- Value Iteration và Policy Iteration cho ra **cùng một optimal value
  function** và **cùng một optimal policy**.
- Đánh giá bằng 1000 episode simulation: **success rate ≈ 75.2%**, mean
  reward ≈ 0.752 cho cả hai policy — so với random policy chỉ đạt
  success rate ≈ 0%.

## So sánh Value Iteration và Policy Iteration

| Thuật toán | Số vòng lặp | Thời gian (s) | Success rate | Mean reward |
|---|---:|---:|---:|---:|
| Value Iteration | 438 | ~0.048 | 0.752 | 0.752 |
| Policy Iteration | 7 | ~0.033 | 0.752 | 0.752 |

Xem nhận xét chi tiết (8 dòng) trong output của `src/bai35.py` / `src/main.py`
và trong notebook.

## Nhận xét

- Cả hai thuật toán Dynamic Programming đều hội tụ về cùng optimal policy
  vì bài toán FrozenLake là một MDP hữu hạn, model transition đã biết đầy
  đủ (`env.unwrapped.P`), thỏa điều kiện áp dụng DP.
- Policy Iteration hội tụ với ít vòng lặp "ngoài" hơn nhưng mỗi vòng lặp
  tốn kém hơn (chứa cả một quá trình Policy Evaluation lặp nhiều sweep).
- `gamma` càng gần 1 thì agent càng quan tâm đến phần thưởng ở tương lai
  xa; `gamma = 0` thì agent chỉ quan tâm đến phần thưởng ngay lập tức.
- `theta` càng nhỏ thì số iteration cần để hội tụ càng nhiều nhưng kết
  quả càng chính xác.

## Tài liệu tham khảo

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction* (2nd ed.). MIT Press.
- Tài liệu chính thức Gymnasium: https://gymnasium.farama.org/
- Tài liệu FrozenLake-v1: https://gymnasium.farama.org/environments/toy_text/frozen_lake/
