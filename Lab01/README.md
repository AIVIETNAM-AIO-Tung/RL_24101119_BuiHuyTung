# Bai thuc hanh so 1 - Lam quen voi Gymnasium

Ho ten: Bùi Huy Tùng
MSSV: 24101119
Lop: EEE.AI-24106.1
GitHub username: [https://github.com/AIVIETNAM-AIO-Tung]
Repository URL: https://github.com/AIVIETNAM-AIO-Tung/RL_24101119_BuiHuyTung.git

Python version: 3.12+ (xem chi tiet bang `python src/bai01.py`)
Gymnasium version: 1.3.0
NumPy version: xem bang `python src/bai01.py`
Matplotlib version: xem bang `pip show matplotlib`

## Cach cai dat

```bash
# Tao virtual environment
python -m venv .venv

# Kich hoat (Linux/macOS)
source .venv/bin/activate
# Kich hoat (Windows)
.venv\Scripts\activate

# Cai thu vien
pip install -r requirements.txt
```

## Cach chay tung bai

Moi bai tap nam trong file rieng `src/baiXX.py` va co the chay doc lap:

```bash
cd Lab01
python src/bai01.py
python src/bai02.py
...
python src/bai36.py   # hoac tuong duong voi src/main.py
```

File chuyen doi API Gym cu sang Gymnasium (Phan 8):

```bash
python src/migration_gym_to_gymnasium.py
```

Chuong trinh khoi dong (khong phai bai nop):

```bash
python src/starter.py
```

## Cach chay chuong trinh tong hop

`src/main.py` la mini-project (Bai 36): pipeline hoan chinh
create_environment -> policy -> run_episode -> evaluate_policy -> plot_results.

```bash
python src/main.py
```

Ket qua in ra thong ke reward (mean, std, min, max, mean length) va luu
bieu do vao `figures/mini_project_rewards.png`.

## Mo ta ket qua

- `figures/reward_cartpole.png`: reward theo tung episode cua random agent
  tren CartPole-v1 (100 episode) - Bai 17.
- `figures/moving_average.png`: reward goc va moving average (window=10)
  cua random agent - Bai 18.
- `figures/comparison_agents.png`: so sanh mean reward giua 3 agent
  (Random, Angle-based, Improved) tren CartPole-v1, moi agent 500 episode
  - Bai 35.
- `figures/mini_project_rewards.png`: reward va moving average cua policy
  cai tien trong mini-project (Bai 36).

Ket qua chinh: policy heuristic dua tren pole_angle va
pole_angular_velocity (Bai 32, 35, 36) cai thien manh hieu suat so voi
random policy, thuong xuyen dat reward toi da 500 tren CartPole-v1.

## Cau tra loi cau hoi ly thuyet

Xem chi tiet trong notebook `notebooks/Lab01_MSSV_HoTen.ipynb`, phan cuoi
cung (Muc 10 cua de bai).

## Kho khan gap phai

[Ghi lai kho khan thuc te ban gap phai khi lam bai, vi du: phan biet
terminated/truncated luc dau con nham lan, can doc lai tai lieu
Gymnasium; chinh cua so moving average sao cho bieu do de nhin hon; v.v.]

## Ket luan

Qua bai thuc hanh nay, sinh vien da lam quen voi vong lap tuong tac co ban
cua Reinforcement Learning (Agent -> Action -> Environment -> Observation +
Reward -> Agent), su dung dung API moi cua Gymnasium (`reset()`, `step()`,
phan biet `terminated`/`truncated`), xay dung random agent, thu thap va
phan tich thong ke reward qua nhieu episode, va buoc dau xay dung cac
policy don gian (always-left/right, heuristic dua tren goc nghieng) cho
thay policy tot hon co the cai thien dang ke hieu suat so voi hanh dong
ngau nhien. Day la nen tang can thiet truoc khi tiep can cac thuat toan RL
chinh thuc nhu Q-Learning, SARSA hay DQN.
