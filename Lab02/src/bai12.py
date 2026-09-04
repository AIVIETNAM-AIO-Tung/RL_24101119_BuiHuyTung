"""Bai 12. Xay dung MDP nho: 2 state, 2 action.

State 0: "Ngoi hoc"
State 1: "Choi game"
Action 0: "Tiep tuc"
Action 1: "Chuyen hoat dong"
"""

N_STATES = 2
N_ACTIONS = 2

# P[state][action] = list cac (probability, next_state, reward, terminated)
P = {
    0: {  # State 0 - Ngoi hoc
        0: [(0.8, 0, 1.0, False), (0.2, 1, 0.0, False)],   # Tiep tuc hoc
        1: [(0.6, 1, -0.5, False), (0.4, 0, 0.0, False)],  # Chuyen sang choi
    },
    1: {  # State 1 - Choi game
        0: [(0.9, 1, -1.0, False), (0.1, 0, 0.5, False)],  # Tiep tuc choi
        1: [(0.7, 0, 0.5, False), (0.3, 1, -1.0, False)],  # Chuyen sang hoc
    },
}


def main():
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            print(f"P[{s}][{a}] = {P[s][a]}")


if __name__ == "__main__":
    main()
