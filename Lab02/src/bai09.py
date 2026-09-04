"""Bai 9. Tinh G_t cho moi t, tinh tu cuoi episode ve dau."""

from mdp_utils import discounted_returns

rewards = [0, 0, 0, 1]
gamma = 0.9


def main():
    G = discounted_returns(rewards, gamma)
    for t, g in enumerate(G):
        print(f"G_{t} = {g:.4f}")


if __name__ == "__main__":
    main()
