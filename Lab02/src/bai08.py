"""Bai 8. Discounted return voi nhieu gia tri gamma."""

from bai07 import rewards
from mdp_utils import compute_return

gammas = [0.0, 0.5, 0.9, 0.99, 1.0]


def main():
    print("| Gamma | Return |")
    print("|---:|---:|")
    for g in gammas:
        G = compute_return(rewards, g)
        print(f"| {g:.2f} | {G:.4f} |")


if __name__ == "__main__":
    main()
