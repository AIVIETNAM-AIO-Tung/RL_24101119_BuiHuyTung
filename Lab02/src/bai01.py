"""Bai 1. Tao transition matrix cho Markov chain 3 trang thai thoi tiet."""

import numpy as np

STATES = ["Sunny", "Cloudy", "Rainy"]

# Hang i = trang thai hien tai, cot j = trang thai ke tiep.
# Sunny -> Sunny/Cloudy/Rainy, Cloudy -> ..., Rainy -> ...
P = np.array([
    [0.7, 0.2, 0.1],   # Sunny
    [0.3, 0.4, 0.3],   # Cloudy
    [0.2, 0.3, 0.5],   # Rainy
])


def main():
    print("Transition matrix P (Sunny, Cloudy, Rainy):")
    print(P)
    print("\nTong moi hang:", P.sum(axis=1))


if __name__ == "__main__":
    main()
