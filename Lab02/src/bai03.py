"""Bai 3. Tinh xac suat trang thai ke tiep sau mot buoc."""

import numpy as np

from bai01 import P, STATES

p0 = np.array([1.0, 0.0, 0.0])


def main():
    p1 = p0 @ P  # KHONG hard-code, tinh bang phep nhan ma tran
    print("p0 =", p0)
    print("p1 =", p1)
    for name, prob in zip(STATES, p1):
        print(f"  P({name}) = {prob:.3f}")


if __name__ == "__main__":
    main()
