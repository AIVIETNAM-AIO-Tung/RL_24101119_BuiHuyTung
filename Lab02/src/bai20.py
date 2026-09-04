"""Bai 20. So sanh model deterministic va stochastic tai state=0, action=RIGHT."""

import gymnasium as gym

STATE = 0
ACTION = 2  # RIGHT


def main():
    for slippery in [False, True]:
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery)
        env.reset()
        P = env.unwrapped.P
        transitions = P[STATE][ACTION]

        print(f"is_slippery={slippery}: so transitions = {len(transitions)}")
        for prob, next_state, reward, terminated in transitions:
            print(f"  prob={prob:.3f}  next_state={next_state}  "
                  f"reward={reward}  terminated={terminated}")
        print()

        env.close()

    print("Ket luan:")
    print("- Khi is_slippery=False: moi (state, action) chi dan den DUY NHAT")
    print("  mot next_state voi xac suat 1.0 (moi truong hoan toan xac dinh).")
    print("- Khi is_slippery=True: moi action co xac suat truot sang huong")
    print("  vuong goc, nen mot action co the dan den 3 next_state khac nhau")
    print("  (huong dinh, va 2 huong vuong goc), moi huong xac suat ~1/3.")
    print("- Vi vay khi lam Dynamic Programming voi is_slippery=True, bat")
    print("  buoc phai duyet va cong theo xac suat cho tung transition, khong")
    print("  the gia dinh chi co mot next_state.")


if __name__ == "__main__":
    main()
