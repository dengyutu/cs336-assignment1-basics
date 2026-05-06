import sympy

if __name__ == "__main__":
    T = 1024
    V = 50257
    # GPT-2 small
    # L = 12
    # d = 768
    # d_ff = 4 * d
    # GPT-2 medium
    # L = 24
    # d = 1024
    # d_ff = 4 * d
    # GPT-2 Large
    # L = 36
    # d = 1280
    # d_ff = 4 * d
    # GPT-2 XL
    L = 48
    d = 1600
    d_ff = 4 * d
    H = 25
    B = sympy.Symbol("B")

    parameters = L * (12 * d**2 + 2 * d) + d + V * d
    gradients = parameters
    optimizer_states = 2 * parameters
    activations = L * (16 * B * T * d + B * H * (T**2)) + 2 * B * T * d + B * T * V
    total = parameters + gradients + optimizer_states + activations
    print(f"{parameters=}")
    print(f"{total=}")
