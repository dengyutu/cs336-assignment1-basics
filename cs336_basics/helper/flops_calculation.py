if __name__ == "__main__":
    # T = 1024
    T = 16384
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

    W_QKV = L * 6 * T * d**2
    QK = L * 2 * T**2 * d
    attn = L * 2 * T**2 * d
    W_O = L * 2 * T * d**2
    SwiGLU = L * 6 * T * d * d_ff
    LM = 2 * T * d * V
    Total = W_QKV + QK + attn + W_O + SwiGLU + LM
    print(f"{Total = }")
    print(f"{W_QKV/Total = }")
    print(f"{QK/Total = }")
    print(f"{attn/Total = }")
    print(f"{W_O/Total = }")
    print(f"{SwiGLU/Total = }")
    print(f"{LM/Total = }")
