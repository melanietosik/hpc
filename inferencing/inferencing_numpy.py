import numpy as np
import time


def multiply(A, B):
    """
    Matrix multiplication
    """
    A_row, A_col = A.shape
    B_row, B_col = B.shape
    if A_col != B_row:
        exit()

    C = np.zeros((A_row, B_col))

    for i in range(A_row):
        for j in range(B_col):
            for k in range(A_col):
                C[i][j] += A[i][k] * B[k][j]
    return C


def relu(A):
    """
    ReLU (rectified linear unit) activation function
    """
    row, col = A.shape
    for i in range(row):
        for j in range(col):
            if A[i, j] < 0:
                A[i, j] = 0
    return A


def init(row, col):
    """
    Initialization
    """
    return np.fromfunction(
        lambda i, j: (0.4 + ((i + j) % 40 - 20) / 40.0),
        (row, col),
        dtype=np.float64,
    )


# Specify dimensions of input and hidden layers
I_0 = 256 * 256  # Input [l=0]
H_0 = 4000       # Hidden layer [l=0]
H_1 = 1000       # Hidden layer [l=1]

"""
C3
"""
print("*C3*")
x_0 = init(I_0, 1)
W_0 = init(H_0, I_0)
W_1 = init(H_1, H_0)

start = time.monotonic()

z_0 = relu(multiply(W_0, x_0))
z_1 = relu(multiply(W_1, z_0))

end = time.monotonic()
print("exectime={}".format(str(round((end - start), 2))))

S = np.sum(z_1)
print("checksum={}".format(round(S, 2)))

"""
C4
"""
print("*C4*")
x_0 = init(I_0, 1)
W_0 = init(H_0, I_0)
W_1 = init(H_1, H_0)

start = time.monotonic()

z_0 = np.clip(np.dot(W_0, x_0), a_min=0, a_max=None)
z_1 = np.clip(np.dot(W_1, z_0), a_min=0, a_max=None)

end = time.monotonic()
print("exectime={}".format(str(round((end - start), 2))))

S = np.sum(z_1)
print("checksum={}".format(round(S, 2)))
