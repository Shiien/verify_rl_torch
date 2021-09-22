import numpy as np
import torch

maze = [
    [-1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
]
torch.manual_seed(2021)
V = torch.normal(0, 1, [5, 5])
A = torch.normal(0, 1, [5, 5, 4])
A = torch.nn.Parameter(A)
optim = torch.optim.SGD([A], 0.1)

R = torch.zeros([5, 5])
R = R - 1
R[4][4] = 10
move = [[0, 1], [1, 0], [-1, 0], [0, -1]]
gamma = 1
P_old = A.softmax(dim=-1).detach()
P = A.softmax(dim=-1).detach()
Q_old = A - (A * P_old).sum(dim=-1, keepdim=True) + V.unsqueeze(-1)
Q = torch.normal(0, 1, [5, 5, 4])
for T in range(2000):
    total_err = 0
    for i in range(5):
        for j in range(5):
            if i == 4 and j == 4:
                V[i][j] = 0
                Q[i][j][:] = 0
                continue
            vans = []
            for k, [x, y] in enumerate(move):
                if 0 <= i + x < 5 and 0 <= y + j < 5:
                    vans.append((R[i + x][y + j] + gamma * V[i + x][y + j]) * P_old[i][j][k])
            V[i][j] = sum(vans)
            for k, [x, y] in enumerate(move):
                if 0 <= i + x < 5 and 0 <= y + j < 5:
                    Q[i][j][k] = R[i + x][y + j] + gamma * V[i + x][j + y]
                else:
                    Q[i][j][k] = 0
    Q_target = Q# - V.unsqueeze(-1)
    Q_new = A - (A * P_old).sum(dim=-1, keepdim=True) + V.unsqueeze(-1)
    A_loss = ((Q_target.detach()-Q_new) ** 2).sum()
    optim.zero_grad()
    A_loss.backward()
    optim.step()
    P_new = A.softmax(dim=-1).detach()

    total_err += (P_new-P).abs().sum().item()
    P = P_new
    print(T, total_err)
    if total_err < 1e-6:
        break

# I_P = 2 * torch.eye(4).unsqueeze(0).unsqueeze(0).expand([5, 5, -1, -1]) - P.unsqueeze(-1).expand([-1, -1, -1, 4])
# ans = torch.inverse(I_P)  # @(Q-V.unsqueeze(-1))
# ans = torch.einsum('ijxy,ijy->ijx', ans, Q - V.unsqueeze(-1))
# A_new = Q-V.unsqueeze(-1)
# Q_new = A_new - (A_new * P).sum(dim=-1, keepdim=True) + V.unsqueeze(-1)
# QQ = Q-(Q*P).sum(dim=-1,keepdim=True)+V.unsqueeze(-1)
print(V, P[4][3], P[3][4], P[4][4])
