from matplotlib import pyplot as plt
import torch
from core.utils import BetaInc, Phi
import numpy as np

def I(l, u):
    """
    Computes the incomplete beta function.
    """
    c = torch.tensor(1 / 2)
    return BetaInc.apply(l, u, c, torch.tensor(1))

def deterministic_bound(Gibbs_risk, l, u, l_1_norm, distribution):
    """
    Computes Ben's bound, given Gibbs risk, u and l.
    """
    if distribution == "gaussian":
        phi_l, phi_u = Phi(l), Phi(u)
        return (Gibbs_risk - phi_u) / (1 - phi_l - phi_u)
    elif distribution == "dirichlet":
        I_l, I_u = I(l_1_norm - l, l), I(u, l_1_norm - u)
        return (Gibbs_risk - I_u) / (I_l - I_u)
    elif distribution == "categorical":
        return (Gibbs_risk - (1 - u)) / (l - (1 - u))

def loop(n, base):
    """
    Main loop
    """
    x = []
    categorical = []
    dirichlet = []
    gaussian = []
    for i in range(500, 1001):
        x.append(i / 1000)
        categorical.append(1000 / i)
        dirichlet.append(deterministic_bound(1, i * base / 1000, base, base, "dirichlet"))
        gaussian.append(
            deterministic_bound(1, torch.tensor((2 * (i * base / 1000) - base) / n ** 0.5), torch.tensor(10), base,
                                "gaussian"))
    return x, categorical, dirichlet, gaussian

x, categorical, dirichlet, gaussian = loop(10, 10)
plt.plot(x, categorical, c='blue')
plt.plot(x, dirichlet, c='orange')
plt.plot(x, gaussian, c='green')

plt.plot([0.5], [2], c='black')
plt.plot([0.5], [2], ls='--', c='black')

*_, dirichlet, gaussian = loop(100, 100)

plt.plot(x, np.array(categorical) + 0.0035, ls='--', c='blue')
plt.plot(x, dirichlet, ls='-', c='orange')
plt.plot(x, gaussian, ls='--', c='green')

plt.legend(['Categorical', 'Dirichlet', 'Gaussian', 'n = $|\mathbf{w}|$ = 10', 'n = $|\mathbf{w}|$ = 100'])

plt.title('Deterministic bounding factor as a function of $\min_{\mathbf{x}} |\mathbf{w}\cdot\mathbf{f}(\mathbf{x})|$ \n when $d(\cdot, h_w) = 0$')
plt.ylabel('Deterministic bounding factor')
plt.xlabel('$\min_{\mathbf{x}} |\mathbf{w}\cdot\mathbf{f}(\mathbf{x})|$')
plt.show()