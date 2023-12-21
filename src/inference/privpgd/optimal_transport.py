from typing import Optional, Tuple

import cvxpy as cp
import numpy as np
import ot
import torch


def project_onto_probability_space_marginal(
    M: np.ndarray, vdp: np.ndarray
) -> np.ndarray:
    """
    Projects a matrix onto the probability space considering the marginal distribution.

    Args:
        M (np.ndarray): A square matrix representing the cost.
        vdp (np.ndarray): The vector of the dual problem.

    Returns:
        np.ndarray: The projected vector.
    """
    n = M.shape[0]

    # dual variables
    x = cp.Variable((n, n), nonneg=True)
    y = cp.Variable(n, nonneg=True)
    v = cp.Variable(n, nonneg=True)
    # The objective is to minimize the dot product of M and x
    objective = cp.Minimize(cp.sum(cp.multiply(M, x)) + 2 * cp.sum(y))

    # Constraints
    constraints = [
        cp.diag(x) == np.zeros(n),
        cp.sum(x, axis=0) - cp.sum(x, axis=1) >= v - vdp - y,  # sum over rows
        cp.sum(v) == 1,
    ]
    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()
    v_value = v.value

    # get rid of numerical issues
    return v_value / sum(v_value)


def sinkhorn_gradient_weights(
    a: np.ndarray, b: np.ndarray, M: np.ndarray, reg: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the gradient of the Sinkhorn algorithm with respect to the weights.

    Args:
        a (np.ndarray): The source distribution.
        b (np.ndarray): The target distribution.
        M (np.ndarray): The cost matrix.
        reg (float): The regularization parameter.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The gradient with respect to the weights, and the log of the dual variable 'u'.
    """
    mat, logs = ot.bregman.sinkhorn_log(a, b, M, reg, log=True)
    return torch.sum(mat * M), reg * torch.log(logs["u"])


def sinkhorn_gradient_locations(
    a: np.ndarray, b: np.ndarray, M: np.ndarray, reg: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the gradient of the Sinkhorn algorithm with respect to the locations.

    Args:
        a (np.ndarray): The source distribution.
        b (np.ndarray): The target distribution.
        M (np.ndarray): The cost matrix.
        reg (float): The regularization parameter.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The gradient with respect to the locations, and the Sinkhorn matrix.
    """
    mat = ot.bregman.sinkhorn_log(a, b, M, reg, log=False)
    return torch.sum(mat * M), mat


def sinkhorn_divergence_gradient(
    a: np.ndarray, b: np.ndarray, M: np.ndarray, Ma: np.ndarray, reg: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the gradient of the Sinkhorn divergence.

    Args:
        a (np.ndarray): The source distribution.
        b (np.ndarray): The target distribution.
        M (np.ndarray): The cost matrix for the (a, b) comparison.
        Ma (np.ndarray): The cost matrix for the (a, a) comparison.
        reg (float): The regularization parameter.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The Sinkhorn distance without entropy, the Sinkhorn
        matrices for (a, b) and (a, a), and the gradient.
    """
    a = torch.abs(a) / torch.sum(torch.abs(a))
    b = torch.abs(b) / torch.sum(torch.abs(b))
    mat, logs = ot.bregman.sinkhorn_log(a, b, M, reg, log=True)
    amat, alogs = ot.bregman.sinkhorn_log(a, a, Ma, reg, log=True)
    f = reg * torch.log(logs["u"] + 1e-20)
    af, ag = reg * torch.log(alogs["u"] + 1e-20), reg * torch.log(
        alogs["v"] + 1e-20
    )
    sinkhorn_distance_without_entropy = torch.sum(mat * M)
    gradw = f - 1 / 2 * (af + ag)
    return sinkhorn_distance_without_entropy, mat, amat, gradw


def sinkhorn_gradient(
    a: np.ndarray, b: np.ndarray, M: np.ndarray, reg: float
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Computes the Sinkhorn gradient.

    Args:
        a (np.ndarray): The source distribution.
        b (np.ndarray): The target distribution.
        M (np.ndarray): The cost matrix.
        reg (float): The regularization parameter.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]: The Sinkhorn distance without entropy,
        the Sinkhorn matrix, None, and the gradient.
    """
    a = torch.abs(a) / torch.sum(torch.abs(a))
    b = torch.abs(b) / torch.sum(torch.abs(b))

    mat, logs = ot.bregman.sinkhorn_log(a, b, M, reg, log=True)
    f = reg * torch.log(logs["u"])
    sinkhorn_distance_without_entropy = torch.sum(mat * M)
    f = torch.clamp(f, -100.0, 100.0)
    gradw = f
    return sinkhorn_distance_without_entropy, mat, None, gradw
