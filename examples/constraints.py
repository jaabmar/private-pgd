import torch


def approx_thres_func_constraint(
    X: torch.Tensor,
    thetas: torch.Tensor,
    taus: torch.Tensor,
    sigma: float = 5.0,
) -> torch.Tensor:
    """
    Compute a smooth sigmoid-based approximation for a given set of particles.

    Args:
    X (torch.Tensor): A tensor representing the particles, of shape [n_particles, n_features].
    thetas (torch.Tensor): A tensor of theta parameters, of shape [n_constraints, n_features].
    taus (torch.Tensor): A tensor of tau parameters, of shape [n_constraints].
    sigma (float, optional): A scaling factor for the exponent in the sigmoid function. Defaults to 5.0.

    Returns:
    torch.Tensor: A tensor representing the mean sigmoid values, of shape [n_constraints].
    """
    # Compute the dot product between the particles and theta
    dot_product = torch.matmul(X, thetas.t())

    # Compute the exponent term
    exponent = -sigma * (dot_product - taus)

    # Compute the sigmoid function
    sigmoid = 1 / (1 + torch.exp(exponent))

    # Calculate the mean of sigmoid values across all particles
    mean_sigmoid = torch.mean(sigmoid, dim=0)

    return mean_sigmoid


class RegularizedGradientDescent:
    def __init__(
        self, constraint_function: callable, noisy_estimate: torch.Tensor
    ):
        """
        Initialize the RegularizedGradientDescent class.

        This class is designed to perform gradient descent with a regularization term
        based on a constraint function and a noisy estimate.

        Args:
        constraint_function (callable): A function that applies a constraint function to the input data.
        noisy_estimate (torch.Tensor): A tensor with a noisy estimate to compare against the constraint function's output.
        """
        self.constraint_function = constraint_function
        self.noisy_estimate = noisy_estimate

    def get_gradient_and_loss(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and loss for the input tensor X.

        This method applies the constraint function to X, computes a constraint-based loss by
        comparing it to the noisy estimate, and then calculates the gradients.

        Args:
        X (torch.Tensor): A tensor representing the input data on which the gradient and loss are computed.

        Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the gradient tensor and the loss tensor.
        """
        # Compute the function values for each row in X
        f_X = self.constraint_function(X)

        # Compute the constraint loss
        constraint_loss = 0.01 / (
            0.0001 + torch.norm(f_X - self.noisy_estimate, 2) ** 2
        )

        # Compute the gradients without modifying X.grad
        (grad_X,) = torch.autograd.grad(
            constraint_loss, X, create_graph=True, retain_graph=True
        )

        # Clone the gradients before any alteration
        modified_grad_X = grad_X.clone()

        return modified_grad_X, constraint_loss
