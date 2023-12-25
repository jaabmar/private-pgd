import math
from functools import partial

import numpy as np
from scipy.special import softmax

from mechanisms.cdp2adp import cdp_rho
from mechanisms.privacy_calibrator import ana_gaussian_mech


class Mechanism:
    def __init__(
        self,
        epsilon: float,
        delta: float,
        bounded: bool = True,
        prng: np.random = np.random,
    ):
        """
        Base class for a mechanism.

        Args:
            epsilon (float): Privacy parameter.
            delta (float): Privacy parameter.
            bounded (bool): Privacy definition (bounded vs unbounded DP).
            prng (np.random): Pseudo-random number generator.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.rho = 0 if delta == 0 else cdp_rho(epsilon, delta)
        self.bounded = bounded
        self.sensitivity = 2.0 if self.bounded else 1.0
        self.marginal_sensitivity = np.sqrt(2) if self.bounded else 1.0
        self.prng = prng

    def run(self, data, workload, engine):
        pass

    def exponential_mechanism(self, qualities, epsilon, base_measure=None):
        if isinstance(qualities, dict):
            keys = list(qualities.keys())
            qualities = np.array([qualities[key] for key in keys])
            if base_measure is not None:
                base_measure = np.log([base_measure[key] for key in keys])
        else:
            qualities = np.array(qualities)
            keys = np.arange(qualities.size)

        q = qualities - qualities.max()
        if base_measure is None:
            p = softmax(0.5 * epsilon / self.sensitivity * q)
        else:
            p = softmax(0.5 * epsilon / self.sensitivity * q + base_measure)

        return keys[self.prng.choice(p.size, p=p)]

    def gaussian_noise_scale(self, l2_sensitivity, epsilon, delta):
        """
        Return the Gaussian noise scale to attain (epsilon, delta)-DP.

        Args:
            l2_sensitivity: L2 sensitivity parameter.
            epsilon: Privacy parameter.
            delta: Privacy parameter.

        Returns:
            Gaussian noise scale.
        """
        if self.bounded:
            l2_sensitivity *= math.sqrt(2.0)
        return l2_sensitivity * ana_gaussian_mech(epsilon, delta)["sigma"]

    def laplace_noise_scale(self, l1_sensitivity, epsilon):
        """
        Return the Laplace noise scale necessary to attain epsilon-DP.

        Args:
            l1_sensitivity: L1 sensitivity parameter.
            epsilon: Privacy parameter.

        Returns:
            Laplace noise scale.
        """
        if self.bounded:
            l1_sensitivity *= 2.0
        return l1_sensitivity / epsilon

    def gaussian_noise(self, sigma, size):
        """
        Generate iid Gaussian noise of a given scale and size.

        Args:
            sigma: Noise scale.
            size: Size of the noise.

        Returns:
            Generated noise.
        """
        return self.prng.normal(0, sigma, size)

    def laplace_noise(self, b, size):
        """
        Generate iid Laplace noise of a given scale and size.

        Args:
            b: Noise scale.
            size: Size of the noise.

        Returns:
            Generated noise.
        """
        return self.prng.laplace(0, b, size)

    def best_noise_distribution(
        self, l1_sensitivity, l2_sensitivity, epsilon, delta
    ):
        """
        Determine the best noise distribution (Laplace or Gaussian).

        Args:
            l1_sensitivity: L1 sensitivity parameter.
            l2_sensitivity: L2 sensitivity parameter.
            epsilon: Privacy parameter.
            delta: Privacy parameter.

        Returns:
            Function that samples from the appropriate noise distribution.
        """
        b = self.laplace_noise_scale(l1_sensitivity, epsilon)
        sigma = self.gaussian_noise_scale(l2_sensitivity, epsilon, delta)
        if np.sqrt(2) * b < sigma:
            return partial(self.laplace_noise, b)
        return partial(self.gaussian_noise, sigma)
