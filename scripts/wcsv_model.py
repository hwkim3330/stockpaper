#!/usr/bin/env python3
"""
WCSV (Weekday-Conditional Stochastic Volatility) Model
======================================================

Author: 김현우 (Hyunwoo Kim)
Email: hwkim3330@gmail.com
Date: 2025-10-23

This module implements the Weekday-Conditional Stochastic Volatility (WCSV) model,
which extends the standard GARCH(1,1) model by allowing weekday-specific parameters.

Model Specification:
    r_t = μ_d + sqrt(h_t) * ε_t
    h_t = ω_d + α_d * r_{t-1}^2 + β_d * h_{t-1}

where d ∈ {0,1,2,3,4} represents the weekday (Mon=0, ..., Fri=4)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class WCSVModel:
    """Weekday-Conditional Stochastic Volatility Model"""

    def __init__(self):
        self.params = None
        self.fitted = False
        self.log_likelihood = None
        self.aic = None
        self.bic = None

    def negative_log_likelihood(self, params, returns, weekday):
        """
        Compute negative log-likelihood for optimization

        Parameters:
        -----------
        params : array-like
            Model parameters [mu_0...mu_4, omega_0...omega_4, alpha_0...alpha_4, beta_0...beta_4]
        returns : array-like
            Return series
        weekday : array-like
            Weekday indicators (0=Mon, ..., 4=Fri)

        Returns:
        --------
        float : Negative log-likelihood
        """
        n = len(returns)

        # Extract parameters
        mu = params[0:5]
        omega = params[5:10]
        alpha = params[10:15]
        beta = params[15:20]

        # Check parameter constraints
        if np.any(omega < 0) or np.any(alpha < 0) or np.any(beta < 0):
            return 1e10
        if np.any(alpha + beta >= 1):
            return 1e10

        # Initialize conditional variance
        h = np.zeros(n)
        h[0] = np.var(returns)

        # Compute log-likelihood
        log_lik = 0
        for t in range(1, n):
            d = int(weekday[t])

            # Update conditional variance
            h[t] = omega[d] + alpha[d] * (returns[t-1] - mu[d])**2 + beta[d] * h[t-1]

            # Ensure positive variance
            if h[t] <= 0:
                h[t] = 1e-6

            # Add to log-likelihood
            log_lik += -0.5 * (np.log(2 * np.pi) + np.log(h[t]) +
                               (returns[t] - mu[d])**2 / h[t])

        return -log_lik

    def fit(self, returns, weekday, initial_params=None, method='L-BFGS-B'):
        """
        Fit WCSV model using Maximum Likelihood Estimation

        Parameters:
        -----------
        returns : array-like
            Return series
        weekday : array-like
            Weekday indicators (0=Mon, ..., 4=Fri)
        initial_params : array-like, optional
            Initial parameter values
        method : str
            Optimization method

        Returns:
        --------
        self : WCSVModel
            Fitted model
        """
        # Default initial parameters
        if initial_params is None:
            initial_params = np.concatenate([
                np.zeros(5),           # mu
                np.ones(5) * 0.0002,   # omega
                np.ones(5) * 0.08,     # alpha
                np.ones(5) * 0.9       # beta
            ])

        # Parameter bounds
        bounds = (
            [(None, None)] * 5 +      # mu: unrestricted
            [(1e-6, None)] * 5 +      # omega: positive
            [(0, 0.3)] * 5 +          # alpha: [0, 0.3]
            [(0, 0.999)] * 5          # beta: [0, 0.999]
        )

        # Optimize
        result = minimize(
            self.negative_log_likelihood,
            initial_params,
            args=(returns, weekday),
            method=method,
            bounds=bounds,
            options={'maxiter': 1000, 'disp': False}
        )

        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")

        # Store results
        self.params = result.x
        self.log_likelihood = -result.fun
        self.fitted = True

        # Compute information criteria
        n_params = len(self.params)
        n_obs = len(returns)
        self.aic = 2 * n_params - 2 * self.log_likelihood
        self.bic = np.log(n_obs) * n_params - 2 * self.log_likelihood

        return self

    def get_parameters(self):
        """Return estimated parameters as a DataFrame"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        params_df = pd.DataFrame({
            'Weekday': weekdays,
            'mu': self.params[0:5],
            'omega': self.params[5:10],
            'alpha': self.params[10:15],
            'beta': self.params[15:20]
        })

        # Compute unconditional volatility for each weekday
        params_df['sigma'] = np.sqrt(
            params_df['omega'] / (1 - params_df['alpha'] - params_df['beta'])
        )

        return params_df

    def predict_variance(self, returns, weekday, h0=None):
        """
        Predict conditional variance series

        Parameters:
        -----------
        returns : array-like
            Return series
        weekday : array-like
            Weekday indicators
        h0 : float, optional
            Initial variance

        Returns:
        --------
        array : Conditional variance series
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        n = len(returns)
        mu = self.params[0:5]
        omega = self.params[5:10]
        alpha = self.params[10:15]
        beta = self.params[15:20]

        h = np.zeros(n)
        h[0] = h0 if h0 is not None else np.var(returns)

        for t in range(1, n):
            d = int(weekday[t])
            h[t] = omega[d] + alpha[d] * (returns[t-1] - mu[d])**2 + beta[d] * h[t-1]

        return h

    def summary(self):
        """Print model summary"""
        if not self.fitted:
            print("Model not fitted yet")
            return

        print("=" * 70)
        print("WCSV Model Estimation Results")
        print("=" * 70)
        print(f"Log-Likelihood: {self.log_likelihood:.4f}")
        print(f"AIC: {self.aic:.4f}")
        print(f"BIC: {self.bic:.4f}")
        print("\n")

        params_df = self.get_parameters()
        print(params_df.to_string(index=False))
        print("=" * 70)


def example_usage():
    """Example usage of WCSVModel"""

    # Generate synthetic data
    np.random.seed(42)
    n_weeks = 500
    n_days = n_weeks * 5

    # True parameters (different volatility for Mon and Fri)
    mu_true = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
    omega_true = np.array([0.00024, 0.00018, 0.00016, 0.00017, 0.00021])
    alpha_true = np.array([0.089, 0.075, 0.071, 0.073, 0.082])
    beta_true = np.array([0.881, 0.898, 0.905, 0.902, 0.887])

    # Simulate returns
    returns = np.zeros(n_days)
    h = np.zeros(n_days)
    weekday = np.tile(np.arange(5), n_weeks)

    h[0] = 0.0001
    returns[0] = np.random.randn() * np.sqrt(h[0])

    for t in range(1, n_days):
        d = weekday[t]
        h[t] = omega_true[d] + alpha_true[d] * returns[t-1]**2 + beta_true[d] * h[t-1]
        returns[t] = mu_true[d] + np.random.randn() * np.sqrt(h[t])

    # Fit model
    print("Fitting WCSV model...")
    model = WCSVModel()
    model.fit(returns, weekday)

    # Print results
    model.summary()

    print("\nTrue parameters (for reference):")
    true_params = pd.DataFrame({
        'Weekday': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'mu': mu_true,
        'omega': omega_true,
        'alpha': alpha_true,
        'beta': beta_true,
        'sigma': np.sqrt(omega_true / (1 - alpha_true - beta_true))
    })
    print(true_params.to_string(index=False))


if __name__ == '__main__':
    example_usage()
