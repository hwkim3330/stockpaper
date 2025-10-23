#!/usr/bin/env python3
"""
Monte Carlo Simulation for WCSV Model
======================================

Author: 김현우 (Hyunwoo Kim)
Email: hwkim3330@gmail.com
Date: 2025-10-23

This module performs Monte Carlo simulations to:
1. Validate the WCSV model
2. Compare WCSV vs standard GARCH
3. Analyze extreme value distributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')


class GARCHSimulator:
    """Standard GARCH(1,1) simulator"""

    def __init__(self, mu, omega, alpha, beta):
        self.mu = mu
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

    def simulate(self, n_days, h0=None, seed=None):
        """Simulate GARCH process"""
        if seed is not None:
            np.random.seed(seed)

        returns = np.zeros(n_days)
        h = np.zeros(n_days)

        h[0] = h0 if h0 is not None else self.omega / (1 - self.alpha - self.beta)
        returns[0] = self.mu + np.random.randn() * np.sqrt(h[0])

        for t in range(1, n_days):
            h[t] = self.omega + self.alpha * (returns[t-1] - self.mu)**2 + self.beta * h[t-1]
            returns[t] = self.mu + np.random.randn() * np.sqrt(h[t])

        return returns, h


class WCSVSimulator:
    """WCSV model simulator"""

    def __init__(self, mu, omega, alpha, beta):
        """
        Parameters should be arrays of length 5 (one for each weekday)
        """
        self.mu = np.array(mu)
        self.omega = np.array(omega)
        self.alpha = np.array(alpha)
        self.beta = np.array(beta)

    def simulate(self, n_weeks, h0=None, seed=None):
        """Simulate WCSV process"""
        if seed is not None:
            np.random.seed(seed)

        n_days = n_weeks * 5
        weekday = np.tile(np.arange(5), n_weeks)

        returns = np.zeros(n_days)
        h = np.zeros(n_days)

        # Initial variance
        h[0] = h0 if h0 is not None else np.mean(
            self.omega / (1 - self.alpha - self.beta)
        )
        returns[0] = self.mu[0] + np.random.randn() * np.sqrt(h[0])

        for t in range(1, n_days):
            d = weekday[t]
            h[t] = self.omega[d] + self.alpha[d] * (returns[t-1] - self.mu[d])**2 + self.beta[d] * h[t-1]
            returns[t] = self.mu[d] + np.random.randn() * np.sqrt(h[t])

        return returns, h, weekday


def analyze_weekly_extremes_simulation(returns, weekday, n_weeks):
    """
    Analyze weekday distribution of weekly extremes from simulation

    Returns:
    --------
    dict : Counts and proportions of extremes by weekday
    """
    # Reshape into weeks
    returns_weekly = returns.reshape(n_weeks, 5)

    # Find day of weekly high and low
    high_days = np.argmax(returns_weekly, axis=1)
    low_days = np.argmin(returns_weekly, axis=1)

    # Count occurrences
    high_counts = np.bincount(high_days, minlength=5)
    low_counts = np.bincount(low_days, minlength=5)

    # Convert to proportions
    high_props = high_counts / n_weeks
    low_props = low_counts / n_weeks

    return {
        'high_counts': high_counts,
        'low_counts': low_counts,
        'high_props': high_props,
        'low_props': low_props
    }


def compare_models(n_simulations=1000, n_weeks=500):
    """
    Compare WCSV vs standard GARCH via simulation

    Parameters:
    -----------
    n_simulations : int
        Number of Monte Carlo simulations
    n_weeks : int
        Number of weeks per simulation

    Returns:
    --------
    dict : Comparison results
    """
    print(f"Running {n_simulations} simulations with {n_weeks} weeks each...")

    # WCSV parameters (Monday and Friday have higher volatility)
    mu_wcsv = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
    omega_wcsv = np.array([0.00024, 0.00018, 0.00016, 0.00017, 0.00021])
    alpha_wcsv = np.array([0.089, 0.075, 0.071, 0.073, 0.082])
    beta_wcsv = np.array([0.881, 0.898, 0.905, 0.902, 0.887])

    # Standard GARCH parameters (average of WCSV)
    mu_garch = np.mean(mu_wcsv)
    omega_garch = np.mean(omega_wcsv)
    alpha_garch = np.mean(alpha_wcsv)
    beta_garch = np.mean(beta_wcsv)

    # Initialize simulators
    wcsv_sim = WCSVSimulator(mu_wcsv, omega_wcsv, alpha_wcsv, beta_wcsv)
    garch_sim = GARCHSimulator(mu_garch, omega_garch, alpha_garch, beta_garch)

    # Store results
    wcsv_high_props = np.zeros((n_simulations, 5))
    garch_high_props = np.zeros((n_simulations, 5))

    for i in range(n_simulations):
        if (i+1) % 100 == 0:
            print(f"  Completed {i+1}/{n_simulations} simulations...")

        # WCSV simulation
        wcsv_returns, wcsv_h, wcsv_weekday = wcsv_sim.simulate(n_weeks, seed=i)
        wcsv_results = analyze_weekly_extremes_simulation(wcsv_returns, wcsv_weekday, n_weeks)
        wcsv_high_props[i, :] = wcsv_results['high_props']

        # GARCH simulation
        garch_returns, garch_h = garch_sim.simulate(n_weeks * 5, seed=i+1000)
        garch_weekday = np.tile(np.arange(5), n_weeks)
        garch_results = analyze_weekly_extremes_simulation(garch_returns, garch_weekday, n_weeks)
        garch_high_props[i, :] = garch_results['high_props']

    return {
        'wcsv_high_props': wcsv_high_props,
        'garch_high_props': garch_high_props,
        'wcsv_params': (mu_wcsv, omega_wcsv, alpha_wcsv, beta_wcsv),
        'garch_params': (mu_garch, omega_garch, alpha_garch, beta_garch)
    }


def plot_comparison(results):
    """Plot simulation comparison results"""

    wcsv_props = results['wcsv_high_props']
    garch_props = results['garch_high_props']
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

    # Compute mean proportions
    wcsv_mean = np.mean(wcsv_props, axis=0)
    garch_mean = np.mean(garch_props, axis=0)
    uniform = np.ones(5) * 0.2

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean proportions
    x = np.arange(5)
    width = 0.25

    axes[0].bar(x - width, uniform, width, label='Uniform (Expected)', color='gray', alpha=0.5)
    axes[0].bar(x, wcsv_mean, width, label='WCSV', color='#3498db')
    axes[0].bar(x + width, garch_mean, width, label='Standard GARCH', color='#e74c3c')

    axes[0].set_xlabel('Weekday')
    axes[0].set_ylabel('Proportion of Weekly Highs')
    axes[0].set_title('Weekday Distribution of Weekly Highs\n(Mean across simulations)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(weekdays)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Distribution of Monday proportions
    axes[1].hist(wcsv_props[:, 0], bins=30, alpha=0.6, label='WCSV (Monday)', color='#3498db', density=True)
    axes[1].hist(garch_props[:, 0], bins=30, alpha=0.6, label='GARCH (Monday)', color='#e74c3c', density=True)
    axes[1].axvline(x=0.2, color='gray', linestyle='--', label='Uniform (0.2)')

    axes[1].set_xlabel('Proportion of Weekly Highs on Monday')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution of Monday High Proportions\n(Across simulations)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main simulation study"""

    print("=" * 70)
    print("Monte Carlo Simulation: WCSV vs Standard GARCH")
    print("=" * 70)

    # Run simulations
    results = compare_models(n_simulations=1000, n_weeks=500)

    # Compute statistics
    wcsv_mean = np.mean(results['wcsv_high_props'], axis=0)
    garch_mean = np.mean(results['garch_high_props'], axis=0)
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    print("\n" + "=" * 70)
    print("Results: Proportion of Weekly Highs by Weekday")
    print("=" * 70)

    print(f"\n{'Weekday':<12} {'WCSV':>10} {'GARCH':>10} {'Uniform':>10}")
    print("-" * 45)
    for i, day in enumerate(weekdays):
        print(f"{day:<12} {wcsv_mean[i]:>10.3f} {garch_mean[i]:>10.3f} {0.2:>10.3f}")

    # Statistical tests
    print("\n" + "=" * 70)
    print("Statistical Tests (Monday vs Uniform)")
    print("=" * 70)

    # KS-test for Monday proportions
    uniform_sample = np.random.beta(10, 40, size=1000)  # Beta dist approximating uniform proportion
    ks_wcsv = ks_2samp(results['wcsv_high_props'][:, 0], uniform_sample)
    ks_garch = ks_2samp(results['garch_high_props'][:, 0], uniform_sample)

    print(f"\nKolmogorov-Smirnov Test:")
    print(f"  WCSV:  statistic = {ks_wcsv.statistic:.4f}, p-value = {ks_wcsv.pvalue:.4f}")
    print(f"  GARCH: statistic = {ks_garch.statistic:.4f}, p-value = {ks_garch.pvalue:.4f}")

    # RMSE from uniform
    wcsv_rmse = np.sqrt(np.mean((wcsv_mean - 0.2)**2))
    garch_rmse = np.sqrt(np.mean((garch_mean - 0.2)**2))

    print(f"\nRMSE from Uniform Distribution:")
    print(f"  WCSV:  {wcsv_rmse:.4f}")
    print(f"  GARCH: {garch_rmse:.4f}")

    # Save plot
    try:
        fig = plot_comparison(results)
        plt.savefig('figures/simulation_comparison.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved to figures/simulation_comparison.png")
    except:
        print("\nCould not save plot (figures/ directory may not exist)")

    print("=" * 70)


if __name__ == '__main__':
    main()
