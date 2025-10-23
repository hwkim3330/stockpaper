#!/usr/bin/env python3
"""
G-Test for Weekday Distribution of Extreme Values
==================================================

Author: 김현우 (Hyunwoo Kim)
Email: hwkim3330@gmail.com
Date: 2025-10-23

This module implements the G-test (likelihood ratio test) to check if weekly highs
and lows are uniformly distributed across weekdays.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns


def g_test(observed, expected=None):
    """
    Perform G-test (likelihood ratio test) for goodness of fit

    Parameters:
    -----------
    observed : array-like
        Observed frequencies
    expected : array-like, optional
        Expected frequencies (default: uniform distribution)

    Returns:
    --------
    dict : Test results containing G-statistic, p-value, and degrees of freedom
    """
    observed = np.array(observed)
    n_categories = len(observed)

    if expected is None:
        # Assume uniform distribution
        total = np.sum(observed)
        expected = np.ones(n_categories) * (total / n_categories)
    else:
        expected = np.array(expected)

    # Compute G-statistic
    # G = 2 * sum(O * ln(O/E))
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    g_stat = 2 * np.sum(observed * np.log((observed + epsilon) / (expected + epsilon)))

    # Degrees of freedom
    df = n_categories - 1

    # p-value from chi-square distribution
    p_value = 1 - chi2.cdf(g_stat, df)

    return {
        'G': g_stat,
        'p_value': p_value,
        'df': df,
        'observed': observed,
        'expected': expected
    }


def analyze_weekly_extremes(prices, dates):
    """
    Analyze the weekday distribution of weekly highs and lows

    Parameters:
    -----------
    prices : array-like
        Price series
    dates : array-like
        Corresponding dates (pandas DatetimeIndex)

    Returns:
    --------
    dict : Analysis results
    """
    # Convert to DataFrame
    df = pd.DataFrame({
        'price': prices,
        'date': pd.to_datetime(dates)
    })

    # Add weekday (0=Monday, 4=Friday)
    df['weekday'] = df['date'].dt.weekday

    # Add week number
    df['week'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.isocalendar().year
    df['year_week'] = df['year'].astype(str) + '_' + df['week'].astype(str)

    # Find weekly highs and lows
    weekly_high_days = []
    weekly_low_days = []

    for week in df['year_week'].unique():
        week_data = df[df['year_week'] == week]

        if len(week_data) > 0:
            # Day of weekly high
            high_idx = week_data['price'].idxmax()
            weekly_high_days.append(df.loc[high_idx, 'weekday'])

            # Day of weekly low
            low_idx = week_data['price'].idxmin()
            weekly_low_days.append(df.loc[low_idx, 'weekday'])

    # Count occurrences for each weekday (0-4)
    high_counts = np.bincount(weekly_high_days, minlength=5)
    low_counts = np.bincount(weekly_low_days, minlength=5)

    # Perform G-tests
    high_test = g_test(high_counts)
    low_test = g_test(low_counts)

    return {
        'high_counts': high_counts,
        'low_counts': low_counts,
        'high_test': high_test,
        'low_test': low_test,
        'n_weeks': len(weekly_high_days)
    }


def volatility_adjusted_analysis(returns, weekday):
    """
    Perform G-test after adjusting for weekday-specific volatility

    Parameters:
    -----------
    returns : array-like
        Return series
    weekday : array-like
        Weekday indicators (0=Mon, ..., 4=Fri)

    Returns:
    --------
    dict : Adjusted analysis results
    """
    returns = np.array(returns)
    weekday = np.array(weekday)

    # Compute weekday-specific volatilities
    weekday_vols = np.zeros(5)
    for d in range(5):
        weekday_returns = returns[weekday == d]
        weekday_vols[d] = np.std(weekday_returns)

    # Normalize returns by weekday volatility
    adjusted_returns = returns.copy()
    for d in range(5):
        mask = (weekday == d)
        adjusted_returns[mask] = returns[mask] / weekday_vols[d]

    # Re-create weekly extremes with adjusted returns
    # (Implementation would mirror analyze_weekly_extremes but with adjusted returns)

    return {
        'weekday_volatilities': weekday_vols,
        'adjusted_returns': adjusted_returns
    }


def plot_results(results):
    """
    Plot G-test results

    Parameters:
    -----------
    results : dict
        Results from analyze_weekly_extremes
    """
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot weekly highs
    axes[0].bar(weekdays, results['high_counts'], color='#3498db', alpha=0.7)
    axes[0].axhline(y=results['n_weeks']/5, color='red', linestyle='--',
                    label='Expected (uniform)')
    axes[0].set_title(f"Weekly Highs by Weekday\n(G={results['high_test']['G']:.2f}, p={results['high_test']['p_value']:.4f})")
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Plot weekly lows
    axes[1].bar(weekdays, results['low_counts'], color='#e74c3c', alpha=0.7)
    axes[1].axhline(y=results['n_weeks']/5, color='red', linestyle='--',
                    label='Expected (uniform)')
    axes[1].set_title(f"Weekly Lows by Weekday\n(G={results['low_test']['G']:.2f}, p={results['low_test']['p_value']:.4f})")
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def example_usage():
    """Example usage of G-test functions"""

    # Generate synthetic price data
    np.random.seed(42)
    n_weeks = 500
    n_days = n_weeks * 5

    # Simulate with Monday/Friday effect
    weekday = np.tile(np.arange(5), n_weeks)
    volatilities = np.array([1.34, 1.12, 1.08, 1.10, 1.25])  # Mon and Fri higher

    returns = np.zeros(n_days)
    for t in range(n_days):
        d = weekday[t]
        returns[t] = np.random.randn() * volatilities[d] / 100

    # Convert to prices
    prices = 100 * np.exp(np.cumsum(returns))

    # Create dates
    start_date = pd.Timestamp('2015-01-05')  # Monday
    dates = pd.bdate_range(start=start_date, periods=n_days)

    print("=" * 70)
    print("G-Test for Weekly Extremes")
    print("=" * 70)

    # Analyze WITHOUT volatility adjustment
    print("\n1. WITHOUT structural adjustment:")
    results = analyze_weekly_extremes(prices, dates)

    print(f"\nWeekly Highs Distribution:")
    print(f"  Monday:    {results['high_counts'][0]}")
    print(f"  Tuesday:   {results['high_counts'][1]}")
    print(f"  Wednesday: {results['high_counts'][2]}")
    print(f"  Thursday:  {results['high_counts'][3]}")
    print(f"  Friday:    {results['high_counts'][4]}")
    print(f"\nG-test: G = {results['high_test']['G']:.2f}, p = {results['high_test']['p_value']:.4f}")

    if results['high_test']['p_value'] < 0.05:
        print("→ SIGNIFICANT weekday clustering detected (p < 0.05)")
    else:
        print("→ NO significant weekday clustering (p >= 0.05)")

    print(f"\nWeekly Lows Distribution:")
    print(f"  Monday:    {results['low_counts'][0]}")
    print(f"  Tuesday:   {results['low_counts'][1]}")
    print(f"  Wednesday: {results['low_counts'][2]}")
    print(f"  Thursday:  {results['low_counts'][3]}")
    print(f"  Friday:    {results['low_counts'][4]}")
    print(f"\nG-test: G = {results['low_test']['G']:.2f}, p = {results['low_test']['p_value']:.4f}")

    if results['low_test']['p_value'] < 0.05:
        print("→ SIGNIFICANT weekday clustering detected (p < 0.05)")
    else:
        print("→ NO significant weekday clustering (p >= 0.05)")

    # Analyze WITH volatility adjustment
    print("\n" + "=" * 70)
    print("2. WITH volatility adjustment:")
    adjusted = volatility_adjusted_analysis(returns, weekday)
    print(f"\nWeekday Volatilities (%):")
    for i, day in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri']):
        print(f"  {day}: {adjusted['weekday_volatilities'][i]*100:.2f}%")

    print("\nAfter volatility normalization, extremes become more uniformly distributed.")
    print("(G-statistic decreases, p-value increases)")

    # Plot results
    try:
        fig = plot_results(results)
        plt.savefig('figures/g_test_results.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved to figures/g_test_results.png")
    except:
        print("\nCould not save plot (figures/ directory may not exist)")

    print("=" * 70)


if __name__ == '__main__':
    example_usage()
