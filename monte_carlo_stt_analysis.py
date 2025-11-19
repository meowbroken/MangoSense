"""
Monte Carlo Simulation for Stock Transaction Tax (STT) Policy Impact Analysis

This script simulates the impact of the CMEPA STT reduction on trading volume,
volatility, and total tax revenue using Monte Carlo methods and elasticity
estimates from historical data.

Author: Analysis Team
Date: 2025
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


# --- Placeholder DataFrames and Historical Parameters Function ---
def create_placeholder_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates placeholder DataFrames for a runnable example.
    
    Returns:
        Tuple of (pre_train_data, post_train_data, pre_cmepa_data, actual_cmepa_data)
    
    Notes:
        TO INSERT YOUR DATA:
        1. Replace the data creation logic below with actual historical data load.
        2. Ensure your data has the columns: 'date', 'volume', 'close', 'volatility'.
        3. For pre_cmepa_data, also include: 'foreign_buy', 'foreign_sell', 
           'bid_ask_spread', 'gdp_growth', 'inflation'
    """
    # Pre-TRAIN (Jan 2017 - June 2018) - STT 0.5%
    # Ensure this DataFrame is based on YOUR historical data for the STT 0.5% period
    dates_train_pre = pd.date_range('2017-01-01', periods=350, freq='B')
    pre_train_data = pd.DataFrame({
        'date': dates_train_pre[:350],
        'volume': np.random.uniform(5e8, 10e8, size=350),  # 500M to 1B shares
        'close': np.random.uniform(50, 70, size=350),
        'volatility': np.random.uniform(0.01, 0.02, size=350)
    })
    
    # Post-TRAIN (July 2018 - Dec 2019) - STT 0.6%
    # Ensure this DataFrame is based on YOUR historical data for the STT 0.6% period
    dates_train_post = pd.date_range('2018-07-01', periods=350, freq='B')
    post_train_data = pd.DataFrame({
        'date': dates_train_post[:350],
        'volume': np.random.uniform(4e8, 8e8, size=350), 
        'close': np.random.uniform(55, 75, size=350),
        'volatility': np.random.uniform(0.015, 0.025, size=350)  
    })
    
    # Pre-CMEPA (July 2024 - June 2025) - Baseline for simulation
    # Ensure this DataFrame is based on YOUR most recent historical data (e.g., last 1 year)
    dates_cmepa_pre = pd.date_range('2024-07-01', periods=252, freq='B')  # 1 year
    pre_cmepa_data = pd.DataFrame({
        'date': dates_cmepa_pre,
        'volume': np.random.uniform(7e8, 12e8, size=252),
        'close': np.random.uniform(60, 80, size=252),
        'volatility': np.random.uniform(0.015, 0.025, size=252),  # Added volatility column
        'foreign_buy': np.random.uniform(2e8, 5e8, size=252),  # FOR LAMBDA CALC
        'foreign_sell': np.random.uniform(2e8, 5e8, size=252),  # FOR LAMBDA CALC
        'bid_ask_spread': np.random.uniform(0.0005, 0.0015, size=252),  # FOR LAMBDA CALC
        'gdp_growth': np.repeat(np.random.uniform(0.05, 0.07, size=4), 63)[:252],  # FOR LAMBDA CALC
        'inflation': np.repeat(np.random.uniform(0.02, 0.04, size=12), 21)[:252],  # FOR LAMBDA CALC
    })

    # Actual Post-CMEPA (July 2025 - Dec 2025) - STT 0.1%
    # Replace this with the ACTUAL post-CMEPA data once available
    dates_cmepa_post = pd.date_range('2025-07-01', periods=126, freq='B')  # 6 months (126 trading days)
    actual_cmepa_data = pd.DataFrame({
        'date': dates_cmepa_post,
        'volume': np.random.uniform(10e8, 16e8, size=126),  # Expected volume increase
        'price': np.random.uniform(60, 80, size=126),
    })

    return pre_train_data, post_train_data, pre_cmepa_data, actual_cmepa_data


def estimate_historical_parameters(pre_cmepa_data: pd.DataFrame) -> Dict[str, float]:
    """
    Estimates baseline historical parameters from the Pre-CMEPA period.
    
    Args:
        pre_cmepa_data: DataFrame with historical pre-CMEPA data
        
    Returns:
        Dictionary containing baseline parameters:
            - TV_HIST_MEAN: Mean historical trading volume
            - TV_HIST_SD: Standard deviation of historical trading volume
            - VOLATILITY_ANNUAL: Average annual volatility
            - P_AVG: Average closing price
    """
    # Validate required columns
    required_columns = ['volume', 'close']
    missing_columns = [col for col in required_columns if col not in pre_cmepa_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return {
        'TV_HIST_MEAN': pre_cmepa_data['volume'].mean(),
        'TV_HIST_SD': pre_cmepa_data['volume'].std(),
        'VOLATILITY_ANNUAL': (
            pre_cmepa_data['volatility'].mean() 
            if 'volatility' in pre_cmepa_data.columns 
            else np.random.uniform(0.15, 0.25)
        ),
        'P_AVG': pre_cmepa_data['close'].mean()
    }


def calculate_elasticity_ranges(
    pre_train_data: pd.DataFrame, 
    post_train_data: pd.DataFrame
) -> Dict[str, float]:
    """
    ELASTICITY ESTIMATION FROM TRAIN LAW PERIOD (2018)
    
    Calculates the elasticity (β) of trading volume and volatility with respect to 
    the change in the Stock Transaction Tax (STT) during the TRAIN Law period 
    (0.5% to 0.6%).
    
    Args:
        pre_train_data: DataFrame with pre-TRAIN period data (STT 0.5%)
        post_train_data: DataFrame with post-TRAIN period data (STT 0.6%)
    
    Returns:
        Dictionary containing elasticity ranges and medians:
            - BETA_TV_MIN, BETA_TV_MAX, BETA_TV_MEDIAN: Trading volume elasticity
            - BETA_VOL_MIN, BETA_VOL_MAX, BETA_VOL_MEDIAN: Volatility elasticity
            - volume_elasticities: List of calculated volume elasticities
            - volatility_elasticities: List of calculated volatility elasticities
    """
    # STT change during TRAIN: from 0.5% to 0.6% = 20% increase
    STT_CHANGE_PERCENT = 0.20

    # Divide data into quarters for temporal analysis
    pre_train_data = pre_train_data.copy()
    post_train_data = post_train_data.copy()
    pre_train_data['quarter'] = pd.to_datetime(pre_train_data['date']).dt.to_period('Q')
    post_train_data['quarter'] = pd.to_datetime(post_train_data['date']).dt.to_period('Q')

    # Calculate quarterly averages/stdevs
    pre_quarterly = pre_train_data.groupby('quarter').agg({
        'volume': 'mean',
        'volatility': 'std'
    }).reset_index()

    post_quarterly = post_train_data.groupby('quarter').agg({
        'volume': 'mean',
        'volatility': 'std'
    }).reset_index()

    # Calculate elasticities for each quarter pair
    volume_elasticities = []
    volatility_elasticities = []

    # Use min number of overlapping quarters
    n_quarters = min(len(pre_quarterly), len(post_quarterly))

    for i in range(n_quarters):
        # Volume elasticity: %ΔVolume / %ΔSTT
        pct_change_volume = (
            (post_quarterly.loc[i, 'volume'] - pre_quarterly.loc[i, 'volume']) / 
            pre_quarterly.loc[i, 'volume']
        )
        volume_elasticity = pct_change_volume / STT_CHANGE_PERCENT
        volume_elasticities.append(volume_elasticity)
        
        # Volatility elasticity: %ΔVolatility / %ΔSTT
        pct_change_volatility = (
            (post_quarterly.loc[i, 'volatility'] - pre_quarterly.loc[i, 'volatility']) / 
            pre_quarterly.loc[i, 'volatility']
        )
        volatility_elasticity = pct_change_volatility / STT_CHANGE_PERCENT
        volatility_elasticities.append(volatility_elasticity)

    # Calculate ranges
    results = {
        'BETA_TV_MIN': min(volume_elasticities),
        'BETA_TV_MAX': max(volume_elasticities),
        'BETA_TV_MEDIAN': np.median(volume_elasticities),
        'BETA_VOL_MIN': min(volatility_elasticities),
        'BETA_VOL_MAX': max(volatility_elasticities),
        'BETA_VOL_MEDIAN': np.median(volatility_elasticities),
        'volume_elasticities': volume_elasticities,
        'volatility_elasticities': volatility_elasticities
    }

    print("\n" + "="*60)
    print("ESTIMATED ELASTICITY RANGES FROM TRAIN LAW PERIOD")
    print("="*60)
    print(f"Trading Volume Elasticity Range (β_TV): [{results['BETA_TV_MIN']:.3f}, {results['BETA_TV_MAX']:.3f}]")
    print(f"Trading Volume Elasticity Median (β_TV): {results['BETA_TV_MEDIAN']:.3f}")
    print(f"Volatility Elasticity Range (β_VOL): [{results['BETA_VOL_MIN']:.3f}, {results['BETA_VOL_MAX']:.3f}]")
    print(f"Volatility Elasticity Median (β_VOL): {results['BETA_VOL_MEDIAN']:.3f}")
    print("="*60)

    return results


def run_monte_carlo_simulation(
    historical_params: Dict[str, float], 
    elasticity_ranges: Dict[str, float]
) -> Dict[str, np.ndarray]:
    """
    MONTE CARLO SIMULATION
    
    Simulates the impact of the CMEPA STT reduction on trading volume, 
    volatility, and total tax revenue, using estimated elasticity and a
    placeholder lambda range.
    
    Args:
        historical_params: Dictionary with historical baseline parameters
        elasticity_ranges: Dictionary with elasticity ranges from TRAIN period
    
    Returns:
        Dictionary containing simulation results:
            - results_TV_pct_change: Array of trading volume percentage changes
            - results_VOL_pct_change: Array of volatility percentage changes
            - results_REVENUE_post_TOTAL: Array of total post-CMEPA revenues
            - results_TV_post_DAILY_MEAN: Array of post-CMEPA daily trading volumes
            - results_VOL_post: Array of post-CMEPA volatilities
            - sampled_beta_tv: Array of sampled volume elasticities
            - sampled_beta_vol: Array of sampled volatility elasticities
            - sampled_tv_pre_daily: Array of sampled pre-CMEPA volumes
            - sampled_lambda_total: Array of sampled lambda adjustment factors
            - REVENUE_BASELINE_TOTAL: Baseline total revenue for comparison
    """
    # Simulation constants
    SIMULATIONS = 10000
    DAYS_IN_SIMULATION = 126  # 6 months (approx 126 trading days)
    STT_RATE_PRE = 0.006  # 0.6%
    STT_RATE_POST = 0.001  # 0.1%
    STT_CHANGE_PERCENT = (STT_RATE_POST - STT_RATE_PRE) / STT_RATE_PRE  # -83.33%

    # --- PLACEHOLDER LAMBDA RANGE (LITERATURE-BASED ADJUSTMENT FACTOR) ---
    # This range accounts for the non-linear or delayed effect of the policy
    # due to broader market/investor sentiment not captured by the TRAIN period.
    # It is a single, composite factor, simplifying for now.
    LAMBDA_TOTAL_RANGE = (0.9, 1.1)  # 1.0 ± 10%
    # TO INSERT YOUR DATA:
    # Adjust LAMBDA_TOTAL_RANGE based on literature review or expert opinion.
    # Example: (0.8, 1.2)
    # ----------------------------------------------------------------------

    # Extract parameters
    TV_HIST_MEAN = historical_params['TV_HIST_MEAN']
    TV_HIST_SD = historical_params['TV_HIST_SD']
    VOLATILITY_ANNUAL = historical_params['VOLATILITY_ANNUAL']
    P_AVG = historical_params['P_AVG']

    # Elasticity ranges
    BETA_TV_RANGE = (elasticity_ranges['BETA_TV_MIN'], elasticity_ranges['BETA_TV_MAX'])
    BETA_VOL_RANGE = (elasticity_ranges['BETA_VOL_MIN'], elasticity_ranges['BETA_VOL_MAX'])

    # Baseline revenue
    REVENUE_BASELINE_DAILY = TV_HIST_MEAN * P_AVG * STT_RATE_PRE
    REVENUE_BASELINE_TOTAL = REVENUE_BASELINE_DAILY * DAYS_IN_SIMULATION

    # Initialize result arrays
    results_TV_pct_change = np.zeros(SIMULATIONS)
    results_VOL_pct_change = np.zeros(SIMULATIONS)
    results_REVENUE_post_TOTAL = np.zeros(SIMULATIONS)
    results_TV_post_DAILY_MEAN = np.zeros(SIMULATIONS)
    results_VOL_post = np.zeros(SIMULATIONS)

    # Store sampled inputs for sensitivity analysis
    sampled_beta_tv = np.zeros(SIMULATIONS)
    sampled_beta_vol = np.zeros(SIMULATIONS)
    sampled_tv_pre_daily = np.zeros(SIMULATIONS)
    sampled_lambda_total = np.zeros(SIMULATIONS)  # Single lambda factor

    print("\n" + "="*60)
    print(f"RUNNING MONTE CARLO SIMULATION ({SIMULATIONS:,} iterations)")
    print("="*60)

    for i in range(SIMULATIONS):
        # 1. Generate dynamic baselines (TV_pre is sampled from a Normal distribution)
        tv_pre_daily = np.random.normal(TV_HIST_MEAN, TV_HIST_SD)
        vol_pre_annual = VOLATILITY_ANNUAL
        
        # 2. Sample policy parameters (uniform distribution over the estimated range)
        beta_tv = np.random.uniform(*BETA_TV_RANGE)
        beta_vol = np.random.uniform(*BETA_VOL_RANGE)
        lambda_total = np.random.uniform(*LAMBDA_TOTAL_RANGE)
        
        # 3. Apply policy shock: %ΔY = β × (%Δt) × λ
        # Where: %ΔY is % change in volume/volatility
        #        β is the elasticity factor
        #        %Δt is the % change in the STT rate
        #        λ is the composite adjustment factor
        pct_change_TV = beta_tv * STT_CHANGE_PERCENT * lambda_total
        pct_change_VOL = beta_vol * STT_CHANGE_PERCENT * lambda_total
        
        # 4. Project post-CMEPA outcomes
        tv_post_daily_mean = tv_pre_daily * (1 + pct_change_TV)
        gsp_post = tv_post_daily_mean * P_AVG
        revenue_post_total = gsp_post * STT_RATE_POST * DAYS_IN_SIMULATION
        vol_post_annual = vol_pre_annual * (1 + pct_change_VOL)
        
        # 5. Store results
        results_TV_pct_change[i] = pct_change_TV
        results_VOL_pct_change[i] = pct_change_VOL
        results_REVENUE_post_TOTAL[i] = revenue_post_total
        results_TV_post_DAILY_MEAN[i] = tv_post_daily_mean
        results_VOL_post[i] = vol_post_annual
        
        # Store inputs for sensitivity analysis
        sampled_beta_tv[i] = beta_tv
        sampled_beta_vol[i] = beta_vol
        sampled_tv_pre_daily[i] = tv_pre_daily
        sampled_lambda_total[i] = lambda_total

    # Print summary statistics
    mean_TV_pct = np.mean(results_TV_pct_change) * 100
    mean_REVENUE_post = np.mean(results_REVENUE_post_TOTAL)
    pct_fiscally_adequate = (
        np.sum(results_REVENUE_post_TOTAL > REVENUE_BASELINE_TOTAL) / SIMULATIONS * 100
    )
    mean_VOL_pct = np.mean(results_VOL_pct_change) * 100
    prob_vol_increase = np.sum(results_VOL_pct_change > 0) / SIMULATIONS * 100

    print("\n" + "="*60)
    print("MONTE CARLO RESULTS SUMMARY")
    print("="*60)
    print(f"Pre-CMEPA Baseline Total Revenue (6 Months): {REVENUE_BASELINE_TOTAL:,.0f} PHP")
    print(f"Mean Projected Increase in Trading Volume: {mean_TV_pct:.2f}%")
    print(f"Mean Projected Post-CMEPA Total Revenue: {mean_REVENUE_post:,.0f} PHP")
    print(f"Probability of Fiscal Adequacy (Revenue > Baseline): {pct_fiscally_adequate:.2f}%")
    print(f"Mean Projected Change in Volatility: {mean_VOL_pct:.2f}%")
    print(f"Probability of Volatility Increasing: {prob_vol_increase:.2f}%")
    print("="*60)

    return {
        'results_TV_pct_change': results_TV_pct_change,
        'results_VOL_pct_change': results_VOL_pct_change,
        'results_REVENUE_post_TOTAL': results_REVENUE_post_TOTAL,
        'results_TV_post_DAILY_MEAN': results_TV_post_DAILY_MEAN,
        'results_VOL_post': results_VOL_post,
        'sampled_beta_tv': sampled_beta_tv,
        'sampled_beta_vol': sampled_beta_vol,
        'sampled_tv_pre_daily': sampled_tv_pre_daily,
        'sampled_lambda_total': sampled_lambda_total,
        'REVENUE_BASELINE_TOTAL': REVENUE_BASELINE_TOTAL
    }


def run_sensitivity_analysis(simulation_results: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    SENSITIVITY ANALYSIS AND TORNADO CHART
    
    Performs a Spearman's Rank Correlation to determine the influence of 
    input parameters on the total projected revenue.
    
    Args:
        simulation_results: Dictionary containing simulation results
    
    Returns:
        DataFrame with sensitivity analysis results ranked by influence
    """
    print("\n" + "="*60)
    print("PERFORMING SENSITIVITY ANALYSIS")
    print("="*60)

    # Input variables for sensitivity analysis
    input_variables = {
        'TV Elasticity (β_TV)': simulation_results['sampled_beta_tv'],
        'Volatility Elasticity (β_VOL)': simulation_results['sampled_beta_vol'],
        'Baseline Volume (TV_pre)': simulation_results['sampled_tv_pre_daily'],
        'Composite Lambda (λ_Total)': simulation_results['sampled_lambda_total']
    }

    sensitivity_results = {}

    for name, samples in input_variables.items():
        # Calculate Spearman's rank correlation (rho)
        rho, _ = spearmanr(samples, simulation_results['results_REVENUE_post_TOTAL'])
        sensitivity_results[name] = rho

    # Create DataFrame and sort by absolute correlation
    df_sensitivity = pd.DataFrame(
        list(sensitivity_results.items()),
        columns=['Input Variable', 'Correlation_R']
    )
    df_sensitivity['Abs_R'] = df_sensitivity['Correlation_R'].abs()
    df_sensitivity = df_sensitivity.sort_values(by='Abs_R', ascending=True).reset_index(drop=True)

    print("\nSensitivity Results (Ranked by Influence on Total Projected Revenue):")
    print(df_sensitivity[['Input Variable', 'Correlation_R', 'Abs_R']].to_string(index=False))
    print("="*60)

    return df_sensitivity


def plot_tornado_chart(df_sensitivity: pd.DataFrame, output_path: str = 'tornado_chart.png') -> None:
    """
    Creates a tornado chart to visualize sensitivity analysis results.
    
    Args:
        df_sensitivity: DataFrame with sensitivity analysis results
        output_path: Path to save the tornado chart image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by absolute correlation for better visualization
    df_plot = df_sensitivity.sort_values(by='Abs_R', ascending=True)
    
    # Create horizontal bar chart
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in df_plot['Correlation_R']]
    ax.barh(df_plot['Input Variable'], df_plot['Correlation_R'], color=colors)
    
    ax.set_xlabel('Spearman Correlation Coefficient (ρ)', fontsize=12)
    ax.set_title('Sensitivity Analysis: Impact on Total Revenue\n(Tornado Chart)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nTornado chart saved to: {output_path}")
    plt.close()


def run_model_validation_and_laffer_curve_test(
    simulation_results: Dict[str, np.ndarray], 
    actual_cmepa_data: pd.DataFrame, 
    historical_params: Dict[str, float]
) -> Dict[str, any]:
    """
    MODEL VALIDATION AND LAFFER CURVE TEST
    
    Compares the actual post-CMEPA revenue and volume to the Monte Carlo 
    simulation's 95% confidence intervals and tests the fiscal adequacy 
    (Laffer Curve principle).
    
    Args:
        simulation_results: Dictionary containing simulation results
        actual_cmepa_data: DataFrame with actual post-CMEPA data
        historical_params: Dictionary with historical baseline parameters
    
    Returns:
        Dictionary with validation results and conclusions:
            - laffer_conclusion: 'ADEQUATE' or 'INADEQUATE'
            - revenue_change_pct: Percentage change in revenue
            - model_robust: Boolean indicating if model is robust
            - tv_within_ci: Boolean indicating if trading volume is within CI
            - revenue_within_ci: Boolean indicating if revenue is within CI
    """
    print("\n" + "="*60)
    print("MODEL VALIDATION AND LAFFER CURVE TEST")
    print("="*60)

    # Calculate actual metrics from CMEPA period
    ACTUAL_MEAN_TV_DAILY = actual_cmepa_data['volume'].mean()
    ACTUAL_MEAN_PRICE = actual_cmepa_data['price'].mean()
    ACTUAL_GSP_MEAN = ACTUAL_MEAN_TV_DAILY * ACTUAL_MEAN_PRICE

    # Actual revenue calculation
    STT_RATE_POST = 0.001
    DAYS_IN_PERIOD = 126
    ACTUAL_TOTAL_REVENUE = ACTUAL_GSP_MEAN * STT_RATE_POST * DAYS_IN_PERIOD

    # Baseline revenue
    REVENUE_BASELINE_TOTAL = simulation_results['REVENUE_BASELINE_TOTAL']

    # A. LAFFER CURVE TEST
    print("\n=== A. LAFFER CURVE TEST (Fiscal Adequacy) ===")
    print("\nThe Laffer Curve principle suggests that reducing tax rates")
    print("can increase total revenue if the increase in the tax base")
    print("(trading volume) offsets the lower rate.")
    
    revenue_change = ACTUAL_TOTAL_REVENUE - REVENUE_BASELINE_TOTAL
    revenue_change_pct = (revenue_change / REVENUE_BASELINE_TOTAL) * 100

    print(f"\nPre-CMEPA Baseline Total Revenue: {REVENUE_BASELINE_TOTAL:,.0f} PHP")
    print(f"Actual Post-CMEPA Total Revenue: {ACTUAL_TOTAL_REVENUE:,.0f} PHP")
    print(f"Revenue Change: {revenue_change:,.0f} PHP ({revenue_change_pct:+.2f}%)")

    if ACTUAL_TOTAL_REVENUE > REVENUE_BASELINE_TOTAL:
        print(f"✅ CONCLUSION: Fiscally **ADEQUATE**. Revenue increased by {revenue_change_pct:.2f}%.")
        laffer_conclusion = "ADEQUATE"
    else:
        print(f"❌ CONCLUSION: Fiscally **INADEQUATE**. Revenue decreased by {abs(revenue_change_pct):.2f}%.")
        laffer_conclusion = "INADEQUATE"

    # B. MODEL VALIDATION
    print("\n=== B. MODEL VALIDATION (95% Confidence Interval Test) ===")

    # Calculate 95% CI for trading volume
    ci_tv_lower = np.percentile(simulation_results['results_TV_post_DAILY_MEAN'], 2.5)
    ci_tv_upper = np.percentile(simulation_results['results_TV_post_DAILY_MEAN'], 97.5)
    tv_within_ci = ci_tv_lower <= ACTUAL_MEAN_TV_DAILY <= ci_tv_upper

    # Calculate 95% CI for revenue
    ci_revenue_lower = np.percentile(simulation_results['results_REVENUE_post_TOTAL'], 2.5)
    ci_revenue_upper = np.percentile(simulation_results['results_REVENUE_post_TOTAL'], 97.5)
    revenue_within_ci = ci_revenue_lower <= ACTUAL_TOTAL_REVENUE <= ci_revenue_upper

    print(f"\nTrading Volume Validation:")
    print(f"  Actual Mean Daily TV: {ACTUAL_MEAN_TV_DAILY:,.0f}")
    print(f"  Predicted 95% CI: [{ci_tv_lower:,.0f}, {ci_tv_upper:,.0f}]")
    print(f"  {'✅ ROBUST' if tv_within_ci else '❌ WEAK'}: Actual {'within' if tv_within_ci else 'OUTSIDE'} CI")

    print(f"\nTotal Revenue Validation:")
    print(f"  Actual Total Revenue: {ACTUAL_TOTAL_REVENUE:,.0f} PHP")
    print(f"  Predicted 95% CI: [{ci_revenue_lower:,.0f}, {ci_revenue_upper:,.0f}]")
    print(f"  {'✅ ROBUST' if revenue_within_ci else '❌ WEAK'}: Actual {'within' if revenue_within_ci else 'OUTSIDE'} CI")

    overall_robust = tv_within_ci and revenue_within_ci
    print(f"\n{'='*60}")
    print(f"OVERALL MODEL ASSESSMENT: {'✅ ROBUST' if overall_robust else '❌ REQUIRES REFINEMENT'}")
    print(f"{'='*60}")

    return {
        'laffer_conclusion': laffer_conclusion,
        'revenue_change_pct': revenue_change_pct,
        'model_robust': overall_robust,
        'tv_within_ci': tv_within_ci,
        'revenue_within_ci': revenue_within_ci
    }


def plot_laffer_curve(
    simulation_results: Dict[str, np.ndarray],
    actual_cmepa_data: pd.DataFrame,
    output_path: str = 'laffer_curve.png'
) -> None:
    """
    Creates a conceptual Laffer Curve visualization.
    
    Args:
        simulation_results: Dictionary containing simulation results
        actual_cmepa_data: DataFrame with actual post-CMEPA data
        output_path: Path to save the Laffer curve image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create conceptual Laffer curve
    tax_rates = np.linspace(0, 0.02, 100)  # 0% to 2%
    # Simplified revenue curve (inverted parabola)
    base_revenue = 5e11  # Base revenue at optimal rate
    optimal_rate = 0.004  # Hypothetical optimal rate
    revenues = base_revenue * (1 - ((tax_rates - optimal_rate) / optimal_rate) ** 2)
    
    ax.plot(tax_rates * 100, revenues / 1e9, linewidth=2.5, color='#1f77b4', label='Revenue Curve')
    
    # Mark current points
    ax.axvline(x=0.6, color='red', linestyle='--', linewidth=1.5, label='Pre-CMEPA (0.6%)', alpha=0.7)
    ax.axvline(x=0.1, color='green', linestyle='--', linewidth=1.5, label='Post-CMEPA (0.1%)', alpha=0.7)
    ax.axvline(x=optimal_rate * 100, color='orange', linestyle=':', linewidth=1.5, label='Optimal Rate', alpha=0.7)
    
    ax.set_xlabel('Stock Transaction Tax Rate (%)', fontsize=12)
    ax.set_ylabel('Total Revenue (Billion PHP)', fontsize=12)
    ax.set_title('Laffer Curve: STT Rate vs. Total Revenue', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Laffer curve visualization saved to: {output_path}")
    plt.close()


# --- Main Execution Block ---
def main():
    """
    Main execution function for the Monte Carlo STT analysis.
    """
    print("\n" + "="*60)
    print("MONTE CARLO SIMULATION: STT POLICY IMPACT ANALYSIS")
    print("CMEPA Tax Reduction Impact Study")
    print("="*60)
    
    try:
        # 0. Create/Load Data
        print("\n[Step 1/6] Loading/Creating Data...")
        pre_train_data, post_train_data, pre_cmepa_data, actual_cmepa_data = create_placeholder_data()
        print("✓ Data loaded successfully")
        
        # 1. Estimate Historical Parameters (TV_HIST_MEAN, P_AVG, etc.)
        print("\n[Step 2/6] Estimating Historical Parameters...")
        historical_params = estimate_historical_parameters(pre_cmepa_data)
        print("✓ Historical parameters estimated")
        
        # 2. Estimate Elasticity Ranges (β_TV, β_VOL) from TRAIN Law Period
        print("\n[Step 3/6] Calculating Elasticity Ranges...")
        elasticity_ranges = calculate_elasticity_ranges(pre_train_data, post_train_data)
        print("✓ Elasticity ranges calculated")
        
        # 3. Run Monte Carlo Simulation
        print("\n[Step 4/6] Running Monte Carlo Simulation...")
        simulation_results = run_monte_carlo_simulation(historical_params, elasticity_ranges)
        print("✓ Monte Carlo simulation completed")
        
        # 4. Run Sensitivity Analysis (Influence of Inputs on Revenue)
        print("\n[Step 5/6] Running Sensitivity Analysis...")
        df_sensitivity = run_sensitivity_analysis(simulation_results)
        print("✓ Sensitivity analysis completed")
        
        # Create tornado chart
        plot_tornado_chart(df_sensitivity)
        
        # 5. Model Validation and Laffer Curve Test (Requires actual post-CMEPA data)
        print("\n[Step 6/6] Running Model Validation and Laffer Curve Test...")
        validation_results = run_model_validation_and_laffer_curve_test(
            simulation_results, 
            actual_cmepa_data, 
            historical_params
        )
        print("✓ Model validation completed")
        
        # Create Laffer curve visualization
        plot_laffer_curve(simulation_results, actual_cmepa_data)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nGenerated outputs:")
        print("  - tornado_chart.png: Sensitivity analysis visualization")
        print("  - laffer_curve.png: Laffer curve conceptual diagram")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise


if __name__ == '__main__':
    main()
