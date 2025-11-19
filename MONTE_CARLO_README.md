# Monte Carlo Simulation: Stock Transaction Tax (STT) Policy Impact Analysis

## Overview

This script performs a comprehensive Monte Carlo simulation to analyze the impact of the CMEPA (Corporate Market Efficiency and Productivity Act) Stock Transaction Tax reduction on trading volume, market volatility, and tax revenue.

## Features

### 1. **Elasticity Estimation**
- Calculates elasticity of trading volume and volatility using historical data from the TRAIN Law period (2018)
- Analyzes quarterly patterns to capture temporal variations
- Provides range and median estimates for robust analysis

### 2. **Monte Carlo Simulation**
- Runs 10,000 iterations to capture uncertainty in projections
- Simulates the impact of STT reduction from 0.6% to 0.1%
- Projects trading volume changes, volatility changes, and tax revenue
- Incorporates lambda adjustment factor for market sentiment effects

### 3. **Sensitivity Analysis**
- Performs Spearman rank correlation analysis
- Identifies which input parameters most strongly influence revenue outcomes
- Generates a tornado chart for visual interpretation

### 4. **Model Validation**
- Tests predictions against actual post-CMEPA data
- Calculates 95% confidence intervals for trading volume and revenue
- Assesses model robustness

### 5. **Laffer Curve Test**
- Evaluates fiscal adequacy of the policy change
- Determines if revenue increased or decreased post-policy
- Tests the principle that lower tax rates can increase total revenue

## Requirements

```bash
pip install pandas numpy scipy matplotlib
```

Or install from the included requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Simply run the script:
```bash
python monte_carlo_stt_analysis.py
```

### Output

The script generates:
1. **Console Output**: Detailed results including:
   - Estimated elasticity ranges
   - Monte Carlo simulation summary statistics
   - Sensitivity analysis results
   - Model validation conclusions

2. **Visualizations**:
   - `tornado_chart.png`: Sensitivity analysis showing which parameters most influence revenue
   - `laffer_curve.png`: Conceptual Laffer curve showing the tax rate-revenue relationship

## Customizing the Analysis

### 1. Replace Placeholder Data

The script uses placeholder data by default. To use your actual data, modify the `create_placeholder_data()` function:

```python
def create_placeholder_data():
    # Load your actual data here
    pre_train_data = pd.read_csv('your_pre_train_data.csv')
    post_train_data = pd.read_csv('your_post_train_data.csv')
    pre_cmepa_data = pd.read_csv('your_pre_cmepa_data.csv')
    actual_cmepa_data = pd.read_csv('your_actual_cmepa_data.csv')
    
    return pre_train_data, post_train_data, pre_cmepa_data, actual_cmepa_data
```

### 2. Required Data Columns

**For pre_train_data and post_train_data:**
- `date`: Trading dates
- `volume`: Daily trading volume
- `close`: Closing price
- `volatility`: Daily volatility measure

**For pre_cmepa_data:**
- All of the above, plus:
- `foreign_buy`: Foreign investor buy volume
- `foreign_sell`: Foreign investor sell volume
- `bid_ask_spread`: Market liquidity measure
- `gdp_growth`: Quarterly GDP growth rate
- `inflation`: Monthly inflation rate

**For actual_cmepa_data:**
- `date`: Trading dates
- `volume`: Daily trading volume
- `price`: Trading price

### 3. Adjust Lambda Range

The lambda adjustment factor accounts for market sentiment and behavioral effects not captured by historical elasticity. Adjust it based on literature review or expert opinion:

```python
# In run_monte_carlo_simulation() function
LAMBDA_TOTAL_RANGE = (0.8, 1.2)  # Adjust these values
```

## Understanding the Results

### Elasticity Interpretation

- **Negative Volume Elasticity**: Trading volume decreases when STT increases (expected)
- **Positive Volatility Elasticity**: Market volatility increases when STT increases (expected)

### Simulation Results

- **Probability of Fiscal Adequacy**: Percentage of simulations where revenue exceeds baseline
- **Mean Projected Increase in Trading Volume**: Expected percentage change in trading activity
- **Mean Projected Change in Volatility**: Expected percentage change in market volatility

### Sensitivity Analysis

Variables are ranked by their influence on total revenue:
- Higher absolute correlation = stronger influence
- Positive correlation = increases revenue when parameter increases
- Negative correlation = decreases revenue when parameter increases

### Model Validation

- **Within CI**: Model predictions align with actual data (robust model)
- **Outside CI**: Model may need refinement or recalibration

## Methodology

### Elasticity Calculation

The elasticity (β) is calculated using the formula:
```
β = (%ΔY / %Δτ)
```
Where:
- %ΔY = percentage change in outcome variable (volume or volatility)
- %Δτ = percentage change in STT rate

### Monte Carlo Formula

Each simulation iteration applies:
```
%ΔY = β × %Δτ × λ
```
Where:
- λ = adjustment factor for market sentiment effects

### Revenue Projection

Post-policy revenue is calculated as:
```
Revenue = Volume × Price × STT_Rate × Trading_Days
```

## Limitations

1. **Placeholder Data**: Uses simulated data by default - replace with actual market data for real analysis
2. **Linear Assumption**: Assumes linear relationship between STT changes and outcomes
3. **Lambda Estimation**: The adjustment factor is parameterized rather than empirically estimated
4. **Market Conditions**: Does not account for concurrent economic shocks or policy changes

## Best Practices

1. **Data Quality**: Ensure high-quality, cleaned historical data
2. **Validation Period**: Use sufficient pre/post periods for robust elasticity estimation
3. **Sensitivity Analysis**: Always run sensitivity analysis to understand parameter importance
4. **Model Validation**: Compare predictions against actual data when available
5. **Documentation**: Document all assumptions and data sources

## References

- Laffer Curve: Laffer, A. B. (2004). The Laffer Curve: Past, Present, and Future
- Monte Carlo Methods: Metropolis, N., & Ulam, S. (1949). The Monte Carlo Method
- Elasticity Estimation: Standard econometric techniques for tax elasticity

## Contact

For questions or issues with this analysis, please contact the development team.

## License

This script is provided as-is for research and policy analysis purposes.
