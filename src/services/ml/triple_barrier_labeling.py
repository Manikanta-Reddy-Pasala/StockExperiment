"""
Triple Barrier Labeling Method
Based on López de Prado's research and Korean stock prediction paper

This labeling technique defines three barriers:
1. Upper barrier (take-profit): +X% price move
2. Lower barrier (stop-loss): -X% price move
3. Time barrier: T days horizon

Label = which barrier is touched FIRST
- Label 1: Upper barrier hit first (profit)
- Label 0: Time barrier hit first (no significant move)
- Label -1: Lower barrier hit first (loss)

This provides more realistic targets that account for actual trading behavior
and risk management, avoiding look-ahead bias and overfitting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TripleBarrierLabeler:
    """
    Triple Barrier Method for financial time series labeling.

    Based on the methodology from:
    - "Advances in Financial Machine Learning" by Marcos López de Prado
    - Korean stock prediction research (2024)

    Parameters:
    -----------
    upper_barrier : float
        Upper barrier as percentage (e.g., 9.0 for +9%)
    lower_barrier : float
        Lower barrier as percentage (e.g., 9.0 for -9%)
    time_horizon : int
        Maximum holding period in days (e.g., 29 days)
    """

    def __init__(self, upper_barrier: float = 9.0, lower_barrier: float = 9.0, time_horizon: int = 29):
        self.upper_barrier = upper_barrier / 100.0  # Convert to decimal
        self.lower_barrier = lower_barrier / 100.0
        self.time_horizon = time_horizon

        logger.info(f"Triple Barrier Labeler initialized: "
                   f"upper={upper_barrier}%, lower={lower_barrier}%, horizon={time_horizon} days")

    def apply_triple_barrier(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Apply triple barrier labeling to a DataFrame with OHLCV data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data, indexed by date
        price_col : str
            Column name for closing price (default: 'close')

        Returns:
        --------
        pd.DataFrame
            Original DataFrame with added columns:
            - 'tbl_label': The barrier label (1, 0, -1)
            - 'tbl_barrier_touched': Which barrier was hit ('upper', 'lower', 'time')
            - 'tbl_days_to_barrier': Days until barrier was hit
            - 'tbl_return': Actual return when barrier was hit
        """
        df = df.copy()

        n = len(df)
        labels = np.zeros(n)
        barrier_touched = np.empty(n, dtype=object)
        days_to_barrier = np.zeros(n)
        returns = np.zeros(n)

        prices = df[price_col].values

        for i in range(n - self.time_horizon):
            current_price = prices[i]

            # Look forward for barriers
            found_barrier = False
            for t in range(1, min(self.time_horizon + 1, n - i)):
                future_price = prices[i + t]
                price_return = (future_price - current_price) / current_price

                # Check if upper barrier is hit
                if price_return >= self.upper_barrier:
                    labels[i] = 1
                    barrier_touched[i] = 'upper'
                    days_to_barrier[i] = t
                    returns[i] = price_return
                    found_barrier = True
                    break

                # Check if lower barrier is hit
                elif price_return <= -self.lower_barrier:
                    labels[i] = -1
                    barrier_touched[i] = 'lower'
                    days_to_barrier[i] = t
                    returns[i] = price_return
                    found_barrier = True
                    break

            # If no barrier hit, use time barrier
            if not found_barrier and i + self.time_horizon < n:
                labels[i] = 0
                barrier_touched[i] = 'time'
                days_to_barrier[i] = self.time_horizon
                returns[i] = (prices[i + self.time_horizon] - current_price) / current_price

        # Mark the last samples that don't have enough forward data as NaN
        labels[n - self.time_horizon:] = np.nan
        barrier_touched[n - self.time_horizon:] = None
        days_to_barrier[n - self.time_horizon:] = np.nan
        returns[n - self.time_horizon:] = np.nan

        # Add columns to DataFrame
        df['tbl_label'] = labels
        df['tbl_barrier_touched'] = barrier_touched
        df['tbl_days_to_barrier'] = days_to_barrier
        df['tbl_return'] = returns

        # Calculate label distribution
        valid_labels = labels[~np.isnan(labels)]
        if len(valid_labels) > 0:
            label_counts = pd.Series(valid_labels).value_counts()
            logger.info(f"Triple Barrier Label Distribution:")
            logger.info(f"  Profit (1):  {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(valid_labels)*100:.1f}%)")
            logger.info(f"  Neutral (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(valid_labels)*100:.1f}%)")
            logger.info(f"  Loss (-1):   {label_counts.get(-1, 0)} ({label_counts.get(-1, 0)/len(valid_labels)*100:.1f}%)")

        return df

    def apply_for_ml(self, df: pd.DataFrame, price_col: str = 'close',
                     return_multiclass: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply triple barrier labeling optimized for ML training.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        price_col : str
            Column name for closing price
        return_multiclass : bool
            If True, returns 3-class labels (0, 1, 2)
            If False, returns original labels (-1, 0, 1)

        Returns:
        --------
        Tuple[pd.DataFrame, np.ndarray]
            - DataFrame with valid samples (dropped NaN labels)
            - Array of labels ready for ML training
        """
        df_labeled = self.apply_triple_barrier(df, price_col)

        # Drop samples without valid labels
        df_clean = df_labeled.dropna(subset=['tbl_label']).copy()

        labels = df_clean['tbl_label'].values

        # Convert to 0-1-2 format for classification if requested
        if return_multiclass:
            # Map: -1 -> 0 (loss), 0 -> 1 (neutral), 1 -> 2 (profit)
            labels = labels + 1
            logger.info("Labels converted to multiclass format: 0=Loss, 1=Neutral, 2=Profit")

        logger.info(f"ML-ready dataset: {len(df_clean)} samples with valid labels")

        return df_clean, labels


def create_balanced_barriers(df: pd.DataFrame, price_col: str = 'close',
                            target_balance: float = 0.33) -> Tuple[float, float, int]:
    """
    Automatically determine barrier parameters to achieve balanced label distribution.

    This function analyzes historical data to find barrier percentages and time horizon
    that produce approximately balanced labels (33% profit, 33% neutral, 33% loss).

    Parameters:
    -----------
    df : pd.DataFrame
        Historical price data
    price_col : str
        Column name for closing price
    target_balance : float
        Target proportion for each class (default: 0.33)

    Returns:
    --------
    Tuple[float, float, int]
        (upper_barrier_pct, lower_barrier_pct, time_horizon_days)
    """
    logger.info("Auto-tuning triple barrier parameters for balanced labels...")

    # Try different parameter combinations
    best_params = None
    best_balance_score = float('inf')

    # Search grid
    barrier_pcts = [5, 7, 9, 11, 13, 15]  # Percentage thresholds
    time_horizons = [14, 21, 29, 42, 60]  # Days

    for barrier_pct in barrier_pcts:
        for horizon in time_horizons:
            labeler = TripleBarrierLabeler(
                upper_barrier=barrier_pct,
                lower_barrier=barrier_pct,
                time_horizon=horizon
            )

            df_labeled = labeler.apply_triple_barrier(df, price_col)
            labels = df_labeled['tbl_label'].dropna()

            if len(labels) < 100:
                continue

            # Calculate balance score (how far from perfect balance)
            label_counts = labels.value_counts(normalize=True)
            balance_score = sum(abs(label_counts.get(i, 0) - target_balance) for i in [-1, 0, 1])

            if balance_score < best_balance_score:
                best_balance_score = balance_score
                best_params = (barrier_pct, barrier_pct, horizon)

    if best_params:
        logger.info(f"Best balanced parameters: upper={best_params[0]}%, "
                   f"lower={best_params[1]}%, horizon={best_params[2]} days")
        logger.info(f"Balance score: {best_balance_score:.4f} (lower is better)")
        return best_params
    else:
        logger.warning("Could not find balanced parameters, using defaults")
        return (9.0, 9.0, 29)


def apply_meta_labeling(df: pd.DataFrame, primary_model_predictions: np.ndarray,
                       price_col: str = 'close') -> pd.DataFrame:
    """
    Apply meta-labeling: use triple barrier to determine bet SIZE, not just direction.

    Meta-labeling is a two-stage approach:
    1. Primary model predicts direction (buy/sell)
    2. Meta-model (using triple barriers) predicts if the bet will be profitable

    This helps filter out low-confidence signals and size positions appropriately.

    Parameters:
    -----------
    df : pd.DataFrame
        Price data
    primary_model_predictions : np.ndarray
        Array of primary model predictions (1 for buy signal, 0 for no signal)
    price_col : str
        Column name for closing price

    Returns:
    --------
    pd.DataFrame
        DataFrame with meta-labeling columns added
    """
    df = df.copy()

    # Apply triple barrier labeling
    labeler = TripleBarrierLabeler()
    df = labeler.apply_triple_barrier(df, price_col)

    # Meta-label: 1 if primary prediction AND barrier is profitable, 0 otherwise
    df['meta_label'] = (
        (primary_model_predictions[:len(df)] == 1) &
        (df['tbl_label'] == 1)
    ).astype(int)

    logger.info("Meta-labeling applied")
    logger.info(f"Primary signals: {primary_model_predictions.sum()}")
    logger.info(f"Profitable signals (meta-label=1): {df['meta_label'].sum()}")

    return df


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')

    # Simulate stock price with trend and noise
    price = 100 + np.cumsum(np.random.randn(500) * 2 + 0.1)

    df = pd.DataFrame({
        'close': price,
        'open': price * (1 + np.random.randn(500) * 0.01),
        'high': price * (1 + np.abs(np.random.randn(500)) * 0.02),
        'low': price * (1 - np.abs(np.random.randn(500)) * 0.02),
        'volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)

    # Test 1: Basic triple barrier labeling
    logger.info("\n=== Test 1: Basic Triple Barrier Labeling ===")
    labeler = TripleBarrierLabeler(upper_barrier=9.0, lower_barrier=9.0, time_horizon=29)
    df_labeled = labeler.apply_triple_barrier(df)

    print("\nSample labeled data:")
    print(df_labeled[['close', 'tbl_label', 'tbl_barrier_touched', 'tbl_days_to_barrier', 'tbl_return']].head(10))

    # Test 2: ML-ready format
    logger.info("\n=== Test 2: ML-Ready Format ===")
    df_ml, labels = labeler.apply_for_ml(df, return_multiclass=True)
    print(f"\nML dataset shape: {df_ml.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels.astype(int))}")

    # Test 3: Auto-balanced parameters
    logger.info("\n=== Test 3: Auto-Balanced Parameters ===")
    best_params = create_balanced_barriers(df)
    print(f"\nBest parameters: {best_params}")
