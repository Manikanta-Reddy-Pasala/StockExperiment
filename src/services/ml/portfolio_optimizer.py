"""
Modern Portfolio Theory Optimizer
Optimal portfolio allocation using mean-variance optimization and risk parity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from datetime import datetime, timedelta
import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Portfolio optimization using Modern Portfolio Theory.

    Methods:
    1. Mean-Variance Optimization (Markowitz)
    2. Maximum Sharpe Ratio
    3. Minimum Variance
    4. Risk Parity
    5. Black-Litterman (with ML predictions)
    """

    def __init__(self, db_session, risk_free_rate: float = 0.06):
        """
        Initialize optimizer.

        Args:
            db_session: Database session
            risk_free_rate: Annual risk-free rate (default: 6% for India)
        """
        self.db = db_session
        self.risk_free_rate = risk_free_rate

    def optimize_portfolio(self,
                          stocks: List[Dict],
                          method: str = 'max_sharpe',
                          target_return: Optional[float] = None,
                          max_weight: float = 0.20,
                          min_weight: float = 0.01) -> Dict:
        """
        Optimize portfolio allocation.

        Args:
            stocks: List of stock dictionaries with ML predictions
            method: 'max_sharpe', 'min_variance', 'risk_parity', 'ml_enhanced'
            target_return: Target return for efficient frontier
            max_weight: Maximum weight per stock (default: 20%)
            min_weight: Minimum weight per stock (default: 1%)

        Returns:
            Dictionary with optimal weights and portfolio metrics
        """
        logger.info(f"Optimizing portfolio for {len(stocks)} stocks using {method}")

        try:
            # Get historical returns and covariance
            symbols = [s['symbol'] for s in stocks]
            returns, cov_matrix = self._get_returns_and_covariance(symbols)

            if returns is None or cov_matrix is None:
                return self._equal_weight_fallback(stocks)

            # ML-enhanced expected returns
            expected_returns = self._calculate_expected_returns(stocks, returns)

            # Optimize based on method
            if method == 'max_sharpe':
                weights = self._optimize_max_sharpe(expected_returns, cov_matrix,
                                                    max_weight, min_weight)
            elif method == 'min_variance':
                weights = self._optimize_min_variance(cov_matrix, max_weight, min_weight)
            elif method == 'risk_parity':
                weights = self._optimize_risk_parity(cov_matrix, max_weight, min_weight)
            elif method == 'ml_enhanced':
                weights = self._optimize_ml_enhanced(stocks, expected_returns, cov_matrix,
                                                     max_weight, min_weight)
            elif method == 'efficient_frontier' and target_return:
                weights = self._optimize_target_return(expected_returns, cov_matrix,
                                                       target_return, max_weight, min_weight)
            else:
                logger.warning(f"Unknown method: {method}, using max_sharpe")
                weights = self._optimize_max_sharpe(expected_returns, cov_matrix,
                                                    max_weight, min_weight)

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            portfolio_sharpe = (portfolio_return - self.risk_free_rate / 252) / portfolio_vol

            # Create allocation dictionary
            allocations = []
            for i, symbol in enumerate(symbols):
                if weights[i] > 0.001:  # Only include meaningful allocations
                    allocations.append({
                        'symbol': symbol,
                        'weight': round(weights[i], 4),
                        'expected_return': round(expected_returns[i] * 252, 4),  # Annualized
                        'ml_score': stocks[i].get('ml_prediction_score', 0.5)
                    })

            # Sort by weight
            allocations = sorted(allocations, key=lambda x: x['weight'], reverse=True)

            result = {
                'method': method,
                'allocations': allocations,
                'metrics': {
                    'expected_return': round(portfolio_return * 252, 4),  # Annualized
                    'volatility': round(portfolio_vol * np.sqrt(252), 4),  # Annualized
                    'sharpe_ratio': round(portfolio_sharpe * np.sqrt(252), 4),  # Annualized
                    'num_stocks': len(allocations),
                    'max_weight': round(max(weights), 4),
                    'min_weight': round(min([w for w in weights if w > 0.001]), 4)
                },
                'optimized_at': datetime.now().isoformat()
            }

            logger.info(f"Portfolio optimized: Sharpe={portfolio_sharpe * np.sqrt(252):.2f}, "
                       f"Return={portfolio_return * 252:.2%}")

            return result

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}", exc_info=True)
            return self._equal_weight_fallback(stocks)

    def _get_returns_and_covariance(self, symbols: List[str],
                                     days: int = 90) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate historical returns and covariance matrix."""
        try:
            # Get historical prices for all symbols
            query = text("""
                SELECT symbol, date, close
                FROM historical_data
                WHERE symbol = ANY(:symbols)
                AND date >= NOW() - INTERVAL ':days days'
                ORDER BY symbol, date
            """)

            result = self.db.execute(query, {'symbols': symbols, 'days': days})
            df = pd.DataFrame(result.fetchall(), columns=['symbol', 'date', 'close'])

            # Pivot to get price matrix
            price_matrix = df.pivot(index='date', columns='symbol', values='close')

            # Filter to symbols with complete data
            price_matrix = price_matrix.dropna(axis=1)

            if len(price_matrix) < 30 or len(price_matrix.columns) < 3:
                logger.warning("Insufficient data for covariance calculation")
                return None, None

            # Calculate returns
            returns = price_matrix.pct_change().dropna()

            # Expected returns (mean)
            expected_returns = returns.mean().values

            # Covariance matrix
            cov_matrix = returns.cov().values

            return expected_returns, cov_matrix

        except Exception as e:
            logger.error(f"Failed to calculate returns/covariance: {e}")
            return None, None

    def _calculate_expected_returns(self, stocks: List[Dict],
                                    historical_returns: np.ndarray) -> np.ndarray:
        """
        Calculate expected returns using ML predictions + historical returns.

        Combines:
        - Historical mean returns (30%)
        - ML predicted returns (70%)
        """
        ml_returns = []
        for stock in stocks:
            predicted_change = float(stock.get('predicted_change_pct', 0)) / 100
            # Convert 2-week prediction to daily
            daily_return = (1 + predicted_change) ** (1/14) - 1
            ml_returns.append(daily_return)

        ml_returns = np.array(ml_returns)

        # Blend historical and ML-predicted returns
        expected_returns = 0.3 * historical_returns + 0.7 * ml_returns

        return expected_returns

    def _optimize_max_sharpe(self, expected_returns: np.ndarray,
                             cov_matrix: np.ndarray,
                             max_weight: float,
                             min_weight: float) -> np.ndarray:
        """Optimize for maximum Sharpe ratio."""
        n = len(expected_returns)

        # Objective: Negative Sharpe ratio (we minimize)
        def neg_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate / 252) / portfolio_vol
            return -sharpe

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]

        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(n))

        # Initial guess: equal weights
        init_weights = np.array([1.0 / n] * n)

        # Optimize
        result = minimize(neg_sharpe, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x if result.success else init_weights

    def _optimize_min_variance(self, cov_matrix: np.ndarray,
                               max_weight: float,
                               min_weight: float) -> np.ndarray:
        """Optimize for minimum variance (lowest risk)."""
        n = cov_matrix.shape[0]

        # Objective: Portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(n))

        # Initial guess
        init_weights = np.array([1.0 / n] * n)

        # Optimize
        result = minimize(portfolio_variance, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x if result.success else init_weights

    def _optimize_risk_parity(self, cov_matrix: np.ndarray,
                              max_weight: float,
                              min_weight: float) -> np.ndarray:
        """
        Risk parity optimization: Equal risk contribution from each asset.
        """
        n = cov_matrix.shape[0]

        # Objective: Difference between risk contributions
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            target_risk = portfolio_vol / n  # Equal risk
            return np.sum((risk_contrib - target_risk) ** 2)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(n))

        # Initial guess
        init_weights = np.array([1.0 / n] * n)

        # Optimize
        result = minimize(risk_parity_objective, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x if result.success else init_weights

    def _optimize_ml_enhanced(self, stocks: List[Dict],
                              expected_returns: np.ndarray,
                              cov_matrix: np.ndarray,
                              max_weight: float,
                              min_weight: float) -> np.ndarray:
        """
        ML-enhanced optimization: Combines Sharpe optimization with ML scores.
        """
        # Extract ML scores
        ml_scores = np.array([s.get('ml_prediction_score', 0.5) for s in stocks])
        ml_confidence = np.array([s.get('ml_confidence', 0.5) for s in stocks])
        ml_risk = np.array([s.get('ml_risk_score', 0.5) for s in stocks])

        # Combined score (higher = better)
        ml_combined = ml_scores * ml_confidence * (1 - ml_risk)
        ml_combined = ml_combined / np.sum(ml_combined)  # Normalize

        # Get Sharpe-optimal weights
        sharpe_weights = self._optimize_max_sharpe(expected_returns, cov_matrix,
                                                    max_weight, min_weight)

        # Blend: 60% Sharpe-optimal, 40% ML-weighted
        blended_weights = 0.6 * sharpe_weights + 0.4 * ml_combined

        # Re-normalize
        blended_weights = blended_weights / np.sum(blended_weights)

        # Ensure bounds
        blended_weights = np.clip(blended_weights, min_weight, max_weight)
        blended_weights = blended_weights / np.sum(blended_weights)

        return blended_weights

    def _optimize_target_return(self, expected_returns: np.ndarray,
                                cov_matrix: np.ndarray,
                                target_return: float,
                                max_weight: float,
                                min_weight: float) -> np.ndarray:
        """Optimize for target return on efficient frontier."""
        n = len(expected_returns)

        # Objective: Minimize variance
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return / 252}
        ]

        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(n))

        # Initial guess
        init_weights = np.array([1.0 / n] * n)

        # Optimize
        result = minimize(portfolio_variance, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x if result.success else init_weights

    def _equal_weight_fallback(self, stocks: List[Dict]) -> Dict:
        """Fallback to equal weighting if optimization fails."""
        n = len(stocks)
        weight = 1.0 / n

        allocations = [
            {
                'symbol': s['symbol'],
                'weight': round(weight, 4),
                'expected_return': 0,
                'ml_score': s.get('ml_prediction_score', 0.5)
            }
            for s in stocks
        ]

        return {
            'method': 'equal_weight_fallback',
            'allocations': allocations,
            'metrics': {
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'num_stocks': n,
                'max_weight': weight,
                'min_weight': weight
            },
            'optimized_at': datetime.now().isoformat()
        }

    def generate_efficient_frontier(self, stocks: List[Dict],
                                    num_points: int = 50) -> pd.DataFrame:
        """
        Generate efficient frontier.

        Returns:
            DataFrame with return, volatility, sharpe for each point
        """
        logger.info(f"Generating efficient frontier with {num_points} points")

        symbols = [s['symbol'] for s in stocks]
        returns, cov_matrix = self._get_returns_and_covariance(symbols)

        if returns is None:
            return pd.DataFrame()

        expected_returns = self._calculate_expected_returns(stocks, returns)

        # Range of target returns
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        target_returns = np.linspace(min_return, max_return, num_points)

        frontier = []

        for target_return in target_returns:
            try:
                weights = self._optimize_target_return(
                    expected_returns, cov_matrix, target_return, 0.20, 0.01
                )

                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                sharpe = (portfolio_return - self.risk_free_rate / 252) / portfolio_vol

                frontier.append({
                    'return': portfolio_return * 252,  # Annualized
                    'volatility': portfolio_vol * np.sqrt(252),  # Annualized
                    'sharpe': sharpe * np.sqrt(252)  # Annualized
                })

            except:
                continue

        return pd.DataFrame(frontier)

    def recommend_allocation(self, capital: float,
                            allocations: List[Dict],
                            current_prices: Dict[str, float]) -> Dict:
        """
        Convert portfolio weights to actual share quantities.

        Args:
            capital: Total capital to invest
            allocations: Portfolio allocations with weights
            current_prices: Dict of {symbol: price}

        Returns:
            Dict with recommended share quantities
        """
        recommendations = []

        for alloc in allocations:
            symbol = alloc['symbol']
            weight = alloc['weight']
            price = current_prices.get(symbol, 0)

            if price > 0:
                target_value = capital * weight
                shares = int(target_value / price)
                actual_value = shares * price
                actual_weight = actual_value / capital if capital > 0 else 0

                recommendations.append({
                    'symbol': symbol,
                    'target_weight': weight,
                    'actual_weight': round(actual_weight, 4),
                    'shares': shares,
                    'price': price,
                    'value': round(actual_value, 2)
                })

        total_invested = sum(r['value'] for r in recommendations)
        cash_remaining = capital - total_invested

        return {
            'capital': capital,
            'total_invested': round(total_invested, 2),
            'cash_remaining': round(cash_remaining, 2),
            'utilization': round(total_invested / capital, 4),
            'recommendations': recommendations
        }
