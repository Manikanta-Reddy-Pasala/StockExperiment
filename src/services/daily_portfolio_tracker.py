"""
Daily Portfolio Tracker Service

This service fetches daily portfolio data from Fyers API and tracks performance metrics over time.
It stores daily snapshots and calculates performance analytics.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class DailyPortfolioTracker:
    """
    Tracks daily portfolio performance using Fyers API data.
    Stores daily snapshots and calculates performance metrics.
    """

    def __init__(self, db_url: str = None):
        try:
            import config
            self.db_url = db_url or config.POSTGRES_URL
        except ImportError:
            self.db_url = db_url or "postgresql://postgres:password@localhost/stockexperiment"

        self.engine = None
        self.cache_manager = None
        self.init_database()
        self.init_cache()

    def init_database(self):
        """Initialize the portfolio tracking database"""
        try:
            # Initialize PostgreSQL only (SQLite removed)
            self.engine = create_engine(self.db_url)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Using PostgreSQL database")
            self._init_postgresql_tables()

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _init_postgresql_tables(self):
        """Initialize PostgreSQL tables"""
        with self.engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS daily_portfolio_snapshots (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        date DATE NOT NULL,
                        total_value DECIMAL(15,2) NOT NULL,
                        total_invested DECIMAL(15,2) NOT NULL,
                        total_pnl DECIMAL(15,2) NOT NULL,
                        total_pnl_percent DECIMAL(8,4) NOT NULL,
                        day_change DECIMAL(15,2) DEFAULT 0,
                        day_change_percent DECIMAL(8,4) DEFAULT 0,
                        holdings_data JSONB NOT NULL,
                        positions_data JSONB NOT NULL,
                        market_data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, date)
                    )
                """))

                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS portfolio_performance_metrics (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        period_start DATE NOT NULL,
                        period_end DATE NOT NULL,
                        period_type VARCHAR(20) NOT NULL,
                        total_return_percent DECIMAL(8,4) NOT NULL,
                        annualized_return DECIMAL(8,4) NOT NULL,
                        volatility DECIMAL(8,4) NOT NULL,
                        sharpe_ratio DECIMAL(8,4) DEFAULT 0,
                        max_drawdown DECIMAL(8,4) DEFAULT 0,
                        win_rate DECIMAL(8,4) DEFAULT 0,
                        best_day_return DECIMAL(8,4) DEFAULT 0,
                        worst_day_return DECIMAL(8,4) DEFAULT 0,
                        total_trading_days INTEGER DEFAULT 0,
                        benchmark_return DECIMAL(8,4) DEFAULT 0,
                        alpha DECIMAL(8,4) DEFAULT 0,
                        beta DECIMAL(8,4) DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, period_start, period_end, period_type)
                    )
                """))

                # Create indexes for better performance
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_daily_snapshots_user_date
                    ON daily_portfolio_snapshots(user_id, date DESC)
                """))

                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_performance_metrics_user_period
                    ON portfolio_performance_metrics(user_id, period_type, period_end DESC)
                """))

                logger.info("Portfolio tracking PostgreSQL database initialized successfully")

    # SQLite fallback removed

    def init_cache(self):
        """Initialize Redis cache manager"""
        try:
            from .redis_cache_manager import get_cache_manager
            self.cache_manager = get_cache_manager()
            logger.info("Redis cache manager initialized for portfolio tracker")
        except Exception as e:
            logger.warning(f"Cache manager initialization failed: {e}")
            self.cache_manager = None

    def fetch_and_store_daily_snapshot(self, user_id: int, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch current portfolio data from Fyers API and store daily snapshot
        """
        try:
            from .brokers.fyers_service import get_fyers_service

            fyers_service = get_fyers_service()
            snapshot_date = date_str if date_str else date.today().isoformat()

            # Get portfolio data from Fyers
            holdings_response = fyers_service.holdings(user_id)
            positions_response = fyers_service.positions(user_id)
            funds_response = fyers_service.funds(user_id)

            # Check if we have valid data
            if holdings_response.get('s') != 'ok':
                logger.warning(f"Holdings data not available: {holdings_response.get('message')}")
                holdings_data = []
            else:
                holdings_data = holdings_response.get('holdings', [])

            if positions_response.get('s') != 'ok':
                logger.warning(f"Positions data not available: {positions_response.get('message')}")
                positions_data = []
            else:
                positions_data = positions_response.get('netPositions', [])

            # Calculate portfolio metrics
            total_value = 0
            total_invested = 0
            total_pnl = 0

            # Process holdings
            for holding in holdings_data:
                market_val = holding.get('market_val', 0)
                cost_price = holding.get('costPrice', 0)
                quantity = holding.get('quantity', 0)

                total_value += market_val
                total_invested += cost_price * quantity
                total_pnl += holding.get('pl', 0)

            # Process positions
            for position in positions_data:
                total_pnl += position.get('pl', 0)
                total_value += position.get('ltp', 0) * position.get('qty', 0)

            # Calculate percentages
            total_pnl_percent = (total_pnl / total_invested * 100) if total_invested > 0 else 0

            # Get previous day's data for day change calculation
            previous_snapshot = self.get_latest_snapshot(user_id, days_back=1)
            day_change = 0
            day_change_percent = 0

            if previous_snapshot:
                day_change = total_value - previous_snapshot['total_value']
                day_change_percent = (day_change / previous_snapshot['total_value'] * 100) if previous_snapshot['total_value'] > 0 else 0

            # Store snapshot
            snapshot_data = {
                'user_id': user_id,
                'date': snapshot_date,
                'total_value': total_value,
                'total_invested': total_invested,
                'total_pnl': total_pnl,
                'total_pnl_percent': total_pnl_percent,
                'day_change': day_change,
                'day_change_percent': day_change_percent,
                'holdings_data': json.dumps(holdings_data),
                'positions_data': json.dumps(positions_data),
                'market_data': json.dumps(funds_response)
            }

            self.store_snapshot(snapshot_data)

            logger.info(f"Portfolio snapshot stored for user {user_id} on {snapshot_date}")
            return {
                'success': True,
                'data': snapshot_data,
                'message': 'Portfolio snapshot stored successfully'
            }

        except Exception as e:
            logger.error(f"Error fetching portfolio snapshot: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to fetch portfolio snapshot'
            }

    def store_snapshot(self, snapshot_data: Dict[str, Any]):
        """Store a portfolio snapshot in the database"""
        try:
            with self.engine.begin() as conn:
                # PostgreSQL uses ON CONFLICT
                conn.execute(text("""
                    INSERT INTO daily_portfolio_snapshots
                    (user_id, date, total_value, total_invested, total_pnl, total_pnl_percent,
                     day_change, day_change_percent, holdings_data, positions_data, market_data)
                    VALUES (:user_id, :date, :total_value, :total_invested, :total_pnl, :total_pnl_percent,
                            :day_change, :day_change_percent, :holdings_data, :positions_data, :market_data)
                    ON CONFLICT (user_id, date) DO UPDATE SET
                        total_value = EXCLUDED.total_value,
                        total_invested = EXCLUDED.total_invested,
                        total_pnl = EXCLUDED.total_pnl,
                        total_pnl_percent = EXCLUDED.total_pnl_percent,
                        day_change = EXCLUDED.day_change,
                        day_change_percent = EXCLUDED.day_change_percent,
                        holdings_data = EXCLUDED.holdings_data,
                        positions_data = EXCLUDED.positions_data,
                        market_data = EXCLUDED.market_data,
                        created_at = CURRENT_TIMESTAMP
                """), {
                    'user_id': snapshot_data['user_id'],
                    'date': snapshot_data['date'],
                    'total_value': snapshot_data['total_value'],
                    'total_invested': snapshot_data['total_invested'],
                    'total_pnl': snapshot_data['total_pnl'],
                    'total_pnl_percent': snapshot_data['total_pnl_percent'],
                    'day_change': snapshot_data['day_change'],
                    'day_change_percent': snapshot_data['day_change_percent'],
                    'holdings_data': json.dumps(json.loads(snapshot_data['holdings_data'])) if isinstance(snapshot_data['holdings_data'], str) else json.dumps(snapshot_data['holdings_data']),
                    'positions_data': json.dumps(json.loads(snapshot_data['positions_data'])) if isinstance(snapshot_data['positions_data'], str) else json.dumps(snapshot_data['positions_data']),
                    'market_data': json.dumps(json.loads(snapshot_data['market_data'])) if isinstance(snapshot_data['market_data'], str) else json.dumps(snapshot_data['market_data']) if snapshot_data.get('market_data') else None
                })

                logger.info(f"Snapshot stored for user {snapshot_data['user_id']} on {snapshot_data['date']}")

                # Clear cache for this user
                if self.cache_manager:
                    self.cache_manager.clear_user_cache(snapshot_data['user_id'])

        except Exception as e:
            logger.error(f"Error storing snapshot: {e}")
            raise

    def get_latest_snapshot(self, user_id: int, days_back: int = 0) -> Optional[Dict[str, Any]]:
        """Get the latest portfolio snapshot for a user"""
        # Try cache first
        if self.cache_manager and days_back == 0:
            cached = self.cache_manager.get_cached_portfolio_snapshot(user_id)
            if cached:
                return cached

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT * FROM daily_portfolio_snapshots
                    WHERE user_id = :user_id
                    ORDER BY date DESC
                    LIMIT 1 OFFSET :days_back
                """), {'user_id': user_id, 'days_back': days_back})

                row = result.fetchone()
                if row:
                    snapshot = dict(row._mapping)
                    # Parse JSON fields
                    if snapshot.get('holdings_data'):
                        snapshot['holdings_data'] = json.loads(snapshot['holdings_data']) if isinstance(snapshot['holdings_data'], str) else snapshot['holdings_data']
                    if snapshot.get('positions_data'):
                        snapshot['positions_data'] = json.loads(snapshot['positions_data']) if isinstance(snapshot['positions_data'], str) else snapshot['positions_data']
                    if snapshot.get('market_data'):
                        snapshot['market_data'] = json.loads(snapshot['market_data']) if isinstance(snapshot['market_data'], str) else snapshot['market_data']

                    # Cache the result
                    if self.cache_manager and days_back == 0:
                        self.cache_manager.cache_portfolio_snapshot(user_id, snapshot, ttl=300)

                    return snapshot
                return None
        except Exception as e:
            logger.error(f"Error getting latest snapshot: {e}")
            return None

    def get_portfolio_history(self, user_id: int, days: int = 30) -> List[Dict[str, Any]]:
        """Get portfolio history for specified number of days"""
        # Try cache first
        if self.cache_manager:
            cached = self.cache_manager.get_cached_portfolio_history(user_id, days)
            if cached:
                return cached

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT * FROM daily_portfolio_snapshots
                    WHERE user_id = :user_id
                    AND date >= CURRENT_DATE - INTERVAL '{} days'
                    ORDER BY date ASC
                """.format(days)), {'user_id': user_id})

                rows = result.fetchall()
                history = []
                for row in rows:
                    snapshot = dict(row._mapping)
                    # Parse JSON fields
                    if snapshot.get('holdings_data'):
                        snapshot['holdings_data'] = json.loads(snapshot['holdings_data']) if isinstance(snapshot['holdings_data'], str) else snapshot['holdings_data']
                    if snapshot.get('positions_data'):
                        snapshot['positions_data'] = json.loads(snapshot['positions_data']) if isinstance(snapshot['positions_data'], str) else snapshot['positions_data']
                    if snapshot.get('market_data'):
                        snapshot['market_data'] = json.loads(snapshot['market_data']) if isinstance(snapshot['market_data'], str) else snapshot['market_data']
                    history.append(snapshot)

                # Cache the result
                if self.cache_manager:
                    self.cache_manager.cache_portfolio_history(user_id, days, history, ttl=1800)

                return history
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return []

    def calculate_performance_metrics(self, user_id: int, period_days: int = 30) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        # Try cache first
        if self.cache_manager:
            cached = self.cache_manager.get_cached_portfolio_performance(user_id, str(period_days))
            if cached:
                return cached

        try:
            # Get historical data
            history = self.get_portfolio_history(user_id, period_days)

            if len(history) < 2:
                return {
                    'success': False,
                    'error': 'Insufficient historical data for performance calculation',
                    'data': {}
                }

            # Convert to DataFrame for easier calculations
            df = pd.DataFrame(history)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # Calculate daily returns
            df['daily_return'] = df['total_value'].pct_change()

            # Calculate metrics
            start_value = df['total_value'].iloc[0]
            end_value = df['total_value'].iloc[-1]
            total_return_percent = ((end_value - start_value) / start_value * 100) if start_value > 0 else 0

            # Annualized return (assuming 252 trading days)
            days_actual = len(df)
            annualized_return = (pow(end_value / start_value, 252 / days_actual) - 1) * 100 if start_value > 0 else 0

            # Volatility (annualized)
            daily_returns = df['daily_return'].dropna()
            volatility = daily_returns.std() * (252 ** 0.5) * 100 if len(daily_returns) > 1 else 0

            # Sharpe ratio (assuming risk-free rate of 5%)
            risk_free_rate = 0.05
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0

            # Maximum drawdown
            df['cumulative_return'] = (1 + df['daily_return'].fillna(0)).cumprod()
            df['running_max'] = df['cumulative_return'].expanding().max()
            df['drawdown'] = (df['cumulative_return'] - df['running_max']) / df['running_max']
            max_drawdown = df['drawdown'].min() * 100

            # Win rate
            positive_days = (daily_returns > 0).sum()
            win_rate = (positive_days / len(daily_returns) * 100) if len(daily_returns) > 0 else 0

            # Best and worst days
            best_day_return = daily_returns.max() * 100 if len(daily_returns) > 0 else 0
            worst_day_return = daily_returns.min() * 100 if len(daily_returns) > 0 else 0

            metrics = {
                'period_days': period_days,
                'total_return_percent': round(total_return_percent, 2),
                'annualized_return': round(annualized_return, 2),
                'volatility': round(volatility, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown, 2),
                'win_rate': round(win_rate, 2),
                'best_day_return': round(best_day_return, 2),
                'worst_day_return': round(worst_day_return, 2),
                'total_trading_days': len(daily_returns),
                'start_value': round(start_value, 2),
                'end_value': round(end_value, 2),
                'current_pnl': round(df['total_pnl'].iloc[-1], 2),
                'current_pnl_percent': round(df['total_pnl_percent'].iloc[-1], 2)
            }

            result = {
                'success': True,
                'data': metrics,
                'history_chart_data': df[['date', 'total_value', 'daily_return', 'drawdown']].to_dict('records')
            }

            # Cache the result
            if self.cache_manager:
                self.cache_manager.cache_portfolio_performance(user_id, str(period_days), result, ttl=600)

            return result

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': {}
            }

    def get_portfolio_performance_summary(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive portfolio performance summary"""
        try:
            # Get multiple period performances
            performance_1w = self.calculate_performance_metrics(user_id, 7)
            performance_1m = self.calculate_performance_metrics(user_id, 30)
            performance_3m = self.calculate_performance_metrics(user_id, 90)
            performance_1y = self.calculate_performance_metrics(user_id, 365)

            # Get latest snapshot
            latest = self.get_latest_snapshot(user_id)

            summary = {
                'current_portfolio': latest,
                'performance': {
                    '1W': performance_1w.get('data', {}),
                    '1M': performance_1m.get('data', {}),
                    '3M': performance_3m.get('data', {}),
                    '1Y': performance_1y.get('data', {})
                },
                'chart_data': performance_1m.get('history_chart_data', []),
                'last_updated': datetime.now().isoformat()
            }

            return {
                'success': True,
                'data': summary
            }

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': {}
            }


# Singleton instance
_portfolio_tracker = None

def get_portfolio_tracker() -> DailyPortfolioTracker:
    """Get the singleton portfolio tracker instance"""
    global _portfolio_tracker
    if _portfolio_tracker is None:
        _portfolio_tracker = DailyPortfolioTracker()
    return _portfolio_tracker