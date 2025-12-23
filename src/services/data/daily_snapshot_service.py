"""
Daily Suggested Stocks Snapshot Service
Stores one snapshot per day with upsert logic (replaces same-day data).
"""

from datetime import datetime, date
from typing import List, Dict, Optional
import logging
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)


class DailySnapshotService:
    """
    Service to save daily snapshots of suggested stocks with technical indicators.
    Implements upsert logic: replaces data for same date/symbol/strategy.
    """

    def __init__(self, db_session):
        self.db = db_session

    def save_daily_snapshot(
        self,
        suggested_stocks: List[Dict],
        snapshot_date: Optional[date] = None
    ) -> Dict:
        """
        Save daily snapshot of suggested stocks with 8-21 EMA strategy indicators.
        Uses INSERT ON CONFLICT to replace same-day data.

        Args:
            suggested_stocks: List of suggested stock dictionaries with EMA strategy indicators
            snapshot_date: Date for snapshot (defaults to today)

        Returns:
            Dictionary with save statistics
        """
        if snapshot_date is None:
            snapshot_date = date.today()
        
        logger.info(f"Saving daily snapshot for {snapshot_date} with {len(suggested_stocks)} stocks")
        
        inserted = 0
        updated = 0
        errors = 0
        
        for stock in suggested_stocks:
            try:
                symbol = stock.get('symbol')

                # Prepare data for insert/update
                data = {
                    'date': snapshot_date,
                    'symbol': symbol,
                    'strategy': stock.get('strategy', 'ema_8_21'),  # Required for unique constraint
                    'model_type': stock.get('model_type', 'traditional'),  # Required for unique constraint
                    'stock_name': stock.get('name'),
                    'current_price': stock.get('current_price'),
                    'market_cap': stock.get('market_cap'),

                    # Selection
                    'selection_score': stock.get('selection_score'),  # EMA strategy score
                    'rank': stock.get('rank'),

                    # 8-21 EMA Strategy Indicators
                    'ema_8': stock.get('ema_8'),
                    'ema_21': stock.get('ema_21'),
                    'ema_trend_score': stock.get('ema_trend_score'),
                    'demarker': stock.get('demarker'),

                    # Fibonacci Targets
                    'fib_target_1': stock.get('fib_target_1'),
                    'fib_target_2': stock.get('fib_target_2'),
                    'fib_target_3': stock.get('fib_target_3'),

                    # Signals (8-21 EMA based)
                    'buy_signal': stock.get('buy_signal', False),
                    'sell_signal': stock.get('sell_signal', False),
                    'signal_quality': stock.get('signal_quality', 'none'),

                    # Technical Indicators (legacy, optional)
                    'rsi_14': stock.get('rsi_14'),
                    'macd': stock.get('macd'),
                    'sma_50': stock.get('sma_50'),
                    'sma_200': stock.get('sma_200'),

                    # Fundamental Metrics
                    'pe_ratio': stock.get('pe_ratio'),
                    'pb_ratio': stock.get('pb_ratio'),
                    'roe': stock.get('roe'),
                    'eps': stock.get('eps'),
                    'beta': stock.get('beta'),

                    # Growth & Profitability
                    'revenue_growth': stock.get('revenue_growth'),
                    'earnings_growth': stock.get('earnings_growth'),
                    'operating_margin': stock.get('operating_margin'),

                    # Trading Signals
                    'target_price': stock.get('target_price'),
                    'stop_loss': stock.get('stop_loss'),
                    'recommendation': stock.get('recommendation'),
                    'reason': stock.get('reason'),

                    # Metadata
                    'sector': stock.get('sector'),
                    'market_cap_category': stock.get('market_cap_category')
                }

                # Insert with ON CONFLICT UPDATE (upsert)
                # Includes strategy and model_type to match unique constraint
                upsert_query = text("""
                    INSERT INTO daily_suggested_stocks (
                        date, symbol, strategy, model_type, stock_name, current_price, market_cap,
                        selection_score, rank,
                        ema_8, ema_21, ema_trend_score, demarker,
                        fib_target_1, fib_target_2, fib_target_3,
                        buy_signal, sell_signal, signal_quality,
                        rsi_14, macd, sma_50, sma_200,
                        pe_ratio, pb_ratio, roe, eps, beta,
                        revenue_growth, earnings_growth, operating_margin,
                        target_price, stop_loss, recommendation, reason,
                        sector, market_cap_category, created_at
                    ) VALUES (
                        :date, :symbol, :strategy, :model_type, :stock_name, :current_price, :market_cap,
                        :selection_score, :rank,
                        :ema_8, :ema_21, :ema_trend_score, :demarker,
                        :fib_target_1, :fib_target_2, :fib_target_3,
                        :buy_signal, :sell_signal, :signal_quality,
                        :rsi_14, :macd, :sma_50, :sma_200,
                        :pe_ratio, :pb_ratio, :roe, :eps, :beta,
                        :revenue_growth, :earnings_growth, :operating_margin,
                        :target_price, :stop_loss, :recommendation, :reason,
                        :sector, :market_cap_category, CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (date, symbol, strategy, model_type)
                    DO UPDATE SET
                        stock_name = EXCLUDED.stock_name,
                        current_price = EXCLUDED.current_price,
                        market_cap = EXCLUDED.market_cap,
                        selection_score = EXCLUDED.selection_score,
                        rank = EXCLUDED.rank,
                        ema_8 = EXCLUDED.ema_8,
                        ema_21 = EXCLUDED.ema_21,
                        ema_trend_score = EXCLUDED.ema_trend_score,
                        demarker = EXCLUDED.demarker,
                        fib_target_1 = EXCLUDED.fib_target_1,
                        fib_target_2 = EXCLUDED.fib_target_2,
                        fib_target_3 = EXCLUDED.fib_target_3,
                        buy_signal = EXCLUDED.buy_signal,
                        sell_signal = EXCLUDED.sell_signal,
                        signal_quality = EXCLUDED.signal_quality,
                        rsi_14 = EXCLUDED.rsi_14,
                        macd = EXCLUDED.macd,
                        sma_50 = EXCLUDED.sma_50,
                        sma_200 = EXCLUDED.sma_200,
                        pe_ratio = EXCLUDED.pe_ratio,
                        pb_ratio = EXCLUDED.pb_ratio,
                        roe = EXCLUDED.roe,
                        eps = EXCLUDED.eps,
                        beta = EXCLUDED.beta,
                        revenue_growth = EXCLUDED.revenue_growth,
                        earnings_growth = EXCLUDED.earnings_growth,
                        operating_margin = EXCLUDED.operating_margin,
                        target_price = EXCLUDED.target_price,
                        stop_loss = EXCLUDED.stop_loss,
                        recommendation = EXCLUDED.recommendation,
                        reason = EXCLUDED.reason,
                        sector = EXCLUDED.sector,
                        market_cap_category = EXCLUDED.market_cap_category
                    RETURNING (xmax = 0) AS inserted
                """)
                
                result = self.db.execute(upsert_query, data)
                row = result.fetchone()
                
                if row and row[0]:  # xmax = 0 means INSERT
                    inserted += 1
                else:  # xmax != 0 means UPDATE
                    updated += 1
                
            except Exception as e:
                logger.error(f"Error saving snapshot for {symbol}: {e}")
                errors += 1
        
        # Commit transaction
        self.db.commit()
        
        stats = {
            'date': snapshot_date.isoformat(),
            'total_stocks': len(suggested_stocks),
            'inserted': inserted,
            'updated': updated,
            'errors': errors
        }
        
        logger.info(f"Snapshot saved: {stats}")
        return stats
    
    def get_latest_snapshot(self, strategy: str = 'default_risk', limit: int = 50) -> List[Dict]:
        """
        Retrieve the latest daily snapshot.
        
        Args:
            strategy: Strategy filter
            limit: Maximum number of stocks to return
            
        Returns:
            List of stock dictionaries from latest snapshot
        """
        query = text("""
            SELECT
                date, symbol, stock_name, current_price, market_cap,
                strategy, selection_score, rank,
                ema_8, ema_21, ema_trend_score, demarker,
                fib_target_1, fib_target_2, fib_target_3,
                buy_signal, sell_signal, signal_quality,
                rsi_14, macd, sma_50, sma_200,
                pe_ratio, pb_ratio, roe, eps, beta,
                revenue_growth, earnings_growth, operating_margin,
                target_price, stop_loss, recommendation, reason,
                sector, market_cap_category, created_at
            FROM daily_suggested_stocks
            WHERE strategy = :strategy
            AND date = (SELECT MAX(date) FROM daily_suggested_stocks WHERE strategy = :strategy)
            ORDER BY selection_score DESC, ema_trend_score DESC
            LIMIT :limit
        """)
        
        result = self.db.execute(query, {'strategy': strategy, 'limit': limit})
        
        stocks = []
        for row in result:
            stock = {
                'date': row.date.isoformat() if row.date else None,
                'symbol': row.symbol,
                'stock_name': row.stock_name,
                'current_price': float(row.current_price) if row.current_price else None,
                'market_cap': float(row.market_cap) if row.market_cap else None,
                'strategy': row.strategy,
                'selection_score': float(row.selection_score) if row.selection_score else None,
                'rank': row.rank,

                # 8-21 EMA Strategy Indicators
                'ema_8': float(row.ema_8) if row.ema_8 else None,
                'ema_21': float(row.ema_21) if row.ema_21 else None,
                'ema_trend_score': float(row.ema_trend_score) if row.ema_trend_score else None,
                'demarker': float(row.demarker) if row.demarker else None,

                # Fibonacci Targets
                'fib_target_1': float(row.fib_target_1) if row.fib_target_1 else None,
                'fib_target_2': float(row.fib_target_2) if row.fib_target_2 else None,
                'fib_target_3': float(row.fib_target_3) if row.fib_target_3 else None,

                # Signals
                'buy_signal': row.buy_signal,
                'sell_signal': row.sell_signal,
                'signal_quality': row.signal_quality,

                # Legacy Technical Indicators (optional)
                'rsi_14': float(row.rsi_14) if row.rsi_14 else None,
                'macd': float(row.macd) if row.macd else None,
                'sma_50': float(row.sma_50) if row.sma_50 else None,
                'sma_200': float(row.sma_200) if row.sma_200 else None,

                # Fundamentals
                'pe_ratio': float(row.pe_ratio) if row.pe_ratio else None,
                'pb_ratio': float(row.pb_ratio) if row.pb_ratio else None,
                'roe': float(row.roe) if row.roe else None,
                'eps': float(row.eps) if row.eps else None,
                'beta': float(row.beta) if row.beta else None,
                'revenue_growth': float(row.revenue_growth) if row.revenue_growth else None,
                'earnings_growth': float(row.earnings_growth) if row.earnings_growth else None,
                'operating_margin': float(row.operating_margin) if row.operating_margin else None,

                # Trading Signals
                'target_price': float(row.target_price) if row.target_price else None,
                'stop_loss': float(row.stop_loss) if row.stop_loss else None,
                'recommendation': row.recommendation,
                'reason': row.reason,

                # Metadata
                'sector': row.sector,
                'market_cap_category': row.market_cap_category,
                'created_at': row.created_at.isoformat() if row.created_at else None
            }
            stocks.append(stock)
        
        return stocks
    
    def get_snapshot_dates(self, strategy: str = 'default_risk') -> List[str]:
        """Get list of available snapshot dates for a strategy."""
        query = text("""
            SELECT DISTINCT date 
            FROM daily_suggested_stocks 
            WHERE strategy = :strategy
            ORDER BY date DESC
        """)
        
        result = self.db.execute(query, {'strategy': strategy})
        return [row.date.isoformat() for row in result]
    
    def delete_old_snapshots(self, keep_days: int = 30) -> int:
        """
        Delete snapshots older than keep_days.
        
        Args:
            keep_days: Number of days to keep (default 30)
            
        Returns:
            Number of rows deleted
        """
        query = text("""
            DELETE FROM daily_suggested_stocks
            WHERE date < CURRENT_DATE - INTERVAL ':keep_days days'
        """)
        
        result = self.db.execute(query, {'keep_days': keep_days})
        deleted = result.rowcount
        self.db.commit()
        
        logger.info(f"Deleted {deleted} old snapshot records (older than {keep_days} days)")
        return deleted
