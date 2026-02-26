"""
Day Trading Service
Gap-up/momentum day trading stock selection strategy.
"""

import logging
from typing import Dict, List, Any
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)


class DayTradingService:
    """Service for selecting day trading stocks using gap-up/momentum strategy."""

    def select_day_trading_stocks(self, session: Session, user_id: int, max_stocks: int = 5) -> Dict[str, Any]:
        """
        Select stocks for day trading based on gap-up/momentum strategy.

        Strategy:
        1. Filter stocks: active, tradeable, price 100-5000
        2. Fetch opening quotes via Fyers API
        3. Filter for gap-up (1-5%), volume surge (1.5x avg), bullish EMA trend
        4. Score and rank, return top candidates

        Args:
            session: Database session
            user_id: User ID for broker API access
            max_stocks: Maximum number of stocks to return

        Returns:
            Dict with 'stocks' list and metadata
        """
        try:
            # Step 1: Get candidate stocks from database
            candidates_query = text("""
                SELECT symbol, stock_name, current_price, ema_8, ema_21,
                       avg_daily_volume_20d, buy_signal
                FROM stocks
                WHERE is_active = TRUE
                  AND is_tradeable = TRUE
                  AND is_suspended = FALSE
                  AND current_price >= 100
                  AND current_price <= 5000
                  AND avg_daily_volume_20d > 0
                  AND ema_8 IS NOT NULL
                  AND ema_21 IS NOT NULL
                ORDER BY avg_daily_volume_20d DESC
                LIMIT 200
            """)

            result = session.execute(candidates_query)
            candidates = [dict(row._mapping) for row in result]

            if not candidates:
                logger.warning("No candidate stocks found for day trading")
                return {'stocks': [], 'strategies_used': ['day_trading']}

            logger.info(f"Day trading: {len(candidates)} candidates from database")

            # Step 2: Fetch live quotes via broker API
            symbols = [c['symbol'] for c in candidates]
            live_quotes = self._fetch_live_quotes(user_id, symbols)

            # Step 3: Filter and score
            scored_stocks = []
            for candidate in candidates:
                symbol = candidate['symbol']
                prev_close = candidate['current_price']  # Last known close from stocks table
                ema_8 = candidate['ema_8']
                ema_21 = candidate['ema_21']
                avg_volume = candidate['avg_daily_volume_20d']
                buy_signal = candidate['buy_signal']

                quote = live_quotes.get(symbol, {})
                open_price = quote.get('open_price')
                ltp = quote.get('ltp')
                current_volume = quote.get('volume')

                if not open_price or not ltp or not prev_close or prev_close <= 0:
                    continue

                # Gap-up filter: 1-5% gap
                gap_pct = ((open_price - prev_close) / prev_close) * 100
                if gap_pct < 1.0 or gap_pct > 5.0:
                    continue

                # Volume surge filter: current volume > 1.5x average
                volume_ratio = 0
                if current_volume and avg_volume and avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    if volume_ratio < 1.5:
                        continue
                else:
                    continue

                # Bullish EMA trend: ema_8 > ema_21
                bullish = ema_8 > ema_21 if ema_8 and ema_21 else False

                # Score: gap_pct * volume_ratio * trend_multiplier
                trend_multiplier = 1.0 if buy_signal else 0.5
                score = gap_pct * volume_ratio * trend_multiplier

                # Day trading targets
                entry_price = ltp
                target_price = entry_price * 1.02   # 2% intraday target
                stop_loss = entry_price * 0.99       # 1% stop loss

                scored_stocks.append({
                    'symbol': symbol,
                    'stock_name': candidate['stock_name'],
                    'current_price': entry_price,
                    'strategy': 'day_trading',
                    'selection_score': round(score, 4),
                    'gap_pct': round(gap_pct, 2),
                    'volume_ratio': round(volume_ratio, 2),
                    'ema_8': ema_8,
                    'ema_21': ema_21,
                    'ema_trend_score': 1.0 if bullish else 0.5,
                    'demarker': None,
                    'signal_quality': 'high' if bullish else 'medium',
                    'target_price': round(target_price, 2),
                    'stop_loss': round(stop_loss, 2),
                    'recommendation': 'BUY',
                    'date': None,
                    'open_price': open_price,
                    'prev_close': prev_close,
                })

            # Sort by score and take top N
            scored_stocks.sort(key=lambda x: x['selection_score'], reverse=True)
            selected = scored_stocks[:max_stocks]

            logger.info(f"Day trading: {len(scored_stocks)} stocks passed filters, "
                       f"selected top {len(selected)}")

            for stock in selected:
                logger.info(f"  {stock['symbol']}: gap={stock['gap_pct']}%, "
                           f"vol_ratio={stock['volume_ratio']}x, score={stock['selection_score']}")

            return {
                'stocks': selected,
                'strategies_used': ['day_trading']
            }

        except Exception as e:
            logger.error(f"Day trading stock selection failed: {e}", exc_info=True)
            return {'stocks': [], 'strategies_used': ['day_trading']}

    def _fetch_live_quotes(self, user_id: int, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch live quotes from broker API for multiple symbols.

        Returns:
            Dict mapping symbol to {ltp, open_price, volume}
        """
        quotes = {}
        try:
            from src.services.brokers.fyers_service import get_fyers_service
            fyers_service = get_fyers_service()

            # Batch fetch quotes (Fyers supports multiple symbols)
            batch_size = 50
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                try:
                    result = fyers_service.quotes_multiple(user_id, batch)
                    if result.get('status') == 'success' and result.get('data'):
                        data = result['data']
                        if isinstance(data, list):
                            for item in data:
                                v = item.get('v', {})
                                sym = item.get('n', '') or v.get('symbol', '')
                                # Try to match back to original symbol
                                matched_symbol = self._match_symbol(sym, batch)
                                if matched_symbol:
                                    quotes[matched_symbol] = {
                                        'ltp': v.get('lp', 0),
                                        'open_price': v.get('open_price', 0),
                                        'volume': v.get('volume', 0),
                                    }
                except Exception as e:
                    logger.debug(f"Batch quote fetch failed for batch {i}: {e}")

        except Exception as e:
            logger.warning(f"Failed to fetch live quotes: {e}")

        return quotes

    def _match_symbol(self, api_symbol: str, candidates: List[str]) -> str:
        """Match API-returned symbol to candidate symbol."""
        if api_symbol in candidates:
            return api_symbol
        # Try without exchange prefix
        for candidate in candidates:
            if candidate in api_symbol or api_symbol in candidate:
                return candidate
        return ''


def get_day_trading_service() -> DayTradingService:
    """Get day trading service instance (stateless)."""
    return DayTradingService()
