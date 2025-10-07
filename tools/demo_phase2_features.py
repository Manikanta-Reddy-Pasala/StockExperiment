#!/usr/bin/env python3
"""
Phase 2 Features Demo Script
Demonstrates backtesting and AI analysis capabilities
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.ml.strategy_backtester import StrategyBacktester
from src.services.ml.ai_stock_analyst import AIStockAnalyst
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_backtesting():
    """Demonstrate strategy backtesting."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO: Advanced Backtesting")
    logger.info("=" * 80)

    db_manager = get_database_manager()

    try:
        with db_manager.get_session() as session:
            # Initialize backtester
            backtester = StrategyBacktester(
                session,
                initial_capital=1000000,  # 10 lakh
                commission_pct=0.1,
                slippage_pct=0.05
            )

            # Run backtest (last 90 days)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

            logger.info(f"\nBacktesting period: {start_date} to {end_date}")
            logger.info("Strategy: ML-based top 10 stocks, rebalance every 14 days")
            logger.info("")

            results = backtester.backtest_ml_strategy(
                start_date=start_date,
                end_date=end_date,
                rebalance_days=14,
                top_n_stocks=10,
                min_confidence=0.6,
                min_ml_score=0.6,
                max_risk_score=0.4
            )

            if results['success']:
                metrics = results['metrics']
                logger.info("\n📊 Backtest Results:")
                logger.info(f"  Initial Capital: ₹{results['initial_capital']:,.2f}")
                logger.info(f"  Final Value: ₹{results['final_value']:,.2f}")
                logger.info(f"  Total Return: ₹{metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")
                logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
                logger.info(f"  Win Rate: {metrics['win_rate']:.1f}%")
                logger.info(f"  Total Trades: {metrics['total_trades']}")
                logger.info(f"  Winning Trades: {metrics['winning_trades']}")
                logger.info(f"  Avg Trade P&L: ₹{metrics['avg_pnl']:,.2f} ({metrics['avg_pnl_pct']:.2f}%)")

                # Show sample trades
                trades = results['trades']
                if trades:
                    logger.info("\n📈 Sample Trades (First 5):")
                    for i, trade in enumerate(trades[:5], 1):
                        logger.info(f"\n  Trade {i}: {trade['symbol']}")
                        logger.info(f"    Entry: {trade['entry_date']} @ ₹{trade['entry_price']:.2f}")
                        logger.info(f"    Exit:  {trade['exit_date']} @ ₹{trade['exit_price']:.2f}")
                        logger.info(f"    P&L:   ₹{trade['pnl']:,.2f} ({trade['pnl_pct']:+.2f}%)")

                return True
            else:
                logger.error("Backtest failed")
                return False

    except Exception as e:
        logger.error(f"Backtest demo failed: {e}", exc_info=True)
        return False


def demo_ai_analysis():
    """Demonstrate AI-powered stock analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO: AI-Powered Stock Analysis")
    logger.info("=" * 80)

    db_manager = get_database_manager()

    try:
        with db_manager.get_session() as session:
            # Get sample stock
            result = session.execute(text("""
                SELECT
                    s.symbol, s.stock_name, s.current_price, s.market_cap,
                    s.pe_ratio, s.pb_ratio, s.roe, s.eps, s.beta,
                    s.debt_to_equity, s.revenue_growth, s.earnings_growth,
                    s.operating_margin, s.net_margin, s.historical_volatility_1y,
                    ti.rsi_14, ti.macd, ti.sma_50, ti.sma_200,
                    dss.ml_prediction_score, dss.ml_confidence, dss.ml_risk_score,
                    dss.ml_price_target
                FROM stocks s
                LEFT JOIN (
                    SELECT DISTINCT ON (symbol) *
                    FROM technical_indicators
                    ORDER BY symbol, date DESC
                ) ti ON s.symbol = ti.symbol
                LEFT JOIN (
                    SELECT DISTINCT ON (symbol) *
                    FROM daily_suggested_stocks
                    ORDER BY symbol, date DESC
                ) dss ON s.symbol = dss.symbol
                WHERE s.current_price IS NOT NULL
                AND s.market_cap IS NOT NULL
                ORDER BY s.market_cap DESC
                LIMIT 3
            """))

            stocks = [dict(row._mapping) for row in result.fetchall()]

            if not stocks:
                logger.warning("No stocks found for AI analysis")
                return False

            # Initialize AI analyst
            analyst = AIStockAnalyst(llm_provider='ollama', model='llama2')

            logger.info(f"\nGenerating AI reports for {len(stocks)} stocks...")
            logger.info("(Note: Ollama must be running locally for LLM analysis)\n")

            for i, stock in enumerate(stocks, 1):
                logger.info(f"\n{'─' * 80}")
                logger.info(f"Stock {i}/{len(stocks)}: {stock['symbol']} - {stock.get('stock_name', 'N/A')}")
                logger.info(f"{'─' * 80}")

                report = analyst.generate_stock_report(stock)

                logger.info(f"\n📊 Technical Analysis:")
                logger.info(f"   {report['technical_analysis']}")

                logger.info(f"\n📈 Fundamental Analysis:")
                logger.info(f"   {report['fundamental_analysis']}")

                logger.info(f"\n🤖 ML Interpretation:")
                logger.info(f"   {report['ml_interpretation']}")

                logger.info(f"\n⚠️  Risk Assessment:")
                for line in report['risk_assessment'].split('\n'):
                    if line.strip():
                        logger.info(f"   {line}")

                logger.info(f"\n💡 Recommendation:")
                rec = report['recommendation']
                logger.info(f"   Action: {rec['action']}")
                logger.info(f"   Rationale: {rec['rationale']}")
                logger.info(f"   Current Price: ₹{rec['current_price']:,.2f}")
                logger.info(f"   Target Price: ₹{rec['target_price']:,.2f}")
                logger.info(f"   Upside: {rec['upside_potential']:+.2f}%")
                logger.info(f"   Stop Loss: ₹{rec['stop_loss']:,.2f}")
                logger.info(f"   Conviction: {rec['conviction']}")

                logger.info(f"\n   Report Confidence: {report['confidence']:.0%}")

            return True

    except Exception as e:
        logger.error(f"AI analysis demo failed: {e}", exc_info=True)
        return False


def main():
    """Run all Phase 2 demos."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2 FEATURES DEMONSTRATION")
    logger.info("=" * 80)
    logger.info("\nThis demo showcases:")
    logger.info("1. Advanced backtesting with realistic trading simulation")
    logger.info("2. AI-powered stock analysis using LLMs")
    logger.info("")

    # Demo 1: Backtesting
    backtesting_success = demo_backtesting()

    # Demo 2: AI Analysis
    ai_success = demo_ai_analysis()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DEMO SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✓ Backtesting Demo: {'Success' if backtesting_success else 'Failed'}")
    logger.info(f"✓ AI Analysis Demo: {'Success' if ai_success else 'Failed'}")
    logger.info("=" * 80)

    if backtesting_success and ai_success:
        logger.info("\n🎉 All Phase 2 features demonstrated successfully!")
        return 0
    else:
        logger.warning("\n⚠️  Some demos failed. Check logs for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
