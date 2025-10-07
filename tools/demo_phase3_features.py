#!/usr/bin/env python3
"""
Phase 3 Features Demo Script
Demonstrates market regime detection, portfolio optimization, sentiment analysis
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.ml.market_regime_detector import MarketRegimeDetector
from src.services.ml.portfolio_optimizer import PortfolioOptimizer
from src.services.ml.sentiment_analyzer import SentimentAnalyzer
from src.services.ml.realtime_stream_processor import RealtimeStreamProcessor, StreamSimulator
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_regime_detection():
    """Demonstrate market regime detection."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 1: Market Regime Detection")
    logger.info("=" * 80)

    db_manager = get_database_manager()

    try:
        with db_manager.get_session() as session:
            detector = MarketRegimeDetector(session)

            # Detect current regime
            result = detector.detect_regime(lookback_days=90)

            logger.info("\n📊 Market Regime Analysis:")
            logger.info(f"  Current Regime: {result['regime']}")
            logger.info(f"  Confidence: {result['confidence']:.0%}")
            logger.info(f"  Description: {result['characteristics'].get('description', 'N/A')}")

            logger.info("\n📈 Regime Characteristics:")
            chars = result['characteristics']
            logger.info(f"  Average Return: {chars.get('avg_return', 0):.2%}")
            logger.info(f"  Volatility: {chars.get('volatility', 0):.2%}")
            logger.info(f"  Trend Strength: {chars.get('trend_strength', 0):.1f}")
            logger.info(f"  Momentum (20d): {chars.get('momentum_20d', 0):.2%}")

            logger.info("\n🔍 Detection Methods:")
            methods = result['methods']
            logger.info(f"  GMM Method: {methods.get('gmm', 'N/A')}")
            logger.info(f"  Technical: {methods.get('technical', 'N/A')}")
            logger.info(f"  Volatility: {methods.get('volatility', 'N/A')}")
            logger.info(f"  Trend: {methods.get('trend', 'N/A')}")

            # Get strategy recommendations
            strategy = detector.get_regime_specific_strategy(result['regime'])

            logger.info("\n💡 Regime-Specific Strategy:")
            logger.info(f"  Position Size: {strategy['position_size']:.0%}")
            logger.info(f"  Stop Loss: {strategy['stop_loss_pct']:.1f}%")
            logger.info(f"  Take Profit: {strategy['take_profit_pct']:.1f}%")
            logger.info(f"  Preferred Strategies: {', '.join(strategy['preferred_strategies'])}")
            logger.info(f"  Risk Tolerance: {strategy['risk_tolerance'].upper()}")
            logger.info(f"  Rebalance Frequency: {strategy['rebalance_frequency']} days")
            logger.info(f"  ML Score Threshold: {strategy['ml_score_threshold']:.2f}")

            return True

    except Exception as e:
        logger.error(f"Regime detection demo failed: {e}", exc_info=True)
        return False


def demo_portfolio_optimization():
    """Demonstrate portfolio optimization."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 2: Modern Portfolio Theory Optimization")
    logger.info("=" * 80)

    db_manager = get_database_manager()

    try:
        with db_manager.get_session() as session:
            # Get top stocks from daily suggestions
            result = session.execute(text("""
                SELECT DISTINCT ON (symbol)
                    symbol, current_price, ml_prediction_score,
                    ml_confidence, ml_risk_score, predicted_change_pct
                FROM daily_suggested_stocks
                WHERE ml_prediction_score >= 0.6
                ORDER BY symbol, date DESC
                LIMIT 15
            """))

            stocks = [dict(row._mapping) for row in result.fetchall()]

            if not stocks:
                logger.warning("No stocks available for optimization")
                return False

            logger.info(f"\nOptimizing portfolio with {len(stocks)} stocks...")

            optimizer = PortfolioOptimizer(session, risk_free_rate=0.06)

            # Try different optimization methods
            methods = ['max_sharpe', 'min_variance', 'risk_parity', 'ml_enhanced']

            for method in methods:
                logger.info(f"\n{'─' * 80}")
                logger.info(f"Method: {method.upper().replace('_', ' ')}")
                logger.info(f"{'─' * 80}")

                portfolio = optimizer.optimize_portfolio(
                    stocks,
                    method=method,
                    max_weight=0.20,
                    min_weight=0.01
                )

                metrics = portfolio['metrics']

                logger.info(f"\n📊 Portfolio Metrics:")
                logger.info(f"  Expected Return: {metrics['expected_return']:.2%} (annualized)")
                logger.info(f"  Volatility: {metrics['volatility']:.2%} (annualized)")
                logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                logger.info(f"  Number of Stocks: {metrics['num_stocks']}")
                logger.info(f"  Max Weight: {metrics['max_weight']:.1%}")

                logger.info(f"\n💼 Top 5 Allocations:")
                for i, alloc in enumerate(portfolio['allocations'][:5], 1):
                    logger.info(f"  {i}. {alloc['symbol']:10} - Weight: {alloc['weight']:.1%}, "
                               f"Expected Return: {alloc['expected_return']:.1%}, "
                               f"ML Score: {alloc['ml_score']:.2f}")

            # Generate efficient frontier
            logger.info(f"\n{'─' * 80}")
            logger.info("Generating Efficient Frontier...")
            logger.info(f"{'─' * 80}")

            frontier = optimizer.generate_efficient_frontier(stocks, num_points=20)

            if not frontier.empty:
                logger.info(f"\n📈 Efficient Frontier (sample points):")
                logger.info(f"{'Return':<10} {'Volatility':<12} {'Sharpe':<10}")
                logger.info("─" * 32)
                for _, row in frontier.head(5).iterrows():
                    logger.info(f"{row['return']:>8.2%}  {row['volatility']:>10.2%}  {row['sharpe']:>8.2f}")

            return True

    except Exception as e:
        logger.error(f"Portfolio optimization demo failed: {e}", exc_info=True)
        return False


def demo_sentiment_analysis():
    """Demonstrate sentiment analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 3: Advanced Sentiment Analysis")
    logger.info("=" * 80)

    db_manager = get_database_manager()

    try:
        with db_manager.get_session() as session:
            # Get sample stocks
            result = session.execute(text("""
                SELECT DISTINCT symbol
                FROM stocks
                WHERE current_price IS NOT NULL
                ORDER BY market_cap DESC
                LIMIT 5
            """))

            symbols = [row[0] for row in result.fetchall()]

            analyzer = SentimentAnalyzer(llm_provider='ollama')

            logger.info(f"\nAnalyzing sentiment for {len(symbols)} stocks...")
            logger.info("(Note: Using simulated news data for demo)\n")

            for i, symbol in enumerate(symbols, 1):
                logger.info(f"{'─' * 80}")
                logger.info(f"Stock {i}/{len(symbols)}: {symbol}")
                logger.info(f"{'─' * 80}")

                sentiment = analyzer.analyze_stock_sentiment(symbol, sources=['news'])

                logger.info(f"\n📰 Overall Sentiment:")
                logger.info(f"  Sentiment: {sentiment['overall_sentiment']}")
                logger.info(f"  Score: {sentiment['overall_score']:.3f} (-1 to +1)")
                logger.info(f"  Confidence: {sentiment['confidence']:.0%}")

                if 'news' in sentiment['sources']:
                    news = sentiment['sources']['news']
                    logger.info(f"\n📄 News Analysis:")
                    logger.info(f"  Headlines Analyzed: {news.get('count', 0)}")
                    logger.info(f"  Positive Headlines: {news.get('positive_count', 0)}")
                    logger.info(f"  Negative Headlines: {news.get('negative_count', 0)}")

                    if 'latest_headlines' in news:
                        logger.info(f"\n📋 Latest Headlines:")
                        for j, headline in enumerate(news['latest_headlines'][:3], 1):
                            score_emoji = "🟢" if headline['score'] > 0.2 else "🔴" if headline['score'] < -0.2 else "⚪"
                            logger.info(f"  {j}. {score_emoji} {headline['text']}")
                            logger.info(f"     Score: {headline['score']:+.2f} | Source: {headline['source']}")

            # Market sentiment
            logger.info(f"\n{'=' * 80}")
            logger.info("Market-Wide Sentiment")
            logger.info(f"{'=' * 80}")

            market_sentiment = analyzer.get_market_sentiment(symbols)

            logger.info(f"\n🌐 Overall Market:")
            logger.info(f"  Sentiment: {market_sentiment['market_sentiment']}")
            logger.info(f"  Score: {market_sentiment['market_score']:.3f}")
            logger.info(f"  Stocks Analyzed: {market_sentiment['total_analyzed']}")

            logger.info(f"\n📊 Distribution:")
            dist = market_sentiment['sentiment_distribution']
            logger.info(f"  Very Positive: {dist['very_positive']}")
            logger.info(f"  Positive: {dist['positive']}")
            logger.info(f"  Neutral: {dist['neutral']}")
            logger.info(f"  Negative: {dist['negative']}")
            logger.info(f"  Very Negative: {dist['very_negative']}")

            return True

    except Exception as e:
        logger.error(f"Sentiment analysis demo failed: {e}", exc_info=True)
        return False


def demo_realtime_streaming():
    """Demonstrate real-time streaming."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 4: Real-Time Stream Processing")
    logger.info("=" * 80)

    db_manager = get_database_manager()

    try:
        with db_manager.get_session() as session:
            # Create processor
            processor = RealtimeStreamProcessor(session, window_size=50)

            # Register callbacks
            alert_count = [0]

            def on_alert(alert):
                alert_count[0] += 1
                logger.info(f"\n🚨 ALERT #{alert_count[0]}: {alert['type']} - {alert['symbol']}")
                logger.info(f"   {alert['message']}")

            processor.register_callback('alert', on_alert)

            # Start processor
            processor.start()

            # Simulate streaming
            logger.info("\nStarting stream simulation (5x speed)...")
            logger.info("Processing 30 days of historical data as real-time stream...")

            simulator = StreamSimulator(session, processor)

            # Get symbols
            result = session.execute(text("""
                SELECT DISTINCT symbol FROM stocks
                WHERE current_price IS NOT NULL
                ORDER BY market_cap DESC
                LIMIT 3
            """))
            symbols = [row[0] for row in result.fetchall()]

            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

            # Run simulation
            import threading
            sim_thread = threading.Thread(
                target=simulator.start_simulation,
                args=(symbols, start_date, 5.0)
            )
            sim_thread.start()

            # Wait a bit
            import time
            time.sleep(10)

            # Stop
            simulator.stop_simulation()
            sim_thread.join(timeout=2)
            processor.stop()

            # Get statistics
            stats = processor.get_statistics()

            logger.info(f"\n📊 Streaming Statistics:")
            logger.info(f"  Messages Processed: {stats['messages_processed']}")
            logger.info(f"  Predictions Made: {stats['predictions_made']}")
            logger.info(f"  Alerts Generated: {stats['alerts_generated']}")
            logger.info(f"  Symbols Tracked: {stats['symbols_tracked']}")
            logger.info(f"  Runtime: {stats.get('runtime_seconds', 0):.1f} seconds")
            logger.info(f"  Throughput: {stats.get('messages_per_second', 0):.1f} msg/sec")

            return True

    except Exception as e:
        logger.error(f"Real-time streaming demo failed: {e}", exc_info=True)
        return False


def main():
    """Run all Phase 3 demos."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: ADVANCED TRADING INTELLIGENCE DEMONSTRATION")
    logger.info("=" * 80)
    logger.info("\nThis demo showcases:")
    logger.info("1. Market Regime Detection (Bull/Bear/Sideways)")
    logger.info("2. Modern Portfolio Theory Optimization")
    logger.info("3. Advanced Sentiment Analysis")
    logger.info("4. Real-Time Stream Processing")
    logger.info("")

    results = {}

    # Demo 1: Regime Detection
    results['regime'] = demo_regime_detection()

    # Demo 2: Portfolio Optimization
    results['portfolio'] = demo_portfolio_optimization()

    # Demo 3: Sentiment Analysis
    results['sentiment'] = demo_sentiment_analysis()

    # Demo 4: Real-Time Streaming
    results['streaming'] = demo_realtime_streaming()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DEMO SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✓ Regime Detection: {'Success' if results['regime'] else 'Failed'}")
    logger.info(f"✓ Portfolio Optimization: {'Success' if results['portfolio'] else 'Failed'}")
    logger.info(f"✓ Sentiment Analysis: {'Success' if results['sentiment'] else 'Failed'}")
    logger.info(f"✓ Real-Time Streaming: {'Success' if results['streaming'] else 'Failed'}")
    logger.info("=" * 80)

    if all(results.values()):
        logger.info("\n🎉 All Phase 3 features demonstrated successfully!")
        return 0
    else:
        logger.warning("\n⚠️  Some demos failed. Check logs for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
