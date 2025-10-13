#!/usr/bin/env python3
"""
Test Ollama Enhancement Service
Tests the Ollama API integration with a sample stock.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.data.strategy_ollama_enhancement_service import get_strategy_ollama_enhancement_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_ollama_enhancement():
    """Test Ollama enhancement with a sample stock."""
    logger.info("=" * 80)
    logger.info("TESTING OLLAMA ENHANCEMENT SERVICE")
    logger.info("=" * 80)

    try:
        # Sample stock recommendation
        sample_recommendation = {
            'symbol': 'NSE:RELIANCE-EQ',
            'name': 'RELIANCE INDUSTRIES LIMITED',
            'current_price': 2850.50,
            'recommended_quantity': 35,
            'investment_amount': 99767.50,
            'selection_score': 85.0,
            'ml_prediction': {
                'predicted_change_percent': 12.5,
                'signal': 'BUY',
                'confidence': 0.82
            }
        }

        logger.info("\n📊 Sample Stock:")
        logger.info(f"  Symbol: {sample_recommendation['symbol']}")
        logger.info(f"  Name: {sample_recommendation['name']}")
        logger.info(f"  Price: ₹{sample_recommendation['current_price']}")
        logger.info(f"  ML Signal: {sample_recommendation['ml_prediction']['signal']}")
        logger.info(f"  ML Confidence: {sample_recommendation['ml_prediction']['confidence']:.2%}")

        # Get Ollama service
        logger.info("\n🔍 Initializing Ollama service...")
        ollama_service = get_strategy_ollama_enhancement_service()

        # Test with "light" enhancement for faster results
        logger.info("\n🚀 Calling Ollama API (light enhancement)...")
        enhanced = ollama_service.enhance_strategy_recommendations(
            [sample_recommendation],
            strategy_type='default_risk',
            enhancement_level='light'
        )

        if enhanced and len(enhanced) > 0:
            enhanced_stock = enhanced[0]

            logger.info("\n✅ OLLAMA ENHANCEMENT SUCCESSFUL!")
            logger.info("=" * 80)

            if 'ollama_enhancement' in enhanced_stock:
                enhancement = enhanced_stock['ollama_enhancement']

                logger.info("\n📈 Enhancement Results:")
                logger.info(f"  Enhancement Score: {enhancement.get('enhancement_score', 0):.2%}")
                logger.info(f"  Strategy Confidence: {enhancement.get('strategy_confidence', 'N/A').upper()}")
                logger.info(f"  Timestamp: {enhancement.get('timestamp', 'N/A')}")

                market_intel = enhancement.get('market_intelligence', {})

                # Show sources
                sources = market_intel.get('sources', [])
                logger.info(f"\n📰 Sources Found: {len(sources)}")
                for i, source in enumerate(sources[:3], 1):
                    logger.info(f"  {i}. {source.get('title', 'N/A')}")

                # Show insights
                insights = market_intel.get('insights', [])
                logger.info(f"\n💡 Key Insights: {len(insights)}")
                for i, insight in enumerate(insights[:3], 1):
                    logger.info(f"  {i}. {insight[:100]}...")

                # Show sentiment
                sentiment_score = market_intel.get('sentiment_score', 0)
                sentiment_label = "BULLISH" if sentiment_score > 0.3 else "BEARISH" if sentiment_score < -0.3 else "NEUTRAL"
                logger.info(f"\n😊 Sentiment Analysis:")
                logger.info(f"  Score: {sentiment_score:.2f} ({sentiment_label})")

                # Show financial metrics
                financial_metrics = market_intel.get('financial_metrics', [])
                if financial_metrics:
                    logger.info(f"\n💰 Financial Metrics Found: {len(financial_metrics)}")
                    for metric in financial_metrics[:5]:
                        logger.info(f"  - {metric}")

            logger.info("\n" + "=" * 80)
            logger.info("✅ TEST PASSED: Ollama is working correctly!")
            logger.info("=" * 80)
            return True

        else:
            logger.warning("\n⚠️  No enhancement data returned")
            return False

    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = test_ollama_enhancement()
    sys.exit(0 if success else 1)
