"""
ML Stock Screening API Routes

This module provides API endpoints for intelligent stock screening
and portfolio recommendations using ML models.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import logging

try:
    from ...services.ml.intelligent_screening_service import (
        get_intelligent_screening_service, RiskProfile
    )
    from ...services.ml.stock_discovery_service import get_stock_discovery_service
except ImportError:
    from services.ml.intelligent_screening_service import (
        get_intelligent_screening_service, RiskProfile
    )
    from services.ml.stock_discovery_service import get_stock_discovery_service

logger = logging.getLogger(__name__)

screening_bp = Blueprint('screening', __name__, url_prefix='/api/v1/screening')


def serialize_stock_analysis(analysis):
    """Serialize MLStockAnalysis object to dictionary."""
    return {
        'symbol': analysis.stock_info.symbol,
        'name': analysis.stock_info.name,
        'market_cap_category': analysis.stock_info.market_cap_category.value,
        'current_price': round(analysis.stock_info.current_price, 2),
        'market_cap_crores': round(analysis.stock_info.market_cap_crores, 2),
        'volume': analysis.stock_info.volume,
        'sector': analysis.stock_info.sector,
        'ml_prediction': round(analysis.ml_prediction, 2),
        'prediction_confidence': round(analysis.prediction_confidence, 3),
        'technical_score': round(analysis.technical_score, 3),
        'risk_score': round(analysis.risk_score, 3),
        'overall_score': round(analysis.overall_score, 3),
        'recommendation': analysis.recommendation,
        'target_price': round(analysis.target_price, 2),
        'stop_loss': round(analysis.stop_loss, 2),
        'expected_return': round(analysis.expected_return * 100, 2),  # As percentage
        'model_status': analysis.model_status,
        'reasons': analysis.reasons
    }


def serialize_stock_info(stock_info):
    """Serialize StockInfo object to dictionary."""
    return {
        'symbol': stock_info.symbol,
        'name': stock_info.name,
        'exchange': stock_info.exchange,
        'market_cap_category': stock_info.market_cap_category.value,
        'current_price': round(stock_info.current_price, 2),
        'market_cap_crores': round(stock_info.market_cap_crores, 2),
        'volume': stock_info.volume,
        'liquidity_score': round(stock_info.liquidity_score, 3),
        'is_tradeable': stock_info.is_tradeable,
        'sector': stock_info.sector
    }


@screening_bp.route('/discover', methods=['GET'])
def discover_stocks():
    """Discover available stocks from broker API."""
    try:
        logger.info("API: Discovering stocks from broker")

        user_id = request.args.get('user_id', 1, type=int)
        force_refresh = request.args.get('refresh', False, type=bool)

        discovery_service = get_stock_discovery_service()

        if force_refresh:
            stocks = discovery_service.refresh_cache(user_id)
        else:
            stocks = discovery_service.discover_tradeable_stocks(user_id)

        # Log discovery statistics to broker response file
        total_discovered = len(stocks)
        logger.info(f"Broker API discovered {total_discovered} stocks from real Fyers API")

        # Categorize stocks
        categorized = {
            'large_cap': [],
            'mid_cap': [],
            'small_cap': []
        }

        for stock in stocks:
            if stock.is_tradeable:
                category = stock.market_cap_category.value
                categorized[category].append(serialize_stock_info(stock))

        # Summary statistics with filtering counts
        tradeable_count = len([s for s in stocks if s.is_tradeable])
        filtered_count = len(categorized['large_cap']) + len(categorized['mid_cap']) + len(categorized['small_cap'])

        summary = {
            'total_stocks': len(stocks),
            'tradeable_stocks': tradeable_count,
            'filtered_stocks': filtered_count,
            'large_cap_count': len(categorized['large_cap']),
            'mid_cap_count': len(categorized['mid_cap']),
            'small_cap_count': len(categorized['small_cap']),
            'discovery_time': datetime.now().isoformat(),
            'data_source': 'Real Fyers Broker API'
        }

        # Log filtering statistics
        logger.info(f"Filtering Stats: Discovered {len(stocks)} → Tradeable {tradeable_count} → Categorized {filtered_count}")

        return jsonify({
            'success': True,
            'summary': summary,
            'stocks': categorized
        })

    except Exception as e:
        logger.error(f"Error in discover_stocks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@screening_bp.route('/screen', methods=['POST'])
def screen_stocks():
    """Screen stocks based on risk profile using ML models."""
    try:
        data = request.get_json()
        risk_profile_str = data.get('risk_profile', 'default')
        user_id = data.get('user_id', 1)
        auto_train = data.get('auto_train', False)

        logger.info(f"API: Screening stocks for {risk_profile_str} risk profile")

        # Parse risk profile
        if risk_profile_str.lower() == 'high_risk':
            risk_profile = RiskProfile.HIGH_RISK
        else:
            risk_profile = RiskProfile.DEFAULT

        # First get stocks from discovery service for statistics
        discovery_service = get_stock_discovery_service()
        discovered_stocks = discovery_service.discover_tradeable_stocks(user_id)

        # Count initial filtering
        total_discovered = len(discovered_stocks)
        tradeable_stocks = [s for s in discovered_stocks if s.is_tradeable]
        tradeable_count = len(tradeable_stocks)

        logger.info(f"ML Screening: Discovered {total_discovered} stocks from broker → {tradeable_count} tradeable")

        # Apply ML screening
        screening_service = get_intelligent_screening_service()
        portfolio = screening_service.screen_stocks_by_risk_profile(
            risk_profile, user_id, auto_train
        )

        # Count ML filtered stocks
        ml_filtered_count = portfolio.total_stocks
        logger.info(f"ML Screening: Tradeable {tradeable_count} → ML analyzed {ml_filtered_count} stocks")

        # Serialize portfolio recommendation with filtering statistics
        result = {
            'risk_profile': portfolio.risk_profile.value,
            'filtering_statistics': {
                'total_discovered': total_discovered,
                'tradeable_stocks': tradeable_count,
                'ml_analyzed': ml_filtered_count,
                'data_source': 'Real Fyers Broker API',
                'filtering_stages': f"Discovered {total_discovered} → Tradeable {tradeable_count} → ML analyzed {ml_filtered_count}"
            },
            'portfolio_metrics': {
                'total_stocks': portfolio.total_stocks,
                'expected_return': round(portfolio.expected_return * 100, 2),  # As percentage
                'portfolio_risk': round(portfolio.portfolio_risk, 3),
                'diversification_score': round(portfolio.diversification_score, 3)
            },
            'allocation_summary': portfolio.allocation_summary,
            'stocks': {
                'large_cap': [serialize_stock_analysis(s) for s in portfolio.large_cap_stocks],
                'mid_cap': [serialize_stock_analysis(s) for s in portfolio.mid_cap_stocks],
                'small_cap': [serialize_stock_analysis(s) for s in portfolio.small_cap_stocks]
            }
        }

        return jsonify({
            'success': True,
            'portfolio': result
        })

    except Exception as e:
        logger.error(f"Error in screen_stocks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@screening_bp.route('/top-picks', methods=['POST'])
def get_top_picks():
    """Get top ML stock picks for immediate trading."""
    try:
        data = request.get_json()
        risk_profile_str = data.get('risk_profile', 'default')
        count = data.get('count', 10)
        user_id = data.get('user_id', 1)

        logger.info(f"API: Getting top {count} picks for {risk_profile_str}")

        # Parse risk profile
        if risk_profile_str.lower() == 'high_risk':
            risk_profile = RiskProfile.HIGH_RISK
        else:
            risk_profile = RiskProfile.DEFAULT

        screening_service = get_intelligent_screening_service()
        top_picks = screening_service.get_top_ml_picks(risk_profile, count, user_id)

        # Serialize top picks
        picks_data = [serialize_stock_analysis(pick) for pick in top_picks]

        # Calculate summary stats
        buy_count = len([p for p in top_picks if p.recommendation == "BUY"])
        avg_expected_return = sum(p.expected_return for p in top_picks) / len(top_picks) if top_picks else 0
        avg_confidence = sum(p.prediction_confidence for p in top_picks) / len(top_picks) if top_picks else 0

        summary = {
            'total_picks': len(top_picks),
            'buy_recommendations': buy_count,
            'avg_expected_return': round(avg_expected_return * 100, 2),
            'avg_ml_confidence': round(avg_confidence, 3),
            'risk_profile': risk_profile.value
        }

        return jsonify({
            'success': True,
            'summary': summary,
            'top_picks': picks_data
        })

    except Exception as e:
        logger.error(f"Error in get_top_picks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@screening_bp.route('/analyze-stock', methods=['POST'])
def analyze_single_stock():
    """Analyze a single stock with ML models."""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        user_id = data.get('user_id', 1)
        auto_train = data.get('auto_train', False)

        if not symbol:
            return jsonify({
                'success': False,
                'error': 'Symbol is required'
            }), 400

        logger.info(f"API: Analyzing single stock {symbol}")

        # Get stock info from discovery service
        discovery_service = get_stock_discovery_service()
        all_stocks = discovery_service.discover_tradeable_stocks(user_id)

        stock_info = None
        for stock in all_stocks:
            if stock.symbol == symbol:
                stock_info = stock
                break

        if not stock_info:
            return jsonify({
                'success': False,
                'error': f'Stock {symbol} not found in tradeable universe'
            }), 404

        # Analyze with ML
        screening_service = get_intelligent_screening_service()
        analysis = screening_service.analyze_stock_with_ml(stock_info, user_id, auto_train)

        if not analysis:
            return jsonify({
                'success': False,
                'error': 'Failed to analyze stock'
            }), 500

        result = serialize_stock_analysis(analysis)

        return jsonify({
            'success': True,
            'analysis': result
        })

    except Exception as e:
        logger.error(f"Error in analyze_single_stock: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@screening_bp.route('/risk-profiles', methods=['GET'])
def get_risk_profiles():
    """Get available risk profiles and their configurations."""
    try:
        profiles = {
            'default': {
                'name': 'Balanced Portfolio',
                'description': 'Conservative approach focusing on stability and steady growth',
                'allocation': {
                    'large_cap': 60,
                    'mid_cap': 30,
                    'small_cap': 10
                },
                'characteristics': [
                    'Lower risk, moderate returns',
                    'High-quality large-cap stocks',
                    'Focus on dividend-paying companies',
                    'Suitable for conservative investors'
                ]
            },
            'high_risk': {
                'name': 'Aggressive Growth Portfolio',
                'description': 'High-risk, high-reward strategy for aggressive growth',
                'allocation': {
                    'large_cap': 0,
                    'mid_cap': 50,
                    'small_cap': 50
                },
                'characteristics': [
                    'Higher risk, higher potential returns',
                    'Focus on growth stocks',
                    'Mid and small-cap opportunities',
                    'Suitable for risk-tolerant investors'
                ]
            }
        }

        return jsonify({
            'success': True,
            'risk_profiles': profiles
        })

    except Exception as e:
        logger.error(f"Error in get_risk_profiles: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@screening_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'service': 'ml_screening',
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


# Test endpoint removed - no hardcoded data allowed
# Use the production /screen endpoint instead