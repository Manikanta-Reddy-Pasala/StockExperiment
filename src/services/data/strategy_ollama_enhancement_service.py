"""
Strategy Ollama Enhancement Service
Integrates Ollama web search API to enhance final strategy stock selections with real-time market intelligence.
This service is specifically designed for strategy execution, not just suggested stocks.
"""

import logging
import requests
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class StrategyOllamaEnhancementService:
    """Service to enhance final strategy stock selections using Ollama web search API."""
    
    def __init__(self, config=None):
        """
        Initialize Strategy Ollama enhancement service.
        
        Args:
            config: OllamaConfig instance. If None, loads default configuration.
        """
        if config is None:
            from src.config.ollama_config import get_ollama_config
            config = get_ollama_config()
        
        self.config = config
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.rate_limit_delay = config.rate_limit_delay
        self.max_retries = config.max_retries
        self.timeout_seconds = config.timeout_seconds
        
    def enhance_strategy_recommendations(self, recommendations: List[Dict[str, Any]], 
                                       strategy_type: str,
                                       enhancement_level: str = "comprehensive") -> List[Dict[str, Any]]:
        """
        Enhance final strategy recommendations with real-time market intelligence.
        
        Args:
            recommendations: List of final strategy stock recommendations
            strategy_type: Type of strategy (e.g., "default_risk", "high_risk")
            enhancement_level: Level of enhancement ("light", "moderate", "comprehensive")
            
        Returns:
            List of enhanced strategy recommendations with market intelligence
        """
        if not recommendations:
            return []
        
        # Check if enhancement level is enabled
        if not self.config.is_enhancement_level_enabled(enhancement_level):
            logger.warning(f"Enhancement level '{enhancement_level}' is disabled, using original recommendations")
            return recommendations
        
        logger.info(f"🔍 Starting Ollama enhancement for {len(recommendations)} strategy recommendations")
        logger.info(f"📊 Strategy: {strategy_type}, Enhancement Level: {enhancement_level}")
        
        enhanced_recommendations = []
        
        for i, recommendation in enumerate(recommendations):
            try:
                # Rate limiting
                if i > 0:
                    time.sleep(self.rate_limit_delay)
                
                # Enhance individual recommendation
                enhanced_recommendation = self._enhance_single_recommendation(
                    recommendation, strategy_type, enhancement_level
                )
                enhanced_recommendations.append(enhanced_recommendation)
                
                logger.info(f"✅ Enhanced {recommendation.get('symbol', 'Unknown')} ({i+1}/{len(recommendations)})")
                
            except Exception as e:
                logger.warning(f"Failed to enhance {recommendation.get('symbol', 'Unknown')}: {e}")
                # Return original recommendation if enhancement fails
                enhanced_recommendations.append(recommendation)
        
        logger.info(f"🎯 Strategy Ollama enhancement completed for {len(enhanced_recommendations)} recommendations")
        return enhanced_recommendations
    
    def _enhance_single_recommendation(self, recommendation: Dict[str, Any], 
                                     strategy_type: str, enhancement_level: str) -> Dict[str, Any]:
        """Enhance a single strategy recommendation with market intelligence."""
        try:
            symbol = recommendation.get('symbol', '')
            name = recommendation.get('name', '')
            current_price = recommendation.get('current_price', 0.0)
            recommended_quantity = recommendation.get('recommended_quantity', 0)
            
            # Create strategy-specific search queries
            search_queries = self._generate_strategy_queries(
                symbol, name, strategy_type, enhancement_level
            )
            
            # Fetch market intelligence
            market_intelligence = {}
            for query_type, query in search_queries.items():
                try:
                    intelligence_data = self._search_market_intelligence(query)
                    if intelligence_data:
                        market_intelligence[query_type] = intelligence_data
                except Exception as e:
                    logger.warning(f"Failed to fetch {query_type} intelligence: {e}")
                    continue
            
            # Calculate strategy-specific enhancement score
            enhancement_score = self._calculate_strategy_enhancement_score(
                market_intelligence, strategy_type, current_price, recommended_quantity
            )
            
            # Enhance recommendation with market intelligence
            enhanced_recommendation = recommendation.copy()
            enhanced_recommendation.update({
                'ollama_enhancement': {
                    'timestamp': datetime.now().isoformat(),
                    'strategy_type': strategy_type,
                    'enhancement_level': enhancement_level,
                    'market_intelligence': market_intelligence,
                    'enhancement_score': enhancement_score,
                    'strategy_confidence': self._calculate_strategy_confidence(enhancement_score, strategy_type)
                }
            })
            
            return enhanced_recommendation
            
        except Exception as e:
            logger.error(f"Error enhancing strategy recommendation {recommendation.get('symbol', 'Unknown')}: {e}")
            return recommendation
    
    def _generate_strategy_queries(self, symbol: str, name: str, strategy_type: str, 
                                 enhancement_level: str) -> Dict[str, str]:
        """Generate strategy-specific search queries."""
        queries = {}
        
        # Get enhancement level configuration
        level_config = self.config.get_enhancement_level_config(enhancement_level)
        query_types = level_config.get('queries', [])
        
        # Clean symbol for search
        clean_symbol = symbol.replace("NSE:", "").replace("-EQ", "")
        
        # Strategy-specific query modifications
        strategy_modifiers = {
            "default_risk": "conservative investment",
            "high_risk": "aggressive growth",
            "momentum": "momentum trading",
            "value": "value investment"
        }
        
        strategy_modifier = strategy_modifiers.get(strategy_type, "investment")
        
        # Generate queries using templates from configuration
        for query_type in query_types:
            template = self.config.get_query_template(query_type)
            if template:
                # Replace placeholders and add strategy context
                query = template.format(
                    company_name=name, 
                    symbol=clean_symbol
                )
                
                # Add strategy-specific context
                if strategy_modifier:
                    query += f" {strategy_modifier} strategy"
                
                queries[query_type] = query
        
        return queries
    
    def _search_market_intelligence(self, query: str) -> Optional[Dict[str, Any]]:
        """Search for market intelligence using Ollama API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {"query": query}
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=self.timeout_seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_ollama_response(result)
            else:
                logger.warning(f"Ollama API returned status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error searching Ollama API: {e}")
            return None
    
    def _parse_ollama_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Ollama API response to extract relevant information."""
        try:
            # Extract key information from response
            sources = response.get('sources', [])
            insights = []
            financial_metrics = []
            sentiment_score = 0.0
            
            for source in sources:
                # Extract insights
                if 'content' in source:
                    content = source['content']
                    insights.append(content[:200] + "..." if len(content) > 200 else content)
                
                # Extract financial metrics (percentages, currencies)
                if 'content' in source:
                    content = source['content']
                    # Look for percentage patterns
                    import re
                    percentages = re.findall(r'(\d+\.?\d*%)', content)
                    financial_metrics.extend(percentages[:3])  # Limit to 3 metrics
                    
                    # Simple sentiment analysis
                    positive_words = ['growth', 'profit', 'increase', 'strong', 'positive', 'bullish']
                    negative_words = ['decline', 'loss', 'decrease', 'weak', 'negative', 'bearish']
                    
                    positive_count = sum(1 for word in positive_words if word.lower() in content.lower())
                    negative_count = sum(1 for word in negative_words if word.lower() in content.lower())
                    
                    if positive_count + negative_count > 0:
                        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
            
            return {
                'sources': sources,
                'insights': insights[:5],  # Limit to 5 insights
                'financial_metrics': financial_metrics[:5],  # Limit to 5 metrics
                'sentiment_score': sentiment_score,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error parsing Ollama response: {e}")
            return {'error': str(e)}
    
    def _calculate_strategy_enhancement_score(self, market_intelligence: Dict[str, Any], 
                                            strategy_type: str, current_price: float, 
                                            recommended_quantity: int) -> float:
        """Calculate strategy-specific enhancement score."""
        try:
            base_score = 0.3
            
            # Score based on sources
            sources = market_intelligence.get('sources', [])
            sources_score = min(len(sources) * 0.1, 0.3)
            
            # Score based on insights
            insights = market_intelligence.get('insights', [])
            insights_score = min(len(insights) * 0.05, 0.2)
            
            # Score based on sentiment
            sentiment_score = market_intelligence.get('sentiment_score', 0.0)
            sentiment_contribution = abs(sentiment_score) * 0.2
            
            # Score based on financial metrics
            financial_metrics = market_intelligence.get('financial_metrics', [])
            financial_score = min(len(financial_metrics) * 0.02, 0.1)
            
            # Strategy-specific scoring
            strategy_multiplier = {
                "default_risk": 1.0,
                "high_risk": 1.2,
                "momentum": 1.1,
                "value": 0.9
            }.get(strategy_type, 1.0)
            
            total_score = (base_score + sources_score + insights_score + 
                          sentiment_contribution + financial_score) * strategy_multiplier
            
            return min(total_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating strategy enhancement score: {e}")
            return 0.3
    
    def _calculate_strategy_confidence(self, enhancement_score: float, strategy_type: str) -> str:
        """Calculate strategy confidence based on enhancement score."""
        if enhancement_score >= 0.8:
            return "high"
        elif enhancement_score >= 0.6:
            return "medium"
        elif enhancement_score >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def enhance_portfolio_recommendations(self, portfolio_recommendations: Dict[str, Any], 
                                       strategy_type: str) -> Dict[str, Any]:
        """Enhance portfolio-level recommendations with market intelligence."""
        try:
            logger.info(f"🔍 Enhancing portfolio recommendations for {strategy_type} strategy")
            
            # Extract individual recommendations
            recommendations = portfolio_recommendations.get('recommendations', [])
            if not recommendations:
                return portfolio_recommendations
            
            # Enhance individual recommendations
            enhanced_recommendations = self.enhance_strategy_recommendations(
                recommendations, strategy_type, "comprehensive"
            )
            
            # Calculate portfolio-level metrics
            total_enhancement_score = sum(
                rec.get('ollama_enhancement', {}).get('enhancement_score', 0.3) 
                for rec in enhanced_recommendations
            ) / len(enhanced_recommendations)
            
            # Update portfolio recommendations
            enhanced_portfolio = portfolio_recommendations.copy()
            enhanced_portfolio['recommendations'] = enhanced_recommendations
            enhanced_portfolio['ollama_enhancement'] = {
                'timestamp': datetime.now().isoformat(),
                'strategy_type': strategy_type,
                'portfolio_enhancement_score': total_enhancement_score,
                'enhanced_recommendations': len(enhanced_recommendations)
            }
            
            logger.info(f"✅ Portfolio enhancement completed with score: {total_enhancement_score:.2f}")
            return enhanced_portfolio
            
        except Exception as e:
            logger.error(f"Error enhancing portfolio recommendations: {e}")
            return portfolio_recommendations


# Global service instance
_strategy_ollama_service = None


def get_strategy_ollama_enhancement_service() -> StrategyOllamaEnhancementService:
    """Get global Strategy Ollama enhancement service instance."""
    global _strategy_ollama_service
    if _strategy_ollama_service is None:
        _strategy_ollama_service = StrategyOllamaEnhancementService()
    return _strategy_ollama_service
