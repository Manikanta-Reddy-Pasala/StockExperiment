"""
ChatGPT Integration for Stock Analysis
Provides AI-powered analysis and recommendations for selected stocks
"""
import requests
import json
from typing import List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ChatGPTAnalyzer:
    """ChatGPT integration for stock analysis and recommendations."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize ChatGPT analyzer.
        
        Args:
            api_key (str): OpenAI API key
            model (str): ChatGPT model to use
        """
        self.api_key = api_key or self._get_api_key()
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        import os
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OpenAI API key not found in environment variables")
        return api_key or "your_openai_api_key_here"
    
    def analyze_stock(self, stock_data: Dict) -> Dict:
        """
        Analyze a single stock using ChatGPT.
        
        Args:
            stock_data (Dict): Stock data including financial metrics
            
        Returns:
            Dict: Analysis results
        """
        try:
            # Prepare stock information for analysis
            stock_info = self._prepare_stock_info(stock_data)
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(stock_info)
            
            # Get ChatGPT analysis
            analysis = self._get_chatgpt_response(prompt)
            
            # Parse and structure the response
            structured_analysis = self._parse_analysis_response(analysis, stock_data)
            
            return structured_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing stock {stock_data.get('symbol', 'unknown')}: {e}")
            return self._get_fallback_analysis(stock_data)
    
    def analyze_portfolio(self, selected_stocks: List[Dict], strategy_name: str) -> Dict:
        """
        Analyze a portfolio of selected stocks.
        
        Args:
            selected_stocks (List[Dict]): List of selected stocks
            strategy_name (str): Name of the strategy used
            
        Returns:
            Dict: Portfolio analysis results
        """
        try:
            # Prepare portfolio information
            portfolio_info = self._prepare_portfolio_info(selected_stocks, strategy_name)
            
            # Create portfolio analysis prompt
            prompt = self._create_portfolio_analysis_prompt(portfolio_info)
            
            # Get ChatGPT analysis
            analysis = self._get_chatgpt_response(prompt)
            
            # Parse and structure the response
            structured_analysis = self._parse_portfolio_analysis_response(analysis, selected_stocks, strategy_name)
            
            return structured_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio for strategy {strategy_name}: {e}")
            return self._get_fallback_portfolio_analysis(selected_stocks, strategy_name)
    
    def compare_strategies(self, strategy_results: Dict[str, List[Dict]]) -> Dict:
        """
        Compare different strategies using ChatGPT.
        
        Args:
            strategy_results (Dict[str, List[Dict]]): Results from different strategies
            
        Returns:
            Dict: Strategy comparison analysis
        """
        try:
            # Prepare strategy comparison information
            comparison_info = self._prepare_strategy_comparison_info(strategy_results)
            
            # Create comparison prompt
            prompt = self._create_strategy_comparison_prompt(comparison_info)
            
            # Get ChatGPT analysis
            analysis = self._get_chatgpt_response(prompt)
            
            # Parse and structure the response
            structured_analysis = self._parse_strategy_comparison_response(analysis, strategy_results)
            
            return structured_analysis
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return self._get_fallback_strategy_comparison(strategy_results)
    
    def _prepare_stock_info(self, stock_data: Dict) -> str:
        """Prepare stock information for ChatGPT analysis."""
        screening_data = stock_data.get('screening_data', {})
        
        info = f"""
Stock: {stock_data.get('symbol', 'Unknown')} - {stock_data.get('name', 'Unknown')}
Sector: {stock_data.get('sector', 'Unknown')}
Market Cap: {stock_data.get('market_cap', 0)} crores

Financial Metrics:
- Current Price: ₹{screening_data.get('current_price', 0)}
- 50-day Moving Average: ₹{screening_data.get('dma_50', 0)}
- Volume: {screening_data.get('volume', 0):,}
- Average Volume (1 week): {screening_data.get('avg_volume_1week', 0):,}

Growth Metrics:
- Sales Growth (YoY): {((screening_data.get('sales_current_year', 0) - screening_data.get('sales_preceding_year', 0)) / screening_data.get('sales_preceding_year', 1) * 100):.2f}%
- Operating Profit Growth: {((screening_data.get('op_profit_latest_quarter', 0) - screening_data.get('op_profit_preceding_quarter', 0)) / screening_data.get('op_profit_preceding_quarter', 1) * 100):.2f}%

Valuation Metrics:
- Intrinsic Value: ₹{screening_data.get('intrinsic_value', 0)}
- Debt to Equity: {screening_data.get('debt_to_equity', 0):.2f}
- Piotroski Score: {screening_data.get('piotroski_score', 0)}
"""
        return info
    
    def _create_analysis_prompt(self, stock_info: str) -> str:
        """Create prompt for stock analysis."""
        return f"""
As a financial analyst, please analyze the following Indian stock and provide your assessment:

{stock_info}

Please provide:
1. Investment Recommendation (BUY/HOLD/SELL) with confidence level (1-10)
2. Key Strengths (3-5 points)
3. Key Risks (3-5 points)
4. Price Target (12-month)
5. Investment Thesis (2-3 sentences)
6. Risk Factors to Monitor

Format your response as JSON with the following structure:
{{
    "recommendation": "BUY/HOLD/SELL",
    "confidence": 8,
    "strengths": ["strength1", "strength2", "strength3"],
    "risks": ["risk1", "risk2", "risk3"],
    "price_target": 1500,
    "investment_thesis": "Brief investment thesis",
    "risk_factors": ["factor1", "factor2"]
}}
"""
    
    def _create_portfolio_analysis_prompt(self, portfolio_info: str) -> str:
        """Create prompt for portfolio analysis."""
        return f"""
As a portfolio manager, please analyze the following stock portfolio selected using a {portfolio_info['strategy_name']} strategy:

{portfolio_info['stock_details']}

Portfolio Summary:
- Number of stocks: {portfolio_info['num_stocks']}
- Strategy: {portfolio_info['strategy_name']}
- Sector distribution: {portfolio_info['sector_distribution']}

Please provide:
1. Overall Portfolio Assessment (1-10 rating)
2. Diversification Analysis
3. Sector Concentration Risks
4. Portfolio Strengths
5. Portfolio Weaknesses
6. Recommendations for Improvement

Format your response as JSON with the following structure:
{{
    "portfolio_rating": 8,
    "diversification_score": 7,
    "sector_risks": ["risk1", "risk2"],
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "recommendations": ["rec1", "rec2"]
}}
"""
    
    def _create_strategy_comparison_prompt(self, comparison_info: str) -> str:
        """Create prompt for strategy comparison."""
        return f"""
As a quantitative analyst, please compare the following trading strategies based on their stock selections:

{comparison_info}

Please provide:
1. Best Performing Strategy and why
2. Risk-Return Analysis for each strategy
3. Market Condition Suitability
4. Diversification Benefits
5. Overall Recommendation

Format your response as JSON with the following structure:
{{
    "best_strategy": "strategy_name",
    "strategy_rankings": [
        {{"strategy": "name", "score": 8, "reason": "explanation"}},
        {{"strategy": "name", "score": 7, "reason": "explanation"}}
    ],
    "market_suitability": {{
        "bull_market": "best_strategy",
        "bear_market": "best_strategy",
        "sideways_market": "best_strategy"
    }},
    "overall_recommendation": "recommendation"
}}
"""
    
    def _get_chatgpt_response(self, prompt: str) -> str:
        """Get response from ChatGPT API."""
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a professional financial analyst specializing in Indian stock markets."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.3
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling ChatGPT API: {e}")
            return "API Error: Unable to get analysis"
        except Exception as e:
            logger.error(f"Error processing ChatGPT response: {e}")
            return "Error: Unable to process analysis"
    
    def _parse_analysis_response(self, response: str, stock_data: Dict) -> Dict:
        """Parse ChatGPT response for stock analysis."""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response
            
            analysis = json.loads(json_str)
            analysis['timestamp'] = datetime.utcnow().isoformat()
            analysis['stock_symbol'] = stock_data.get('symbol', 'Unknown')
            
            return analysis
            
        except json.JSONDecodeError:
            # Fallback to text parsing
            return {
                'recommendation': 'HOLD',
                'confidence': 5,
                'strengths': ['Analysis pending'],
                'risks': ['Analysis pending'],
                'price_target': stock_data.get('screening_data', {}).get('current_price', 0),
                'investment_thesis': 'AI analysis temporarily unavailable',
                'risk_factors': ['Analysis pending'],
                'timestamp': datetime.utcnow().isoformat(),
                'stock_symbol': stock_data.get('symbol', 'Unknown'),
                'raw_response': response
            }
    
    def _parse_portfolio_analysis_response(self, response: str, selected_stocks: List[Dict], strategy_name: str) -> Dict:
        """Parse ChatGPT response for portfolio analysis."""
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response
            
            analysis = json.loads(json_str)
            analysis['timestamp'] = datetime.utcnow().isoformat()
            analysis['strategy_name'] = strategy_name
            analysis['num_stocks'] = len(selected_stocks)
            
            return analysis
            
        except json.JSONDecodeError:
            return {
                'portfolio_rating': 5,
                'diversification_score': 5,
                'sector_risks': ['Analysis pending'],
                'strengths': ['Analysis pending'],
                'weaknesses': ['Analysis pending'],
                'recommendations': ['Analysis pending'],
                'timestamp': datetime.utcnow().isoformat(),
                'strategy_name': strategy_name,
                'num_stocks': len(selected_stocks),
                'raw_response': response
            }
    
    def _parse_strategy_comparison_response(self, response: str, strategy_results: Dict) -> Dict:
        """Parse ChatGPT response for strategy comparison."""
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response
            
            analysis = json.loads(json_str)
            analysis['timestamp'] = datetime.utcnow().isoformat()
            analysis['strategies_compared'] = list(strategy_results.keys())
            
            return analysis
            
        except json.JSONDecodeError:
            return {
                'best_strategy': 'momentum',
                'strategy_rankings': [],
                'market_suitability': {
                    'bull_market': 'momentum',
                    'bear_market': 'value',
                    'sideways_market': 'growth'
                },
                'overall_recommendation': 'Analysis pending',
                'timestamp': datetime.utcnow().isoformat(),
                'strategies_compared': list(strategy_results.keys()),
                'raw_response': response
            }
    
    def _get_fallback_analysis(self, stock_data: Dict) -> Dict:
        """Get fallback analysis when ChatGPT is unavailable."""
        return {
            'recommendation': 'HOLD',
            'confidence': 5,
            'strengths': ['Screening criteria met', 'Financial metrics positive'],
            'risks': ['Market volatility', 'Sector risks'],
            'price_target': stock_data.get('screening_data', {}).get('current_price', 0) * 1.1,
            'investment_thesis': 'Stock meets screening criteria but AI analysis unavailable',
            'risk_factors': ['Market conditions', 'Company performance'],
            'timestamp': datetime.utcnow().isoformat(),
            'stock_symbol': stock_data.get('symbol', 'Unknown'),
            'fallback': True
        }
    
    def _get_fallback_portfolio_analysis(self, selected_stocks: List[Dict], strategy_name: str) -> Dict:
        """Get fallback portfolio analysis."""
        return {
            'portfolio_rating': 6,
            'diversification_score': 5,
            'sector_risks': ['Concentration risk'],
            'strengths': ['Diversified selection', 'Strategy criteria met'],
            'weaknesses': ['Limited analysis available'],
            'recommendations': ['Monitor performance', 'Review regularly'],
            'timestamp': datetime.utcnow().isoformat(),
            'strategy_name': strategy_name,
            'num_stocks': len(selected_stocks),
            'fallback': True
        }
    
    def _get_fallback_strategy_comparison(self, strategy_results: Dict) -> Dict:
        """Get fallback strategy comparison."""
        return {
            'best_strategy': 'momentum',
            'strategy_rankings': [
                {'strategy': name, 'score': 6, 'reason': 'Limited analysis available'}
                for name in strategy_results.keys()
            ],
            'market_suitability': {
                'bull_market': 'momentum',
                'bear_market': 'value',
                'sideways_market': 'growth'
            },
            'overall_recommendation': 'Use multiple strategies for diversification',
            'timestamp': datetime.utcnow().isoformat(),
            'strategies_compared': list(strategy_results.keys()),
            'fallback': True
        }
    
    def _prepare_portfolio_info(self, selected_stocks: List[Dict], strategy_name: str) -> Dict:
        """Prepare portfolio information for analysis."""
        stock_details = ""
        sector_distribution = {}
        
        for stock in selected_stocks:
            sector = stock.get('sector', 'Unknown')
            sector_distribution[sector] = sector_distribution.get(sector, 0) + 1
            
            stock_details += f"- {stock.get('symbol', 'Unknown')} ({sector})\n"
        
        return {
            'strategy_name': strategy_name,
            'num_stocks': len(selected_stocks),
            'stock_details': stock_details,
            'sector_distribution': sector_distribution
        }
    
    def _prepare_strategy_comparison_info(self, strategy_results: Dict) -> str:
        """Prepare strategy comparison information."""
        info = "Strategy Comparison:\n\n"
        
        for strategy_name, stocks in strategy_results.items():
            info += f"{strategy_name.upper()} Strategy:\n"
            info += f"- Number of stocks: {len(stocks)}\n"
            
            if stocks:
                sectors = {}
                for stock in stocks:
                    sector = stock.get('sector', 'Unknown')
                    sectors[sector] = sectors.get(sector, 0) + 1
                
                info += f"- Sector distribution: {sectors}\n"
                
                # Add top stocks
                info += f"- Top stocks: {', '.join([s.get('symbol', 'Unknown') for s in stocks[:3]])}\n"
            
            info += "\n"
        
        return info


if __name__ == "__main__":
    # Test the ChatGPT analyzer
    analyzer = ChatGPTAnalyzer()
    
    # Mock stock data
    mock_stock = {
        'symbol': 'TEST',
        'name': 'Test Company',
        'sector': 'IT',
        'market_cap': 8000,
        'screening_data': {
            'current_price': 100,
            'dma_50': 95,
            'volume': 1000000,
            'avg_volume_1week': 500000,
            'sales_current_year': 1000,
            'sales_preceding_year': 800,
            'op_profit_latest_quarter': 100,
            'op_profit_preceding_quarter': 80,
            'intrinsic_value': 120,
            'debt_to_equity': 0.1,
            'piotroski_score': 7
        }
    }
    
    # Test analysis
    analysis = analyzer.analyze_stock(mock_stock)
    print(f"Stock analysis: {analysis}")
