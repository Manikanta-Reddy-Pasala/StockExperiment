"""
AI-Powered Stock Analysis Service
Uses LLMs (Ollama/OpenAI) to generate stock reports and insights
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import requests
import json

logger = logging.getLogger(__name__)


class AIStockAnalyst:
    """
    AI-powered stock analysis using LLMs.

    Features:
    - Automated stock reports
    - Technical analysis interpretation
    - Fundamental analysis insights
    - Risk assessment commentary
    - Trading recommendations with rationale
    """

    def __init__(self, llm_provider: str = 'ollama', model: str = 'llama2'):
        """
        Initialize AI analyst.

        Args:
            llm_provider: 'ollama', 'openai', or 'anthropic'
            model: Model name to use
        """
        self.llm_provider = llm_provider
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"

    def generate_stock_report(self, stock_data: Dict) -> Dict:
        """
        Generate comprehensive stock analysis report.

        Args:
            stock_data: Dictionary with stock metrics

        Returns:
            Dictionary with AI-generated insights
        """
        symbol = stock_data.get('symbol', 'UNKNOWN')
        logger.info(f"Generating AI report for {symbol}")

        try:
            # Generate different sections
            technical_analysis = self._analyze_technical(stock_data)
            fundamental_analysis = self._analyze_fundamental(stock_data)
            ml_interpretation = self._interpret_ml_predictions(stock_data)
            risk_assessment = self._assess_risk(stock_data)
            recommendation = self._generate_recommendation(stock_data)

            report = {
                'symbol': symbol,
                'generated_at': datetime.now().isoformat(),
                'technical_analysis': technical_analysis,
                'fundamental_analysis': fundamental_analysis,
                'ml_interpretation': ml_interpretation,
                'risk_assessment': risk_assessment,
                'recommendation': recommendation,
                'confidence': self._calculate_report_confidence(stock_data)
            }

            logger.info(f"✓ AI report generated for {symbol}")
            return report

        except Exception as e:
            logger.error(f"Failed to generate AI report for {symbol}: {e}")
            return self._generate_fallback_report(stock_data)

    def _analyze_technical(self, stock_data: Dict) -> str:
        """Generate technical analysis commentary."""
        rsi = stock_data.get('rsi_14', 50)
        macd = stock_data.get('macd', 0)
        sma_50 = stock_data.get('sma_50', 0)
        sma_200 = stock_data.get('sma_200', 0)
        current_price = stock_data.get('current_price', 0)

        prompt = f"""Analyze the following technical indicators for stock {stock_data.get('symbol')}:

Current Price: ₹{current_price:.2f}
RSI (14): {rsi:.2f}
MACD: {macd:.2f}
SMA 50: ₹{sma_50:.2f}
SMA 200: ₹{sma_200:.2f}

Provide a brief technical analysis (2-3 sentences) covering:
1. Momentum (RSI interpretation)
2. Trend (SMA crossover)
3. Overall technical outlook

Be concise and professional."""

        analysis = self._call_llm(prompt)

        if not analysis:
            # Fallback rule-based analysis
            if rsi > 70:
                rsi_view = "overbought territory"
            elif rsi < 30:
                rsi_view = "oversold territory"
            else:
                rsi_view = "neutral zone"

            trend = "bullish" if sma_50 > sma_200 else "bearish"

            analysis = f"RSI at {rsi:.1f} indicates {rsi_view}. The stock is in a {trend} trend with SMA 50 {'above' if sma_50 > sma_200 else 'below'} SMA 200. MACD at {macd:.2f} suggests {'positive' if macd > 0 else 'negative'} momentum."

        return analysis

    def _analyze_fundamental(self, stock_data: Dict) -> str:
        """Generate fundamental analysis commentary."""
        pe = stock_data.get('pe_ratio', 0)
        pb = stock_data.get('pb_ratio', 0)
        roe = stock_data.get('roe', 0)
        revenue_growth = stock_data.get('revenue_growth', 0)
        operating_margin = stock_data.get('operating_margin', 0)

        prompt = f"""Analyze the fundamental metrics for {stock_data.get('symbol')}:

P/E Ratio: {pe:.2f}
P/B Ratio: {pb:.2f}
ROE: {roe:.2f}%
Revenue Growth: {revenue_growth:.2f}%
Operating Margin: {operating_margin:.2f}%

Provide a brief fundamental analysis (2-3 sentences) covering:
1. Valuation (P/E, P/B)
2. Profitability (ROE, margins)
3. Growth prospects

Be concise and professional."""

        analysis = self._call_llm(prompt)

        if not analysis:
            # Fallback rule-based
            valuation = "fairly valued" if 10 < pe < 25 else ("undervalued" if pe < 10 else "overvalued")
            profitability = "strong" if roe > 15 else ("moderate" if roe > 10 else "weak")

            analysis = f"With a P/E of {pe:.1f}, the stock appears {valuation}. ROE of {roe:.1f}% indicates {profitability} profitability. Revenue growth at {revenue_growth:.1f}% shows {'healthy' if revenue_growth > 10 else 'modest'} expansion."

        return analysis

    def _interpret_ml_predictions(self, stock_data: Dict) -> str:
        """Interpret ML model predictions."""
        ml_score = stock_data.get('ml_prediction_score', 0.5)
        ml_confidence = stock_data.get('ml_confidence', 0.5)
        ml_risk = stock_data.get('ml_risk_score', 0.5)
        predicted_change = stock_data.get('predicted_change_pct', 0)

        prompt = f"""Interpret these ML prediction metrics for {stock_data.get('symbol')}:

ML Prediction Score: {ml_score:.2f} (0-1, higher = better)
ML Confidence: {ml_confidence:.2f} (0-1)
ML Risk Score: {ml_risk:.2f} (0-1, lower = safer)
Predicted 2-Week Change: {predicted_change:+.2f}%

Provide a brief interpretation (2-3 sentences) explaining:
1. What the ML prediction suggests
2. Confidence level and reliability
3. Risk considerations

Be concise and professional."""

        interpretation = self._call_llm(prompt)

        if not interpretation:
            # Fallback
            outlook = "positive" if ml_score > 0.6 else ("negative" if ml_score < 0.4 else "neutral")
            confidence_level = "high" if ml_confidence > 0.7 else ("moderate" if ml_confidence > 0.5 else "low")
            risk_level = "low" if ml_risk < 0.3 else ("moderate" if ml_risk < 0.6 else "high")

            interpretation = f"ML models suggest a {outlook} outlook with {confidence_level} confidence. Predicted 2-week return of {predicted_change:+.1f}%. Risk assessment indicates {risk_level} downside potential."

        return interpretation

    def _assess_risk(self, stock_data: Dict) -> str:
        """Generate risk assessment."""
        beta = stock_data.get('beta', 1.0)
        volatility = stock_data.get('historical_volatility_1y', 0)
        debt_to_equity = stock_data.get('debt_to_equity', 0)
        ml_risk = stock_data.get('ml_risk_score', 0.5)

        # Rule-based risk assessment
        volatility_risk = "High" if volatility > 30 else ("Moderate" if volatility > 15 else "Low")
        leverage_risk = "High" if debt_to_equity > 2 else ("Moderate" if debt_to_equity > 1 else "Low")
        market_risk = "High" if beta > 1.5 else ("Moderate" if beta > 0.8 else "Low")

        assessment = f"""
Risk Profile:
- Volatility Risk: {volatility_risk} (Historical volatility: {volatility:.1f}%)
- Market Risk: {market_risk} (Beta: {beta:.2f})
- Leverage Risk: {leverage_risk} (Debt/Equity: {debt_to_equity:.2f})
- ML Risk Score: {ml_risk:.2f}/1.0

Overall: {'High Risk' if ml_risk > 0.6 else ('Moderate Risk' if ml_risk > 0.3 else 'Low Risk')}
        """.strip()

        return assessment

    def _generate_recommendation(self, stock_data: Dict) -> Dict:
        """Generate trading recommendation."""
        ml_score = stock_data.get('ml_prediction_score', 0.5)
        ml_confidence = stock_data.get('ml_confidence', 0.5)
        ml_risk = stock_data.get('ml_risk_score', 0.5)
        current_price = stock_data.get('current_price', 0)
        predicted_target = stock_data.get('ml_price_target', current_price)

        # Decision logic
        if ml_score >= 0.7 and ml_confidence >= 0.7 and ml_risk <= 0.3:
            action = "STRONG BUY"
            rationale = "High ML prediction score with high confidence and low risk."
        elif ml_score >= 0.6 and ml_confidence >= 0.6 and ml_risk <= 0.4:
            action = "BUY"
            rationale = "Positive ML outlook with acceptable risk levels."
        elif ml_score <= 0.3 or ml_risk >= 0.7:
            action = "SELL"
            rationale = "Low ML prediction score or high risk detected."
        elif ml_score <= 0.4 and ml_confidence >= 0.6:
            action = "AVOID"
            rationale = "ML models predict underperformance with reasonable confidence."
        else:
            action = "HOLD"
            rationale = "Mixed signals. Wait for clearer trend."

        # Calculate targets
        upside = ((predicted_target - current_price) / current_price) * 100 if current_price > 0 else 0
        stop_loss = current_price * 0.92  # 8% stop loss
        target_price = predicted_target

        return {
            'action': action,
            'rationale': rationale,
            'current_price': round(current_price, 2),
            'target_price': round(target_price, 2),
            'upside_potential': round(upside, 2),
            'stop_loss': round(stop_loss, 2),
            'time_horizon': '2 weeks',
            'conviction': 'High' if ml_confidence > 0.7 else ('Medium' if ml_confidence > 0.5 else 'Low')
        }

    def _call_llm(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """
        Call LLM API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens in response

        Returns:
            LLM response text or None if failed
        """
        if self.llm_provider == 'ollama':
            try:
                response = requests.post(
                    self.ollama_url,
                    json={
                        'model': self.model,
                        'prompt': prompt,
                        'stream': False,
                        'options': {
                            'num_predict': max_tokens,
                            'temperature': 0.3  # Lower temperature for factual analysis
                        }
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '').strip()
                else:
                    logger.warning(f"Ollama API returned status {response.status_code}")
                    return None

            except Exception as e:
                logger.warning(f"Failed to call Ollama API: {e}")
                return None

        else:
            # Placeholder for other providers
            logger.warning(f"LLM provider '{self.llm_provider}' not implemented")
            return None

    def _calculate_report_confidence(self, stock_data: Dict) -> float:
        """Calculate overall report confidence based on data quality."""
        ml_confidence = stock_data.get('ml_confidence', 0.5)

        # Check data completeness
        key_fields = ['current_price', 'pe_ratio', 'roe', 'rsi_14', 'macd']
        present = sum(1 for field in key_fields if stock_data.get(field) is not None)
        data_quality = present / len(key_fields)

        # Combined confidence
        confidence = (ml_confidence * 0.7 + data_quality * 0.3)

        return round(confidence, 2)

    def _generate_fallback_report(self, stock_data: Dict) -> Dict:
        """Generate basic report when AI fails."""
        return {
            'symbol': stock_data.get('symbol', 'UNKNOWN'),
            'generated_at': datetime.now().isoformat(),
            'technical_analysis': 'Unable to generate AI analysis. Please check manually.',
            'fundamental_analysis': 'Unable to generate AI analysis. Please check manually.',
            'ml_interpretation': f"ML Score: {stock_data.get('ml_prediction_score', 0):.2f}",
            'risk_assessment': self._assess_risk(stock_data),
            'recommendation': self._generate_recommendation(stock_data),
            'confidence': 0.3
        }

    def batch_analyze(self, stocks: List[Dict], top_n: int = 10) -> List[Dict]:
        """
        Generate AI reports for multiple stocks.

        Args:
            stocks: List of stock data dictionaries
            top_n: Generate detailed reports for top N stocks only

        Returns:
            List of AI reports
        """
        logger.info(f"Generating AI reports for top {top_n} stocks...")

        reports = []
        for i, stock in enumerate(stocks[:top_n]):
            logger.info(f"Processing {i+1}/{top_n}: {stock.get('symbol')}")
            report = self.generate_stock_report(stock)
            reports.append(report)

        return reports
