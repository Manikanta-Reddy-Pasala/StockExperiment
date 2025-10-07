"""
Advanced Sentiment Analysis for Stocks
Analyzes news, social media, and earnings reports for sentiment
"""

import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re
from collections import Counter

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Multi-source sentiment analysis.

    Sources:
    - News headlines (API or web scraping)
    - Social media (Twitter/Reddit via APIs)
    - Earnings call transcripts
    - Analyst reports

    Methods:
    - Rule-based sentiment (keyword matching)
    - ML-based sentiment (using Ollama/LLM)
    - Aggregate sentiment scoring
    """

    def __init__(self, llm_provider: str = 'ollama'):
        self.llm_provider = llm_provider
        self.ollama_url = "http://localhost:11434/api/generate"

        # Sentiment lexicons
        self.positive_words = {
            'bullish', 'growth', 'profit', 'gain', 'beat', 'strong', 'outperform',
            'upgrade', 'surge', 'rally', 'boom', 'increase', 'rise', 'up', 'positive',
            'optimistic', 'confidence', 'momentum', 'breakthrough', 'recovery'
        }

        self.negative_words = {
            'bearish', 'loss', 'decline', 'fall', 'miss', 'weak', 'underperform',
            'downgrade', 'plunge', 'crash', 'slump', 'decrease', 'drop', 'down', 'negative',
            'pessimistic', 'concern', 'risk', 'warning', 'recession', 'crisis'
        }

    def analyze_stock_sentiment(self, symbol: str, sources: List[str] = None) -> Dict:
        """
        Analyze overall sentiment for a stock.

        Args:
            symbol: Stock symbol
            sources: List of sources to analyze ('news', 'social', 'earnings')

        Returns:
            Sentiment analysis results
        """
        if sources is None:
            sources = ['news']  # Default to news only

        logger.info(f"Analyzing sentiment for {symbol} from sources: {sources}")

        results = {}

        # Analyze each source
        if 'news' in sources:
            results['news'] = self._analyze_news_sentiment(symbol)

        if 'social' in sources:
            results['social'] = self._analyze_social_sentiment(symbol)

        if 'earnings' in sources:
            results['earnings'] = self._analyze_earnings_sentiment(symbol)

        # Aggregate sentiment
        aggregate = self._aggregate_sentiment(results)

        return {
            'symbol': symbol,
            'overall_sentiment': aggregate['sentiment'],
            'overall_score': aggregate['score'],
            'confidence': aggregate['confidence'],
            'sources': results,
            'analyzed_at': datetime.now().isoformat()
        }

    def _analyze_news_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """Analyze news sentiment."""
        try:
            # Get news headlines (simulate for now)
            headlines = self._fetch_news_headlines(symbol, days)

            if not headlines:
                return self._neutral_sentiment('news')

            # Analyze each headline
            sentiments = []
            for headline in headlines:
                score = self._score_text_sentiment(headline['text'])
                sentiments.append({
                    'text': headline['text'],
                    'score': score,
                    'source': headline.get('source', 'Unknown'),
                    'date': headline.get('date', datetime.now().isoformat())
                })

            # Aggregate
            avg_score = sum(s['score'] for s in sentiments) / len(sentiments)

            return {
                'sentiment': self._score_to_label(avg_score),
                'score': round(avg_score, 3),
                'count': len(sentiments),
                'latest_headlines': sentiments[:5],
                'positive_count': len([s for s in sentiments if s['score'] > 0.2]),
                'negative_count': len([s for s in sentiments if s['score'] < -0.2])
            }

        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return self._neutral_sentiment('news')

    def _analyze_social_sentiment(self, symbol: str) -> Dict:
        """Analyze social media sentiment."""
        try:
            # Simulate social media mentions
            # In production, integrate with Twitter/Reddit APIs
            mentions = self._fetch_social_mentions(symbol)

            if not mentions:
                return self._neutral_sentiment('social')

            sentiments = []
            for mention in mentions:
                score = self._score_text_sentiment(mention['text'])
                sentiments.append({'score': score, 'platform': mention.get('platform')})

            avg_score = sum(s['score'] for s in sentiments) / len(sentiments)

            return {
                'sentiment': self._score_to_label(avg_score),
                'score': round(avg_score, 3),
                'count': len(sentiments),
                'platforms': Counter([s['platform'] for s in sentiments]),
                'buzz_level': 'high' if len(sentiments) > 50 else 'medium' if len(sentiments) > 10 else 'low'
            }

        except Exception as e:
            logger.error(f"Social sentiment analysis failed: {e}")
            return self._neutral_sentiment('social')

    def _analyze_earnings_sentiment(self, symbol: str) -> Dict:
        """Analyze earnings call sentiment."""
        try:
            # Get latest earnings transcript (simulate)
            transcript = self._fetch_earnings_transcript(symbol)

            if not transcript:
                return self._neutral_sentiment('earnings')

            # Analyze using LLM if available
            sentiment_score = self._analyze_with_llm(transcript['text'], context='earnings')

            if sentiment_score is None:
                sentiment_score = self._score_text_sentiment(transcript['text'])

            return {
                'sentiment': self._score_to_label(sentiment_score),
                'score': round(sentiment_score, 3),
                'quarter': transcript.get('quarter', 'Q4 2024'),
                'date': transcript.get('date', ''),
                'key_topics': self._extract_key_topics(transcript['text'])
            }

        except Exception as e:
            logger.error(f"Earnings sentiment analysis failed: {e}")
            return self._neutral_sentiment('earnings')

    def _score_text_sentiment(self, text: str) -> float:
        """
        Score text sentiment using keyword matching.

        Returns:
            Score from -1 (very negative) to +1 (very positive)
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        positive_count = len(words & self.positive_words)
        negative_count = len(words & self.negative_words)

        total_count = positive_count + negative_count

        if total_count == 0:
            return 0.0

        # Calculate sentiment score
        score = (positive_count - negative_count) / total_count

        return score

    def _analyze_with_llm(self, text: str, context: str = 'general') -> Optional[float]:
        """Analyze sentiment using LLM."""
        if self.llm_provider != 'ollama':
            return None

        try:
            prompt = f"""Analyze the sentiment of the following {context} text about a stock.
Respond with only a sentiment score from -1 (very negative) to +1 (very positive).

Text: {text[:500]}

Sentiment score:"""

            response = requests.post(
                self.ollama_url,
                json={
                    'model': 'llama2',
                    'prompt': prompt,
                    'stream': False,
                    'options': {'num_predict': 10, 'temperature': 0.3}
                },
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()
                text_response = result.get('response', '').strip()

                # Extract numeric score
                import re
                numbers = re.findall(r'-?\d+\.?\d*', text_response)
                if numbers:
                    score = float(numbers[0])
                    return max(-1.0, min(1.0, score))  # Clamp to [-1, 1]

        except Exception as e:
            logger.warning(f"LLM sentiment analysis failed: {e}")

        return None

    def _fetch_news_headlines(self, symbol: str, days: int) -> List[Dict]:
        """Fetch news headlines (simulated for demo)."""
        # In production, integrate with:
        # - NewsAPI (https://newsapi.org/)
        # - Google News RSS
        # - Financial news APIs (Bloomberg, Reuters)

        # Simulate headlines
        headlines = [
            {
                'text': f'{symbol} reports strong quarterly earnings, beats estimates',
                'source': 'Economic Times',
                'date': (datetime.now() - timedelta(days=1)).isoformat()
            },
            {
                'text': f'{symbol} announces new product line, analysts optimistic',
                'source': 'Business Standard',
                'date': (datetime.now() - timedelta(days=2)).isoformat()
            },
            {
                'text': f'Market concerns over {symbol} debt levels',
                'source': 'Reuters',
                'date': (datetime.now() - timedelta(days=3)).isoformat()
            }
        ]

        return headlines

    def _fetch_social_mentions(self, symbol: str) -> List[Dict]:
        """Fetch social media mentions (simulated)."""
        # In production, integrate with:
        # - Twitter API v2
        # - Reddit API (praw)
        # - StockTwits API

        mentions = [
            {'text': f'$${symbol} looking bullish! Strong momentum', 'platform': 'twitter'},
            {'text': f'Buying more {symbol} on this dip', 'platform': 'reddit'},
            {'text': f'{symbol} earnings beat was impressive', 'platform': 'stocktwits'}
        ]

        return mentions

    def _fetch_earnings_transcript(self, symbol: str) -> Optional[Dict]:
        """Fetch earnings call transcript (simulated)."""
        # In production, integrate with:
        # - Seeking Alpha transcripts
        # - Company investor relations pages
        # - Financial data APIs

        transcript = {
            'text': f'We are pleased to report strong growth in Q4. Revenue increased 15% year-over-year. '
                   f'Operating margins improved significantly. We remain optimistic about future prospects.',
            'quarter': 'Q4 2024',
            'date': (datetime.now() - timedelta(days=30)).isoformat()
        }

        return transcript

    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from text."""
        # Simple keyword extraction
        keywords = {
            'growth', 'revenue', 'profit', 'margin', 'expansion',
            'cost', 'competition', 'market share', 'innovation'
        }

        text_lower = text.lower()
        found_topics = [word for word in keywords if word in text_lower]

        return found_topics[:5]

    def _aggregate_sentiment(self, sources: Dict) -> Dict:
        """Aggregate sentiment across sources."""
        scores = []
        weights = {'news': 0.5, 'social': 0.2, 'earnings': 0.3}

        total_weight = 0
        weighted_sum = 0

        for source, data in sources.items():
            if 'score' in data:
                weight = weights.get(source, 0.33)
                weighted_sum += data['score'] * weight
                total_weight += weight
                scores.append(data['score'])

        if total_weight == 0:
            return {'sentiment': 'NEUTRAL', 'score': 0.0, 'confidence': 0.3}

        avg_score = weighted_sum / total_weight

        # Confidence based on agreement between sources
        if len(scores) > 1:
            variance = np.var(scores) if 'np' in dir() else 0.1
            confidence = max(0.3, min(1.0, 1.0 - variance))
        else:
            confidence = 0.5

        return {
            'sentiment': self._score_to_label(avg_score),
            'score': round(avg_score, 3),
            'confidence': round(confidence, 2)
        }

    def _score_to_label(self, score: float) -> str:
        """Convert numeric score to sentiment label."""
        if score > 0.3:
            return 'VERY_POSITIVE'
        elif score > 0.1:
            return 'POSITIVE'
        elif score > -0.1:
            return 'NEUTRAL'
        elif score > -0.3:
            return 'NEGATIVE'
        else:
            return 'VERY_NEGATIVE'

    def _neutral_sentiment(self, source: str) -> Dict:
        """Return neutral sentiment when analysis fails."""
        return {
            'sentiment': 'NEUTRAL',
            'score': 0.0,
            'count': 0,
            'source': source
        }

    def batch_analyze(self, symbols: List[str]) -> Dict[str, Dict]:
        """Analyze sentiment for multiple stocks."""
        logger.info(f"Batch analyzing sentiment for {len(symbols)} symbols")

        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.analyze_stock_sentiment(symbol, sources=['news'])
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                results[symbol] = {
                    'overall_sentiment': 'NEUTRAL',
                    'overall_score': 0.0,
                    'confidence': 0.0
                }

        return results

    def get_market_sentiment(self, index_stocks: List[str]) -> Dict:
        """
        Analyze overall market sentiment from index components.

        Args:
            index_stocks: List of index component symbols

        Returns:
            Market-wide sentiment analysis
        """
        logger.info(f"Analyzing market sentiment from {len(index_stocks)} index stocks")

        sentiments = self.batch_analyze(index_stocks)

        scores = [s['overall_score'] for s in sentiments.values()]
        avg_score = sum(scores) / len(scores) if scores else 0

        positive_count = len([s for s in scores if s > 0.2])
        negative_count = len([s for s in scores if s < -0.2])

        return {
            'market_sentiment': self._score_to_label(avg_score),
            'market_score': round(avg_score, 3),
            'positive_stocks': positive_count,
            'negative_stocks': negative_count,
            'neutral_stocks': len(scores) - positive_count - negative_count,
            'total_analyzed': len(scores),
            'sentiment_distribution': {
                'very_positive': len([s for s in scores if s > 0.3]),
                'positive': len([s for s in scores if 0.1 < s <= 0.3]),
                'neutral': len([s for s in scores if -0.1 <= s <= 0.1]),
                'negative': len([s for s in scores if -0.3 <= s < -0.1]),
                'very_negative': len([s for s in scores if s < -0.3])
            },
            'analyzed_at': datetime.now().isoformat()
        }


# Import numpy if available
try:
    import numpy as np
except ImportError:
    logger.warning("NumPy not available for sentiment analysis")
    np = None
