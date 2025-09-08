"""
ChatGPT Stock Selection Validator
"""
import openai
import json
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from data_provider.data_manager import get_data_manager


class ChatGPTValidator:
    """Validate stock selections using ChatGPT."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ChatGPT validator.
        
        Args:
            api_key (Optional[str]): OpenAI API key
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if self.api_key:
            openai.api_key = self.api_key
        
        self.data_manager = get_data_manager()
    
    def validate_stocks(self, selected_stocks: List[Dict[str, Any]], 
                       market_context: str = "Indian stock market") -> List[Dict[str, Any]]:
        """
        Validate stock selections using ChatGPT.
        
        Args:
            selected_stocks (List[Dict[str, Any]]): List of selected stocks
            market_context (str): Market context for validation
            
        Returns:
            List[Dict[str, Any]]: Validated stocks with ChatGPT scores
        """
        if not self.api_key:
            # Return stocks with default validation if no API key
            return self._default_validation(selected_stocks)
        
        try:
            # Prepare stock data for ChatGPT
            stock_info = self._prepare_stock_info(selected_stocks)
            
            # Create prompt for ChatGPT
            prompt = self._create_validation_prompt(stock_info, market_context)
            
            # Call ChatGPT API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert in stock market analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse response
            validation_result = response.choices[0].message.content
            validated_stocks = self._parse_validation_response(validation_result, selected_stocks)
            
            return validated_stocks
        except Exception as e:
            print(f"Error validating stocks with ChatGPT: {e}")
            # Return stocks with default validation if API call fails
            return self._default_validation(selected_stocks)
    
    def _prepare_stock_info(self, selected_stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare detailed stock information for validation.
        
        Args:
            selected_stocks (List[Dict[str, Any]]): Selected stocks
            
        Returns:
            List[Dict[str, Any]]: Detailed stock information
        """
        stock_info = []
        
        for stock in selected_stocks:
            symbol = stock.get('symbol', '')
            if not symbol:
                continue
                
            # Get current price and basic info
            try:
                current_price = self.data_manager.get_current_price(symbol)
                historical_data = self.data_manager.get_historical_data(symbol, period="1mo", interval="1d")
                
                # Calculate basic metrics
                if not historical_data.empty:
                    avg_volume = historical_data['Volume'].mean() if 'Volume' in historical_data.columns else 0
                    price_change = ((current_price - historical_data['Close'].iloc[0]) / historical_data['Close'].iloc[0] * 100) if 'Close' in historical_data.columns and len(historical_data) > 0 else 0
                else:
                    avg_volume = 0
                    price_change = 0
                
                stock_info.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'momentum': stock.get('momentum', 0),
                    'signal': stock.get('signal', 'HOLD'),
                    'avg_volume': avg_volume,
                    'price_change_pct': price_change
                })
            except Exception as e:
                print(f"Error getting data for {symbol}: {e}")
                # Add stock with minimal info
                stock_info.append({
                    'symbol': symbol,
                    'current_price': 0,
                    'momentum': stock.get('momentum', 0),
                    'signal': stock.get('signal', 'HOLD'),
                    'avg_volume': 0,
                    'price_change_pct': 0
                })
        
        return stock_info
    
    def _create_validation_prompt(self, stock_info: List[Dict[str, Any]], 
                                market_context: str) -> str:
        """
        Create prompt for ChatGPT validation.
        
        Args:
            stock_info (List[Dict[str, Any]]): Stock information
            market_context (str): Market context
            
        Returns:
            str: Validation prompt
        """
        prompt = f"""
        As a financial analyst, please evaluate the following stock selections in the {market_context} context.
        For each stock, provide a validation score between 0-100, where:
        - 0-30: Strong Sell
        - 31-50: Sell
        - 51-70: Hold
        - 71-85: Buy
        - 86-100: Strong Buy
        
        Consider factors like:
        1. Fundamental analysis (if available)
        2. Technical indicators
        3. Market sentiment
        4. Volume trends
        5. Risk factors
        
        Stock Information:
        """
        
        for stock in stock_info:
            prompt += f"""
        - Symbol: {stock['symbol']}
          Current Price: ₹{stock['current_price']:.2f}
          Momentum Score: {stock['momentum']:.4f}
          Signal: {stock['signal']}
          Avg Volume: {stock['avg_volume']:,.0f}
          Price Change: {stock['price_change_pct']:.2f}%
        """
        
        prompt += """
        
        Please respond in JSON format with the following structure:
        {
          "validations": [
            {
              "symbol": "STOCK_SYMBOL",
              "validation_score": 85,
              "recommendation": "Strong Buy",
              "reasoning": "Brief reasoning for the score"
            }
          ]
        }
        
        Only return the JSON object, nothing else.
        """
        
        return prompt
    
    def _parse_validation_response(self, response_text: str, 
                                 selected_stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse ChatGPT validation response.
        
        Args:
            response_text (str): ChatGPT response
            selected_stocks (List[Dict[str, Any]]): Original selected stocks
            
        Returns:
            List[Dict[str, Any]]: Validated stocks
        """
        try:
            # Try to parse JSON from response
            response_json = json.loads(response_text)
            validations = response_json.get('validations', [])
            
            # Create validation map
            validation_map = {v['symbol']: v for v in validations}
            
            # Merge validation with original stocks
            validated_stocks = []
            for stock in selected_stocks:
                symbol = stock.get('symbol', '')
                validation = validation_map.get(symbol, {})
                
                validated_stock = stock.copy()
                validated_stock['chatgpt_score'] = validation.get('validation_score', 50)
                validated_stock['chatgpt_recommendation'] = validation.get('recommendation', 'Hold')
                validated_stock['chatgpt_reasoning'] = validation.get('reasoning', 'No reasoning provided')
                
                validated_stocks.append(validated_stock)
            
            return validated_stocks
        except Exception as e:
            print(f"Error parsing ChatGPT response: {e}")
            # Return stocks with default scores
            return self._default_validation(selected_stocks)
    
    def _default_validation(self, selected_stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Provide default validation when ChatGPT is not available.
        
        Args:
            selected_stocks (List[Dict[str, Any]]): Selected stocks
            
        Returns:
            List[Dict[str, Any]]: Stocks with default validation
        """
        validated_stocks = []
        
        for stock in selected_stocks:
            # Use momentum score as a proxy for validation if available
            momentum = stock.get('momentum', 0)
            # Convert momentum to a 0-100 score (simple linear mapping)
            chatgpt_score = max(0, min(100, 50 + (momentum * 1000)))
            
            validated_stock = stock.copy()
            validated_stock['chatgpt_score'] = chatgpt_score
            validated_stock['chatgpt_recommendation'] = self._get_recommendation_from_score(chatgpt_score)
            validated_stock['chatgpt_reasoning'] = "Default validation based on momentum score"
            
            validated_stocks.append(validated_stock)
        
        return validated_stocks
    
    def _get_recommendation_from_score(self, score: float) -> str:
        """
        Get recommendation text from score.
        
        Args:
            score (float): Validation score
            
        Returns:
            str: Recommendation
        """
        if score >= 86:
            return "Strong Buy"
        elif score >= 71:
            return "Buy"
        elif score >= 51:
            return "Hold"
        elif score >= 31:
            return "Sell"
        else:
            return "Strong Sell"
    
    def get_market_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get market sentiment analysis for symbols.
        
        Args:
            symbols (List[str]): List of symbols
            
        Returns:
            Dict[str, Any]: Sentiment analysis
        """
        if not self.api_key:
            return {"sentiment": "neutral", "confidence": 0.5}
        
        try:
            prompt = f"""
            Analyze the current market sentiment for these Indian stocks: {', '.join(symbols)}
            Consider recent news, earnings reports, and market trends.
            
            Respond in JSON format:
            {{
              "sentiment": "bullish/bearish/neutral",
              "confidence": 0.0-1.0,
              "key_factors": ["factor1", "factor2"],
              "outlook": "Brief market outlook"
            }}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial market sentiment analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            sentiment_result = response.choices[0].message.content
            return json.loads(sentiment_result)
        except Exception as e:
            print(f"Error getting market sentiment: {e}")
            return {"sentiment": "neutral", "confidence": 0.5}