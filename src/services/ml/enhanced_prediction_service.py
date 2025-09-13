query(MLPrediction).filter(
                    MLPrediction.is_validated == True
                )
                
                if symbol:
                    query = query.join(Stock).filter(Stock.symbol == symbol)
                
                predictions = query.all()
                
                if not predictions:
                    return {"error": "No validated predictions found"}
                
                errors = [p.prediction_error for p in predictions if p.prediction_error is not None]
                
                performance = {
                    "total_validated_predictions": len(predictions),
                    "symbol": symbol or "All stocks",
                    "performance_metrics": {}
                }
                
                if errors:
                    performance["performance_metrics"] = {
                        "mean_absolute_error": np.mean(errors),
                        "median_absolute_error": np.median(errors),
                        "std_error": np.std(errors),
                        "min_error": np.min(errors),
                        "max_error": np.max(errors),
                        "accuracy_within_5pct": len([e for e in errors if e <= 5]) / len(errors) * 100,
                        "accuracy_within_10pct": len([e for e in errors if e <= 10]) / len(errors) * 100,
                        "accuracy_within_20pct": len([e for e in errors if e <= 20]) / len(errors) * 100
                    }
                
                # Signal accuracy
                correct_signals = 0
                total_signals = 0
                
                for pred in predictions:
                    if pred.actual_price and pred.current_price:
                        actual_change = (pred.actual_price - pred.current_price) / pred.current_price
                        predicted_signal = pred.signal
                        
                        if predicted_signal == "BUY" and actual_change > 0.02:  # > 2% gain
                            correct_signals += 1
                        elif predicted_signal == "SELL" and actual_change < -0.02:  # > 2% loss
                            correct_signals += 1
                        elif predicted_signal == "HOLD" and -0.02 <= actual_change <= 0.02:
                            correct_signals += 1
                        
                        total_signals += 1
                
                if total_signals > 0:
                    performance["signal_accuracy"] = correct_signals / total_signals * 100
                
                return performance
                
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {"error": str(e)}
    
    def _get_basic_prediction(self, symbol: str, user_id: int) -> Dict:
        """Get basic prediction using existing prediction service."""
        try:
            # Use the existing get_prediction function from prediction_service.py
            from . import prediction_service
            return prediction_service.get_prediction(symbol, user_id)
        except Exception as e:
            logger.warning(f"Error getting basic prediction for {symbol}: {e}")
            return self._get_fallback_prediction(symbol)
    
    def _enhance_prediction(self, symbol: str, basic_prediction: Dict, user_id: int) -> Dict:
        """Enhance basic prediction with additional metrics and analysis."""
        try:
            enhanced = basic_prediction.copy()
            
            # Add confidence metrics
            enhanced["confidence"] = self._calculate_confidence(basic_prediction)
            enhanced["risk_score"] = self._calculate_risk_score(symbol, basic_prediction)
            enhanced["volatility_score"] = self._calculate_volatility_score(symbol, user_id)
            
            # Add model agreement metrics
            model_predictions = [
                basic_prediction.get("rf_predicted_price", 0),
                basic_prediction.get("xgb_predicted_price", 0),
                basic_prediction.get("lstm_predicted_price", 0)
            ]
            model_predictions = [p for p in model_predictions if p > 0]
            
            if len(model_predictions) > 1:
                enhanced["model_agreement"] = self._calculate_model_agreement(model_predictions)
            else:
                enhanced["model_agreement"] = 0.5
            
            # Add enhanced signal with strength
            enhanced["signal_strength"] = self._calculate_signal_strength(enhanced)
            enhanced["investment_recommendation"] = self._generate_investment_recommendation(enhanced)
            
            # Add time horizon recommendations
            enhanced["time_horizon"] = self._recommend_time_horizon(enhanced)
            
            # Add risk management suggestions
            enhanced["risk_management"] = self._generate_risk_management(symbol, enhanced)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing prediction for {symbol}: {e}")
            return basic_prediction
    
    def _calculate_confidence(self, prediction: Dict) -> float:
        """Calculate prediction confidence based on multiple factors."""
        try:
            confidence = 0.5  # Base confidence
            
            # Factor 1: Magnitude of predicted change
            predicted_change = abs(prediction.get("predicted_change_percent", 0))
            if predicted_change > 15:
                confidence += 0.3
            elif predicted_change > 10:
                confidence += 0.2
            elif predicted_change > 5:
                confidence += 0.1
            
            # Factor 2: Model agreement (if available)
            model_agreement = prediction.get("model_agreement", 0.5)
            confidence += (model_agreement - 0.5) * 0.4
            
            # Factor 3: Signal consistency
            signal = prediction.get("signal", "HOLD")
            predicted_change = prediction.get("predicted_change_percent", 0)
            
            if (signal == "BUY" and predicted_change > 2) or \
               (signal == "SELL" and predicted_change < -2) or \
               (signal == "HOLD" and -2 <= predicted_change <= 2):
                confidence += 0.1
            else:
                confidence -= 0.1
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_risk_score(self, symbol: str, prediction: Dict) -> float:
        """Calculate risk score for the prediction."""
        try:
            risk_score = 5.0  # Base risk score (1-10 scale)
            
            # Factor in volatility
            volatility = prediction.get("volatility_score", 0.5)
            risk_score += volatility * 3
            
            # Factor in predicted change magnitude
            predicted_change = abs(prediction.get("predicted_change_percent", 0))
            if predicted_change > 20:
                risk_score += 2
            elif predicted_change > 10:
                risk_score += 1
            
            # Factor in signal type
            signal = prediction.get("signal", "HOLD")
            if signal == "SELL":
                risk_score += 1
            elif signal == "BUY":
                predicted_change = prediction.get("predicted_change_percent", 0)
                if predicted_change > 15:
                    risk_score += 1.5  # High return expectations = higher risk
            
            return min(max(risk_score, 1.0), 10.0)
            
        except Exception as e:
            logger.warning(f"Error calculating risk score: {e}")
            return 5.0
    
    def _calculate_volatility_score(self, symbol: str, user_id: int) -> float:
        """Calculate volatility score based on historical price data."""
        try:
            # Get historical data
            df = get_stock_data(symbol, period="3m", user_id=user_id)  # 3 months of data
            
            if df is None or len(df) < 20:
                return 0.5  # Default medium volatility
            
            # Calculate daily returns
            df['daily_return'] = df['Close'].pct_change()
            
            # Calculate volatility (standard deviation of daily returns)
            volatility = df['daily_return'].std()
            
            # Normalize volatility to 0-1 scale
            # Typical stock volatility ranges from 0.01 to 0.05 (1% to 5% daily)
            normalized_volatility = min(volatility / 0.05, 1.0)
            
            return normalized_volatility
            
        except Exception as e:
            logger.warning(f"Error calculating volatility for {symbol}: {e}")
            return 0.5
    
    def _calculate_model_agreement(self, model_predictions: List[float]) -> float:
        """Calculate how much the different models agree."""
        try:
            if len(model_predictions) < 2:
                return 0.5
            
            # Calculate coefficient of variation (std/mean)
            mean_pred = np.mean(model_predictions)
            std_pred = np.std(model_predictions)
            
            if mean_pred == 0:
                return 0.5
            
            cv = std_pred / abs(mean_pred)
            
            # Convert to agreement score (lower CV = higher agreement)
            # CV of 0.1 (10%) = high agreement, CV of 0.3 (30%) = low agreement
            agreement = max(0, 1 - (cv / 0.3))
            
            return min(agreement, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating model agreement: {e}")
            return 0.5
    
    def _calculate_signal_strength(self, prediction: Dict) -> float:
        """Calculate strength of the trading signal."""
        try:
            predicted_change = abs(prediction.get("predicted_change_percent", 0))
            confidence = prediction.get("confidence", 0.5)
            model_agreement = prediction.get("model_agreement", 0.5)
            
            # Combine factors
            strength = (predicted_change / 20) * confidence * model_agreement
            
            return min(strength, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating signal strength: {e}")
            return 0.5
    
    def _generate_investment_recommendation(self, prediction: Dict) -> Dict:
        """Generate detailed investment recommendation."""
        try:
            signal = prediction.get("signal", "HOLD")
            confidence = prediction.get("confidence", 0.5)
            signal_strength = prediction.get("signal_strength", 0.5)
            risk_score = prediction.get("risk_score", 5.0)
            predicted_change = prediction.get("predicted_change_percent", 0)
            
            recommendation = {
                "action": signal,
                "strength": "Strong" if signal_strength > 0.7 else "Medium" if signal_strength > 0.4 else "Weak",
                "confidence_level": "High" if confidence > 0.75 else "Medium" if confidence > 0.5 else "Low",
                "risk_level": "High" if risk_score > 7 else "Medium" if risk_score > 4 else "Low"
            }
            
            # Generate position sizing recommendation
            if signal == "BUY":
                if confidence > 0.8 and signal_strength > 0.7:
                    recommendation["position_size"] = "Large (5-8% of portfolio)"
                elif confidence > 0.6 and signal_strength > 0.5:
                    recommendation["position_size"] = "Medium (3-5% of portfolio)"
                else:
                    recommendation["position_size"] = "Small (1-3% of portfolio)"
            else:
                recommendation["position_size"] = "N/A"
            
            # Generate rationale
            if signal == "BUY":
                recommendation["rationale"] = f"Models predict {predicted_change:.1f}% upside with {confidence*100:.0f}% confidence"
            elif signal == "SELL":
                recommendation["rationale"] = f"Models predict {abs(predicted_change):.1f}% downside with {confidence*100:.0f}% confidence"
            else:
                recommendation["rationale"] = f"Neutral outlook with {abs(predicted_change):.1f}% expected movement"
            
            return recommendation
            
        except Exception as e:
            logger.warning(f"Error generating investment recommendation: {e}")
            return {"action": "HOLD", "strength": "Weak", "confidence_level": "Low"}
    
    def _recommend_time_horizon(self, prediction: Dict) -> Dict:
        """Recommend appropriate time horizon for the prediction."""
        try:
            predicted_change = abs(prediction.get("predicted_change_percent", 0))
            volatility = prediction.get("volatility_score", 0.5)
            signal_strength = prediction.get("signal_strength", 0.5)
            
            # Base recommendation on prediction characteristics
            if predicted_change > 15 and signal_strength > 0.6:
                horizon = "Short-term (1-3 months)"
                rationale = "Strong predicted movement suggests short-term opportunity"
            elif predicted_change > 8 and volatility < 0.6:
                horizon = "Medium-term (3-6 months)"
                rationale = "Moderate predicted movement with stable volatility"
            elif predicted_change > 3:
                horizon = "Long-term (6-12 months)"
                rationale = "Conservative prediction suitable for longer holding period"
            else:
                horizon = "Hold or avoid"
                rationale = "Minimal predicted movement"
            
            return {
                "recommended_horizon": horizon,
                "rationale": rationale,
                "monitoring_frequency": "Weekly" if "Short-term" in horizon else "Bi-weekly" if "Medium-term" in horizon else "Monthly"
            }
            
        except Exception as e:
            logger.warning(f"Error recommending time horizon: {e}")
            return {"recommended_horizon": "Medium-term (3-6 months)", "rationale": "Default recommendation"}
    
    def _generate_risk_management(self, symbol: str, prediction: Dict) -> Dict:
        """Generate risk management recommendations."""
        try:
            current_price = prediction.get("last_close_price", 0)
            predicted_change = prediction.get("predicted_change_percent", 0)
            risk_score = prediction.get("risk_score", 5.0)
            volatility = prediction.get("volatility_score", 0.5)
            
            risk_mgmt = {}
            
            if current_price > 0:
                # Stop loss recommendation
                if risk_score > 7:
                    stop_loss_pct = 0.08  # 8% stop loss for high risk
                elif risk_score > 4:
                    stop_loss_pct = 0.12  # 12% stop loss for medium risk
                else:
                    stop_loss_pct = 0.15  # 15% stop loss for low risk
                
                risk_mgmt["stop_loss"] = {
                    "percentage": stop_loss_pct * 100,
                    "price": current_price * (1 - stop_loss_pct),
                    "rationale": f"Based on {risk_score:.1f}/10 risk score"
                }
                
                # Target price recommendation
                if predicted_change > 0:
                    target_multiplier = 2.0 if risk_score > 6 else 1.5  # Risk-reward ratio
                    target_price = current_price * (1 + (predicted_change/100) * target_multiplier)
                    
                    risk_mgmt["target_price"] = {
                        "price": target_price,
                        "upside_potential": ((target_price - current_price) / current_price) * 100,
                        "risk_reward_ratio": target_multiplier
                    }
            
            # Position sizing recommendation
            if risk_score > 7:
                max_position = 3  # Max 3% for high risk
            elif risk_score > 4:
                max_position = 5  # Max 5% for medium risk
            else:
                max_position = 8  # Max 8% for low risk
            
            risk_mgmt["position_sizing"] = {
                "max_portfolio_percentage": max_position,
                "rationale": f"Based on {risk_score:.1f}/10 risk assessment"
            }
            
            # Diversification advice
            risk_mgmt["diversification"] = {
                "sector_limit": "Max 20% in any single sector",
                "correlation_advice": "Avoid highly correlated positions",
                "rebalancing": "Review monthly for high volatility stocks" if volatility > 0.6 else "Review quarterly"
            }
            
            return risk_mgmt
            
        except Exception as e:
            logger.warning(f"Error generating risk management: {e}")
            return {"stop_loss": {"percentage": 10, "rationale": "Default recommendation"}}
    
    def _apply_strategy_adjustments(self, prediction: Dict, strategy_type: str) -> Dict:
        """Apply strategy-specific adjustments to predictions."""
        try:
            adjusted = prediction.copy()
            
            if strategy_type == "high_risk":
                # For high risk strategy, boost small cap predictions and reduce large cap
                # This would require market cap information which we'd get from the stock data
                predicted_change = prediction.get("predicted_change_percent", 0)
                
                # Increase confidence for higher predicted returns (high risk seeks high returns)
                if predicted_change > 10:
                    adjusted["confidence"] = min(adjusted.get("confidence", 0.5) * 1.2, 1.0)
                
                # Adjust signal threshold (more aggressive)
                if predicted_change > 8 and prediction.get("signal") == "HOLD":
                    adjusted["signal"] = "BUY"
                elif predicted_change < -5 and prediction.get("signal") == "HOLD":
                    adjusted["signal"] = "SELL"
                    
            elif strategy_type == "default_risk":
                # For default risk strategy, be more conservative
                predicted_change = prediction.get("predicted_change_percent", 0)
                
                # Require higher confidence for buy signals
                if prediction.get("signal") == "BUY" and adjusted.get("confidence", 0.5) < 0.65:
                    adjusted["signal"] = "HOLD"
                
                # Reduce confidence for very high predictions (seems too good to be true)
                if predicted_change > 20:
                    adjusted["confidence"] = min(adjusted.get("confidence", 0.5) * 0.8, 1.0)
            
            return adjusted
            
        except Exception as e:
            logger.warning(f"Error applying strategy adjustments: {e}")
            return prediction
    
    def _save_prediction_to_db(self, symbol: str, prediction: Dict, user_id: int):
        """Save prediction to database for tracking and validation."""
        try:
            with self.db_manager.get_session() as session:
                # Get or create stock record
                stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                
                if not stock:
                    logger.warning(f"Stock {symbol} not found in database, skipping prediction save")
                    return
                
                # Create ML prediction record
                ml_prediction = MLPrediction(
                    stock_id=stock.id,
                    user_id=user_id,
                    current_price=prediction.get("last_close_price", 0),
                    rf_predicted_price=prediction.get("rf_predicted_price", 0),
                    xgb_predicted_price=prediction.get("xgb_predicted_price", 0),
                    lstm_predicted_price=prediction.get("lstm_predicted_price", 0),
                    ensemble_predicted_price=prediction.get("final_predicted_price", 0),
                    prediction_confidence=prediction.get("confidence", 0.5),
                    prediction_std=0,  # Could be calculated from model variance
                    signal=prediction.get("signal", "HOLD"),
                    signal_strength=prediction.get("signal_strength", 0.5),
                    expected_return=prediction.get("predicted_change_percent", 0),
                    risk_reward_ratio=prediction.get("risk_score", 5.0) / 10.0,
                    model_version="enhanced_v1.0",
                    features_used=json.dumps(["technical", "fundamental", "sentiment"]),
                    training_data_period="1y"
                )
                
                session.add(ml_prediction)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error saving prediction to database: {e}")
    
    def _get_fallback_prediction(self, symbol: str) -> Dict:
        """Generate fallback prediction when models fail."""
        import random
        
        # Generate conservative fallback prediction
        predicted_change = random.uniform(-3, 8)  # Slight positive bias
        
        return {
            "symbol": symbol,
            "rf_predicted_price": 0,
            "xgb_predicted_price": 0,
            "lstm_predicted_price": 0,
            "final_predicted_price": 0,
            "last_close_price": 0,
            "predicted_change_percent": predicted_change,
            "signal": "BUY" if predicted_change > 3 else "HOLD" if predicted_change > -2 else "SELL",
            "confidence": 0.4,  # Low confidence for fallback
            "model_agreement": 0.3,
            "risk_score": 6.0,
            "volatility_score": 0.5,
            "signal_strength": 0.3,
            "investment_recommendation": {
                "action": "HOLD",
                "strength": "Weak",
                "confidence_level": "Low"
            },
            "is_fallback": True
        }


def get_enhanced_prediction_service() -> EnhancedPredictionService:
    """Get enhanced prediction service instance."""
    return EnhancedPredictionService()


# Backward compatibility function
def get_prediction(symbol: str, user_id: int = 1) -> Dict:
    """Get prediction using enhanced service (backward compatible)."""
    service = get_enhanced_prediction_service()
    return service.get_enhanced_prediction(symbol, user_id)
