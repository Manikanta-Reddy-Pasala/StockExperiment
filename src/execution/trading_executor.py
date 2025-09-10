"""
Main Trading Execution Engine
Orchestrates the complete trading workflow: screening, strategy application, dry run, and evaluation
"""
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from datastore.database import get_database_manager
from screening.stock_screener import StockScreener
from strategies.strategy_engine import StrategyEngine
from portfolio.dry_run_manager import DryRunManager
from analysis.chatgpt_analyzer import ChatGPTAnalyzer
import json

logger = logging.getLogger(__name__)


class TradingExecutor:
    """Main trading execution engine."""
    
    def __init__(self, user_id: int = None):
        """
        Initialize trading executor.
        
        Args:
            user_id (int): User ID for multi-user support
        """
        self.user_id = user_id
        self.db_manager = get_database_manager()
        
        # Initialize components
        self.screener = StockScreener()
        self.strategy_engine = StrategyEngine()
        self.dry_run_manager = DryRunManager()
        self.chatgpt_analyzer = ChatGPTAnalyzer()
        
        # Execution state
        self.is_running = False
        self.last_execution = None
        self.execution_history = []
        
        # Performance tracking
        self.performance_data = {
            'daily_returns': {},
            'strategy_performance': {},
            'portfolio_values': {},
            'risk_metrics': {}
        }
    
    def run_complete_workflow(self) -> Dict:
        """
        Run the complete trading workflow.
        
        Returns:
            Dict: Execution results and analysis
        """
        try:
            logger.info("Starting complete trading workflow")
            start_time = datetime.utcnow()
            
            # Step 1: Stock Screening
            logger.info("Step 1: Running stock screening")
            screened_stocks = self.screener.run_daily_screening()
            
            if not screened_stocks:
                logger.warning("No stocks passed screening criteria")
                return self._create_empty_result("No stocks passed screening")
            
            # Step 2: Strategy Application
            logger.info("Step 2: Applying trading strategies")
            strategy_results = self.strategy_engine.run_strategies(screened_stocks)
            
            # Step 3: ChatGPT Analysis
            logger.info("Step 3: Getting AI analysis")
            ai_analysis = self._get_ai_analysis(strategy_results)
            
            # Step 4: Dry Run Portfolio Creation
            logger.info("Step 4: Creating dry run portfolios")
            dry_run_results = self._create_dry_run_portfolios(strategy_results)
            
            # Step 5: Performance Evaluation
            logger.info("Step 5: Evaluating performance")
            performance_evaluation = self._evaluate_performance(strategy_results, dry_run_results)
            
            # Step 6: Store Results
            logger.info("Step 6: Storing execution results")
            self._store_execution_results(screened_stocks, strategy_results, ai_analysis, dry_run_results)
            
            # Create comprehensive result
            execution_result = {
                'execution_id': f"exec_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': start_time.isoformat(),
                'duration_seconds': (datetime.utcnow() - start_time).total_seconds(),
                'screened_stocks': {
                    'count': len(screened_stocks),
                    'stocks': screened_stocks
                },
                'strategy_results': strategy_results,
                'ai_analysis': ai_analysis,
                'dry_run_results': dry_run_results,
                'performance_evaluation': performance_evaluation,
                'recommendations': self._generate_recommendations(strategy_results, ai_analysis)
            }
            
            # Update execution state
            self.last_execution = execution_result
            self.execution_history.append(execution_result)
            
            logger.info(f"Trading workflow completed successfully in {execution_result['duration_seconds']:.2f} seconds")
            return execution_result
            
        except Exception as e:
            logger.error(f"Error in trading workflow: {e}")
            return self._create_error_result(str(e))
    
    def run_dry_run_only(self, strategy_name: str = None) -> Dict:
        """
        Run dry run mode only (for testing strategies).
        
        Args:
            strategy_name (str): Specific strategy to test, or None for all
            
        Returns:
            Dict: Dry run results
        """
        try:
            logger.info(f"Running dry run mode for strategy: {strategy_name or 'all'}")
            
            # Get previously screened stocks
            screened_stocks = self._get_latest_screened_stocks()
            if not screened_stocks:
                logger.warning("No screened stocks available for dry run")
                return self._create_empty_result("No screened stocks available")
            
            # Apply strategies
            if strategy_name:
                strategy = self.strategy_engine.strategies.get(strategy_name)
                if strategy:
                    suggested_stocks = strategy.select_stocks(screened_stocks)
                    strategy_results = {strategy_name: suggested_stocks}
                else:
                    return self._create_error_result(f"Strategy {strategy_name} not found")
            else:
                strategy_results = self.strategy_engine.run_strategies(screened_stocks)
            
            # Create dry run portfolios
            dry_run_results = self._create_dry_run_portfolios(strategy_results)
            
            # Get AI analysis
            ai_analysis = self._get_ai_analysis(strategy_results)
            
            return {
                'mode': 'dry_run',
                'timestamp': datetime.utcnow().isoformat(),
                'strategy_results': strategy_results,
                'dry_run_results': dry_run_results,
                'ai_analysis': ai_analysis,
                'recommendations': self._generate_recommendations(strategy_results, ai_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error in dry run mode: {e}")
            return self._create_error_result(str(e))
    
    def start_scheduled_execution(self, interval_hours: int = 1):
        """
        Start scheduled execution of trading workflow.
        
        Args:
            interval_hours (int): Execution interval in hours
        """
        if self.is_running:
            logger.warning("Scheduled execution already running")
            return
        
        self.is_running = True
        logger.info(f"Starting scheduled execution every {interval_hours} hours")
        
        # Schedule the workflow
        schedule.every(interval_hours).hours.do(self._scheduled_workflow)
        
        # Start the scheduler in a separate thread
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Scheduled execution started")
    
    def stop_scheduled_execution(self):
        """Stop scheduled execution."""
        self.is_running = False
        schedule.clear()
        logger.info("Scheduled execution stopped")
    
    def _scheduled_workflow(self):
        """Execute the trading workflow on schedule."""
        try:
            logger.info("Executing scheduled trading workflow")
            result = self.run_complete_workflow()
            
            # Log key metrics
            if result.get('screened_stocks', {}).get('count', 0) > 0:
                logger.info(f"Scheduled execution completed: {result['screened_stocks']['count']} stocks screened")
            else:
                logger.warning("Scheduled execution completed with no stocks selected")
                
        except Exception as e:
            logger.error(f"Error in scheduled workflow: {e}")
    
    def _get_ai_analysis(self, strategy_results: Dict[str, List[Dict]]) -> Dict:
        """Get AI analysis for strategy results."""
        try:
            # Analyze each strategy's portfolio
            portfolio_analysis = {}
            for strategy_name, suggested_stocks in strategy_results.items():
                if suggested_stocks:
                    analysis = self.chatgpt_analyzer.analyze_portfolio(suggested_stocks, strategy_name)
                    portfolio_analysis[strategy_name] = analysis
            
            # Compare strategies
            strategy_comparison = self.chatgpt_analyzer.compare_strategies(strategy_results)
            
            return {
                'portfolio_analysis': portfolio_analysis,
                'strategy_comparison': strategy_comparison,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting AI analysis: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    def _create_dry_run_portfolios(self, strategy_results: Dict[str, List[Dict]]) -> Dict:
        """Create dry run portfolios for each strategy."""
        try:
            dry_run_results = {}
            
            for strategy_name, suggested_stocks in strategy_results.items():
                if suggested_stocks:
                    # Create portfolio for this strategy
                    success = self.dry_run_manager.execute_strategy(
                        strategy_name, 
                        suggested_stocks, 
                        allocation_strategy='equal_weight'
                    )
                    
                    if success:
                        # Get portfolio performance
                        performance = self.dry_run_manager.get_strategy_performance(strategy_name)
                        dry_run_results[strategy_name] = {
                            'portfolio_created': True,
                            'num_stocks': len(suggested_stocks),
                            'performance': performance,
                            'suggested_stocks': suggested_stocks
                        }
                    else:
                        dry_run_results[strategy_name] = {
                            'portfolio_created': False,
                            'error': 'Failed to create portfolio'
                        }
            
            return dry_run_results
            
        except Exception as e:
            logger.error(f"Error creating dry run portfolios: {e}")
            return {'error': str(e)}
    
    def _evaluate_performance(self, strategy_results: Dict, dry_run_results: Dict) -> Dict:
        """Evaluate performance of strategies and portfolios."""
        try:
            evaluation = {
                'timestamp': datetime.utcnow().isoformat(),
                'strategy_metrics': {},
                'portfolio_metrics': {},
                'risk_analysis': {},
                'performance_ranking': []
            }
            
            # Evaluate each strategy
            for strategy_name, suggested_stocks in strategy_results.items():
                if suggested_stocks:
                    # Strategy metrics
                    strategy_metrics = self.strategy_engine.get_strategy_performance_metrics(
                        strategy_name, suggested_stocks
                    )
                    evaluation['strategy_metrics'][strategy_name] = strategy_metrics
                    
                    # Portfolio metrics
                    if strategy_name in dry_run_results and dry_run_results[strategy_name].get('portfolio_created'):
                        portfolio_performance = dry_run_results[strategy_name]['performance']
                        evaluation['portfolio_metrics'][strategy_name] = portfolio_performance
                        
                        # Risk analysis
                        risk_metrics = self._calculate_risk_metrics(suggested_stocks, portfolio_performance)
                        evaluation['risk_analysis'][strategy_name] = risk_metrics
            
            # Create performance ranking
            evaluation['performance_ranking'] = self._create_performance_ranking(evaluation)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    def _calculate_risk_metrics(self, suggested_stocks: List[Dict], portfolio_performance: Dict) -> Dict:
        """Calculate risk metrics for a portfolio."""
        try:
            # Calculate sector concentration
            sector_distribution = {}
            for stock in suggested_stocks:
                sector = stock.get('sector', 'Unknown')
                sector_distribution[sector] = sector_distribution.get(sector, 0) + 1
            
            # Calculate concentration risk
            max_sector_weight = max(sector_distribution.values()) / len(suggested_stocks) if suggested_stocks else 0
            concentration_risk = 'High' if max_sector_weight > 0.4 else 'Medium' if max_sector_weight > 0.2 else 'Low'
            
            # Calculate market cap distribution
            market_cap_distribution = {'small_cap': 0, 'mid_cap': 0, 'large_cap': 0}
            for stock in suggested_stocks:
                market_cap = stock.get('market_cap', 0)
                if market_cap < 10000:
                    market_cap_distribution['small_cap'] += 1
                elif market_cap < 20000:
                    market_cap_distribution['mid_cap'] += 1
                else:
                    market_cap_distribution['large_cap'] += 1
            
            return {
                'sector_concentration': sector_distribution,
                'concentration_risk': concentration_risk,
                'market_cap_distribution': market_cap_distribution,
                'num_positions': len(suggested_stocks),
                'diversification_score': 1 - max_sector_weight
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'error': str(e)}
    
    def _create_performance_ranking(self, evaluation: Dict) -> List[Dict]:
        """Create performance ranking of strategies."""
        try:
            ranking = []
            
            for strategy_name in evaluation['strategy_metrics'].keys():
                strategy_metrics = evaluation['strategy_metrics'][strategy_name]
                portfolio_metrics = evaluation['portfolio_metrics'].get(strategy_name, {})
                
                # Calculate composite score
                score = 0
                score += strategy_metrics.get('num_stocks', 0) * 0.2  # Stock count
                score += strategy_metrics.get('avg_score', 0) * 0.3  # Strategy score
                
                if portfolio_metrics:
                    return_pct = portfolio_metrics.get('return_percentage', 0)
                    score += max(0, return_pct) * 0.5  # Return performance
                
                ranking.append({
                    'strategy': strategy_name,
                    'score': score,
                    'num_stocks': strategy_metrics.get('num_stocks', 0),
                    'avg_strategy_score': strategy_metrics.get('avg_score', 0),
                    'return_percentage': portfolio_metrics.get('return_percentage', 0)
                })
            
            # Sort by score (descending)
            ranking.sort(key=lambda x: x['score'], reverse=True)
            
            return ranking
            
        except Exception as e:
            logger.error(f"Error creating performance ranking: {e}")
            return []
    
    def _generate_recommendations(self, strategy_results: Dict, ai_analysis: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        try:
            # Count total stocks selected
            total_stocks = sum(len(stocks) for stocks in strategy_results.values())
            
            if total_stocks == 0:
                recommendations.append("No stocks selected by any strategy. Consider relaxing screening criteria.")
                return recommendations
            
            # Strategy-specific recommendations
            for strategy_name, suggested_stocks in strategy_results.items():
                if suggested_stocks:
                    recommendations.append(f"{strategy_name.title()} strategy suggested {len(suggested_stocks)} stocks")
                    
                    # Check for concentration
                    sectors = [s.get('sector', 'Unknown') for s in suggested_stocks]
                    if len(set(sectors)) < len(suggested_stocks) * 0.5:
                        recommendations.append(f"Consider diversifying {strategy_name} strategy across more sectors")
            
            # AI-based recommendations
            if 'strategy_comparison' in ai_analysis:
                comparison = ai_analysis['strategy_comparison']
                if 'best_strategy' in comparison:
                    recommendations.append(f"AI recommends {comparison['best_strategy']} strategy as best performing")
            
            # Performance recommendations
            if len(strategy_results) > 1:
                recommendations.append("Consider using multiple strategies for better diversification")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def _get_latest_screened_stocks(self) -> List[Dict]:
        """Get the latest screened stocks from database."""
        try:
            # This would typically query the database for the most recent screening results
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"Error getting latest screened stocks: {e}")
            return []
    
    def _store_execution_results(self, screened_stocks: List[Dict], strategy_results: Dict, 
                                ai_analysis: Dict, dry_run_results: Dict):
        """Store execution results in database."""
        try:
            # This would store results in the database
            # For now, just log the results
            logger.info(f"Storing execution results: {len(screened_stocks)} screened stocks, {len(strategy_results)} strategies")
        except Exception as e:
            logger.error(f"Error storing execution results: {e}")
    
    def _create_empty_result(self, message: str) -> Dict:
        """Create empty result structure."""
        return {
            'execution_id': f"exec_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'empty',
            'message': message,
            'screened_stocks': {'count': 0, 'stocks': []},
            'strategy_results': {},
            'ai_analysis': {},
            'dry_run_results': {},
            'performance_evaluation': {},
            'recommendations': [message]
        }
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result structure."""
        return {
            'execution_id': f"exec_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'error',
            'error': error_message,
            'screened_stocks': {'count': 0, 'stocks': []},
            'strategy_results': {},
            'ai_analysis': {},
            'dry_run_results': {},
            'performance_evaluation': {},
            'recommendations': [f"Error: {error_message}"]
        }
    
    def get_execution_status(self) -> Dict:
        """Get current execution status."""
        return {
            'is_running': self.is_running,
            'last_execution': self.last_execution,
            'execution_count': len(self.execution_history),
            'performance_data': self.performance_data
        }
    
    def cleanup_dry_run_portfolios(self):
        """Clean up all dry run portfolios."""
        self.dry_run_manager.cleanup_all_portfolios()
        logger.info("Cleaned up all dry run portfolios")


if __name__ == "__main__":
    # Test the trading executor
    executor = TradingExecutor()
    
    # Run complete workflow
    result = executor.run_complete_workflow()
    print(f"Execution result: {result.get('status', 'unknown')}")
    
    # Test dry run mode
    dry_run_result = executor.run_dry_run_only('momentum')
    print(f"Dry run result: {dry_run_result.get('mode', 'unknown')}")
