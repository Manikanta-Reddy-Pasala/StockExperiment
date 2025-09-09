"""
Performance Tracking and Analytics Module
Tracks strategy performance over time and generates visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import logging
from datastore.database import get_database_manager

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks and analyzes strategy performance over time."""
    
    def __init__(self):
        """Initialize performance tracker."""
        self.db_manager = get_database_manager()
        self.performance_data = {}
        self.strategy_metrics = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def record_daily_performance(self, strategy_name: str, performance_data: Dict):
        """
        Record daily performance data for a strategy.
        
        Args:
            strategy_name (str): Name of the strategy
            performance_data (Dict): Performance metrics
        """
        try:
            date_key = datetime.utcnow().strftime('%Y-%m-%d')
            
            if strategy_name not in self.performance_data:
                self.performance_data[strategy_name] = {}
            
            self.performance_data[strategy_name][date_key] = {
                'timestamp': datetime.utcnow().isoformat(),
                'portfolio_value': performance_data.get('current_value', 0),
                'total_return': performance_data.get('total_return', 0),
                'return_percentage': performance_data.get('return_percentage', 0),
                'num_positions': performance_data.get('num_positions', 0),
                'positions': performance_data.get('positions', {})
            }
            
            logger.info(f"Recorded daily performance for {strategy_name}")
            
        except Exception as e:
            logger.error(f"Error recording daily performance for {strategy_name}: {e}")
    
    def calculate_strategy_metrics(self, strategy_name: str, lookback_days: int = 90) -> Dict:
        """
        Calculate comprehensive metrics for a strategy.
        
        Args:
            strategy_name (str): Name of the strategy
            lookback_days (int): Number of days to look back
            
        Returns:
            Dict: Strategy performance metrics
        """
        try:
            if strategy_name not in self.performance_data:
                return self._get_empty_metrics()
            
            # Get performance data for the lookback period
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)
            
            daily_data = []
            for date_str, data in self.performance_data[strategy_name].items():
                date = datetime.strptime(date_str, '%Y-%m-%d')
                if start_date <= date <= end_date:
                    daily_data.append(data)
            
            if not daily_data:
                return self._get_empty_metrics()
            
            # Calculate metrics
            portfolio_values = [d['portfolio_value'] for d in daily_data]
            returns = [d['return_percentage'] for d in daily_data]
            
            # Basic metrics
            total_return = returns[-1] if returns else 0
            avg_daily_return = np.mean(returns) if returns else 0
            volatility = np.std(returns) if len(returns) > 1 else 0
            
            # Risk metrics
            sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            
            # Win rate
            positive_returns = [r for r in returns if r > 0]
            win_rate = len(positive_returns) / len(returns) if returns else 0
            
            # Position metrics
            avg_positions = np.mean([d['num_positions'] for d in daily_data])
            
            metrics = {
                'strategy_name': strategy_name,
                'period_days': len(daily_data),
                'total_return': total_return,
                'avg_daily_return': avg_daily_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_positions': avg_positions,
                'best_day': max(returns) if returns else 0,
                'worst_day': min(returns) if returns else 0,
                'current_value': portfolio_values[-1] if portfolio_values else 0
            }
            
            self.strategy_metrics[strategy_name] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {strategy_name}: {e}")
            return self._get_empty_metrics()
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from peak."""
        if not portfolio_values:
            return 0
        
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd * 100  # Return as percentage
    
    def _get_empty_metrics(self) -> Dict:
        """Get empty metrics structure."""
        return {
            'strategy_name': 'unknown',
            'period_days': 0,
            'total_return': 0,
            'avg_daily_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'avg_positions': 0,
            'best_day': 0,
            'worst_day': 0,
            'current_value': 0
        }
    
    def compare_strategies(self, strategy_names: List[str] = None) -> Dict:
        """
        Compare performance of multiple strategies.
        
        Args:
            strategy_names (List[str]): List of strategies to compare, or None for all
            
        Returns:
            Dict: Strategy comparison results
        """
        try:
            if strategy_names is None:
                strategy_names = list(self.performance_data.keys())
            
            comparison = {
                'strategies': {},
                'rankings': {},
                'summary': {}
            }
            
            # Calculate metrics for each strategy
            for strategy_name in strategy_names:
                metrics = self.calculate_strategy_metrics(strategy_name)
                comparison['strategies'][strategy_name] = metrics
            
            # Create rankings
            comparison['rankings'] = {
                'by_return': self._rank_by_metric(strategy_names, 'total_return', reverse=True),
                'by_sharpe': self._rank_by_metric(strategy_names, 'sharpe_ratio', reverse=True),
                'by_win_rate': self._rank_by_metric(strategy_names, 'win_rate', reverse=True),
                'by_drawdown': self._rank_by_metric(strategy_names, 'max_drawdown', reverse=False)
            }
            
            # Summary statistics
            comparison['summary'] = {
                'best_return': max(comparison['strategies'].values(), key=lambda x: x['total_return']),
                'best_sharpe': max(comparison['strategies'].values(), key=lambda x: x['sharpe_ratio']),
                'lowest_drawdown': min(comparison['strategies'].values(), key=lambda x: x['max_drawdown']),
                'most_consistent': max(comparison['strategies'].values(), key=lambda x: x['win_rate'])
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return {'error': str(e)}
    
    def _rank_by_metric(self, strategy_names: List[str], metric: str, reverse: bool = True) -> List[Dict]:
        """Rank strategies by a specific metric."""
        rankings = []
        for strategy_name in strategy_names:
            if strategy_name in self.strategy_metrics:
                metrics = self.strategy_metrics[strategy_name]
                rankings.append({
                    'strategy': strategy_name,
                    'value': metrics.get(metric, 0),
                    'rank': 0  # Will be set after sorting
                })
        
        rankings.sort(key=lambda x: x['value'], reverse=reverse)
        
        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def generate_performance_charts(self, strategy_names: List[str] = None, 
                                  save_path: str = None) -> Dict[str, str]:
        """
        Generate performance visualization charts.
        
        Args:
            strategy_names (List[str]): Strategies to include in charts
            save_path (str): Path to save charts
            
        Returns:
            Dict[str, str]: Paths to generated chart files
        """
        try:
            if strategy_names is None:
                strategy_names = list(self.performance_data.keys())
            
            chart_paths = {}
            
            # 1. Portfolio Value Over Time
            chart_paths['portfolio_values'] = self._create_portfolio_value_chart(strategy_names, save_path)
            
            # 2. Returns Distribution
            chart_paths['returns_distribution'] = self._create_returns_distribution_chart(strategy_names, save_path)
            
            # 3. Risk-Return Scatter
            chart_paths['risk_return'] = self._create_risk_return_chart(strategy_names, save_path)
            
            # 4. Drawdown Chart
            chart_paths['drawdown'] = self._create_drawdown_chart(strategy_names, save_path)
            
            # 5. Strategy Comparison
            chart_paths['strategy_comparison'] = self._create_strategy_comparison_chart(strategy_names, save_path)
            
            logger.info(f"Generated {len(chart_paths)} performance charts")
            return chart_paths
            
        except Exception as e:
            logger.error(f"Error generating performance charts: {e}")
            return {'error': str(e)}
    
    def _create_portfolio_value_chart(self, strategy_names: List[str], save_path: str = None) -> str:
        """Create portfolio value over time chart."""
        plt.figure(figsize=(12, 8))
        
        for strategy_name in strategy_names:
            if strategy_name in self.performance_data:
                dates = []
                values = []
                
                for date_str, data in sorted(self.performance_data[strategy_name].items()):
                    dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
                    values.append(data['portfolio_value'])
                
                if dates and values:
                    plt.plot(dates, values, label=strategy_name, linewidth=2)
        
        plt.title('Portfolio Value Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value (INR)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f"portfolio_values_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = f"{save_path}/{filename}" if save_path else filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_returns_distribution_chart(self, strategy_names: List[str], save_path: str = None) -> str:
        """Create returns distribution chart."""
        plt.figure(figsize=(12, 8))
        
        for strategy_name in strategy_names:
            if strategy_name in self.performance_data:
                returns = []
                for data in self.performance_data[strategy_name].values():
                    returns.append(data['return_percentage'])
                
                if returns:
                    plt.hist(returns, alpha=0.7, label=strategy_name, bins=20)
        
        plt.title('Daily Returns Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Daily Return (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"returns_distribution_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = f"{save_path}/{filename}" if save_path else filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_risk_return_chart(self, strategy_names: List[str], save_path: str = None) -> str:
        """Create risk-return scatter plot."""
        plt.figure(figsize=(10, 8))
        
        returns = []
        volatilities = []
        labels = []
        
        for strategy_name in strategy_names:
            metrics = self.calculate_strategy_metrics(strategy_name)
            if metrics['period_days'] > 0:
                returns.append(metrics['total_return'])
                volatilities.append(metrics['volatility'])
                labels.append(strategy_name)
        
        if returns and volatilities:
            plt.scatter(volatilities, returns, s=100, alpha=0.7)
            
            # Add labels
            for i, label in enumerate(labels):
                plt.annotate(label, (volatilities[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points')
        
        plt.title('Risk vs Return Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Volatility (%)', fontsize=12)
        plt.ylabel('Total Return (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"risk_return_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = f"{save_path}/{filename}" if save_path else filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_drawdown_chart(self, strategy_names: List[str], save_path: str = None) -> str:
        """Create drawdown chart."""
        plt.figure(figsize=(12, 8))
        
        for strategy_name in strategy_names:
            if strategy_name in self.performance_data:
                dates = []
                drawdowns = []
                
                portfolio_values = []
                for date_str, data in sorted(self.performance_data[strategy_name].items()):
                    dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
                    portfolio_values.append(data['portfolio_value'])
                
                if portfolio_values:
                    peak = portfolio_values[0]
                    for value in portfolio_values:
                        if value > peak:
                            peak = value
                        drawdown = (peak - value) / peak * 100
                        drawdowns.append(drawdown)
                    
                    plt.plot(dates, drawdowns, label=strategy_name, linewidth=2)
        
        plt.title('Drawdown Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f"drawdown_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = f"{save_path}/{filename}" if save_path else filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_strategy_comparison_chart(self, strategy_names: List[str], save_path: str = None) -> str:
        """Create strategy comparison bar chart."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data
        metrics_data = {}
        for strategy_name in strategy_names:
            metrics = self.calculate_strategy_metrics(strategy_name)
            metrics_data[strategy_name] = metrics
        
        # 1. Total Returns
        strategies = list(metrics_data.keys())
        returns = [metrics_data[s]['total_return'] for s in strategies]
        axes[0, 0].bar(strategies, returns, color='skyblue')
        axes[0, 0].set_title('Total Returns (%)', fontweight='bold')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Sharpe Ratios
        sharpe_ratios = [metrics_data[s]['sharpe_ratio'] for s in strategies]
        axes[0, 1].bar(strategies, sharpe_ratios, color='lightgreen')
        axes[0, 1].set_title('Sharpe Ratios', fontweight='bold')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Win Rates
        win_rates = [metrics_data[s]['win_rate'] * 100 for s in strategies]
        axes[1, 0].bar(strategies, win_rates, color='orange')
        axes[1, 0].set_title('Win Rates (%)', fontweight='bold')
        axes[1, 0].set_ylabel('Win Rate (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Max Drawdowns
        max_drawdowns = [metrics_data[s]['max_drawdown'] for s in strategies]
        axes[1, 1].bar(strategies, max_drawdowns, color='lightcoral')
        axes[1, 1].set_title('Max Drawdowns (%)', fontweight='bold')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Strategy Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f"strategy_comparison_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = f"{save_path}/{filename}" if save_path else filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def generate_performance_report(self, strategy_names: List[str] = None, 
                                  lookback_days: int = 90) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            strategy_names (List[str]): Strategies to include in report
            lookback_days (int): Lookback period in days
            
        Returns:
            Dict: Comprehensive performance report
        """
        try:
            if strategy_names is None:
                strategy_names = list(self.performance_data.keys())
            
            # Calculate metrics for all strategies
            strategy_metrics = {}
            for strategy_name in strategy_names:
                strategy_metrics[strategy_name] = self.calculate_strategy_metrics(strategy_name, lookback_days)
            
            # Compare strategies
            comparison = self.compare_strategies(strategy_names)
            
            # Generate charts
            chart_paths = self.generate_performance_charts(strategy_names)
            
            # Create report
            report = {
                'report_id': f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'generated_at': datetime.utcnow().isoformat(),
                'period_days': lookback_days,
                'strategies_analyzed': strategy_names,
                'strategy_metrics': strategy_metrics,
                'strategy_comparison': comparison,
                'chart_paths': chart_paths,
                'summary': {
                    'best_performing_strategy': comparison['summary']['best_return']['strategy_name'],
                    'most_consistent_strategy': comparison['summary']['most_consistent']['strategy_name'],
                    'lowest_risk_strategy': comparison['summary']['lowest_drawdown']['strategy_name'],
                    'total_strategies': len(strategy_names)
                }
            }
            
            logger.info(f"Generated performance report for {len(strategy_names)} strategies")
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def export_performance_data(self, strategy_names: List[str] = None, 
                              format: str = 'json') -> str:
        """
        Export performance data to file.
        
        Args:
            strategy_names (List[str]): Strategies to export
            format (str): Export format ('json', 'csv')
            
        Returns:
            str: Path to exported file
        """
        try:
            if strategy_names is None:
                strategy_names = list(self.performance_data.keys())
            
            export_data = {}
            for strategy_name in strategy_names:
                if strategy_name in self.performance_data:
                    export_data[strategy_name] = self.performance_data[strategy_name]
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            
            if format == 'json':
                filename = f"performance_data_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format == 'csv':
                # Convert to DataFrame and export
                all_data = []
                for strategy_name, daily_data in export_data.items():
                    for date_str, data in daily_data.items():
                        row = {
                            'strategy': strategy_name,
                            'date': date_str,
                            **data
                        }
                        all_data.append(row)
                
                df = pd.DataFrame(all_data)
                filename = f"performance_data_{timestamp}.csv"
                df.to_csv(filename, index=False)
            
            logger.info(f"Exported performance data to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting performance data: {e}")
            return ""


if __name__ == "__main__":
    # Test the performance tracker
    tracker = PerformanceTracker()
    
    # Mock some performance data
    mock_data = {
        'portfolio_value': 105000,
        'total_return': 5000,
        'return_percentage': 5.0,
        'num_positions': 5,
        'positions': {}
    }
    
    tracker.record_daily_performance('momentum', mock_data)
    tracker.record_daily_performance('value', mock_data)
    
    # Generate report
    report = tracker.generate_performance_report(['momentum', 'value'])
    print(f"Performance report generated: {report.get('report_id', 'unknown')}")
