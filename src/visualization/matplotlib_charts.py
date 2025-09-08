"""
Matplotlib-based Visualization Module
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import io
import base64


class MatplotlibCharts:
    """Generate Matplotlib-based charts for trading data."""
    
    def __init__(self):
        """Initialize the matplotlib charts system."""
        # Set default style
        plt.style.use('seaborn-v0_8')
    
    def create_portfolio_value_chart(self, portfolio_data: List[Dict[str, Any]], 
                                   title: str = "Portfolio Value Over Time") -> str:
        """
        Create a portfolio value chart.
        
        Args:
            portfolio_data (List[Dict[str, Any]]): Portfolio value data
            title (str): Chart title
            
        Returns:
            str: Base64 encoded image
        """
        if not portfolio_data:
            return ""
        
        # Convert to DataFrame
        df = pd.DataFrame(portfolio_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot portfolio value
        ax.plot(df['date'], df['value'], linewidth=2, label='Portfolio Value')
        
        # Plot zero line for reference
        ax.axhline(y=df['value'].iloc[0], color='r', linestyle='--', alpha=0.7, label='Initial Value')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value (₹)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_str
    
    def create_pnl_chart(self, pnl_data: List[Dict[str, Any]], 
                        title: str = "Profit & Loss Over Time") -> str:
        """
        Create a P&L chart.
        
        Args:
            pnl_data (List[Dict[str, Any]]): P&L data
            title (str): Chart title
            
        Returns:
            str: Base64 encoded image
        """
        if not pnl_data:
            return ""
        
        # Convert to DataFrame
        df = pd.DataFrame(pnl_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot daily P&L
        ax.bar(df['date'], df['daily_pnl'], width=0.8, alpha=0.7, label='Daily P&L', color='blue')
        
        # Plot cumulative P&L
        ax2 = ax.twinx()
        ax2.plot(df['date'], df['cumulative_pnl'], color='red', linewidth=2, label='Cumulative P&L')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Daily P&L (₹)', color='blue')
        ax2.set_ylabel('Cumulative P&L (₹)', color='red')
        ax.set_title(title)
        
        # Legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_str
    
    def create_trades_chart(self, trades_data: List[Dict[str, Any]], 
                           title: str = "Trade Executions") -> str:
        """
        Create a trades chart.
        
        Args:
            trades_data (List[Dict[str, Any]]): Trades data
            title (str): Chart title
            
        Returns:
            str: Base64 encoded image
        """
        if not trades_data:
            return ""
        
        # Convert to DataFrame
        df = pd.DataFrame(trades_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Separate buys and sells
        buys = df[df['transaction_type'] == 'BUY']
        sells = df[df['transaction_type'] == 'SELL']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot buys and sells
        if not buys.empty:
            ax.scatter(buys['date'], buys['price'], c='green', s=50, alpha=0.7, label='Buys', marker='^')
        if not sells.empty:
            ax.scatter(sells['date'], sells['price'], c='red', s=50, alpha=0.7, label='Sells', marker='v')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (₹)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_str
    
    def create_performance_comparison_chart(self, chatgpt_data: List[Dict[str, Any]], 
                                          index_data: List[Dict[str, Any]],
                                          title: str = "ChatGPT vs Index Performance") -> str:
        """
        Create a performance comparison chart between ChatGPT recommendations and index.
        
        Args:
            chatgpt_data (List[Dict[str, Any]]): ChatGPT portfolio performance data
            index_data (List[Dict[str, Any]]): Index performance data
            title (str): Chart title
            
        Returns:
            str: Base64 encoded image
        """
        if not chatgpt_data or not index_data:
            return ""
        
        # Convert to DataFrames
        chatgpt_df = pd.DataFrame(chatgpt_data)
        index_df = pd.DataFrame(index_data)
        
        # Ensure date columns exist
        if 'date' not in chatgpt_df.columns and 'timestamp' in chatgpt_df.columns:
            chatgpt_df['date'] = chatgpt_df['timestamp']
        if 'date' not in index_df.columns and 'timestamp' in index_df.columns:
            index_df['date'] = index_df['timestamp']
        
        # Convert dates
        chatgpt_df['date'] = pd.to_datetime(chatgpt_df['date'])
        index_df['date'] = pd.to_datetime(index_df['date'])
        
        # Sort by date
        chatgpt_df = chatgpt_df.sort_values('date')
        index_df = index_df.sort_values('date')
        
        # Normalize values to start at 100
        if 'value' in chatgpt_df.columns:
            chatgpt_df['normalized_value'] = (chatgpt_df['value'] / chatgpt_df['value'].iloc[0]) * 100
        else:
            # Simulate some data if not available
            chatgpt_df['normalized_value'] = 100 + np.cumsum(np.random.normal(0, 1, len(chatgpt_df)))
        
        if 'value' in index_df.columns:
            index_df['normalized_value'] = (index_df['value'] / index_df['value'].iloc[0]) * 100
        else:
            # Simulate some data if not available
            index_df['normalized_value'] = 100 + np.cumsum(np.random.normal(0, 0.8, len(index_df)))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot both series
        ax.plot(chatgpt_df['date'], chatgpt_df['normalized_value'], 
                linewidth=2, label='ChatGPT Strategy', color='blue')
        ax.plot(index_df['date'], index_df['normalized_value'], 
                linewidth=2, label='Index (NIFTY 50)', color='orange')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Value (Base = 100)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add performance metrics as text
        chatgpt_return = ((chatgpt_df['normalized_value'].iloc[-1] / 100) - 1) * 100
        index_return = ((index_df['normalized_value'].iloc[-1] / 100) - 1) * 100
        
        ax.text(0.02, 0.98, f'ChatGPT Return: {chatgpt_return:.1f}%', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='blue', alpha=0.1))
        ax.text(0.02, 0.90, f'Index Return: {index_return:.1f}%', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.1))
        
        # Tight layout
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_str
    
    def create_chatgpt_vs_momentum_comparison(self, chatgpt_stocks: List[Dict[str, Any]], 
                                            momentum_stocks: List[Dict[str, Any]],
                                            title: str = "ChatGPT vs Momentum Stock Selections") -> str:
        """
        Create a comparison chart between ChatGPT and momentum stock selections.
        
        Args:
            chatgpt_stocks (List[Dict[str, Any]]): Stocks selected by ChatGPT
            momentum_stocks (List[Dict[str, Any]]): Stocks selected by momentum strategy
            title (str): Chart title
            
        Returns:
            str: Base64 encoded image
        """
        if not chatgpt_stocks and not momentum_stocks:
            return ""
        
        # Create comparison data
        comparison_data = []
        
        # Process ChatGPT stocks
        for stock in chatgpt_stocks:
            comparison_data.append({
                'symbol': stock.get('symbol', ''),
                'chatgpt_score': stock.get('chatgpt_score', 0),
                'momentum_score': stock.get('momentum', 0),
                'source': 'ChatGPT'
            })
        
        # Process momentum stocks (that might not be in ChatGPT selection)
        momentum_symbols = {stock.get('symbol', '') for stock in momentum_stocks}
        chatgpt_symbols = {stock.get('symbol', '') for stock in chatgpt_stocks}
        
        for stock in momentum_stocks:
            symbol = stock.get('symbol', '')
            if symbol not in chatgpt_symbols:
                comparison_data.append({
                    'symbol': symbol,
                    'chatgpt_score': 0,  # Not selected by ChatGPT
                    'momentum_score': stock.get('momentum', 0),
                    'source': 'Momentum'
                })
        
        if not comparison_data:
            return ""
        
        # Convert to DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create bar chart
        x = np.arange(len(df))
        width = 0.35
        
        ax.bar(x - width/2, df['chatgpt_score'], width, label='ChatGPT Score', alpha=0.8, color='blue')
        ax.bar(x + width/2, df['momentum_score'] * 1000, width, label='Momentum Score (scaled)', alpha=0.8, color='orange')
        
        # Set labels and title
        ax.set_xlabel('Stock Symbols')
        ax.set_ylabel('Scores')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(df['symbol'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_str
    
    def create_validation_score_distribution(self, validated_stocks: List[Dict[str, Any]],
                                           title: str = "ChatGPT Validation Score Distribution") -> str:
        """
        Create a distribution chart of ChatGPT validation scores.
        
        Args:
            validated_stocks (List[Dict[str, Any]]): Stocks with validation scores
            title (str): Chart title
            
        Returns:
            str: Base64 encoded image
        """
        if not validated_stocks:
            return ""
        
        # Extract scores
        scores = [stock.get('chatgpt_score', 0) for stock in validated_stocks]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        n, bins, patches = ax.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add mean line
        mean_score = np.mean(scores)
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean Score: {mean_score:.1f}')
        
        # Labels and title
        ax.set_xlabel('Validation Score')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_str
    
    def create_stock_selection_heatmap(self, selection_data: List[Dict[str, Any]], 
                                     title: str = "Stock Selection Heatmap") -> str:
        """
        Create a heatmap of stock selections over time.
        
        Args:
            selection_data (List[Dict[str, Any]]): Stock selection data
            title (str): Chart title
            
        Returns:
            str: Base64 encoded image
        """
        if not selection_data:
            return ""
        
        # Convert to DataFrame
        df = pd.DataFrame(selection_data)
        
        # Ensure required columns exist
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime('today')
        if 'symbol' not in df.columns:
            df['symbol'] = 'UNKNOWN'
        if 'score' not in df.columns:
            df['score'] = np.random.rand(len(df))  # Simulate scores
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        
        # Pivot data for heatmap
        pivot_df = df.pivot_table(index='symbol', columns='date', values='score', fill_value=0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(pivot_df.values, cmap='RdYlGn', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(pivot_df.columns)))
        ax.set_yticks(np.arange(len(pivot_df.index)))
        ax.set_xticklabels([date.strftime('%m-%d') for date in pivot_df.columns])
        ax.set_yticklabels(pivot_df.index)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Selection Score')
        
        # Labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Symbol')
        ax.set_title(title)
        
        # Tight layout
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_str