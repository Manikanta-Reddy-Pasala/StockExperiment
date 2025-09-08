"""
Trade Logging Example
"""
import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from logging.trade_logger import TradeLogger
from datastore.database import get_database_manager


def run_trade_logging_example():
    """Run a trade logging example."""
    print("Running Trade Logging Example")
    print("=" * 50)
    
    # Initialize database and trade logger
    db_manager = get_database_manager()
    db_manager.create_tables()  # Ensure tables exist
    trade_logger = TradeLogger(db_manager)
    
    # Example 1: Log a trade execution
    print("1. Logging Trade Execution")
    print("-" * 20)
    trade_data = {
        'order_id': 'ORDER_12345',
        'broker_order_id': 'BROKER_67890',
        'symbol': 'INFY',
        'quantity': 100,
        'price': 1500.50,
        'transaction_type': 'BUY',
        'timestamp': datetime.now().isoformat(),
        'exchange': 'NSE'
    }
    
    log_id = trade_logger.log_trade_execution(trade_data)
    print(f"Trade execution logged with ID: {log_id}")
    
    # Example 2: Log an order placement
    print("\n2. Logging Order Placement")
    print("-" * 20)
    order_data = {
        'order_id': 'ORDER_12346',
        'symbol': 'RELIANCE',
        'quantity': 50,
        'order_type': 'LIMIT',
        'transaction_type': 'SELL',
        'price': 2400.00,
        'trigger_price': 0.0,
        'product_type': 'MIS'
    }
    
    log_id = trade_logger.log_order_placement(order_data)
    print(f"Order placement logged with ID: {log_id}")
    
    # Example 3: Log an order modification
    print("\n3. Logging Order Modification")
    print("-" * 20)
    modification_data = {
        'order_id': 'ORDER_12346',
        'broker_order_id': 'BROKER_67891',
        'symbol': 'RELIANCE',
        'old_quantity': 50,
        'new_quantity': 75,
        'old_price': 2400.00,
        'new_price': 2390.00,
        'status': 'SUCCESS',
        'message': 'Order modified successfully'
    }
    
    log_id = trade_logger.log_order_modification(modification_data)
    print(f"Order modification logged with ID: {log_id}")
    
    # Example 4: Log an order cancellation
    print("\n4. Logging Order Cancellation")
    print("-" * 20)
    cancellation_data = {
        'order_id': 'ORDER_12347',
        'broker_order_id': 'BROKER_67892',
        'symbol': 'TCS',
        'quantity': 25,
        'status': 'SUCCESS',
        'message': 'Order cancelled by user'
    }
    
    log_id = trade_logger.log_order_cancellation(cancellation_data)
    print(f"Order cancellation logged with ID: {log_id}")
    
    # Example 5: Log a position update
    print("\n5. Logging Position Update")
    print("-" * 20)
    position_data = {
        'symbol': 'INFY',
        'old_quantity': 100,
        'new_quantity': 150,
        'average_price': 1500.25,
        'market_value': 225037.50,
        'pnl': 5000.00,
        'timestamp': datetime.now().isoformat()
    }
    
    log_id = trade_logger.log_position_update(position_data)
    print(f"Position update logged with ID: {log_id}")
    
    # Example 6: Log a risk violation
    print("\n6. Logging Risk Violation")
    print("-" * 20)
    violation_data = {
        'violation_type': 'DAILY_LOSS_LIMIT_EXCEEDED',
        'symbol': 'HDFCBANK',
        'current_loss': -25000.00,
        'loss_limit': -20000.00,
        'action_taken': 'TRADING_HALTED',
        'timestamp': datetime.now().isoformat()
    }
    
    log_id = trade_logger.log_risk_violation(violation_data)
    print(f"Risk violation logged with ID: {log_id}")
    
    # Example 7: Log a system event
    print("\n7. Logging System Event")
    print("-" * 20)
    log_id = trade_logger.log_system_event(
        event_type="SYSTEM_STARTUP",
        message="Trading system started successfully",
        details={
            'version': '1.0.0',
            'mode': 'development',
            'timestamp': datetime.now().isoformat()
        }
    )
    print(f"System event logged with ID: {log_id}")
    
    # Example 8: Retrieve recent trade logs
    print("\n8. Retrieving Recent Trade Logs")
    print("-" * 20)
    trade_logs = trade_logger.get_trade_logs(limit=5)
    print(f"Retrieved {len(trade_logs)} recent trade logs:")
    for log in trade_logs:
        print(f"  - [{log['timestamp']}] {log['module']}: {log['message']}")
    
    # Example 9: Retrieve recent risk logs
    print("\n9. Retrieving Recent Risk Logs")
    print("-" * 20)
    risk_logs = trade_logger.get_risk_logs(limit=5)
    print(f"Retrieved {len(risk_logs)} recent risk logs:")
    for log in risk_logs:
        print(f"  - [{log['timestamp']}] {log['module']}: {log['message']}")
    
    print("\nTrade Logging Example Completed!")


if __name__ == "__main__":
    run_trade_logging_example()