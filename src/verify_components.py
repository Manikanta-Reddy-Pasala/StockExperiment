"""
Simple script to verify that core components work
"""
def test_config_manager():
    """Test configuration manager."""
    print("Testing ConfigManager...")
    try:
        from config_manager import ConfigManager
        config = ConfigManager("development")
        name = config.get("system.name")
        print(f"  Config system name: {name}")
        print("  ConfigManager: PASSED")
        return True
    except Exception as e:
        print(f"  ConfigManager: FAILED - {e}")
        return False

def test_simulator():
    """Test simulator."""
    print("Testing Simulator...")
    try:
        from simulator.simulator import Simulator
        simulator = Simulator()
        profile = simulator.get_profile()
        print(f"  Simulator profile status: {profile['status']}")
        print("  Simulator: PASSED")
        return True
    except Exception as e:
        print(f"  Simulator: FAILED - {e}")
        return False

def test_selector_engine():
    """Test selector engine."""
    print("Testing SelectorEngine...")
    try:
        from selector.selector_engine import SelectorEngine
        selector = SelectorEngine()
        strategies = selector.get_available_strategies()
        print(f"  Available strategies: {strategies}")
        print("  SelectorEngine: PASSED")
        return True
    except Exception as e:
        print(f"  SelectorEngine: FAILED - {e}")
        return False

def test_risk_manager():
    """Test risk manager."""
    print("Testing RiskManager...")
    try:
        from risk.risk_manager import RiskManager
        config = {"max_capital_per_trade": 0.01}
        risk_manager = RiskManager(config)
        risk_manager.set_total_equity(1000000)
        position_size = risk_manager.calculate_position_size(10.0)
        print(f"  Position size for ATR=10.0: {position_size}")
        print("  RiskManager: PASSED")
        return True
    except Exception as e:
        print(f"  RiskManager: FAILED - {e}")
        return False

def test_order_router():
    """Test order router."""
    print("Testing OrderRouter...")
    try:
        from order.order_router import OrderRouter, Order, OrderType, ProductType
        from simulator.simulator import Simulator
        
        # Use simulator as broker
        simulator = Simulator()
        simulator.set_market_data({"INFY": 1500.0})
        
        order_router = OrderRouter(simulator)
        
        # Create an order
        order = Order(
            symbol="INFY",
            quantity=10,
            order_type=OrderType.MARKET,
            transaction_type="BUY",
            product_type=ProductType.MIS
        )
        
        # Place the order
        order_id = order_router.place_order(order)
        print(f"  Order placed with ID: {order_id}")
        print(f"  Order status: {order.status}")
        print("  OrderRouter: PASSED")
        return True
    except Exception as e:
        print(f"  OrderRouter: FAILED - {e}")
        return False

def main():
    """Main function to run all tests."""
    print("Verifying Automated Trading System Components")
    print("=" * 50)
    
    tests = [
        test_config_manager,
        test_simulator,
        test_selector_engine,
        test_risk_manager,
        test_order_router
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All components verified successfully!")
        return 0
    else:
        print("Some components failed verification.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())