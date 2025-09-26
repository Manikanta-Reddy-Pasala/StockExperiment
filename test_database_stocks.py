#!/usr/bin/env python3
"""
Test script to verify that stocks are loaded in the database
and the enhanced discovery system can find them
"""

import sys
import os
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_stocks():
    """Test if stocks are loaded in the database."""
    try:
        from src.models.database import get_database_manager
        from src.models.stock_models import Stock
        
        logger.info("Testing database stock loading...")
        
        db_manager = get_database_manager()
        
        with db_manager.get_session() as session:
            # Count total stocks
            total_stocks = session.query(Stock).count()
            logger.info(f"Total stocks in database: {total_stocks}")
            
            if total_stocks > 0:
                # Get a sample of stocks
                sample_stocks = session.query(Stock).limit(5).all()
                logger.info(f"Sample stocks:")
                for stock in sample_stocks:
                    logger.info(f"  - {stock.symbol}: {stock.name} (Price: {stock.current_price}, Volume: {stock.volume})")
                
                return True
            else:
                logger.warning("No stocks found in database")
                return False
                
    except Exception as e:
        logger.error(f"Error testing database stocks: {e}")
        return False

def test_enhanced_discovery():
    """Test the enhanced discovery system."""
    try:
        from src.services.stock_filtering import get_enhanced_discovery_service
        
        logger.info("Testing enhanced discovery system...")
        
        discovery_service = get_enhanced_discovery_service()
        result = discovery_service.discover_stocks(user_id=1)
        
        logger.info(f"Enhanced discovery result:")
        logger.info(f"  - Total processed: {result.total_processed}")
        logger.info(f"  - Stage 1 passed: {result.stage1_passed}")
        logger.info(f"  - Stage 2 passed: {result.stage2_passed}")
        logger.info(f"  - Final selected: {result.final_selected}")
        logger.info(f"  - Execution time: {result.execution_time:.2f}s")
        
        if result.final_selected > 0:
            logger.info("✅ Enhanced discovery system is working!")
            return True
        else:
            logger.warning("⚠️ Enhanced discovery system found no stocks")
            return False
            
    except Exception as e:
        logger.error(f"Error testing enhanced discovery: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Testing Database Stock Loading and Enhanced Discovery")
    logger.info("=" * 60)
    
    tests = [
        ("Database Stock Loading", test_database_stocks),
        ("Enhanced Discovery System", test_enhanced_discovery)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} PASSED")
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is working correctly.")
        return True
    else:
        logger.error(f"⚠️  {total - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
