"""
Startup Service for Application Initialization
Handles automatic data loading when brokers are connected
"""
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class StartupService:
    """Service for handling application startup initialization."""

    def __init__(self, db_manager, broker_service, stock_master_service):
        self.db_manager = db_manager
        self.broker_service = broker_service
        self.stock_master_service = stock_master_service

    def initialize_application(self) -> Dict[str, Any]:
        """
        Initialize application on startup.
        Checks for connected brokers and automatically loads stock data.
        """
        logger.info("🚀 Starting application initialization...")

        results = {
            'initialization_time': datetime.now().isoformat(),
            'broker_connections': {},
            'stock_data_loaded': False,
            'market_snapshots': False,
            'errors': []
        }

        try:
            # Check for connected brokers across all users
            connected_brokers = self._check_broker_connections()
            results['broker_connections'] = connected_brokers

            if connected_brokers:
                logger.info(f"📡 Found {len(connected_brokers)} connected broker(s)")

                # Load stock master data if brokers are connected
                if self._load_stock_master_data():
                    results['stock_data_loaded'] = True
                    logger.info("📊 Stock master data loaded successfully")

                    # Get market snapshots
                    if self._capture_market_snapshots():
                        results['market_snapshots'] = True
                        logger.info("📈 Market snapshots captured")
                else:
                    results['errors'].append("Failed to load stock master data")
            else:
                logger.info("🔌 No connected brokers found - skipping data load")

        except Exception as e:
            error_msg = f"Error during startup initialization: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

        # Log startup summary
        self._log_startup_summary(results)
        return results

    def _check_broker_connections(self) -> Dict[str, Any]:
        """Check which brokers are connected across all users."""
        connected_brokers = {}

        try:
            with self.db_manager.get_session() as session:
                from ...models.models import BrokerConfiguration

                # Find all active broker configurations
                broker_configs = session.query(BrokerConfiguration).filter(
                    BrokerConfiguration.is_active == True,
                    BrokerConfiguration.is_connected == True
                ).all()

                for config in broker_configs:
                    broker_name = config.broker_name
                    if broker_name not in connected_brokers:
                        connected_brokers[broker_name] = []

                    connected_brokers[broker_name].append({
                        'user_id': config.user_id,
                        'client_id': config.client_id,
                        'last_connection_test': config.last_connection_test.isoformat() if config.last_connection_test else None,
                        'connection_status': config.connection_status
                    })

        except Exception as e:
            logger.error(f"Error checking broker connections: {e}")

        return connected_brokers

    def _load_stock_master_data(self) -> bool:
        """Load comprehensive stock master data."""
        try:
            # Check if refresh is needed
            if self.stock_master_service.is_refresh_needed():
                logger.info("🔄 Stock master data refresh needed - starting comprehensive data load...")

                # Perform full stock master refresh
                refresh_result = self.stock_master_service.refresh_all_stocks()

                if refresh_result.get('success'):
                    symbols_added = refresh_result.get('symbols_added', 0)
                    stocks_processed = refresh_result.get('stocks_processed', 0)
                    logger.info(f"✅ Stock master refresh completed: {symbols_added} symbols, {stocks_processed} stocks processed")
                    return True
                else:
                    error = refresh_result.get('error', 'Unknown error')
                    logger.error(f"❌ Stock master refresh failed: {error}")
                    return False
            else:
                logger.info("✅ Stock master data is up to date")
                return True

        except Exception as e:
            logger.error(f"Error loading stock master data: {e}")
            return False

    def _capture_market_snapshots(self) -> bool:
        """Capture current market snapshots for benchmarking."""
        try:
            # Use the first available connected user for market data
            with self.db_manager.get_session() as session:
                from ...models.models import BrokerConfiguration

                # Get first connected broker
                config = session.query(BrokerConfiguration).filter(
                    BrokerConfiguration.is_active == True,
                    BrokerConfiguration.is_connected == True,
                    BrokerConfiguration.broker_name == 'fyers'  # Focus on Fyers for now
                ).first()

                if not config:
                    logger.warning("No connected Fyers broker found for market snapshots")
                    return False

                # Get major indices quotes
                indices_symbols = 'NSE:NIFTY50-INDEX,NSE:SENSEX-INDEX,NSE:NIFTYMIDCAP150-INDEX,NSE:NIFTYSMLCAP250-INDEX'
                quotes_result = self.broker_service.get_fyers_quotes(config.user_id, indices_symbols)

                if quotes_result.get('s') == 'ok' and quotes_result.get('d'):
                    # Save market snapshot
                    snapshot_data = self._process_market_quotes(quotes_result['d'])
                    self._save_market_snapshot(snapshot_data)
                    logger.info("📊 Market snapshot captured and saved")
                    return True
                else:
                    logger.warning("Failed to get market quotes for snapshot")
                    return False

        except Exception as e:
            logger.error(f"Error capturing market snapshots: {e}")
            return False

    def _process_market_quotes(self, quotes_data) -> Dict[str, Any]:
        """Process raw market quotes into snapshot format."""
        snapshot = {
            'snapshot_date': datetime.now().strftime('%Y-%m-%d'),
            'nifty_50': None,
            'sensex': None,
            'nifty_midcap': None,
            'nifty_smallcap': None,
            'data_source': 'fyers'
        }

        try:
            # Map symbols to snapshot fields
            symbol_mapping = {
                'NSE:NIFTY50-INDEX': 'nifty_50',
                'NSE:SENSEX-INDEX': 'sensex',
                'NSE:NIFTYMIDCAP150-INDEX': 'nifty_midcap',
                'NSE:NIFTYSMLCAP250-INDEX': 'nifty_smallcap'
            }

            for quote in quotes_data:
                symbol = quote.get('n', '')
                if symbol in symbol_mapping:
                    price = quote.get('v', {}).get('lp', 0)
                    snapshot[symbol_mapping[symbol]] = price

        except Exception as e:
            logger.error(f"Error processing market quotes: {e}")

        return snapshot

    def _save_market_snapshot(self, snapshot_data: Dict[str, Any]):
        """Save market snapshot to database."""
        try:
            with self.db_manager.get_session() as session:
                from ...models.stock_models import MarketDataSnapshot

                # Check if snapshot for today already exists
                existing = session.query(MarketDataSnapshot).filter(
                    MarketDataSnapshot.snapshot_date == snapshot_data['snapshot_date']
                ).first()

                if existing:
                    # Update existing snapshot
                    for key, value in snapshot_data.items():
                        if hasattr(existing, key) and value is not None:
                            setattr(existing, key, value)
                else:
                    # Create new snapshot
                    snapshot = MarketDataSnapshot(**snapshot_data)
                    session.add(snapshot)

                session.commit()
                logger.info(f"Market snapshot saved for {snapshot_data['snapshot_date']}")

        except Exception as e:
            logger.error(f"Error saving market snapshot: {e}")

    def _log_startup_summary(self, results: Dict[str, Any]):
        """Log startup initialization summary."""
        logger.info("=" * 50)
        logger.info("🚀 STARTUP INITIALIZATION SUMMARY")
        logger.info("=" * 50)

        # Broker connections
        brokers = results['broker_connections']
        if brokers:
            logger.info(f"📡 Connected Brokers: {len(brokers)}")
            for broker, users in brokers.items():
                logger.info(f"  • {broker}: {len(users)} user(s)")
        else:
            logger.info("📡 Connected Brokers: None")

        # Data loading
        logger.info(f"📊 Stock Data Loaded: {'✅ Yes' if results['stock_data_loaded'] else '❌ No'}")
        logger.info(f"📈 Market Snapshots: {'✅ Yes' if results['market_snapshots'] else '❌ No'}")

        # Errors
        if results['errors']:
            logger.info(f"❌ Errors: {len(results['errors'])}")
            for error in results['errors']:
                logger.info(f"  • {error}")
        else:
            logger.info("❌ Errors: None")

        logger.info("=" * 50)


def get_startup_service(db_manager=None, broker_service=None, stock_master_service=None):
    """Get startup service instance."""
    if db_manager is None:
        from ...models.database import get_database_manager
        db_manager = get_database_manager()

    if broker_service is None:
        from ...services.core.broker_service import get_broker_service
        broker_service = get_broker_service()

    if stock_master_service is None:
        from ...services.ml.stock_master_service import get_stock_master_service
        stock_master_service = get_stock_master_service()

    return StartupService(db_manager, broker_service, stock_master_service)