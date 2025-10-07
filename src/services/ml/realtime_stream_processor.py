"""
Real-Time Market Data Stream Processor
Processes live market data for real-time predictions and alerts
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
import threading
import queue
import time
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)


class RealtimeStreamProcessor:
    """
    Real-time market data processing with online learning capabilities.

    Features:
    - Live data ingestion
    - Real-time feature calculation
    - Online ML prediction updates
    - Alert generation
    - Event-driven architecture
    """

    def __init__(self, db_session, predictor=None, window_size: int = 100):
        """
        Initialize stream processor.

        Args:
            db_session: Database session
            predictor: ML predictor for real-time predictions
            window_size: Rolling window size for calculations
        """
        self.db = db_session
        self.predictor = predictor
        self.window_size = window_size

        # Streaming state
        self.is_running = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.event_queue = queue.Queue(maxsize=100)

        # Rolling windows per symbol
        self.price_windows = {}  # {symbol: deque([prices])}
        self.feature_windows = {}  # {symbol: deque([features])}

        # Callbacks
        self.callbacks = {
            'price_update': [],
            'prediction_update': [],
            'alert': []
        }

        # Statistics
        self.stats = {
            'messages_processed': 0,
            'predictions_made': 0,
            'alerts_generated': 0,
            'start_time': None
        }

    def start(self):
        """Start real-time processing."""
        if self.is_running:
            logger.warning("Stream processor already running")
            return

        self.is_running = True
        self.stats['start_time'] = datetime.now()

        # Start processing threads
        self.processing_thread = threading.Thread(target=self._process_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.event_thread = threading.Thread(target=self._process_events)
        self.event_thread.daemon = True
        self.event_thread.start()

        logger.info("Real-time stream processor started")

    def stop(self):
        """Stop real-time processing."""
        self.is_running = False
        logger.info("Real-time stream processor stopped")

    def ingest(self, data: Dict):
        """
        Ingest real-time market data.

        Args:
            data: Market data tick with format:
                {
                    'symbol': 'RELIANCE',
                    'price': 2450.50,
                    'volume': 1000,
                    'timestamp': '2025-01-07T10:30:15'
                }
        """
        try:
            self.data_queue.put(data, block=False)
        except queue.Full:
            logger.warning("Data queue full, dropping tick")

    def _process_stream(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Get data with timeout
                data = self.data_queue.get(timeout=1.0)

                # Process tick
                self._process_tick(data)

                self.stats['messages_processed'] += 1

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Stream processing error: {e}")

    def _process_tick(self, data: Dict):
        """Process a single market data tick."""
        symbol = data['symbol']
        price = data['price']
        volume = data.get('volume', 0)
        timestamp = data.get('timestamp', datetime.now().isoformat())

        # Initialize windows if needed
        if symbol not in self.price_windows:
            self.price_windows[symbol] = deque(maxlen=self.window_size)
            self.feature_windows[symbol] = deque(maxlen=self.window_size)

        # Add to price window
        self.price_windows[symbol].append({
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        })

        # Calculate real-time features
        features = self._calculate_realtime_features(symbol)

        if features:
            self.feature_windows[symbol].append(features)

            # Make prediction if we have enough data
            if len(self.price_windows[symbol]) >= 20:
                self._make_realtime_prediction(symbol, features, price)

            # Check for alerts
            self._check_alerts(symbol, price, features)

        # Trigger callbacks
        self._trigger_callbacks('price_update', {
            'symbol': symbol,
            'price': price,
            'timestamp': timestamp
        })

    def _calculate_realtime_features(self, symbol: str) -> Optional[Dict]:
        """Calculate features from streaming data."""
        try:
            window = self.price_windows[symbol]

            if len(window) < 10:
                return None

            prices = np.array([tick['price'] for tick in window])
            volumes = np.array([tick['volume'] for tick in window])

            # Calculate features
            features = {}

            # Returns
            returns = np.diff(prices) / prices[:-1]
            features['current_return'] = returns[-1] if len(returns) > 0 else 0
            features['mean_return_5'] = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            features['mean_return_20'] = np.mean(returns[-20:]) if len(returns) >= 20 else 0

            # Volatility
            features['volatility_5'] = np.std(returns[-5:]) if len(returns) >= 5 else 0
            features['volatility_20'] = np.std(returns[-20:]) if len(returns) >= 20 else 0

            # Price levels
            features['current_price'] = prices[-1]
            features['sma_5'] = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
            features['sma_20'] = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]

            # Volume
            features['current_volume'] = volumes[-1]
            features['avg_volume'] = np.mean(volumes)
            features['volume_ratio'] = volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 1

            # Momentum
            if len(prices) >= 5:
                features['momentum_5'] = (prices[-1] - prices[-5]) / prices[-5]
            else:
                features['momentum_5'] = 0

            # Price vs SMAs
            features['price_vs_sma5'] = (prices[-1] - features['sma_5']) / features['sma_5']
            features['price_vs_sma20'] = (prices[-1] - features['sma_20']) / features['sma_20']

            features['timestamp'] = datetime.now().isoformat()

            return features

        except Exception as e:
            logger.error(f"Feature calculation error for {symbol}: {e}")
            return None

    def _make_realtime_prediction(self, symbol: str, features: Dict, price: float):
        """Make real-time ML prediction."""
        if not self.predictor:
            return

        try:
            # Convert features to format expected by predictor
            stock_data = {
                'symbol': symbol,
                'current_price': price,
                **features
            }

            # Make prediction
            prediction = self.predictor.predict(stock_data)

            # Store prediction
            prediction['timestamp'] = datetime.now().isoformat()
            prediction['symbol'] = symbol

            self.stats['predictions_made'] += 1

            # Trigger prediction callback
            self._trigger_callbacks('prediction_update', prediction)

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")

    def _check_alerts(self, symbol: str, price: float, features: Dict):
        """Check for alert conditions."""
        alerts = []

        # Price spike alert
        if 'volume_ratio' in features and features['volume_ratio'] > 3.0:
            alerts.append({
                'type': 'VOLUME_SPIKE',
                'symbol': symbol,
                'message': f"High volume spike: {features['volume_ratio']:.1f}x normal",
                'severity': 'HIGH',
                'price': price
            })

        # Momentum alert
        if 'momentum_5' in features:
            if features['momentum_5'] > 0.02:  # 2% move in 5 ticks
                alerts.append({
                    'type': 'MOMENTUM_BREAKOUT',
                    'symbol': symbol,
                    'message': f"Strong upward momentum: {features['momentum_5']:.2%}",
                    'severity': 'MEDIUM',
                    'price': price
                })
            elif features['momentum_5'] < -0.02:
                alerts.append({
                    'type': 'MOMENTUM_BREAKDOWN',
                    'symbol': symbol,
                    'message': f"Strong downward momentum: {features['momentum_5']:.2%}",
                    'severity': 'MEDIUM',
                    'price': price
                })

        # Volatility alert
        if 'volatility_5' in features and features['volatility_5'] > 0.03:
            alerts.append({
                'type': 'HIGH_VOLATILITY',
                'symbol': symbol,
                'message': f"Elevated volatility detected: {features['volatility_5']:.2%}",
                'severity': 'MEDIUM',
                'price': price
            })

        # Generate events for alerts
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            self.event_queue.put(alert)
            self.stats['alerts_generated'] += 1

    def _process_events(self):
        """Process events and alerts."""
        while self.is_running:
            try:
                event = self.event_queue.get(timeout=1.0)

                # Trigger alert callbacks
                self._trigger_callbacks('alert', event)

                logger.info(f"ALERT: {event['type']} - {event['symbol']} - {event['message']}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")

    def register_callback(self, event_type: str, callback: Callable):
        """
        Register callback for events.

        Args:
            event_type: 'price_update', 'prediction_update', or 'alert'
            callback: Function to call with event data
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.info(f"Callback registered for {event_type}")
        else:
            logger.warning(f"Unknown event type: {event_type}")

    def _trigger_callbacks(self, event_type: str, data: Dict):
        """Trigger callbacks for an event type."""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_current_features(self, symbol: str) -> Optional[Dict]:
        """Get current features for a symbol."""
        if symbol in self.feature_windows and len(self.feature_windows[symbol]) > 0:
            return self.feature_windows[symbol][-1]
        return None

    def get_statistics(self) -> Dict:
        """Get processor statistics."""
        stats = self.stats.copy()

        if stats['start_time']:
            runtime = (datetime.now() - stats['start_time']).total_seconds()
            stats['runtime_seconds'] = runtime
            stats['messages_per_second'] = stats['messages_processed'] / runtime if runtime > 0 else 0

        stats['queue_size'] = self.data_queue.qsize()
        stats['event_queue_size'] = self.event_queue.qsize()
        stats['symbols_tracked'] = len(self.price_windows)

        return stats

    def export_state(self) -> Dict:
        """Export current state for persistence."""
        state = {
            'statistics': self.get_statistics(),
            'symbols': list(self.price_windows.keys()),
            'latest_features': {
                symbol: list(window)[-1] if window else None
                for symbol, window in self.feature_windows.items()
            },
            'exported_at': datetime.now().isoformat()
        }

        return state


class StreamSimulator:
    """Simulate real-time stream from historical data for testing."""

    def __init__(self, db_session, processor: RealtimeStreamProcessor):
        self.db = db_session
        self.processor = processor
        self.is_running = False

    def start_simulation(self, symbols: List[str], start_date: str,
                        speed_multiplier: float = 1.0):
        """
        Simulate real-time stream from historical data.

        Args:
            symbols: List of symbols to simulate
            start_date: Start date for simulation
            speed_multiplier: Playback speed (1.0 = real-time, 10.0 = 10x faster)
        """
        logger.info(f"Starting stream simulation for {len(symbols)} symbols")
        logger.info(f"Speed: {speed_multiplier}x")

        self.is_running = True

        # Get historical data
        from sqlalchemy import text

        query = text("""
            SELECT symbol, date, close, volume
            FROM historical_data
            WHERE symbol = ANY(:symbols)
            AND date >= :start_date
            ORDER BY date, symbol
        """)

        result = self.db.execute(query, {'symbols': symbols, 'start_date': start_date})
        data = result.fetchall()

        logger.info(f"Loaded {len(data)} historical ticks")

        # Replay data
        for row in data:
            if not self.is_running:
                break

            tick = {
                'symbol': row[0],
                'price': float(row[2]),
                'volume': int(row[3]),
                'timestamp': row[1].isoformat()
            }

            self.processor.ingest(tick)

            # Simulate timing
            time.sleep(1.0 / speed_multiplier)

        logger.info("Stream simulation complete")

    def stop_simulation(self):
        """Stop simulation."""
        self.is_running = False
