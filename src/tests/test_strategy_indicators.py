import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Import the services to be tested
from src.services.stock_screening_service import StockScreeningService
from src.services.enhanced_strategy_service import AdvancedStrategyService

@pytest.fixture
def sample_stock_data():
    """Create a sample pandas DataFrame of historical stock data with a clear uptrend at the end."""
    close_prices = [100] * 35
    for _ in range(5):
        close_prices.append(close_prices[-1] - 1)
    for _ in range(20):
        close_prices.append(close_prices[-1] + 2)

    # Generate ascending timestamps
    timestamps = [int((datetime.now() - timedelta(days=59-i)).timestamp()) for i in range(60)]

    data = {
        'timestamp': timestamps,
        'open': [p - 2 for p in close_prices],
        'high': [p + 3 for p in close_prices],
        'low': [p - 3 for p in close_prices],
        'close': close_prices,
        'volume': [100000 + i * 1000 for i in range(60)]
    }
    df = pd.DataFrame(data)
    return df

class TestStrategyIndicators:
    """Unit tests for technical indicator calculations."""

    def test_atr_calculation(self, sample_stock_data):
        """Test the ATR(14) calculation in StockScreeningService."""
        mock_fyers_connector = MagicMock()
        mock_fyers_connector.get_history.return_value = {
            'candles': sample_stock_data.values.tolist()
        }

        service = StockScreeningService()
        service.fyers_connector = mock_fyers_connector

        indicators = service._get_technical_indicators('TEST-EQ')

        assert indicators is not None
        assert 'atr_14' in indicators
        assert indicators['atr_14'] > 0

    def test_entry_indicators_calculation(self, sample_stock_data):
        """Test EMA, RSI, and other calculations in AdvancedStrategyService."""
        mock_fyers_connector = MagicMock()
        mock_fyers_connector.get_history.return_value = {
            'candles': sample_stock_data.values.tolist()
        }

        service = AdvancedStrategyService()
        service.fyers_connector = mock_fyers_connector

        indicators = service._get_entry_indicators('TEST-EQ')

        assert indicators is not None

        # Test RSI
        assert 'rsi_14' in indicators
        assert indicators['rsi_14'] > 80

        # Test EMAs
        last_close = sample_stock_data['close'].iloc[-1]
        assert 'ema_20' in indicators
        assert indicators['ema_20'] < last_close
        assert 'ema_50' in indicators
        assert indicators['ema_50'] < last_close
        assert indicators['ema_20'] > indicators['ema_50']

        # Test 20-day high
        expected_high_20d = sample_stock_data['high'].tail(20).max()
        assert 'high_20d' in indicators
        assert indicators['high_20d'] == expected_high_20d

        # Test avg volume
        expected_avg_volume_20d = sample_stock_data['volume'].tail(20).mean()
        assert 'avg_volume_20d' in indicators
        assert indicators['avg_volume_20d'] == expected_avg_volume_20d

    @patch('src.services.enhanced_strategy_service.AdvancedStrategyService._get_entry_indicators')
    def test_passes_entry_rules(self, mock_get_entry_indicators):
        """Test the logic of the _passes_entry_rules method."""
        service = AdvancedStrategyService()

        stock_data = MagicMock()
        stock_data.symbol = 'TEST-EQ'
        stock_data.current_price = 165
        stock_data.volume = 200000

        # Case 1: All rules pass
        mock_get_entry_indicators.return_value = {
            "ema_20": 160, "ema_50": 155,
            "high_20d": 164, "avg_volume_20d": 100000,
            "rsi_14": 60
        }
        assert service._passes_entry_rules(stock_data, 1) is True

        # Case 2: Price below EMA
        mock_get_entry_indicators.return_value['ema_20'] = 170
        assert service._passes_entry_rules(stock_data, 1) is False
        mock_get_entry_indicators.return_value['ema_20'] = 160

        # Case 3: Price not above 20-day high
        mock_get_entry_indicators.return_value['high_20d'] = 166
        assert service._passes_entry_rules(stock_data, 1) is False
        mock_get_entry_indicators.return_value['high_20d'] = 164

        # Case 4: Volume too low
        stock_data.volume = 140000
        assert service._passes_entry_rules(stock_data, 1) is False
        stock_data.volume = 200000

        # Case 5: RSI out of range
        mock_get_entry_indicators.return_value['rsi_14'] = 45
        assert service._passes_entry_rules(stock_data, 1) is False
        mock_get_entry_indicators.return_value['rsi_14'] = 75
        assert service._passes_entry_rules(stock_data, 1) is False
