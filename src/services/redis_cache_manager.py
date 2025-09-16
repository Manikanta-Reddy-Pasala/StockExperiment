"""
Redis Cache Manager for Stock Experiment Application

Provides centralized caching functionality using Redis for improved performance
and reduced database load.
"""

import json
import logging
import redis
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pickle

logger = logging.getLogger(__name__)

class RedisCacheManager:
    """
    Centralized Redis cache management for the Stock Experiment application.
    Handles caching of portfolio data, market data, and performance metrics.
    """

    def __init__(self, redis_url: str = None, host: str = 'localhost',
                 port: int = 6379, db: int = 0, password: str = None):
        """Initialize Redis cache manager"""
        self.redis_client = None
        self.is_available = False

        try:
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
            else:
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=False
                )

            # Test connection
            self.redis_client.ping()
            self.is_available = True
            logger.info("Redis cache manager initialized successfully")

        except Exception as e:
            logger.warning(f"Redis not available, cache disabled: {e}")
            self.redis_client = None
            self.is_available = False

    def _get_key(self, prefix: str, user_id: int, identifier: str = None) -> str:
        """Generate standardized cache key"""
        if identifier:
            return f"{prefix}:user:{user_id}:{identifier}"
        return f"{prefix}:user:{user_id}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.is_available:
            return None

        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Redis get error for key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache with TTL (default 5 minutes)"""
        if not self.is_available:
            return False

        try:
            serialized_data = pickle.dumps(value)
            self.redis_client.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            logger.warning(f"Redis set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.is_available:
            return False

        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete error for key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.is_available:
            return False

        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.warning(f"Redis exists error for key {key}: {e}")
            return False

    # Portfolio-specific cache methods
    def cache_portfolio_snapshot(self, user_id: int, snapshot_data: Dict[str, Any], ttl: int = 300):
        """Cache portfolio snapshot data"""
        key = self._get_key("portfolio_snapshot", user_id)
        return self.set(key, snapshot_data, ttl)

    def get_cached_portfolio_snapshot(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get cached portfolio snapshot"""
        key = self._get_key("portfolio_snapshot", user_id)
        return self.get(key)

    def cache_portfolio_performance(self, user_id: int, period: str, performance_data: Dict[str, Any], ttl: int = 600):
        """Cache portfolio performance data for specific period"""
        key = self._get_key("portfolio_performance", user_id, period)
        return self.set(key, performance_data, ttl)

    def get_cached_portfolio_performance(self, user_id: int, period: str) -> Optional[Dict[str, Any]]:
        """Get cached portfolio performance for specific period"""
        key = self._get_key("portfolio_performance", user_id, period)
        return self.get(key)

    def cache_portfolio_history(self, user_id: int, days: int, history_data: List[Dict[str, Any]], ttl: int = 1800):
        """Cache portfolio history data"""
        key = self._get_key("portfolio_history", user_id, str(days))
        return self.set(key, history_data, ttl)

    def get_cached_portfolio_history(self, user_id: int, days: int) -> Optional[List[Dict[str, Any]]]:
        """Get cached portfolio history"""
        key = self._get_key("portfolio_history", user_id, str(days))
        return self.get(key)

    # Market data cache methods
    def cache_market_data(self, symbol: str, market_data: Dict[str, Any], ttl: int = 60):
        """Cache market data for symbol"""
        key = f"market_data:{symbol}"
        return self.set(key, market_data, ttl)

    def get_cached_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data for symbol"""
        key = f"market_data:{symbol}"
        return self.get(key)

    def cache_holdings(self, user_id: int, holdings_data: Dict[str, Any], ttl: int = 120):
        """Cache holdings data"""
        key = self._get_key("holdings", user_id)
        return self.set(key, holdings_data, ttl)

    def get_cached_holdings(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get cached holdings data"""
        key = self._get_key("holdings", user_id)
        return self.get(key)

    def cache_positions(self, user_id: int, positions_data: Dict[str, Any], ttl: int = 120):
        """Cache positions data"""
        key = self._get_key("positions", user_id)
        return self.set(key, positions_data, ttl)

    def get_cached_positions(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get cached positions data"""
        key = self._get_key("positions", user_id)
        return self.get(key)

    # Dashboard cache methods
    def cache_dashboard_metrics(self, user_id: int, metrics_data: Dict[str, Any], ttl: int = 60):
        """Cache dashboard metrics"""
        key = self._get_key("dashboard_metrics", user_id)
        return self.set(key, metrics_data, ttl)

    def get_cached_dashboard_metrics(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get cached dashboard metrics"""
        key = self._get_key("dashboard_metrics", user_id)
        return self.get(key)

    # Utility methods
    def clear_user_cache(self, user_id: int):
        """Clear all cached data for a user"""
        if not self.is_available:
            return

        try:
            pattern = f"*:user:{user_id}*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Cleared cache for user {user_id}, {len(keys)} keys deleted")
        except Exception as e:
            logger.warning(f"Error clearing user cache for user {user_id}: {e}")

    def clear_all_cache(self):
        """Clear all cache data (use with caution)"""
        if not self.is_available:
            return

        try:
            self.redis_client.flushdb()
            logger.info("All cache data cleared")
        except Exception as e:
            logger.warning(f"Error clearing all cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.is_available:
            return {'status': 'unavailable'}

        try:
            info = self.redis_client.info()
            return {
                'status': 'available',
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'total_commands_processed': info.get('total_commands_processed', 0)
            }
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {'status': 'error', 'error': str(e)}


# Singleton instance
_cache_manager = None

def get_cache_manager() -> RedisCacheManager:
    """Get the singleton cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        try:
            import config
            _cache_manager = RedisCacheManager(
                redis_url=config.REDIS_URL,
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                password=config.REDIS_PASSWORD
            )
        except ImportError:
            # Fallback if config not available
            _cache_manager = RedisCacheManager()
    return _cache_manager