"""Strategy model registry — single source of truth for the 4 active models.

Each entry exposes the wiring needed to run a model independently:
    - name: registry key (matches dict key)
    - description: short human label
    - strategy_class: the live strategy class (instantiated by callers)
    - config_class: the dataclass type for the strategy's config
    - default_config: a ready-to-use config instance with KISS defaults
    - bars_interval: "1h" | "daily" | "5m"
    - default_window_days: backtest window default
    - harness_module: dotted path to the backtest CLI module
"""
from src.services.technical.models import (
    ema_200_400,
    ema_9_21,
    swing_pullback,
    orb_15min,
)


def _entry(mod) -> dict:
    return {
        "name": mod.NAME,
        "description": mod.DESCRIPTION,
        "strategy_class": mod.STRATEGY_CLASS,
        "config_class": mod.CONFIG_CLASS,
        "default_config": mod.DEFAULT_CONFIG,
        "bars_interval": mod.BARS_INTERVAL,
        "default_window_days": mod.DEFAULT_WINDOW_DAYS,
        "harness_module": mod.HARNESS,
    }


MODELS: dict[str, dict] = {
    "ema_200_400":    _entry(ema_200_400),
    "ema_9_21":       _entry(ema_9_21),
    "swing_pullback": _entry(swing_pullback),
    "orb_15min":      _entry(orb_15min),
}


def get_model(key: str) -> dict:
    """Return registry entry for ``key``; raises KeyError if unknown."""
    if key not in MODELS:
        raise KeyError(f"Unknown model '{key}'. Known: {sorted(MODELS)}")
    return MODELS[key]


def list_models() -> list[str]:
    """Return the registered model keys in insertion order."""
    return list(MODELS.keys())


__all__ = ["MODELS", "get_model", "list_models"]
