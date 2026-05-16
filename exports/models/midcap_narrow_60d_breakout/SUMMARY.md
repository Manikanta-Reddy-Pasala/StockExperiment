# midcap_narrow_60d_breakout — Backtest Summary

**Period**: 2023-05-15 → 2026-05-15 (3 years).
**Universe**: midcap_narrow (~100 NSE midcaps).
**Capital**: ₹2,00,000.

## Result

| Metric | Value |
|---|---:|
| Final NAV | ₹21,79,348 |
| Total return | +989.67% |
| **CAGR** | **+121.66%** ✅ |
| **Max DD** | **-20.43%** ✅ |
| Calmar | 5.96 |
| Trades | 34 (≈11/yr) |

## Per-year ROI

| Year | Start NAV | End NAV | Return |
|---|---:|---:|---:|
| 2023 (May-Dec) | ~200,000 | ~390,620 | +95.31% |
| 2024 | ~395,500 | ~832,255 | +110.42% |
| 2025 | ~842,000 | ~1,305,090 | +55.01% |
| 2026 (Jan-May) | ~1,270,310 | ~2,179,348 | +71.58% |

All four years positive — no down year.

## Top 10 Trades (by PnL)

| Symbol | Hold days | Return | PnL |
|---|---:|---:|---:|
| HINDCOPPER | 30 | +57.22% | ₹6,68,467 |
| GALLANTT | 30 | +34.87% | ₹5,91,648 |
| GALLANTT | 31 | +31.02% | ₹2,23,972 |
| KALYANKJIL | 30 | +29.65% | ₹1,84,223 |
| ABB | 31 | +9.87% | ₹1,77,633 |
| HINDCOPPER | 30 | +18.63% | ₹1,74,272 |
| HINDZINC | 30 | +31.24% | ₹1,50,328 |
| HINDCOPPER | 30 | +44.45% | ₹1,17,529 |
| ASHOKLEY | 31 | +10.60% | ₹1,10,435 |
| GVT&D | 32 | +28.27% | ₹1,07,421 |

Top names: HINDCOPPER, GALLANTT, KALYANKJIL — commodity/industrial/PSU. All exits via MAX_HOLD (30-day cap).

## Configuration

```
hh:              60       # 60-day high breakout
vol_mult:        2.0      # volume > 2x 20-day avg
sma_long:        200      # close > 200-day SMA
max_conc:        1        # single concurrent position
trail_pct:       0.15     # 15% trail from peak
profit_trigger:  0.10     # trail activates after +10% gain
target_pct:      0.60     # +60% target lock
max_hold:        30       # 30-trading-day cap
regime_gate:     OFF      # NIFTY regime filter disabled (boosts CAGR)
slip_bps:        10       # 10bps slippage per fill
brokerage:       ₹20      # per order
stt_pct:         0.10     # 0.10% STT on sells
```
