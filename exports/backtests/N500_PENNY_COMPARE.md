# Nifty 500 — Penny Filter ON vs OFF (₹2L, max_concurrent=2)

Partial report (swing + ORB only; EMA models pending — still running with
`--min-price 0`). Cache reused from earlier runs.

## Swing Pullback Breakout — N500

Penny filter compares ON (default, skip stocks < ₹50) vs OFF (no min_price).

| Year | ON: Taken | ON: ROI% | ON: MDD% | OFF: Taken | OFF: ROI% | OFF: MDD% |
|------|----------:|---------:|---------:|-----------:|----------:|----------:|
| 2023-24 | 35 | **+13.51%** | 11.53% | 31 | +12.81% | 20.40% |
| 2024-25 | 31 | +0.17% | 16.30% | 26 | -0.64% | 15.66% |
| 2025-26 | 29 | +0.77% | 14.70% | 18 | +5.10% | 9.85% |
| **3-yr avg** | — | **+4.82%** | 16.30% | — | **+5.76%** | 20.40% |

**Reads:**
- 3-yr avg slightly higher OFF (+5.76% vs +4.82%) — penny filter cost ~0.94% / yr on swing.
- BUT: 2023-24 ON has lower MDD (11.53% vs 20.40%) — penny filter protects against drawdown in bull year.
- 2025-26 OFF wins clearly (+5.10% vs +0.77%, lower MDD 9.85%) — recent year sees more small-cap winners.
- 2024-25 OFF marginally worse — penny filter neutral in flat regime.

**Verdict (swing):** OFF wins gross ROI by ~1%/yr, but ON has more stable MDD profile. If you prioritize sleep, keep filter ON. If you chase gross gains, turn it OFF.

## 15-min ORB intraday — N500

Penny filter compares — **ORB nofilter run FAILED** (Fyers rate-limit hit 429 on every 5m chunk; 5m bars not cached because no `historical_data_5m` table exists). Result: 0 symbols processed in nofilter set.

| Year | ON: Taken | ON: ROI% | ON: MDD% | OFF: Taken | OFF: ROI% | OFF: MDD% |
|------|----------:|---------:|---------:|-----------:|----------:|----------:|
| 2023-24 | 1372 | -4.66% | 16.83% | 0 | _failed_ | — |
| 2024-25 | 1525 | -11.31% | 25.58% | 0 | _failed_ | — |
| 2025-26 | 1478 | +1.86% | 15.36% | 0 | _failed_ | — |

**Note:** To compare ORB with/without penny filter properly, need either:
1. Add `historical_data_5m` cache table + bulk prefetch 5m bars for full 3y window (~10h Fyers fetch time)
2. Run during a Fyers-throttle-free window (off-hours, less concurrent demand)

Will retry ORB nofilter once Fyers token has fresh quota.

## ORB filtered baseline (for reference)

3-yr avg ROI: -4.70% (1/3 years positive — only 2025-26 marginally +1.86%)

ORB struggles on N500 regardless of penny filter — single-day mean-reversion within 15m ORB window kills most setups. Filter has limited impact since signal density is low to start with.

## Pending — full update coming when EMA runs done

EMA 200/400 + EMA 9/21 × 3 years each (6 backtests) currently running on prod with `--min-price 0`. ETA ~3h. Will update this doc with full 4-model nofilter comparison + final commit.

## Files

- `exports/backtests/yearly/` — penny filter ON (committed earlier)
- `exports/backtests/yearly_nofilter/` — penny filter OFF (partial, pulled)
- `exports/backtests/yearly_nofilter/nifty500_yearly_partial.md` — partial summary

## Reproducibility

```bash
# Penny filter ON (default ₹50)
docker exec trading_system_app python tools/backtests/run_yearly_backtest.py \
  --universe nifty500 --years 3 --capital 200000 \
  --models swing_pullback,orb_15min,ema_200_400,ema_9_21

# Penny filter OFF (--min-price 0)
docker exec trading_system_app python tools/backtests/run_yearly_backtest.py \
  --universe nifty500 --years 3 --capital 200000 \
  --models swing_pullback,orb_15min,ema_200_400,ema_9_21 \
  --min-price 0 \
  --out-root /app/exports/backtests/yearly_nofilter
```
