# n20_daily_v2_large_only

**v2 of `n20_daily_30d_mc1_uptrend`** with NSE Nifty 100 (Large-cap) membership filter. Halves Max DD vs v1 baseline.

## What's different from v1

| Knob | v1 (n20_daily_30d_mc1_uptrend) | **v2 (this)** |
|---|---|---|
| Universe pool | Top-20 ADV from N500 | Same |
| Uptrend filter | close > 200d SMA | Same |
| **NSE Nifty 100 filter** | — | **Stock must be in NSE Nifty 100** |
| Lookback | 30 days | Same |
| Position | top-1 (max_concurrent=1) | Same |
| Rebalance | Daily | Same |

The Nifty 100 filter cuts mid-caps and small-caps from the candidate pool. Strategy still ranks top-20 ADV + uptrend by 30d return, but must be a Large-cap per NSE official index membership.

## Stock pick logic (plain English)

1. **Universe build (per day)**: top-20 N500 stocks by 20-day ADV
2. **Uptrend filter**: keep only stocks where close > 200d SMA
3. **NEW v2 — Large-cap filter**: keep only stocks in NSE Nifty 100 (`src/data/symbols/nifty100.csv`)
4. **Rank by 30d return** (highest first)
5. **Pick top-1** from filtered set; if empty, hold cash
6. **Rebalance daily**

## Why Large-only

Tested 15+ pure-number DD-reduction filters (hard SL, trail SL, mc>1 diversification, vol caps, port-DD circuit breakers, combos). All harmed CAGR more than they cut DD. Only NSE Nifty 100 filter halved DD with modest CAGR cost.

| Filter approach | CAGR | DD | Calmar | Result |
|---|---:|---:|---:|---|
| v1 baseline (no filter) | +157.27% | 50.61% | 3.10 | high return, high DD |
| Hard SL -5%/-7% | +157.11% | 50.61% | 3.10 | doesn't fire (rotation faster) |
| Trail SL -10% | +157.11% | 50.61% | 3.10 | never fires |
| mc=2 (2 positions) | +75.39% | 32.32% | 2.33 | dilutes top-1 edge |
| mc=3 | +42.25% | 29.59% | 1.43 | worse |
| Max daily vol 4% | +68.07% | 32.42% | 2.10 | kills high-vol winners |
| Halt on port DD>15% | +24.56% | 17.88% | 1.37 | stops trading too long |
| **NSE Nifty 100 filter (v2)** | **+140.78%** | **26.92%** | **5.23** ✅ | **winner** |

NSE Nifty 100 filter excludes structurally volatile small/mid stocks (RPOWER, OLAELEC, IDEA, IRB, SCI etc.) that trigger most of the v1 50% DD events. Large-caps deliver more orderly trends.

## Backtest result (₹10L, 2023-05-15 → 2026-05-12)

| Metric | v1 baseline | **v2 Large-only** | Δ |
|---|---:|---:|---:|
| Final NAV | ₹1.70 Cr | **₹1.40 Cr** | -₹30 L |
| CAGR | +157.27% | **+140.78%** | -16.5pp |
| Max DD (NAV-based) | 50.61% | **26.92%** | **-23.7pp** ✅ |
| Max DD (cash-based) | 50.61% | 25.52% | -25.1pp ✅ |
| Calmar | 3.10 | **5.23** | **+2.13** ✅ |
| Trades | 134 | 139 | +5 |
| WR | 47.8% | 43.1% | -4.7pp |

**Risk-adjusted winner.** Same strategy machinery, just constrained universe.

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹10,00,000 | ₹55,58,933 | **+455.89%** | 36 |
| 2024-25 | ₹55,58,933 | ₹1,07,91,611 | **+94.13%** | 52 |
| 2025-26 | ₹1,07,91,611 | ₹1,39,59,936 | **+29.36%** | 51 |

## Files

| File | Purpose |
|---|---|
| `backtest.py` | Standalone reproducer (v2 large-only config) |
| `trade_ledger.json` | 139 trades raw |
| `__init__.py` | Module marker |

See `exports/models/n20_daily_v2_large_only/{SUMMARY.md, TRADE_LEDGER.md}` for full per-trade table with NSE cap classification + invested ₹.

## Reproduce

```bash
docker exec trading_system_app python tools/models/n20_daily_v2_large_only/backtest.py
```

## Caveats

- v1 still exists as `n20_daily_30d_mc1_uptrend` (high-CAGR / high-DD variant). v2 is Calmar-optimized alternative.
- Nifty 100 list refreshes quarterly (NSE Mar/Sep rebalance). Run `tools/refresh_nifty100.py` to keep universe current.
- WR drops 47.8% → 43.1% — strategy enters more often but wins less often (Large-caps less explosive). Net CAGR/DD ratio still better.
- 26.9% DD still substantial for single-stock concentration. If lower DD needed, use mc=2 large-only (would cut CAGR further).
