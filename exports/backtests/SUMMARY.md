# StockExperiment — Summary

_Updated: 2026-05-12 | Capital: ₹10,00,000 | Backtest window: May 2025 → May 2026_

## Production config

```yaml
strategy:        ema_9_21
universe:        selector_top10    # multi-param ranked N500, monthly refresh
filters:         sector_RS + calendar (expiry+budget)
overlay:         vol_sizing 2% per trade
max_concurrent:  2
capital_inr:     1_000_000
min_price:       50
min_adv_lakh:    100
kill_switch:     -5% daily loss
```

## 1-year backtest result

| Metric | Value |
|--------|------:|
| Final equity | ₹13,33,156 |
| Profit | +₹3,33,156 |
| **ROI** | **+33.32%** |
| **MaxDD** | **8.20%** (~₹82K worst dip) |
| Trades | 140 |
| Win rate | 59.7% |

## Top-10 watchlist (selector @ 2025-05-12)

SWIGGY · VMM · AEGISLOG · ANGELONE · SAILIFE · ITI · IKS · AMBER · NTPCGREEN · BSE

## Alternative paths considered

| Path | ROI% | DD% | Win% | Trades | Note |
|------|-----:|----:|-----:|-------:|------|
| **B ⭐ prod** | **+33.32** | **8.20** | 60 | 140 | Best risk-adjusted |
| A Max ROI | +46.87 | 12.56 | 62 | 209 | EMA9/21 raw |
| D BB Squeeze N50 | +32.73 | 3.24 | 80 | 5 | Lowest DD but low confidence |
| E EMA200/400+filters | +29.35 | 9.58 | 71 | 24 | Old winner |

See `path_returns/MASTER_RETURNS.md` for month-by-month.

## Workflow

```bash
# Monthly (1st of month, pre-market)
./tools/live/run_daily.sh selector

# Daily market days
./tools/live/run_daily.sh prefetch    # 09:00 IST
./tools/live/run_daily.sh signals     # 09:30 IST
./tools/live/run_daily.sh paper       # 09:35 IST
*/5 09-15 ./tools/live/run_daily.sh monitor
./tools/live/run_daily.sh report      # 15:35 IST

# Going live (after 4-week paper validates)
LIVE_TRADING=true ./tools/live/run_daily.sh live
```

## Hard truths

1. **5-10%/mo (60-120%/yr) target unreachable** on cash equity ₹10L single strategy
2. **Live gap = 30-40% below backtest** due to slippage/STT/STCG
3. **Realistic forward: 22-28%/yr** at 8-10% DD
4. **Beats Nifty 50 (~12%/yr)** by 10-15pp alpha — solid result

## Files

| Path | Purpose |
|------|---------|
| `tools/live/` | Production scripts (signal, paper, fyers executors) |
| `tools/backtests/stock_selector.py` | Monthly stock ranker |
| `tools/backtests/sector_rs.py` | Sector RS filter |
| `tools/backtests/realistic_capital_sim_v2.py` | Cap-sim with overlays |
| `tools/backtests/path_returns_analysis.py` | Year+month breakdowns |
| `tools/backtests/multi_year_aggregator.py` | N-year consolidation |
| `path_returns/` | Per-path detailed monthly breakdowns |
| `MULTI_YEAR_REPORT.md` | 3-year backtest (pending generation) |

## Repo: github.com/Manikanta-Reddy-Pasala/StockExperiment

## Findings log (chronological)

| Phase | Finding | Outcome |
|-------|---------|---------|
| 0 | 1-yr baseline matrix | Best: ema_200_400 N50 +7.30% |
| 1 | Pattern mining | Top-19 contributors = +1100% sum%; bottom-34 = -400% drag |
| 2 | Top-N sweep | N50 top-19 max=3 = +13.20%, N500 top-20 max=2 = +14.15% |
| 3 | Regime gate (negative) | Nifty trend/ATR gate HURTS — EMA SELL alpha lives in bear |
| 4 | Multi-param selector | Top-10 selector + max=2 = +21.85% |
| 5 | Sector RS + calendar | +29.35% (sector filter +7.5pp boost) |
| 6 | Risk overlays | Vol-sizing 2% wins, win rate 50% → 73.8% |
| 7 | EMA 9/21 on selector | **NEW WINNER: +33.32% / 8.20% DD** |
| 8 | Month-by-month returns | All paths analyzed, Path B best Sharpe |
| 9 | Multi-year (2023-2026) | RUNNING — pending report |

Full historic details preserved in git history.
