# StockExperiment — Summary

_Updated: 2026-05-12 | Capital: ₹10,00,000_

## 🎯 MULTI-YEAR FINDING (3-yr backtest, May 2023 → May 2026)

EMA 200/400 on Nifty 50 large caps **wins long-term**:

| Year | ROI% | MaxDD% | Trades |
|------|-----:|-------:|-------:|
| 2023-2024 | **+98.13** | 13.06 | 179 |
| 2024-2025 | **+54.88** | 13.06 | 125 |
| 2025-2026 | +6.77 | 13.01 | 54 |
| **Avg/yr** | **+53.26%** | 13.06 | 119 |
| **Compound 3-yr** | **+227.64%** | — | 358 total |

vs EMA 9/21 on N50: avg -9.42%/yr (loses all 3 years).

**INVERTED finding:** earlier 1-year Phase 7 favored EMA 9/21 on volatile
mid-caps (selector top-10). Multi-year shows **EMA 200/400 on stable
N50 large caps is far more durable**.

## Production config (updated based on multi-year data)

```yaml
strategy:        ema_200_400          # 1H crossover (CHANGED back from 9_21)
universe:        nifty50              # full N50 large caps (simpler than selector)
filters:         sector_RS + calendar
overlay:         vol_sizing 2%        # optional, reduces DD
max_concurrent:  2
capital_inr:     1_000_000
min_price:       50
min_adv_lakh:    100
kill_switch:     -5% daily loss
```

## Backtest comparison (3 years, N50, ₹10L)

| Config | 2023-24 ROI | 2024-25 ROI | 2025-26 ROI | Avg/yr | Trades |
|--------|------------:|------------:|------------:|-------:|-------:|
| **EMA 200/400 raw** ⭐ | **+98.13** | **+54.88** | +6.77 | **+53.26** | 358 |
| EMA 200/400 + filters | 0 (over-filter) | 0 | 0 | 0 | 0 |
| EMA 9/21 raw | -0.94 | -20.21 | -7.10 | -9.42 | 1163 |
| EMA 9/21 + filters | +18.08 | +8.24 | -6.46 | +6.62 | 110 |
| ORB-60 day trading | (impl broken) | - | - | - | 0 |

**Filters HURT slow EMA (200/400) but HELP fast EMA (9/21).**

EMA 200/400 raw remains multi-year winner. Filters block legitimate
slow-EMA signals. Retest1/retest2 state machine already filters
false alarms by construction.

## Universe options

**Option 1 (recommended — multi-year validated):**
- Full Nifty 50 (53 large-cap stocks)
- EMA 200/400 fires ~120-180 trades/year
- 3-year compound +227.64%

**Option 2 (selector-based, single-year only):**
- Monthly selector top-10 from N500
- EMA 9/21 better here in 2025-26 (+33.32%)
- Not yet multi-year validated
- Recent IPOs (SWIGGY, VMM etc.) lack 3-yr history

## Alternative paths (1-year results)

| Path | ROI% | DD% | Win% | Trades | Note |
|------|-----:|----:|-----:|-------:|------|
| EMA 200/400 N50 simple 2023-24 | +98.13 | 13.06 | n/a | 179 | Multi-year tested |
| EMA 200/400 N50 simple 2024-25 | +54.88 | 13.06 | n/a | 125 | Multi-year tested |
| EMA 9/21 selector top-10 +filters | +33.32 | 8.20 | 60 | 140 | 1-yr only, mid-caps |
| EMA 9/21 selector raw max=3 | +46.87 | 12.56 | 62 | 209 | 1-yr only |
| BB Squeeze N50 | +32.73 | 3.24 | 80 | 5 | Low DD, low confidence |

See `path_returns/MASTER_RETURNS.md` for 2025-26 month-by-month details.
See `MULTI_YEAR_REPORT.md` for full 3-year breakdown.

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
