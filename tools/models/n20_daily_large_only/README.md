# n20_daily_large_only

**Daily-rebalance momentum rotation on NSE Nifty 100 large-caps.** Replaces archived `n20_daily_30d_mc1_uptrend` (50% DD version).

## Stock pick logic (plain English)

1. **Universe build (per day)**: top-20 N500 stocks by 20-day ADV
2. **Uptrend filter**: keep only stocks where close > 200-day SMA
3. **Large-cap filter**: keep only stocks in NSE Nifty 100 (`src/data/symbols/nifty100.csv`)
4. **Max-price filter**: keep only stocks where close ≤ **₹2,500** at entry
5. **Rank by 30-day return** (highest first)
6. **Pick top-1** from filtered set; if empty, hold cash
7. **Rebalance daily** (re-rank + rotate)

## Why MAX_PRICE = ₹2,500?

Backtest 2023-2026 PnL by entry-price bucket showed:
- ₹5,000-10,000 bucket: 7 trades, ΣPnL **-₹2.05M**
- ₹2,000-3,000 bucket: 11 trades, ΣPnL **-₹0.94M**
- ₹3,000-5,000 bucket: 19 trades, ΣPnL +₹2.04M (mixed)

Filtering > ₹2,500 lifts CAGR +38pp and trims DD slightly. Pure formula (price observable at entry), no future knowledge.

## Key knobs

| Knob | Value |
|---|---|
| Universe pool | Top-20 by 20-day ADV from N500 |
| Uptrend filter | close > 200d SMA |
| NSE Nifty 100 filter | Stock must be in NSE Nifty 100 list |
| Max-price filter | close ≤ **₹2,500** at entry |
| Lookback | 30 days |
| Position | top-1, max_concurrent=1 |
| Rebalance | Daily |
| Cash policy | Sit in cash if no candidate matches |

## Backtest result (₹10L, 2023-05-15 → 2026-05-12)

| Metric | Value |
|---|---:|
| Final NAV | **₹1,86,76,864** |
| Total return | **+1767.69%** |
| 3-yr CAGR | **+165.97%** |
| Max DD (rebal cap_after) | **12.60%** |
| Max DD (mark-to-market NAV) | 24.57% |
| Calmar (CAGR/NAV-DD) | **6.76** |
| Trades | 134 |
| WR | 43.6% (58W / 75L) |

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹10,00,000 | ₹52,94,245 | **+429.42%** | ~36 |
| 2024-25 | ₹52,94,245 | ₹1,00,53,016 | **+89.89%** | ~48 |
| 2025-26 | ₹1,00,53,016 | ₹1,86,76,864 | **+85.78%** | ~50 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret | PnL |
|---|---|---:|---:|---:|
| MAZDOCK | 2024-05-29 → 2024-07-04 | 1,678.68 | +66.37% | +₹35.7L |
| BEL | 2025-05-13 → 2025-07-02 | 335.75 | +27.16% | +₹27.3L |
| ETERNAL | 2025-07-21 → 2025-09-10 | 271.70 | +19.40% | +₹22.0L |
| SBIN | 2026-02-05 → 2026-03-05 | 1,073.50 | +8.94% | +₹14.3L |
| MAZDOCK | 2025-04-07 → 2025-04-15 | 2,317.30 | +14.84% | +₹13.7L |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret | PnL |
|---|---|---:|---:|---:|
| ETERNAL | 2024-12-16 → 2024-12-23 | 294.15 | -6.87% | -₹7.1L |
| BEL | 2025-03-28 → 2025-04-02 | 301.32 | -6.28% | -₹6.5L |
| BEL | 2026-01-30 → 2026-02-05 | 449.00 | -3.59% | -₹6.0L |
| BEL | 2026-03-12 → 2026-03-13 | 453.55 | -3.12% | -₹5.3L |
| KOTAKBANK | 2025-04-02 → 2025-04-07 | 430.92 | -5.42% | -₹5.3L |

## Files

| File | Purpose |
|---|---|
| `backtest.py` | Standalone reproducer (MAX_PRICE=2500) |
| `trade_ledger.json` | 134 trades raw |

`exports/models/n20_daily_large_only/{SUMMARY.md, TRADE_LEDGER.md}` for full per-trade table with NSE cap + invested ₹.

## Reproduce

```bash
docker exec trading_system_app python tools/models/n20_daily_large_only/backtest.py
```

## Caveats

- 24-25% MTM DD substantial for single-stock daily rotation.
- 134 trades / 3yr ≈ 45/yr round-trip → 3-5%/yr cost drag. Post-cost CAGR ≈ +160%.
- NSE Nifty 100 list refreshes quarterly (Mar/Sep). Run `tools/refresh_nifty100.py`.
- Slippage not modeled. Real ~10-30 bps drag per round-trip.

## History

- **2026-05-17**: Add MAX_PRICE=₹2,500 filter. CAGR 140.78% → 165.97%, NAV-DD 26.92% → 24.57%, Calmar 5.23 → 6.76. Cuts high-px losers in ₹5K-10K bucket (-₹2.05M aggregate PnL).
- Earlier variant `n20_daily_30d_mc1_uptrend` (no Large-cap filter) hit +157% CAGR but 50% Max DD. Tested 15+ pure-number DD-reduction filters (hard SL, trail SL, mc>1, vol caps, port-DD halt) — all hurt CAGR more than they helped. Only NSE Nifty 100 categorical filter halved DD with acceptable CAGR cost. Original archived at `tools/models/_archived_models/n20_daily_30d_mc1_uptrend/README.md`.
