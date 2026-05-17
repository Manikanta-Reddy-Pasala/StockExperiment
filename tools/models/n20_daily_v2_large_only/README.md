# n20_daily_v2_large_only

**Daily-rebalance momentum rotation on NSE Nifty 100 large-caps.** Replaces archived `n20_daily_30d_mc1_uptrend` (50% DD version).

## Stock pick logic (plain English)

1. **Universe build (per day)**: top-20 N500 stocks by 20-day ADV
2. **Uptrend filter**: keep only stocks where close > 200-day SMA
3. **Large-cap filter**: keep only stocks in NSE Nifty 100 (`src/data/symbols/nifty100.csv`)
4. **Rank by 30-day return** (highest first)
5. **Pick top-1** from filtered set; if empty, hold cash
6. **Rebalance daily** (re-rank + rotate)

## Key knobs

| Knob | Value |
|---|---|
| Universe pool | Top-20 by 20-day ADV from N500 |
| Uptrend filter | close > 200d SMA |
| NSE Nifty 100 filter | Stock must be in NSE Nifty 100 list |
| Lookback | 30 days |
| Position | top-1, max_concurrent=1 |
| Rebalance | Daily |
| Cash policy | Sit in cash if no large-cap candidate matches |

## Backtest result (₹10L, 2023-05-15 → 2026-05-12)

| Metric | Value |
|---|---:|
| Final NAV | **₹1,39,59,936** |
| Total return | **+1295.99%** |
| 3-yr CAGR | **+140.78%** |
| Max DD (cash NAV) | **25.52%** |
| Max DD (mark-to-market NAV) | 26.92% |
| Calmar (CAGR/DD) | **5.23** |
| Trades | 139 |
| WR | 43.1% (59W / 78L) |

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹10,00,000 | ₹55,58,933 | **+455.89%** | 36 |
| 2024-25 | ₹55,58,933 | ₹1,07,91,611 | **+94.13%** | 52 |
| 2025-26 | ₹1,07,91,611 | ₹1,39,59,936 | **+29.36%** | 51 |

## Files

| File | Purpose |
|---|---|
| `backtest.py` | Standalone reproducer |
| `trade_ledger.json` | 139 trades raw |

`exports/models/n20_daily_v2_large_only/{SUMMARY.md, TRADE_LEDGER.md}` for full per-trade table with NSE cap + invested ₹.

## Reproduce

```bash
docker exec trading_system_app python tools/models/n20_daily_v2_large_only/backtest.py
```

## Caveats

- 25-27% DD still substantial for single-stock daily rotation.
- 139 trades / 3yr = ~46/yr round-trip → 3-5%/yr cost drag. Post-cost CAGR ≈ +135%.
- NSE Nifty 100 list refreshes quarterly (Mar/Sep rebalance). Run `tools/refresh_nifty100.py` to keep current.
- Slippage not modeled. Real ~10-30 bps drag per round-trip.

## History

Earlier variant `n20_daily_30d_mc1_uptrend` (no Large-cap filter) hit +157% CAGR but 50% Max DD. Tested 15+ pure-number DD-reduction filters (hard SL, trail SL, mc>1, vol caps, port-DD halt) — all hurt CAGR more than they helped. Only NSE Nifty 100 categorical filter halved DD with acceptable CAGR cost. Original archived at `tools/models/_archived_models/n20_daily_30d_mc1_uptrend/README.md`.
