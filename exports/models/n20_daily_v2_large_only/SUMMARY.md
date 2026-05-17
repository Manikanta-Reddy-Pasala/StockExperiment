# n20_daily_v2_large_only — SUMMARY

Daily-rebalance momentum rotation on NSE Nifty 100 large-caps. Top-20 ADV pool + uptrend + Nifty 100 filter → top-1 by 30d return → daily rotate.

## Stock pick logic (plain English)

1. **Universe (per day)**: top-20 N500 stocks by 20-day ADV
2. **Uptrend filter**: keep stocks where close > 200d SMA
3. **Large-cap filter**: keep stocks in NSE Nifty 100 (`src/data/symbols/nifty100.csv`)
4. **Rank by 30d return** (highest first)
5. **Pick top-1**; sit in cash if no large-cap matches
6. **Rebalance daily**

## Key knobs

| Knob | Value |
|---|---|
| Universe pool | Top-20 by 20-day ADV from N500 |
| Uptrend filter | close > 200d SMA |
| Large-cap filter | NSE Nifty 100 membership |
| Lookback | 30 days |
| Position | top-1, max_concurrent=1 |
| Rebalance period | **Daily** |
| Cash policy | Sit in cash if no candidate matches |

## Headline result (₹10L, 2023-05-15 → 2026-05-12)

| Metric | Value |
|---|---:|
| Final NAV | **₹1,39,59,936** |
| Total return | **+1295.99%** |
| **3-yr CAGR** | **+140.78%/yr** |
| Max DD (cash NAV) | 25.52% |
| Max DD (mark-to-market) | 26.92% |
| Trades | 139 |
| WR | 43.1% (59W / 78L) |
| Calmar (CAGR/DD) | **5.52** |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 139 | 59 | 78 | 43% | +11,076,686 |

All trades are Large-cap by construction (NSE Nifty 100 filter enforced).

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹5,558,933 | **+455.89%** | 36 |
| 2024-25 | ₹5,558,933 | ₹10,791,611 | **+94.13%** | 52 |
| 2025-26 | ₹10,791,611 | ₹12,076,686 | **+11.91%** | 51 |

## Caveats

- WR 43% — strategy enters more often, wins less, but high win sizes compensate.
- 25-27% Max DD still substantial for single-stock concentration. Plan for 30% DD periods.
- 139 trades / 3yr = ~46/yr round-trip → ~3-5%/yr cost drag. Post-cost CAGR ≈ +135%.
- NSE Nifty 100 list refreshes quarterly (Mar/Sep). Run `tools/refresh_nifty100.py` to keep current.
- Slippage not modeled — real ~10-30 bps drag per round-trip.
- Survivorship: stocks delisted from N500 mid-period missing.

## History

Earlier `n20_daily_30d_mc1_uptrend` (no Large-cap filter) hit +157% CAGR but 50% DD. Pure-number DD-reduction filter sweep (15+ variants: hard SL, trail SL, mc>1, vol caps, port-DD halt, combos) all harmed CAGR more than helped DD. Only NSE Nifty 100 membership filter (categorical) halved DD with acceptable CAGR cost. Original archived at `tools/models/_archived_models/n20_daily_30d_mc1_uptrend/README.md`.

Full ledger: `TRADE_LEDGER.md`