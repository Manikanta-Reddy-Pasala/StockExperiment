# n20_daily_30d_mc1_uptrend — SUMMARY

**3-year backtest** (2023-05-15 → 2026-05-12, ₹10L start, Top-20 ADV + uptrend filter, daily rotation)

| Metric | Value |
|---|---:|
| Final NAV (cash+open) | **~₹1.70 Cr** |
| Total return | **+1599.57%** |
| **3-yr CAGR** | **+157.27%/yr** |
| Max DD (cash NAV) | 50.61% |
| Round-trips | 134 |
| Win rate | 47.8% (64W / 70L) |
| Calmar | 3.11 |

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹3,922,517 | **+292.25%** | 45 |
| 2024-25 | ₹3,922,517 | ₹12,728,837 | **+224.51%** | 34 |
| 2025-26 | ₹12,728,837 | ₹16,818,564 | **+32.13%** | 55 |

## Strategy parameters

| Knob | Value |
|---|---|
| Universe size | Top **20** by 20-day ADV (much smaller than pseudo-N100's 100) |
| Filter | Close > 200-day SMA (uptrend gate) |
| Lookback | 30-day return ranking |
| Position | Hold top-1 (`max_concurrent=1`) |
| Rebalance | **Daily** (every trading day) |
| Universe rebuild | Daily PIT-strict (no lookahead) |

## Universe source

- Source: `src/data/symbols/nifty500.csv`
- Per trading day: compute 20-day ADV (close × volume), sort desc, take **top 20**
- Apply uptrend filter: keep only stocks where close > 200d SMA
- Rank remaining by 30-day return, pick top-1
- Universe rebuilds daily — strictly PIT (uses only prior-day data)

## Top 5 winners

| Symbol | Entry → Exit | PnL ₹ | Ret % |
|---|---|---:|---:|
| GRSE | 2025-05-16 → 2025-06-11 | +3,882,769 | +24.86% |
| NETWEB | 2025-09-18 → 2025-10-20 | +3,486,894 | +28.31% |
| ITI | 2024-12-31 → 2025-01-06 | +3,357,320 | +40.60% |
| HINDCOPPER | 2025-12-26 → 2026-02-11 | +3,011,503 | +26.93% |
| BSE | 2025-04-17 → 2025-05-16 | +2,892,787 | +22.73% |

## Top 5 losses

| Symbol | Entry → Exit | PnL ₹ | Ret % |
|---|---|---:|---:|
| RPOWER | 2025-06-11 → 2025-06-17 | -2,090,831 | -10.72% |
| CHENNPETRO | 2025-11-17 → 2025-11-26 | -2,069,222 | -16.66% |
| NETWEB | 2025-10-28 → 2025-11-04 | -1,953,758 | -13.70% |
| MCX | 2025-07-01 → 2025-07-18 | -1,565,974 | -9.11% |
| JPPOWER | 2025-07-18 → 2025-07-24 | -1,195,913 | -7.66% |

## Most-traded stocks (top 10)

| Symbol | Trades | Net PnL ₹ |
|---|---:|---:|
| BSE | 14 | +4,119,385 |
| PAYTM | 9 | +2,885,585 |
| ETERNAL | 8 | +453,055 |
| IRFC | 8 | +2,009,189 |
| MAZDOCK | 6 | +3,111,935 |
| NETWEB | 6 | -759,282 |
| ITI | 5 | +2,555,989 |
| ADANIPOWER | 5 | +2,531,765 |
| IDEA | 5 | +281,048 |
| NBCC | 4 | -394,915 |

## Caveats

- **Max DD 50%** — single-stock concentration with daily rotation. Whipsaw in regime shifts.
- **134 trades / 3yr** = ~45 round-trips/yr. STT + brokerage drag ~3-5%/yr. Post-cost CAGR ≈ +150%.
- **Slippage not modeled**: backtest fills at close. Real ~10-30 bps drag per round-trip.
- **Y3 weakness**: only +34% as small-cap rotation cooled. Strategy regime-dependent.
- **Survivorship**: N500 base list is current. ~5-10% upward bias from delisted names.
- **Universe rebuilds daily**: PIT-strict, uses only prior-day data. No lookahead.

Full ledger: `TRADE_LEDGER.md`