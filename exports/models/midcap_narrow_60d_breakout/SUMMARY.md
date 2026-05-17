# midcap_narrow_60d_breakout — SUMMARY (ANGELONE EXCLUDED)

**Honest result with corp-action anomaly removed.** ANGELONE excluded from pseudo-midcap universe (entry ₹316 → exit ₹2856 = 9x in 2 months in `historical_data` — likely unadjusted bonus/split).

**3-year backtest** (2023-05-15 → 2026-05-15, ₹10L start, Pseudo-midcap V1 winner config, ANGELONE filtered)

| Metric | Value |
|---|---:|
| Final NAV | **₹4,792,492** |
| Total return | **+379.25%** |
| **3-yr CAGR** | **+68.60%/yr** (below 80% threshold) |
| Max DD (cash NAV) | 17.83% |
| Round-trips | 12 |
| Win rate | 75.0% (9W / 3L) |
| Calmar | 3.85 |
| Exit reasons | {'TARGET': 1, 'MAX_HOLD': 11} |

## With vs without ANGELONE

| Variant | CAGR | Final NAV | Max DD | Trades | WR |
|---|---:|---:|---:|---:|---:|
| With ANGELONE (full V1) | +337.62% | ₹8.38 Cr | 6.76% | 13 | 92.3% |
| **ANGELONE excluded** | **+68.60%** | **₹4,792,492** | 17.83% | 12 | 75.0% |

ANGELONE alone contributed ~₹7.9 Cr — without it, strategy delivers honest +68% CAGR (still strong but below user's 80% bar).

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹3,343,026 | **+234.30%** | 4 |
| 2024-25 | ₹3,343,026 | ₹5,073,918 | **+51.78%** | 4 |
| 2025-26 | ₹5,073,918 | ₹4,792,492 | **-5.55%** | 4 |

Y3 turned negative — strategy fragility without anomalous winner.

## All 12 trades (no ANGELONE)

| # | Symbol | Entry → Exit | Qty | Entry ₹ | Exit ₹ | PnL ₹ | Ret % | Reason |
|--:|---|---|---:|---:|---:|---:|---:|---|
| 1 | MAZDOCK | 2023-05-17 → 2023-07-12 | 2,454 | 407.39 | 865.06 | +1,120,997 | +112.56% | TARGET |
| 2 | INDIANB | 2023-07-13 → 2023-10-11 | 6,519 | 325.32 | 422.73 | +632,187 | +30.07% | MAX_HOLD |
| 3 | GMDCLTD | 2023-10-13 → 2024-01-11 | 6,812 | 404.15 | 466.38 | +420,710 | +15.51% | MAX_HOLD |
| 4 | CHENNPETRO | 2024-01-12 → 2024-04-12 | 3,739 | 848.75 | 894.90 | +169,212 | +5.54% | MAX_HOLD |
| 5 | HINDZINC | 2024-04-15 → 2024-07-15 | 7,858 | 425.42 | 659.04 | +1,830,550 | +55.07% | MAX_HOLD |
| 6 | OFSS | 2024-07-16 → 2024-10-14 | 471 | 10,960.95 | 11,719.72 | +351,840 | +7.03% | MAX_HOLD |
| 7 | HDFCAMC | 2024-10-16 → 2025-01-14 | 2,420 | 2,282.28 | 1,930.60 | -855,764 | -15.32% | MAX_HOLD |
| 8 | INDUSTOWER | 2025-01-21 → 2025-04-21 | 12,476 | 374.27 | 407.09 | +404,346 | +8.88% | MAX_HOLD |
| 9 | TATACONSUM | 2025-04-25 → 2025-07-24 | 4,368 | 1,161.36 | 1,071.73 | -396,218 | -7.63% | MAX_HOLD |
| 10 | PAYTM | 2025-07-25 → 2025-10-23 | 4,248 | 1,101.10 | 1,282.82 | +766,460 | +16.62% | MAX_HOLD |
| 11 | SCI | 2025-10-24 → 2026-01-22 | 21,930 | 248.25 | 207.23 | -904,033 | -16.44% | MAX_HOLD |
| 12 | INDIANB | 2026-01-23 → 2026-04-23 | 5,020 | 904.35 | 914.04 | +43,993 | +1.17% | MAX_HOLD |

## Top 3 winners (ex-ANGELONE)

| Symbol | Entry → Exit | PnL ₹ | Ret % |
|---|---|---:|---:|
| HINDZINC | 2024-04-15 → 2024-07-15 | +1,830,550 | +55.07% |
| MAZDOCK | 2023-05-17 → 2023-07-12 | +1,120,997 | +112.56% |
| PAYTM | 2025-07-25 → 2025-10-23 | +766,460 | +16.62% |

HINDZINC +55%, MAZDOCK +112%, INDIANB +30% — honest mid-cap winners.

## All 3 losses

| Symbol | Entry → Exit | PnL ₹ | Ret % |
|---|---|---:|---:|
| SCI | 2025-10-24 → 2026-01-22 | -904,033 | -16.44% |
| HDFCAMC | 2024-10-16 → 2025-01-14 | -855,764 | -15.32% |
| TATACONSUM | 2025-04-25 → 2025-07-24 | -396,218 | -7.63% |

## Strategy parameters

Same as V1 winner: hh=40 day breakout, vm=2.0 vol confirm, tg=+100%, tr=-20% from peak, mh=90d, NO SMA20 exit. Universe = pseudo-midcap (N500 skip-30 ADV, take next 100) MINUS ANGELONE.

## Caveats

- **CAGR drops from +337% to +68%** when 1 anomalous trade removed. Strategy fragility exposed.
- **Real Nifty Midcap 150 (NSE official) result on same strategy: -18% CAGR**. Pseudo-midcap universe lookahead is doing most of the work.
- **Y3 -5.55%** — strategy struggling in latest year.
- **Below 80% CAGR threshold** — would normally be discarded per user's filter rule.
- **Not production-ready**.

Full ledger with ANGELONE included: `TRADE_LEDGER.md`