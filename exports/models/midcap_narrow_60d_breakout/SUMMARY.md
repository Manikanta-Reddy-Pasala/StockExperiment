# midcap_narrow_60d_breakout — SUMMARY

**3-year backtest** (2023-05-15 → 2026-05-15, ₹10L start, Pseudo-midcap universe, V1 WINNER config)

| Metric | Value |
|---|---:|
| Final NAV | **₹83,811,502** |
| Total return | **+8281.15%** |
| **3-yr CAGR** | **+337.62%/yr** |
| Max DD (cash NAV) | 6.76% |
| Round-trips | 13 (+1 open) |
| Win rate | 92.3% (12W / 1L) |
| Calmar (CAGR/MaxDD) | 49.94 |
| Exit reasons | {'TARGET': 2, 'MAX_HOLD': 10, 'TRAIL': 1} |

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹3,343,026 | **+234.30%** | 4 |
| 2024-25 | ₹3,343,026 | ₹53,590,783 | **+1503.06%** | 4 |
| 2025-26 | ₹53,590,783 | ₹83,811,502 | **+56.39%** | 5 (+1 open) |

## Strategy parameters (V1 WINNER from 486-config sweep)

| Param | Value | vs original V1 |
|---|---|---|
| Breakout window | 40-day high | (was 60) |
| Volume confirm | 2.0× 20d avg | same |
| Stage-2 filter | close > 200d SMA | same |
| Target | +100% | (was +60%) |
| Trail | -20% from peak after +10% | (was -15%) |
| Max hold | 90 trading days | (was 30) |
| SMA20 exit | **DISABLED** | **was enabled** |
| Universe | Pseudo-midcap (N500 skip-30 ADV, take next 100) | same |
| Costs | 10 bps slip + ₹20 brk + 0.10% STT | same |

**Key sweep finding**: removing SMA20 exit was the single biggest CAGR boost. Strategy was chopping winners on routine pullbacks. MAX_HOLD or TARGET take it instead.

## Top 5 winners

| Symbol | Entry → Exit | PnL ₹ | Ret % | Reason |
|---|---|---:|---:|---|
| ANGELONE | 2024-10-16 → 2024-12-23 | +44,245,561 | +802.59% | TARGET |
| BHARATFORG | 2026-02-04 → 2026-05-05 | +14,647,105 | +21.09% | MAX_HOLD |
| HINDCOPPER | 2025-12-30 → 2026-02-01 | +13,467,157 | +23.96% | TRAIL |
| INDIGO | 2024-12-26 → 2025-03-26 | +3,819,886 | +7.89% | MAX_HOLD |
| HINDPETRO | 2025-09-30 → 2025-12-29 | +3,686,300 | +7.15% | MAX_HOLD |

**⚠️ ANGELONE caveat**: trade #7 added ₹4.42 Cr (~53% of total profit). Entry ₹316.82 → exit ₹2856.69 = 9x in 2 months — likely corporate-action data anomaly (bonus/split unadjusted). Real returns on this single trade would be a fraction.

## Only loss

| Symbol | Entry → Exit | PnL ₹ | Ret % |
|---|---|---:|---:|
| HDFCLIFE | 2025-06-30 → 2025-09-29 | -3,849,236 | -6.57% |

## Universe source

- Source: `src/data/symbols/nifty500.csv` (NSE 500)
- Method: Compute 20-day ADV → sort desc → **skip top-30** (large-caps, in N100 model) → **take next 100** = pseudo-midcap (ADV-rank 31-130)
- Universe frozen at end-of-backtest snapshot (lookahead bias acknowledged)
- Build: `python tools/models/midcap_narrow_60d_breakout/build_universe.py --skip-top 30 --top 100`
- First 10 (end-2026): ADANIGREEN, SUZLON, ADANIPORTS, SHRIRAMFIN, JIOFIN, NETWEB, WAAREEENER, SCI, ITC, SAIL

## Caveats

- **ANGELONE data anomaly**: 53% of returns from one trade with suspicious 9x in 2 months.
- **Lookahead universe**: pseudo-midcap built with end-of-data ADV. Real-time would have different ranking.
- **Real Nifty Midcap 150 (NSE official) result on same strategy: -18.18% CAGR**. Strategy entirely dependent on lookahead universe.
- **Not production-ready** — backtest result is upper-bound exploration.

Full ledger: `TRADE_LEDGER.md`