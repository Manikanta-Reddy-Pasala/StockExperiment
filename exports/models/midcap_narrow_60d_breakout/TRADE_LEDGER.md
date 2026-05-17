# midcap_narrow_60d_breakout — Full Trade Ledger (V2: Mid+Small only, ANGELONE data-fixed)

**Data matches SUMMARY.md** — V2 cap filter (excludes Large from pseudo-midcap pool) + ANGELONE data fix applied (prices ÷10 in 2024-12-23 → 2026-02-25 window for split-adjustment inconsistency).

Capital ₹10L → ₹6,500,421 (+550.04%) · CAGR +86.63% · 12 trades · Max DD 15.15% · Calmar 5.72

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Mid** | 3 | 2 | 1 | 67% | +1,377,589 |
| **Small** | 9 | 7 | 2 | 78% | +4,208,705 |

All trades Mid + Small (Large filtered by V2 cap rule).

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹1,909,855 | **+90.99%** | 3 |
| 2024-25 | ₹1,909,855 | ₹2,487,636 | **+30.25%** | 4 |
| 2025-26 | ₹2,487,636 | ₹6,500,421 | **+161.31%** | 5 |

## All 12 trades

| # | Symbol | Cap | Index | Entry Date | Entry ₹ | Qty | Invested ₹ | Exit Date | Exit ₹ | PnL ₹ | Ret % | Reason |
|--:|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| 1 | CEMPRO | **Small** | Nifty Smallcap 250 | 2023-05-17 | 140.54 | 7,115 | ₹999,942 | 2023-08-16 | 199.90 | +420,901 | +42.38% | MAX_HOLD |
| 2 | COHANCE | **Small** | Nifty Smallcap 250 | 2023-08-17 | 525.52 | 2,703 | ₹1,420,481 | 2023-11-15 | 569.63 | +117,656 | +8.50% | MAX_HOLD |
| 3 | ARE&M | **Small** | Nifty Smallcap 250 | 2023-11-24 | 670.72 | 2,293 | ₹1,537,961 | 2024-02-22 | 833.52 | +371,359 | +24.40% | MAX_HOLD |
| 4 | DATAPATTNS | **Small** | Nifty Smallcap 250 | 2024-02-23 | 2,267.26 | 842 | ₹1,909,033 | 2024-05-23 | 3,092.55 | +692,270 | +36.54% | MAX_HOLD |
| 5 | SCI | **Small** | Nifty Smallcap 250 | 2024-05-24 | 262.21 | 9,923 | ₹2,601,910 | 2024-08-22 | 273.63 | +110,527 | +4.46% | MAX_HOLD |
| 6 | CDSL | **Small** | Nifty Smallcap 250 | 2024-08-26 | 1,601.60 | 1,693 | ₹2,711,509 | 2024-11-25 | 1,553.00 | -84,937 | -2.94% | MAX_HOLD |
| 7 | PERSISTENT | **Mid** | Nifty Midcap 150 | 2024-11-26 | 5,945.94 | 441 | ₹2,622,160 | 2025-02-24 | 5,634.16 | -140,000 | -5.15% | MAX_HOLD |
| 8 | CEMPRO | **Small** | Nifty Smallcap 250 | 2025-03-12 | 548.50 | 4,535 | ₹2,487,448 | 2025-06-10 | 807.24 | +1,169,723 | +47.32% | MAX_HOLD |
| 9 | WOCKPHARMA | **Small** | Nifty Smallcap 250 | 2025-06-12 | 1,834.83 | 1,993 | ₹3,656,816 | 2025-09-10 | 1,558.34 | -554,176 | -14.98% | MAX_HOLD |
| 10 | INDIANB | **Mid** | Nifty Midcap 150 | 2025-09-11 | 695.64 | 4,460 | ₹3,102,554 | 2025-12-10 | 782.02 | +381,712 | +12.53% | MAX_HOLD |
| 11 | HINDCOPPER | **Small** | Nifty Smallcap 250 | 2025-12-15 | 382.38 | 9,113 | ₹3,484,629 | 2026-02-01 | 598.65 | +1,965,382 | +56.72% | TRAIL |
| 12 | BHARATFORG | **Mid** | Nifty Midcap 150 | 2026-02-04 | 1,541.54 | 3,535 | ₹5,449,344 | 2026-05-05 | 1,864.73 | +1,135,877 | +21.09% | MAX_HOLD |

## Notes

- V2 = Exclude Large from pseudo-midcap pool. Cap-filter sweep tested 6 variants; V2 wins on all 3 metrics (CAGR, DD, Calmar).
- ANGELONE data fix applied (prices ÷10 in corrupted window). With clean data, ANGELONE never qualifies for breakout entry — explicit exclusion no longer needed.
- Universe = pseudo-midcap (N500 skip-30 ADV, take next 100) MINUS NSE Nifty 100 (ANGELONE data fixed, eligible but not picked).
- Full V1 (all caps + ANGELONE): +337% CAGR / ₹8.38 Cr but inflated by anomaly. See git history for original.