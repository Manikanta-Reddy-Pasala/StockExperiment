# midcap_narrow_60d_breakout — Honest Trade Ledger (REAL Nifty Midcap 150)

**Strategy STATUS: BROKEN.** Honest backtest on real NSE Nifty Midcap 150 universe yields -18.18% CAGR (₹2L → ₹1.10L). Recommend archive.

**FIX 2026-05-17**: Universe = REAL NSE Nifty Midcap 150 (`src/data/symbols/nifty_midcap150.csv`). Previously used pseudo-midcap ("skip top-30 ADV from N500, take next 100") with TODAY's ADV applied retroactively = lookahead.

**Capital:** ₹2,00,000 start | **Window:** May 15 2023 → May 15 2026 | **Round-trips:** 49 (+1 open) | **WR:** 18/49 = 36.7%

Final NAV: **₹109,536** | Total: **-45.23%** | CAGR: **-18.18%/yr** | Max DD: **70.39%**

**Exit reasons**: {'SMA': 35, 'MAX_HOLD': 14}. **ZERO target hits, ZERO trail hits.** All exits are SMA20 whipsaw or MAX_HOLD timeout. Strategy fails to capture +60% target.

---

## What went wrong: initial claim vs honest reality

| Metric | Original claim (lookahead pseudo-midcap) | Honest (real Nifty Midcap 150) | Δ |
|---|---:|---:|---:|
| CAGR | +121.66% | **-18.18%** | -140pp |
| Total return | +989.67% | -45.23% | wipeout |
| Max DD | -20.43% | -70.39% | 3.5× worse |
| Trades | 34 | 49 | more whipsaws |
| Win rate | unstated | 36.7% | poor |
| Y1 ROI | +95.31% | +34.19% | -61pp |
| Y2 ROI | +110.42% | **-31.57%** | -142pp |
| Y3 ROI | +55-71% | **-40.36%** | -111pp |

### Root causes

1. **Lookahead universe** (same bug as momentum_n100): old `build_universe.py` rebuilt midcap_narrow using TODAY's ADV applied retroactively. Strategy 'knew' which mid-caps would become winners.
2. **Universe membership matters more than rules**: pseudo-midcap caught BSE, MAZDOCK, NETWEB, COCHINSHIP, ETERNAL etc. — stocks that 3-10x'd post-2023. Real Nifty Midcap 150 (current list applied back) doesn't contain those pre-graduation surge winners.
3. **60d breakout in midcap = high failure rate**: breakouts to new highs in mid-caps are noisy. Volume confirmation (vol > 2× 20d avg) doesn't filter out fake breakouts. SMA20 exit triggers on every dip = whipsaw losses.
4. **No +60% targets hit**: across 49 trades over 3 years, zero positions reached +60%. Trail stop at -15% from peak after +10% never armed. MAX_HOLD 30d captures only modest moves.

### Worst trades

| Symbol | Entry → Exit | PnL | Ret | Reason |
|---|---|---:|---:|---|
| BHEL | 2024-03-05 → 2024-03-13 | -42,777 | -15.23% | SMA |
| KPRMILL | 2024-08-07 → 2024-08-13 | -28,065 | -9.39% | SMA |
| KPRMILL | 2025-05-12 → 2025-05-29 | -21,902 | -11.82% | SMA |
| SUNDARMFIN | 2025-03-24 → 2025-03-27 | -20,306 | -9.31% | SMA |
| PAGEIND | 2024-11-11 → 2024-11-18 | -19,210 | -7.84% | SMA |
| TATAINVEST | 2025-10-01 → 2025-10-16 | -18,425 | -15.09% | SMA |

### Best trades (still modest)

| Symbol | Entry → Exit | PnL | Ret | Reason |
|---|---|---:|---:|---|
| EXIDEIND | 2024-04-10 → 2024-05-10 | +39,815 | +17.69% | MAX_HOLD |
| OIL | 2023-12-21 → 2024-01-20 | +39,772 | +15.64% | MAX_HOLD |
| THERMAX | 2026-04-10 → 2026-05-11 | +27,469 | +32.63% | MAX_HOLD |
| SRF | 2025-01-10 → 2025-02-10 | +26,603 | +12.91% | MAX_HOLD |
| IPCALAB | 2023-10-25 → 2023-11-24 | +21,638 | +9.29% | MAX_HOLD |

Best trade in 3 years: THERMAX +32.63% / +₹27,469. Not close to +60% target.

---

## Money Flow Summary

| Year | Open Capital | Close Capital | ROI | Trades |
|------|------------:|--------------:|-----:|------:|
| 2023-24 | ₹200,000 | ₹268,376 | **+34.19%** | 14 |
| 2024-25 | ₹268,356 | ₹183,657 | **-31.56%** | 17 |
| 2025-26 | ₹183,637 | ₹109,536 | **-40.35%** | 18 |
| **3-yr** | **₹200,000** | **₹109,536** | **-45.23%** | **49** |

## Year 1: 2023-24 (May 15 2023 → May 12 2024) — ₹200,000 → ₹268,376 (+34.19%)

| # | Symbol | Entry | Entry ₹ | Shares | Exit | Exit ₹ | P&L ₹ | Ret % | Reason |
|--:|--------|-------|--------:|-------:|------|-------:|------:|------:|--------|
| 1 | BALKRISIND | 2023-05-23 | 2,302.30 | 86 | 2023-05-30 | 2,223.32 | -7,003 | -3.33% | SMA |
| 2 | JKCEMENT | 2023-05-31 | 3,160.71 | 61 | 2023-06-30 | 3,378.52 | +13,060 | +7.00% | MAX_HOLD |
| 3 | NAM-INDIA | 2023-07-03 | 290.24 | 709 | 2023-08-02 | 312.74 | +15,709 | +7.86% | MAX_HOLD |
| 4 | BERGEPAINT | 2023-08-03 | 600.60 | 369 | 2023-09-04 | 589.24 | -4,429 | -1.79% | MAX_HOLD |
| 5 | RVNL | 2023-09-05 | 158.16 | 1,373 | 2023-10-05 | 170.03 | +16,047 | +7.61% | MAX_HOLD |
| 6 | GODREJIND | 2023-10-06 | 602.15 | 387 | 2023-10-23 | 617.13 | +5,539 | +2.59% | SMA |
| 7 | IPCALAB | 2023-10-25 | 999.00 | 239 | 2023-11-24 | 1,090.71 | +21,638 | +9.29% | MAX_HOLD |
| 8 | GICRE | 2023-11-28 | 307.31 | 847 | 2023-12-20 | 305.19 | -2,068 | -0.59% | SMA |
| 9 | OIL | 2023-12-21 | 242.24 | 1,066 | 2024-01-20 | 279.85 | +39,772 | +15.64% | MAX_HOLD |
| 10 | NHPC | 2024-01-23 | 83.38 | 3,574 | 2024-02-12 | 80.92 | -9,117 | -2.86% | SMA |
| 11 | MEDANTA | 2024-02-13 | 1,347.85 | 214 | 2024-03-01 | 1,297.35 | -11,104 | -3.65% | SMA |
| 12 | BHEL | 2024-03-05 | 265.77 | 1,045 | 2024-03-13 | 225.07 | -42,777 | -15.23% | SMA |
| 13 | COLPAL | 2024-03-15 | 2,690.69 | 87 | 2024-04-09 | 2,619.68 | -6,426 | -2.54% | SMA |
| 14 | EXIDEIND | 2024-04-10 | 384.28 | 594 | 2024-05-10 | 451.80 | +39,815 | +17.69% | MAX_HOLD |

---

## Year 2: 2024-25 (May 13 2024 → May 12 2025) — ₹268,356 → ₹183,657 (-31.56%)

| # | Symbol | Entry | Entry ₹ | Shares | Exit | Exit ₹ | P&L ₹ | Ret % | Reason |
|--:|--------|-------|--------:|-------:|------|-------:|------:|------:|--------|
| 1 | POLYCAB | 2024-05-13 | 6,275.27 | 42 | 2024-06-04 | 6,449.74 | +7,037 | +2.88% | SMA |
| 2 | DABUR | 2024-06-06 | 600.85 | 458 | 2024-07-02 | 602.20 | +321 | +0.32% | SMA |
| 3 | NLCINDIA | 2024-07-03 | 261.26 | 1,055 | 2024-08-02 | 278.62 | +18,001 | +6.75% | SMA |
| 4 | KPRMILL | 2024-08-07 | 961.01 | 305 | 2024-08-13 | 869.93 | -28,065 | -9.39% | SMA |
| 5 | IPCALAB | 2024-08-14 | 1,407.46 | 188 | 2024-09-13 | 1,472.68 | +11,964 | +4.74% | MAX_HOLD |
| 6 | KALYANKJIL | 2024-09-16 | 723.92 | 383 | 2024-10-07 | 701.50 | -8,878 | -3.00% | SMA |
| 7 | DIXON | 2024-10-09 | 14,564.55 | 18 | 2024-10-25 | 13,923.26 | -11,814 | -4.31% | SMA |
| 8 | PAGEIND | 2024-11-11 | 47,847.80 | 5 | 2024-11-18 | 44,053.95 | -19,210 | -7.84% | SMA |
| 9 | FORTIS | 2024-11-22 | 686.69 | 345 | 2024-12-17 | 686.26 | -403 | +0.04% | SMA |
| 10 | KPRMILL | 2024-12-19 | 1,109.61 | 213 | 2024-12-26 | 1,031.67 | -16,841 | -6.93% | SMA |
| 11 | COFORGE | 2024-12-31 | 1,941.95 | 113 | 2025-01-09 | 1,854.74 | -10,084 | -4.40% | SMA |
| 12 | SRF | 2025-01-10 | 2,529.58 | 83 | 2025-02-10 | 2,853.19 | +26,603 | +12.91% | MAX_HOLD |
| 13 | SBICARD | 2025-02-14 | 861.86 | 274 | 2025-03-03 | 827.12 | -9,765 | -3.93% | SMA |
| 14 | MEDANTA | 2025-03-07 | 1,244.34 | 182 | 2025-03-11 | 1,173.33 | -13,159 | -5.61% | SMA |
| 15 | SUNDARMFIN | 2025-03-24 | 4,969.96 | 43 | 2025-03-27 | 4,502.69 | -20,306 | -9.31% | SMA |
| 16 | BANKINDIA | 2025-04-04 | 114.86 | 1,684 | 2025-05-05 | 116.24 | +2,106 | +1.30% | MAX_HOLD |
| 17 | HINDPETRO | 2025-05-06 | 410.41 | 476 | 2025-05-09 | 385.86 | -11,888 | -5.89% | SMA |

---

## Year 3: 2025-26 (May 13 2025 → May 15 2026) — ₹183,637 → ₹109,536 (-40.35%)

| # | Symbol | Entry | Entry ₹ | Shares | Exit | Exit ₹ | P&L ₹ | Ret % | Reason |
|--:|--------|-------|--------:|-------:|------|-------:|------:|------:|--------|
| 1 | KPRMILL | 2025-05-12 | 1,303.30 | 140 | 2025-05-29 | 1,148.15 | -21,902 | -11.82% | SMA |
| 2 | ICICIPRULI | 2025-05-30 | 666.87 | 242 | 2025-06-12 | 631.02 | -8,848 | -5.28% | SMA |
| 3 | OIL | 2025-06-13 | 480.48 | 318 | 2025-06-24 | 445.40 | -11,316 | -7.21% | SMA |
| 4 | ENDURANCE | 2025-06-26 | 2,697.80 | 52 | 2025-07-11 | 2,620.38 | -4,182 | -2.77% | SMA |
| 5 | GLENMARK | 2025-07-14 | 2,172.27 | 63 | 2025-08-01 | 2,065.73 | -6,862 | -4.81% | SMA |
| 6 | RADICO | 2025-08-04 | 2,867.86 | 45 | 2025-08-28 | 2,837.16 | -1,529 | -0.97% | SMA |
| 7 | WAAREEENER | 2025-08-29 | 3,418.82 | 37 | 2025-09-26 | 3,204.39 | -8,072 | -6.18% | SMA |
| 8 | TATAINVEST | 2025-10-01 | 1,067.07 | 113 | 2025-10-16 | 905.09 | -18,425 | -15.09% | SMA |
| 9 | PHOENIXLTD | 2025-10-17 | 1,676.67 | 61 | 2025-11-17 | 1,743.85 | +3,972 | +4.11% | MAX_HOLD |
| 10 | HEROMOTOCO | 2025-11-18 | 5,825.82 | 18 | 2025-12-09 | 5,995.00 | +2,917 | +3.01% | SMA |
| 11 | AUBANK | 2025-12-11 | 981.98 | 111 | 2026-01-12 | 1,006.79 | +2,622 | +2.63% | MAX_HOLD |
| 12 | AIIL | 2026-01-14 | 645.64 | 173 | 2026-01-20 | 609.49 | -6,380 | -5.51% | SMA |
| 13 | DALBHARAT | 2026-01-22 | 2,245.84 | 46 | 2026-01-23 | 2,082.82 | -7,615 | -7.17% | SMA |
| 14 | APLAPOLLO | 2026-01-27 | 2,032.63 | 48 | 2026-02-26 | 2,224.17 | +9,067 | +9.53% | MAX_HOLD |
| 15 | SCHAEFFLER | 2026-02-27 | 4,349.34 | 24 | 2026-03-11 | 4,046.35 | -7,389 | -6.87% | SMA |
| 16 | ASTRAL | 2026-03-12 | 1,720.72 | 57 | 2026-03-13 | 1,610.19 | -6,412 | -6.33% | SMA |
| 17 | THERMAX | 2026-03-17 | 3,262.36 | 28 | 2026-03-23 | 3,098.00 | -4,709 | -4.94% | SMA |
| 18 | THERMAX | 2026-04-10 | 3,538.64 | 24 | 2026-05-11 | 4,688.71 | +27,469 | +32.63% | MAX_HOLD |

---

## Recommendation

**Archive this model.** 3-year unprofitable on honest universe. The strategy depends on lookahead universe drift (pseudo-midcap containing pre-graduation winners) — without that, mean-reversion in the actual midcap-150 wipes out gains.

Alternatives:
- Use `momentum_n100_top5_max1` (real Nifty 100, +80% CAGR honest) for large-cap exposure.
- For midcap exposure, try a different signal (e.g., relative-strength rank vs Nifty Midcap 100 index, not breakout).
- Or buy Nifty Midcap 150 ETF for passive exposure (+18-25% CAGR Indian midcap returns 2023-2026 broadly).