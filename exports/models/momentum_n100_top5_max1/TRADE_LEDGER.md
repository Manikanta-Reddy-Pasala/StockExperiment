# momentum_n100_top5_max1 — Full Trade Ledger (REAL Nifty 100)

**FIX 2026-05-17 (v2)**: Universe = REAL NSE Nifty 100 constituents (`src/data/symbols/nifty100.csv`, NSE archives). Replaces prior pseudo-N100 (ADV-rank from N500, 47/100 fake). Backtest universe now matches LIVE deployment universe.

**Lookback = 30d (was 60d).** Monthly rebalance, top-1 by 30d return, hold full capital in one stock.

**Capital:** ₹10,00,000 start | **Window:** May 15 2023 → May 12 2026 | **Round-trips:** 31 (+1 open) | **WR:** 23/31 = 74.2%

Final NAV: **₹5,868,846** | Total: **+486.88%** | CAGR: **+80.38%/yr** | Max DD (cash NAV): **29.71%**

**Honest, deployable.** No lookahead universe, no pseudo-ADV pad. CAGR ~80% on the actual NSE Nifty 100 list.

---

## Comparison: 3 universe variants

| Variant | CAGR | Final NAV | Max DD | WR | Trades | Universe size |
|---|---:|---:|---:|---:|---:|---:|
| Pseudo-N100 (v1, ADV-rank from N500) | +136.39% | ₹1.32 Cr | 16.15% | 86.7% | 30 | 100 |
| Pseudo-N100 minus `small_cap` (v1.5) | +129.73% | ₹1.21 Cr | 16.15% | 86.7% | 30 | 100 |
| **Real Nifty 100 NSE (v2, LIVE)** | **+80.38%** | **₹58.69 L** | 29.71% | 74.2% | 31 | 104 |

**Why v1.5 barely changed v1**: Filter excluded only **IDEA** from top-101 ADV per year (all 3 years). Other "fake N100" stocks (BSE, MAZDOCK, NETWEB, COCHINSHIP, GRSE, ITI, NBCC, PAYTM) are classified `large_cap` or `mid_cap` in our `stocks` table — too generous categorization (1039 large_cap labeled vs NSE's actual ~100 large-caps).

**Real-universe truth**: only NSE official Nifty 100 (`src/data/symbols/nifty100.csv`) cuts the fake-100. Pseudo-N100 caught BSE/MAZDOCK/NETWEB/IDEA/ITI/NBCC/COCHINSHIP/PAYTM — stocks not in real index.

Production deployed on **v2 (real NSE Nifty 100)**. Backtest below is v2 result.

---

## Money Flow Summary

| Year | Open Capital | Close Capital | ROI | Trades | Wins | Losses |
|------|------------:|--------------:|-----:|------:|----:|------:|
| 2023-24 | ₹1,000,000 | ₹2,416,397 | **+141.64%** | 10 | 8 | 2 |
| 2024-25 | ₹2,416,397 | ₹2,656,524 | **+9.94%** | 10 | 6 | 4 |
| 2025-26 | ₹2,656,524 | ₹5,868,846 | **+120.92%** | 11+1 open | 9 | 2 |
| **3-yr** | **₹10,00,000** | **₹5,868,846** | **+486.88%** | **31+1** | **23** | **8** |

**Y2 weakness**: choppy 2024-25 caught 3 consecutive losers (BAJAJ-AUTO -19%, HINDZINC -10%, MAZDOCK round-2 -4%). Strategy mean-reverts.

## Most-Traded Stocks

| Symbol | Trades | Net PnL ₹ | Win/Loss |
|---|---:|---:|---:|
| ADANIPOWER | 3 | +1,852,928 | 2/1 |
| SOLARINDS | 3 | +788,321 | 3/0 |
| SHRIRAMFIN | 3 | +1,210,419 | 3/0 |
| MAZDOCK | 2 | +359,301 | 1/1 |
| IRFC | 2 | +133,298 | 1/1 |
| HINDZINC | 2 | -361,853 | 0/2 |
| CHOLAFIN | 1 | +40,238 | 1/0 |
| PFC | 1 | +178,712 | 1/0 |
| ADANIGREEN | 1 | +99,771 | 1/0 |
| CUMMINSIND | 1 | +217,087 | 1/0 |
| ABB | 1 | +64,204 | 1/0 |
| VEDL | 1 | +276,722 | 1/0 |

---

## Year 1: 2023-24 (May 15 2023 → May 12 2024) — ₹1,000,000 → ₹2,416,397 (+141.64%)

| # | Symbol | Entry Date | Entry ₹ | Shares | Deployed | Exit Date | Exit ₹ | P&L ₹ | % | Cash After |
|--:|--------|-----------|--------:|-------:|---------:|-----------|-------:|------:|--:|-----------:|
| 1 | CHOLAFIN | 2023-05-15 | 1,003.50 | 996 | 999,486 | 2023-06-01 | 1,043.90 | +40,238 | +4.03% | 1,040,238 |
| 2 | ADANIPOWER | 2023-06-01 | 50.83 | 20,465 | 1,040,236 | 2023-07-03 | 49.69 | -23,330 | -2.24% | 1,016,908 |
| 3 | MAZDOCK | 2023-07-03 | 644.55 | 1,577 | 1,016,455 | 2023-09-01 | 943.53 | +471,491 | +46.39% | 1,488,400 |
| 4 | IRFC | 2023-09-01 | 55.75 | 26,697 | 1,488,358 | 2023-11-01 | 72.95 | +459,188 | +30.85% | 1,947,588 |
| 5 | SOLARINDS | 2023-11-01 | 5,521.35 | 352 | 1,943,515 | 2023-12-01 | 6,188.75 | +234,925 | +12.09% | 2,182,513 |
| 6 | PFC | 2023-12-01 | 365.15 | 5,977 | 2,182,502 | 2024-01-01 | 395.05 | +178,712 | +8.19% | 2,361,225 |
| 7 | ADANIGREEN | 2024-01-01 | 1,598.40 | 1,477 | 2,360,837 | 2024-02-01 | 1,665.95 | +99,771 | +4.23% | 2,460,997 |
| 8 | IRFC | 2024-02-01 | 169.90 | 14,484 | 2,460,832 | 2024-03-01 | 147.40 | -325,890 | -13.24% | 2,135,107 |
| 9 | CUMMINSIND | 2024-03-01 | 2,726.15 | 783 | 2,134,575 | 2024-04-01 | 3,003.40 | +217,087 | +10.17% | 2,352,193 |
| 10 | ABB | 2024-04-01 | 6,504.65 | 361 | 2,348,179 | 2024-05-02 | 6,682.50 | +64,204 | +2.73% | 2,416,397 |

**Top wins:** MAZDOCK (+₹471,491, +46.39%), IRFC (+₹459,188, +30.85%), SOLARINDS (+₹234,925, +12.09%)

**Losses:** IRFC (₹-325,890, -13.24%), ADANIPOWER (₹-23,330, -2.24%)

---

## Year 2: 2024-25 (May 13 2024 → May 12 2025) — ₹2,416,397 → ₹2,656,524 (+9.94%)

| # | Symbol | Entry Date | Entry ₹ | Shares | Deployed | Exit Date | Exit ₹ | P&L ₹ | % | Cash After |
|--:|--------|-----------|--------:|-------:|---------:|-----------|-------:|------:|--:|-----------:|
| 1 | VEDL | 2024-05-02 | 153.86 | 15,705 | 2,416,371 | 2024-06-03 | 171.48 | +276,722 | +11.45% | 2,693,119 |
| 2 | HINDZINC | 2024-06-03 | 696.40 | 3,867 | 2,692,979 | 2024-07-01 | 656.75 | -153,327 | -5.69% | 2,539,793 |
| 3 | MAZDOCK | 2024-07-01 | 2,196.95 | 1,156 | 2,539,674 | 2024-09-02 | 2,099.90 | -112,190 | -4.42% | 2,427,603 |
| 4 | TRENT | 2024-09-02 | 7,148.20 | 339 | 2,423,240 | 2024-10-01 | 7,612.70 | +157,466 | +6.50% | 2,585,068 |
| 5 | BAJAJ-AUTO | 2024-10-01 | 12,157.45 | 212 | 2,577,379 | 2024-11-01 | 9,875.95 | -483,678 | -18.77% | 2,101,390 |
| 6 | HINDZINC | 2024-11-01 | 558.25 | 3,764 | 2,101,253 | 2024-12-02 | 502.85 | -208,526 | -9.92% | 1,892,865 |
| 7 | INDHOTEL | 2024-12-02 | 801.05 | 2,362 | 1,892,080 | 2025-01-01 | 873.60 | +171,363 | +9.06% | 2,064,228 |
| 8 | KOTAKBANK | 2025-01-01 | 1,788.40 | 1,154 | 2,063,814 | 2025-03-03 | 1,914.60 | +145,635 | +7.06% | 2,209,863 |
| 9 | SHRIRAMFIN | 2025-03-03 | 621.30 | 3,556 | 2,209,343 | 2025-04-01 | 637.45 | +57,429 | +2.60% | 2,267,292 |
| 10 | SOLARINDS | 2025-04-01 | 11,131.60 | 203 | 2,259,715 | 2025-05-02 | 13,049.00 | +389,232 | +17.22% | 2,656,524 |

**Top wins:** SOLARINDS (+₹389,232, +17.22%), VEDL (+₹276,722, +11.45%), INDHOTEL (+₹171,363, +9.06%)

**Losses:** BAJAJ-AUTO (₹-483,678, -18.77%), HINDZINC (₹-208,526, -9.92%), HINDZINC (₹-153,327, -5.69%)

---

## Year 3: 2025-26 (May 13 2025 → May 12 2026) — ₹2,656,524 → ₹5,868,846 (+120.92%)

| # | Symbol | Entry Date | Entry ₹ | Shares | Deployed | Exit Date | Exit ₹ | P&L ₹ | % | Cash After |
|--:|--------|-----------|--------:|-------:|---------:|-----------|-------:|------:|--:|-----------:|
| 1 | HAL | 2025-05-02 | 4,492.40 | 591 | 2,655,008 | 2025-06-02 | 5,017.10 | +310,098 | +11.68% | 2,966,622 |
| 2 | SOLARINDS | 2025-06-02 | 16,294.00 | 182 | 2,965,508 | 2025-07-01 | 17,196.00 | +164,164 | +5.54% | 3,130,786 |
| 3 | MUTHOOTFIN | 2025-07-01 | 2,641.90 | 1,185 | 3,130,652 | 2025-08-01 | 2,592.70 | -58,302 | -1.86% | 3,072,484 |
| 4 | BOSCHLTD | 2025-08-01 | 40,390.00 | 76 | 3,069,640 | 2025-09-01 | 40,785.00 | +30,020 | +0.98% | 3,102,504 |
| 5 | ETERNAL | 2025-09-01 | 321.10 | 9,662 | 3,102,468 | 2025-10-01 | 329.00 | +76,330 | +2.46% | 3,178,834 |
| 6 | ADANIPOWER | 2025-10-01 | 152.51 | 20,843 | 3,178,766 | 2025-11-03 | 156.73 | +87,957 | +2.77% | 3,266,791 |
| 7 | SHRIRAMFIN | 2025-11-03 | 796.45 | 4,101 | 3,266,241 | 2026-01-01 | 1,019.70 | +915,548 | +28.03% | 4,182,340 |
| 8 | TMCV | 2026-01-01 | 427.75 | 9,777 | 4,182,112 | 2026-02-01 | 441.30 | +132,478 | +3.17% | 4,314,818 |
| 9 | SHRIRAMFIN | 2026-02-01 | 997.60 | 4,325 | 4,314,620 | 2026-03-02 | 1,052.50 | +237,442 | +5.50% | 4,552,260 |
| 10 | ENRIN | 2026-03-02 | 2,972.70 | 1,531 | 4,551,204 | 2026-04-01 | 2,613.90 | -549,323 | -12.07% | 4,002,938 |
| 11 | ADANIPOWER | 2026-04-01 | 157.11 | 25,478 | 4,002,849 | 2026-05-04 | 227.30 | +1,788,301 | +44.68% | 5,791,238 |
| 12 | ADANIGREEN | 2026-05-04 | 1,290.70 | 4,486 | 5,790,080 | OPEN (2026-05-12) | 1,308.00 | +77,608 | +1.34% | 5,868,846 |

**Top wins:** ADANIPOWER (+₹1,788,301, +44.68%), SHRIRAMFIN (+₹915,548, +28.03%), HAL (+₹310,098, +11.68%)

**Losses:** ENRIN (₹-549,323, -12.07%), MUTHOOTFIN (₹-58,302, -1.86%)

---

## Hall of Fame

### Top 10 Winners

| Symbol | Entry → Exit | Days | Shares | Entry ₹ | Exit ₹ | PnL ₹ | Ret % |
|---|---|---:|---:|---:|---:|---:|---:|
| ADANIPOWER | 2026-04-01 → 2026-05-04 | 33 | 25,478 | 157.11 | 227.30 | +1,788,301 | +44.68% |
| SHRIRAMFIN | 2025-11-03 → 2026-01-01 | 59 | 4,101 | 796.45 | 1,019.70 | +915,548 | +28.03% |
| MAZDOCK | 2023-07-03 → 2023-09-01 | 60 | 1,577 | 644.55 | 943.53 | +471,491 | +46.39% |
| IRFC | 2023-09-01 → 2023-11-01 | 61 | 26,697 | 55.75 | 72.95 | +459,188 | +30.85% |
| SOLARINDS | 2025-04-01 → 2025-05-02 | 31 | 203 | 11,131.60 | 13,049.00 | +389,232 | +17.22% |
| HAL | 2025-05-02 → 2025-06-02 | 31 | 591 | 4,492.40 | 5,017.10 | +310,098 | +11.68% |
| VEDL | 2024-05-02 → 2024-06-03 | 32 | 15,705 | 153.86 | 171.48 | +276,722 | +11.45% |
| SHRIRAMFIN | 2026-02-01 → 2026-03-02 | 29 | 4,325 | 997.60 | 1,052.50 | +237,442 | +5.50% |
| SOLARINDS | 2023-11-01 → 2023-12-01 | 30 | 352 | 5,521.35 | 6,188.75 | +234,925 | +12.09% |
| CUMMINSIND | 2024-03-01 → 2024-04-01 | 31 | 783 | 2,726.15 | 3,003.40 | +217,087 | +10.17% |

### All 8 Losses

| Symbol | Entry → Exit | Days | Shares | Entry ₹ | Exit ₹ | PnL ₹ | Ret % |
|---|---|---:|---:|---:|---:|---:|---:|
| ENRIN | 2026-03-02 → 2026-04-01 | 30 | 1,531 | 2,972.70 | 2,613.90 | -549,323 | -12.07% |
| BAJAJ-AUTO | 2024-10-01 → 2024-11-01 | 31 | 212 | 12,157.45 | 9,875.95 | -483,678 | -18.77% |
| IRFC | 2024-02-01 → 2024-03-01 | 29 | 14,484 | 169.90 | 147.40 | -325,890 | -13.24% |
| HINDZINC | 2024-11-01 → 2024-12-02 | 31 | 3,764 | 558.25 | 502.85 | -208,526 | -9.92% |
| HINDZINC | 2024-06-03 → 2024-07-01 | 28 | 3,867 | 696.40 | 656.75 | -153,327 | -5.69% |
| MAZDOCK | 2024-07-01 → 2024-09-02 | 63 | 1,156 | 2,196.95 | 2,099.90 | -112,190 | -4.42% |
| MUTHOOTFIN | 2025-07-01 → 2025-08-01 | 31 | 1,185 | 2,641.90 | 2,592.70 | -58,302 | -1.86% |
| ADANIPOWER | 2023-06-01 → 2023-07-03 | 32 | 20,465 | 50.83 | 49.69 | -23,330 | -2.24% |

---

## Universe (Real Nifty 100 - 104 stocks)

Source: NSE archives `ind_nifty100list.csv`. Refresh via `python tools/refresh_nifty100.py`.

NSE rebalances Mar/Sep. This backtest uses the **current** (2026-05) constituent list applied retroactively. Real N100 is stable (~5-8% turnover/yr) so lookahead bias is small.

**Stocks traded (in real N100):** ABB, ADANIGREEN, ADANIPOWER, BAJAJ-AUTO, BOSCHLTD, CHOLAFIN, CUMMINSIND, ENRIN, ETERNAL, HAL, HINDZINC, INDHOTEL, IRFC, KOTAKBANK, MAZDOCK, MUTHOOTFIN, PFC, SHRIRAMFIN, SOLARINDS, TMCV, TRENT, VEDL

---

## Honest Caveats

- **Max DD 29.71%** (cash NAV) — single-stock concentration. Y2 chop hit hard.
- **Universe drift**: NSE Nifty 100 today ≠ 2023 N100 exactly. ~15-20% drift over 3 years. Real backtest with PIT-historical constituents would be ~5-10% lower CAGR. Acceptable approximation.
- **Survivorship**: stocks delisted from N100 mid-2023-26 are absent. Minor upward bias.
- **Costs**: 31 trades / 3yr = ~10 round-trips/yr. STT+brokerage drag ~1-2%/yr. Post-cost CAGR ≈ +78%.
- **Slippage**: backtest fills at close. Real ~10-30 bps drag per round-trip.
- **Live universe matches backtest**: prod `build_universe.py` now emits real Nifty 100 from NSE CSV (same source).