# BPCL (BPCL)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 302.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 6 |
| PENDING | 30 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 10 |
| ENTRY2 | 14 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 20 / 9
- **Target hits / Stop hits / Partials:** 0 / 21 / 8
- **Avg / median % per leg:** 6.43% / 7.89%
- **Sum % (uncompounded):** 186.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 16 | 72.7% | 0 | 16 | 6 | 6.32% | 139.1% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 0 | 5 | 2 | 6.13% | 42.9% |
| BUY @ 3rd Alert (retest2) | 15 | 10 | 66.7% | 0 | 11 | 4 | 6.41% | 96.2% |
| SELL (all) | 7 | 4 | 57.1% | 0 | 5 | 2 | 6.77% | 47.4% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 13.63% | 54.5% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.37% | -7.1% |
| retest1 (combined) | 11 | 10 | 90.9% | 0 | 7 | 4 | 8.85% | 97.4% |
| retest2 (combined) | 18 | 10 | 55.6% | 0 | 14 | 4 | 4.95% | 89.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 11:15:00 | 191.95 | 177.59 | 177.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 10:15:00 | 194.27 | 179.40 | 178.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 14:15:00 | 297.10 | 300.72 | 277.79 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-03-14 14:15:00 | 304.30 | 300.78 | 278.61 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-14 15:15:00 | 304.00 | 300.81 | 278.74 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 14:15:00 | 278.60 | 298.30 | 279.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-19 14:15:00 | 278.60 | 298.30 | 279.52 | SL hit (close<ema400) qty=1.00 sl=279.52 alert=retest1 |
| Cross detected — sustain check pending | 2024-03-20 09:15:00 | 281.62 | 297.94 | 279.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 10:15:00 | 281.75 | 297.78 | 279.54 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 12:15:00 | 324.01 | 305.10 | 297.73 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-31 09:15:00 | 309.75 | 311.91 | 303.34 | SL hit (close<ema200) qty=0.50 sl=311.91 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-05 10:15:00 | 282.02 | 311.89 | 304.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 11:15:00 | 281.55 | 311.59 | 304.15 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:15:00 | 323.78 | 307.62 | 305.75 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-13 11:15:00 | 325.95 | 326.61 | 318.17 | SL hit (close<ema200) qty=0.50 sl=326.61 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 11:15:00 | 310.85 | 336.28 | 336.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 13:15:00 | 308.55 | 335.75 | 336.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 12:15:00 | 303.85 | 303.70 | 313.91 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-12-13 10:15:00 | 298.10 | 303.64 | 312.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:15:00 | 298.60 | 303.59 | 312.25 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-16 09:15:00 | 298.05 | 303.46 | 311.97 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:15:00 | 296.20 | 303.39 | 311.89 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 09:15:00 | 253.81 | 279.64 | 290.52 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 09:15:00 | 251.77 | 279.64 | 290.52 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 260.96 | 256.40 | 268.34 | SL hit (close>ema200) qty=0.50 sl=256.40 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 260.96 | 256.40 | 268.34 | SL hit (close>ema200) qty=0.50 sl=256.40 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 267.35 | 257.75 | 267.50 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-03-12 11:15:00 | 264.27 | 257.90 | 267.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 12:15:00 | 264.39 | 257.96 | 267.47 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-13 09:15:00 | 265.44 | 258.27 | 267.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:15:00 | 264.00 | 258.33 | 267.41 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 270.96 | 259.53 | 266.86 | SL hit (close>static) qty=1.00 sl=270.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 270.96 | 259.53 | 266.86 | SL hit (close>static) qty=1.00 sl=270.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 286.05 | 271.60 | 271.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 11:15:00 | 289.50 | 272.14 | 271.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 309.90 | 310.83 | 300.80 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-04 11:15:00 | 311.75 | 310.80 | 300.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 12:15:00 | 311.30 | 310.81 | 301.04 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-06 09:15:00 | 313.60 | 310.90 | 301.61 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:15:00 | 315.75 | 310.95 | 301.68 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 309.45 | 314.25 | 304.91 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-13 11:15:00 | 313.15 | 314.19 | 304.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:15:00 | 311.80 | 314.17 | 305.01 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-16 12:15:00 | 311.50 | 313.98 | 305.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 316.25 | 314.00 | 305.28 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-23 10:15:00 | 310.75 | 313.90 | 306.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 311.10 | 313.87 | 306.55 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 12:15:00 | 357.99 | 324.89 | 315.13 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 12:15:00 | 358.57 | 324.89 | 315.13 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 12:15:00 | 357.77 | 324.89 | 315.13 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 336.40 | 336.68 | 325.93 | SL hit (close<ema200) qty=0.50 sl=336.68 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 336.40 | 336.68 | 325.93 | SL hit (close<ema200) qty=0.50 sl=336.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 336.40 | 336.68 | 325.93 | SL hit (close<ema200) qty=0.50 sl=336.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 326.90 | 336.27 | 327.11 | SL hit (close<ema400) qty=1.00 sl=327.11 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-07 14:15:00 | 310.35 | 329.96 | 325.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-07 15:15:00 | 310.10 | 329.77 | 325.28 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-08 09:15:00 | 316.80 | 329.64 | 325.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 317.95 | 329.52 | 325.20 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 318.40 | 329.41 | 325.17 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-08-11 14:15:00 | 321.10 | 328.50 | 324.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 15:15:00 | 320.95 | 328.42 | 324.89 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 315.10 | 327.07 | 324.55 | SL hit (close<static) qty=1.00 sl=317.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-19 14:15:00 | 320.95 | 325.84 | 324.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 15:15:00 | 321.90 | 325.80 | 324.05 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-20 12:15:00 | 322.10 | 325.60 | 323.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-20 13:15:00 | 320.90 | 325.55 | 323.97 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-21 09:15:00 | 321.85 | 325.40 | 323.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:15:00 | 322.95 | 325.38 | 323.91 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 317.40 | 324.98 | 323.76 | SL hit (close<static) qty=1.00 sl=317.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 317.40 | 324.98 | 323.76 | SL hit (close<static) qty=1.00 sl=317.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-28 14:15:00 | 310.55 | 322.58 | 322.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-28 14:15:00 | 310.55 | 322.58 | 322.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 310.55 | 322.58 | 322.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 308.45 | 322.33 | 322.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 322.90 | 318.94 | 320.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 322.90 | 318.94 | 320.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 322.90 | 318.94 | 320.46 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-12 15:15:00 | 318.00 | 319.11 | 320.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-15 09:15:00 | 318.80 | 319.11 | 320.45 | ENTRY2 sustain failed after 3960m |
| Cross detected — sustain check pending | 2025-09-15 10:15:00 | 317.85 | 319.09 | 320.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 11:15:00 | 317.45 | 319.08 | 320.42 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-16 11:15:00 | 317.45 | 319.01 | 320.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-16 12:15:00 | 318.15 | 319.00 | 320.33 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-16 13:15:00 | 317.70 | 318.99 | 320.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-16 14:15:00 | 318.40 | 318.99 | 320.31 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 323.80 | 319.07 | 320.33 | SL hit (close>static) qty=1.00 sl=323.10 alert=retest2 |

### Cycle 5 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 330.65 | 321.42 | 321.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 335.20 | 322.85 | 322.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 13:15:00 | 331.55 | 332.17 | 327.90 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-15 10:15:00 | 338.95 | 332.25 | 328.02 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:15:00 | 338.40 | 332.31 | 328.07 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-17 13:15:00 | 335.65 | 332.84 | 328.67 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 14:15:00 | 335.80 | 332.87 | 328.71 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 334.00 | 333.32 | 329.31 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-27 09:15:00 | 339.85 | 333.28 | 329.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:15:00 | 337.85 | 333.33 | 329.47 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 350.70 | 358.14 | 351.27 | SL hit (close<ema400) qty=1.00 sl=351.27 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 350.70 | 358.14 | 351.27 | SL hit (close<ema400) qty=1.00 sl=351.27 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 09:15:00 | 388.53 | 363.57 | 360.68 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 370.20 | 370.77 | 365.39 | SL hit (close<ema200) qty=0.50 sl=370.77 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-09 11:15:00 | 334.55 | 369.09 | 366.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 12:15:00 | 334.45 | 368.75 | 366.68 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 325.40 | 366.81 | 365.75 | SL hit (close<static) qty=1.00 sl=327.40 alert=retest2 |

### Cycle 6 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 325.15 | 364.42 | 364.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 323.90 | 364.02 | 364.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 14:15:00 | 310.30 | 307.33 | 325.98 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-24 09:15:00 | 301.90 | 309.08 | 323.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:15:00 | 305.65 | 309.04 | 323.47 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-29 14:15:00 | 303.85 | 309.04 | 321.77 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 15:15:00 | 304.95 | 309.00 | 321.69 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-07 11:15:00 | 305.65 | 307.36 | 318.97 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-07 12:15:00 | 308.50 | 307.38 | 318.92 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-08 09:15:00 | 303.00 | 307.36 | 318.68 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:15:00 | 305.15 | 307.34 | 318.61 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-03-14 15:15:00 | 304.00 | 2024-03-19 14:15:00 | 278.60 | STOP_HIT | 1.00 | -8.36% |
| BUY | retest2 | 2024-03-20 10:15:00 | 281.75 | 2024-05-21 12:15:00 | 324.01 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-03-20 10:15:00 | 281.75 | 2024-05-31 09:15:00 | 309.75 | STOP_HIT | 0.50 | 9.94% |
| BUY | retest2 | 2024-06-05 11:15:00 | 281.55 | 2024-07-25 10:15:00 | 323.78 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-06-05 11:15:00 | 281.55 | 2024-08-13 11:15:00 | 325.95 | STOP_HIT | 0.50 | 15.77% |
| SELL | retest1 | 2024-12-13 11:15:00 | 298.60 | 2025-01-29 09:15:00 | 253.81 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2024-12-16 10:15:00 | 296.20 | 2025-01-29 09:15:00 | 251.77 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2024-12-13 11:15:00 | 298.60 | 2025-03-06 09:15:00 | 260.96 | STOP_HIT | 0.50 | 12.61% |
| SELL | retest1 | 2024-12-16 10:15:00 | 296.20 | 2025-03-06 09:15:00 | 260.96 | STOP_HIT | 0.50 | 11.90% |
| SELL | retest2 | 2025-03-12 12:15:00 | 264.39 | 2025-03-20 10:15:00 | 270.96 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-03-13 10:15:00 | 264.00 | 2025-03-20 10:15:00 | 270.96 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest1 | 2025-06-04 12:15:00 | 311.30 | 2025-07-08 12:15:00 | 357.99 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-06-06 10:15:00 | 315.75 | 2025-07-08 12:15:00 | 358.57 | PARTIAL | 0.50 | 13.56% |
| BUY | retest2 | 2025-06-13 12:15:00 | 311.80 | 2025-07-08 12:15:00 | 357.77 | PARTIAL | 0.50 | 14.74% |
| BUY | retest1 | 2025-06-04 12:15:00 | 311.30 | 2025-07-25 09:15:00 | 336.40 | STOP_HIT | 0.50 | 8.06% |
| BUY | retest1 | 2025-06-06 10:15:00 | 315.75 | 2025-07-25 09:15:00 | 336.40 | STOP_HIT | 0.50 | 6.54% |
| BUY | retest2 | 2025-06-13 12:15:00 | 311.80 | 2025-07-25 09:15:00 | 336.40 | STOP_HIT | 0.50 | 7.89% |
| BUY | retest2 | 2025-06-16 13:15:00 | 316.25 | 2025-07-31 09:15:00 | 326.90 | STOP_HIT | 1.00 | 3.37% |
| BUY | retest2 | 2025-06-23 11:15:00 | 311.10 | 2025-08-18 09:15:00 | 315.10 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2025-08-08 10:15:00 | 317.95 | 2025-08-22 10:15:00 | 317.40 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-08-11 15:15:00 | 320.95 | 2025-08-22 10:15:00 | 317.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-08-19 15:15:00 | 321.90 | 2025-08-28 14:15:00 | 310.55 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2025-08-21 10:15:00 | 322.95 | 2025-08-28 14:15:00 | 310.55 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-09-15 11:15:00 | 317.45 | 2025-09-17 10:15:00 | 323.80 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest1 | 2025-10-15 11:15:00 | 338.40 | 2025-12-11 09:15:00 | 350.70 | STOP_HIT | 1.00 | 3.63% |
| BUY | retest1 | 2025-10-17 14:15:00 | 335.80 | 2025-12-11 09:15:00 | 350.70 | STOP_HIT | 1.00 | 4.44% |
| BUY | retest2 | 2025-10-27 10:15:00 | 337.85 | 2026-02-05 09:15:00 | 388.53 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-10-27 10:15:00 | 337.85 | 2026-02-16 09:15:00 | 370.20 | STOP_HIT | 0.50 | 9.58% |
| BUY | retest2 | 2026-03-09 12:15:00 | 334.45 | 2026-03-10 10:15:00 | 325.40 | STOP_HIT | 1.00 | -2.71% |
