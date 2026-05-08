# BPCL (BPCL)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 302.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
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
| PENDING | 26 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 10 |
| ENTRY2 | 11 |
| PARTIAL | 6 |
| TARGET_HIT | 10 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 16 / 8
- **Target hits / Stop hits / Partials:** 10 / 8 / 6
- **Avg / median % per leg:** 4.46% / 5.00%
- **Sum % (uncompounded):** 106.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 12 | 70.6% | 8 | 5 | 4 | 4.94% | 84.1% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 2 | 3 | 4 | 3.23% | 29.1% |
| BUY @ 3rd Alert (retest2) | 8 | 6 | 75.0% | 6 | 2 | 0 | 6.87% | 55.0% |
| SELL (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 3.27% | 22.9% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.37% | -7.1% |
| retest1 (combined) | 13 | 10 | 76.9% | 4 | 3 | 6 | 4.54% | 59.1% |
| retest2 (combined) | 11 | 6 | 54.5% | 6 | 5 | 0 | 4.35% | 47.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 11:15:00 | 192.30 | 178.58 | 178.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 12:15:00 | 192.75 | 178.72 | 178.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 14:15:00 | 297.10 | 300.72 | 277.82 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-03-14 14:15:00 | 304.30 | 300.78 | 278.63 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-14 15:15:00 | 304.00 | 300.81 | 278.76 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 14:15:00 | 278.60 | 298.30 | 279.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-19 14:15:00 | 278.60 | 298.30 | 279.55 | SL hit (close<ema400) qty=1.00 sl=279.55 alert=retest1 |
| Cross detected — sustain check pending | 2024-03-20 09:15:00 | 281.62 | 297.94 | 279.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 10:15:00 | 281.75 | 297.78 | 279.56 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2024-04-02 10:15:00 | 309.93 | 298.13 | 283.65 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-05 10:15:00 | 282.02 | 311.89 | 304.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 11:15:00 | 281.55 | 311.59 | 304.15 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2024-06-12 09:15:00 | 309.70 | 307.80 | 303.20 | Target hit (10%) qty=1.00 alert=retest2 |

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
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 283.67 | 301.46 | 310.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 11:15:00 | 281.39 | 296.35 | 304.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-01-13 13:15:00 | 268.74 | 292.14 | 300.71 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2025-01-13 14:15:00 | 266.58 | 291.88 | 300.53 | Target hit (10%) qty=0.50 alert=retest1 |
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
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 09:15:00 | 326.86 | 312.63 | 303.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 09:15:00 | 331.54 | 312.63 | 303.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 309.45 | 314.25 | 304.91 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 309.45 | 314.25 | 304.91 | SL hit (close<ema200) qty=0.50 sl=314.25 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 309.45 | 314.25 | 304.91 | SL hit (close<ema200) qty=0.50 sl=314.25 alert=retest1 |
| Cross detected — sustain check pending | 2025-06-13 11:15:00 | 313.15 | 314.19 | 304.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:15:00 | 311.80 | 314.17 | 305.01 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-16 12:15:00 | 311.50 | 313.98 | 305.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 316.25 | 314.00 | 305.28 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-23 10:15:00 | 310.75 | 313.90 | 306.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 311.10 | 313.87 | 306.55 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2025-07-04 11:15:00 | 342.98 | 320.93 | 312.47 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-07-04 11:15:00 | 342.21 | 320.93 | 312.47 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-07-07 09:15:00 | 347.88 | 322.03 | 313.24 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-07 14:15:00 | 310.35 | 329.96 | 325.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-07 15:15:00 | 310.10 | 329.77 | 325.28 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-08 09:15:00 | 316.80 | 329.64 | 325.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 317.95 | 329.52 | 325.20 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 325.25 | 328.36 | 324.89 | EMA400 retest candle locked (from upside) |
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
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 09:15:00 | 352.59 | 335.25 | 330.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 10:15:00 | 355.32 | 335.47 | 330.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-11-04 09:15:00 | 372.24 | 340.25 | 333.89 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2025-11-04 09:15:00 | 369.38 | 340.25 | 333.89 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2025-11-04 09:15:00 | 371.64 | 340.25 | 333.89 | Target hit (10%) qty=1.00 alert=retest2 |
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
| BUY | retest2 | 2024-03-20 10:15:00 | 281.75 | 2024-04-02 10:15:00 | 309.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 11:15:00 | 281.55 | 2024-06-12 09:15:00 | 309.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2024-12-13 11:15:00 | 298.60 | 2024-12-19 09:15:00 | 283.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-12-16 10:15:00 | 296.20 | 2025-01-07 11:15:00 | 281.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-12-13 11:15:00 | 298.60 | 2025-01-13 13:15:00 | 268.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-12-16 10:15:00 | 296.20 | 2025-01-13 14:15:00 | 266.58 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-12 12:15:00 | 264.39 | 2025-03-20 10:15:00 | 270.96 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-03-13 10:15:00 | 264.00 | 2025-03-20 10:15:00 | 270.96 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest1 | 2025-06-04 12:15:00 | 311.30 | 2025-06-11 09:15:00 | 326.86 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-06 10:15:00 | 315.75 | 2025-06-11 09:15:00 | 331.54 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-04 12:15:00 | 311.30 | 2025-06-13 09:15:00 | 309.45 | STOP_HIT | 0.50 | -0.59% |
| BUY | retest1 | 2025-06-06 10:15:00 | 315.75 | 2025-06-13 09:15:00 | 309.45 | STOP_HIT | 0.50 | -2.00% |
| BUY | retest2 | 2025-06-13 12:15:00 | 311.80 | 2025-07-04 11:15:00 | 342.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-16 13:15:00 | 316.25 | 2025-07-04 11:15:00 | 342.21 | TARGET_HIT | 1.00 | 8.21% |
| BUY | retest2 | 2025-06-23 11:15:00 | 311.10 | 2025-07-07 09:15:00 | 347.88 | TARGET_HIT | 1.00 | 11.82% |
| BUY | retest2 | 2025-08-08 10:15:00 | 317.95 | 2025-08-28 14:15:00 | 310.55 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-09-15 11:15:00 | 317.45 | 2025-09-17 10:15:00 | 323.80 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest1 | 2025-10-15 11:15:00 | 338.40 | 2025-10-30 09:15:00 | 352.59 | PARTIAL | 0.50 | 4.19% |
| BUY | retest1 | 2025-10-17 14:15:00 | 335.80 | 2025-10-30 10:15:00 | 355.32 | PARTIAL | 0.50 | 5.81% |
| BUY | retest1 | 2025-10-15 11:15:00 | 338.40 | 2025-11-04 09:15:00 | 372.24 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-10-17 14:15:00 | 335.80 | 2025-11-04 09:15:00 | 369.38 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-10-27 10:15:00 | 337.85 | 2025-11-04 09:15:00 | 371.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-09 12:15:00 | 334.45 | 2026-03-10 10:15:00 | 325.40 | STOP_HIT | 1.00 | -2.71% |
