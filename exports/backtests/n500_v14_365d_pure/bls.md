# BLS International Services Ltd. (BLS)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 290.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 17 |
| TARGET_HIT | 7 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 9
- **Target hits / Stop hits / Partials:** 7 / 17 / 17
- **Avg / median % per leg:** 3.34% / 5.00%
- **Sum % (uncompounded):** 136.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 41 | 32 | 78.0% | 7 | 17 | 17 | 3.34% | 136.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 41 | 32 | 78.0% | 7 | 17 | 17 | 3.34% | 136.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 41 | 32 | 78.0% | 7 | 17 | 17 | 3.34% | 136.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 419.40 | 383.69 | 383.54 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 364.90 | 386.27 | 386.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 362.10 | 385.81 | 386.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 378.75 | 377.41 | 381.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 378.75 | 377.41 | 381.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 378.75 | 377.41 | 381.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 381.35 | 377.41 | 381.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 370.55 | 377.36 | 381.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 366.30 | 376.94 | 380.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 388.50 | 374.35 | 378.42 | SL hit (close>static) qty=1.00 sl=382.40 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:30:00 | 368.00 | 378.05 | 379.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 14:30:00 | 367.70 | 377.94 | 379.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 15:00:00 | 367.15 | 377.94 | 379.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 378.05 | 374.78 | 377.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:45:00 | 376.50 | 374.78 | 377.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 379.80 | 374.83 | 377.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:45:00 | 380.00 | 374.83 | 377.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 381.05 | 374.89 | 377.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 381.05 | 374.89 | 377.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 377.10 | 374.91 | 377.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 373.10 | 374.95 | 377.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 383.10 | 375.14 | 377.52 | SL hit (close>static) qty=1.00 sl=382.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 383.10 | 375.14 | 377.52 | SL hit (close>static) qty=1.00 sl=382.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 383.10 | 375.14 | 377.52 | SL hit (close>static) qty=1.00 sl=382.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 383.10 | 375.14 | 377.52 | SL hit (close>static) qty=1.00 sl=381.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 10:45:00 | 374.40 | 375.59 | 377.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 382.95 | 375.72 | 377.69 | SL hit (close>static) qty=1.00 sl=381.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 12:15:00 | 397.50 | 379.60 | 379.52 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 373.00 | 379.92 | 379.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 368.85 | 379.67 | 379.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 15:15:00 | 355.25 | 353.95 | 363.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:15:00 | 356.80 | 353.95 | 363.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 337.50 | 323.97 | 337.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:00:00 | 337.50 | 323.97 | 337.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 337.40 | 324.10 | 337.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:15:00 | 335.05 | 324.10 | 337.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 336.00 | 324.22 | 337.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 09:15:00 | 319.20 | 324.44 | 337.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 329.00 | 324.48 | 336.97 | SL hit (close>ema200) qty=0.50 sl=324.48 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 11:15:00 | 318.30 | 324.39 | 334.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 331.65 | 322.30 | 332.14 | SL hit (close>ema200) qty=0.50 sl=322.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 10:30:00 | 335.35 | 323.50 | 331.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:15:00 | 335.50 | 323.63 | 331.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 331.45 | 323.99 | 331.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 331.90 | 323.99 | 331.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 329.85 | 324.05 | 331.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:30:00 | 332.10 | 324.05 | 331.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 330.80 | 324.12 | 331.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 330.35 | 324.12 | 331.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 332.50 | 324.20 | 331.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 328.70 | 324.51 | 331.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 318.58 | 324.38 | 331.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 318.72 | 324.38 | 331.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 312.26 | 324.02 | 331.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 334.50 | 321.09 | 327.65 | SL hit (close>ema200) qty=0.50 sl=321.09 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 334.50 | 321.09 | 327.65 | SL hit (close>ema200) qty=0.50 sl=321.09 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 334.50 | 321.09 | 327.65 | SL hit (close>ema200) qty=0.50 sl=321.09 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:30:00 | 330.35 | 321.16 | 327.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:30:00 | 329.65 | 321.35 | 327.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:00:00 | 329.75 | 321.35 | 327.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 328.15 | 321.56 | 327.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:45:00 | 328.80 | 321.56 | 327.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 328.40 | 321.62 | 327.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 328.40 | 321.62 | 327.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 327.10 | 321.84 | 327.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 15:00:00 | 327.10 | 321.84 | 327.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 327.60 | 321.90 | 327.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 324.00 | 321.90 | 327.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 325.45 | 321.94 | 327.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 11:15:00 | 322.80 | 321.97 | 327.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 09:45:00 | 322.55 | 322.00 | 327.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:30:00 | 323.10 | 322.04 | 327.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 13:15:00 | 323.00 | 322.06 | 327.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:15:00 | 313.83 | 321.62 | 326.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 12:15:00 | 313.17 | 321.53 | 326.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 12:15:00 | 313.26 | 321.53 | 326.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 321.70 | 321.29 | 326.25 | SL hit (close>ema200) qty=0.50 sl=321.29 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 321.70 | 321.29 | 326.25 | SL hit (close>ema200) qty=0.50 sl=321.29 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 321.70 | 321.29 | 326.25 | SL hit (close>ema200) qty=0.50 sl=321.29 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 14:15:00 | 306.66 | 319.08 | 324.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 14:15:00 | 306.42 | 319.08 | 324.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 14:15:00 | 306.94 | 319.08 | 324.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 14:15:00 | 306.85 | 319.08 | 324.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 14:15:00 | 290.52 | 311.88 | 319.03 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 14:15:00 | 290.30 | 311.88 | 319.03 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 14:15:00 | 290.79 | 311.88 | 319.03 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 14:15:00 | 290.70 | 311.88 | 319.03 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 296.85 | 288.32 | 302.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:30:00 | 287.80 | 290.08 | 301.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:15:00 | 273.41 | 288.95 | 300.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 293.85 | 284.78 | 295.93 | SL hit (close>ema200) qty=0.50 sl=284.78 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 289.45 | 284.78 | 295.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:00:00 | 288.70 | 284.82 | 295.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 274.98 | 284.75 | 295.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 285.25 | 284.55 | 295.22 | SL hit (close>ema200) qty=0.50 sl=284.55 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 14:00:00 | 289.15 | 284.71 | 294.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 274.26 | 284.19 | 293.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 274.69 | 284.19 | 293.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 259.83 | 282.92 | 292.90 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 260.24 | 282.92 | 292.90 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 275.40 | 258.96 | 274.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 275.90 | 258.96 | 274.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 272.45 | 259.10 | 274.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:15:00 | 269.55 | 259.10 | 274.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:15:00 | 256.07 | 259.42 | 273.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-30 09:15:00 | 242.60 | 258.77 | 272.94 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2026-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 13:15:00 | 293.15 | 278.16 | 278.14 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-15 11:00:00 | 381.90 | 2025-05-15 12:15:00 | 394.30 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-07-01 09:15:00 | 366.30 | 2025-07-10 09:15:00 | 388.50 | STOP_HIT | 1.00 | -6.06% |
| SELL | retest2 | 2025-07-23 13:30:00 | 368.00 | 2025-08-01 09:15:00 | 383.10 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2025-07-23 14:30:00 | 367.70 | 2025-08-01 09:15:00 | 383.10 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2025-07-23 15:00:00 | 367.15 | 2025-08-01 09:15:00 | 383.10 | STOP_HIT | 1.00 | -4.34% |
| SELL | retest2 | 2025-07-31 09:15:00 | 373.10 | 2025-08-01 09:15:00 | 383.10 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-08-04 10:45:00 | 374.40 | 2025-08-04 13:15:00 | 382.95 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-11-12 14:15:00 | 335.05 | 2025-11-14 09:15:00 | 319.20 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2025-11-12 14:15:00 | 335.05 | 2025-11-14 10:15:00 | 329.00 | STOP_HIT | 0.50 | 1.81% |
| SELL | retest2 | 2025-11-12 15:00:00 | 336.00 | 2025-11-21 11:15:00 | 318.30 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2025-11-12 15:00:00 | 336.00 | 2025-11-28 09:15:00 | 331.65 | STOP_HIT | 0.50 | 1.29% |
| SELL | retest2 | 2025-12-04 10:30:00 | 335.35 | 2025-12-08 13:15:00 | 318.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 12:15:00 | 335.50 | 2025-12-08 13:15:00 | 318.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 09:15:00 | 328.70 | 2025-12-09 09:15:00 | 312.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 10:30:00 | 335.35 | 2025-12-19 09:15:00 | 334.50 | STOP_HIT | 0.50 | 0.25% |
| SELL | retest2 | 2025-12-04 12:15:00 | 335.50 | 2025-12-19 09:15:00 | 334.50 | STOP_HIT | 0.50 | 0.30% |
| SELL | retest2 | 2025-12-08 09:15:00 | 328.70 | 2025-12-19 09:15:00 | 334.50 | STOP_HIT | 0.50 | -1.76% |
| SELL | retest2 | 2025-12-19 10:30:00 | 330.35 | 2025-12-30 11:15:00 | 313.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-19 12:30:00 | 329.65 | 2025-12-30 12:15:00 | 313.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-19 13:00:00 | 329.75 | 2025-12-30 12:15:00 | 313.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-19 10:30:00 | 330.35 | 2025-12-31 15:15:00 | 321.70 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2025-12-19 12:30:00 | 329.65 | 2025-12-31 15:15:00 | 321.70 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2025-12-19 13:00:00 | 329.75 | 2025-12-31 15:15:00 | 321.70 | STOP_HIT | 0.50 | 2.44% |
| SELL | retest2 | 2025-12-23 11:15:00 | 322.80 | 2026-01-08 14:15:00 | 306.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 09:45:00 | 322.55 | 2026-01-08 14:15:00 | 306.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 11:30:00 | 323.10 | 2026-01-08 14:15:00 | 306.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 13:15:00 | 323.00 | 2026-01-08 14:15:00 | 306.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 11:15:00 | 322.80 | 2026-01-20 14:15:00 | 290.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-24 09:45:00 | 322.55 | 2026-01-20 14:15:00 | 290.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-24 11:30:00 | 323.10 | 2026-01-20 14:15:00 | 290.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-24 13:15:00 | 323.00 | 2026-01-20 14:15:00 | 290.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-11 09:30:00 | 287.80 | 2026-02-13 09:15:00 | 273.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 09:30:00 | 287.80 | 2026-02-23 10:15:00 | 293.85 | STOP_HIT | 0.50 | -2.10% |
| SELL | retest2 | 2026-02-23 11:15:00 | 289.45 | 2026-02-24 09:15:00 | 274.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:15:00 | 289.45 | 2026-02-24 14:15:00 | 285.25 | STOP_HIT | 0.50 | 1.45% |
| SELL | retest2 | 2026-02-23 12:00:00 | 288.70 | 2026-03-02 09:15:00 | 274.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 14:00:00 | 289.15 | 2026-03-02 09:15:00 | 274.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 12:00:00 | 288.70 | 2026-03-04 09:15:00 | 259.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 14:00:00 | 289.15 | 2026-03-04 09:15:00 | 260.24 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-25 11:15:00 | 269.55 | 2026-03-27 09:15:00 | 256.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 11:15:00 | 269.55 | 2026-03-30 09:15:00 | 242.60 | TARGET_HIT | 0.50 | 10.00% |
