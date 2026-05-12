# Aptus Value Housing Finance India Ltd. (APTUS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 282.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 24 |
| PARTIAL | 14 |
| TARGET_HIT | 2 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 18
- **Target hits / Stop hits / Partials:** 1 / 24 / 7
- **Avg / median % per leg:** -0.16% / -1.45%
- **Sum % (uncompounded):** -5.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 0 | 0.0% | 0 | 15 | 0 | -3.15% | -47.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 0 | 0.0% | 0 | 15 | 0 | -3.15% | -47.3% |
| SELL (all) | 17 | 14 | 82.4% | 1 | 9 | 7 | 2.49% | 42.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.51% | -3.5% |
| SELL @ 3rd Alert (retest2) | 16 | 14 | 87.5% | 1 | 8 | 7 | 2.86% | 45.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.51% | -3.5% |
| retest2 (combined) | 31 | 14 | 45.2% | 1 | 23 | 7 | -0.05% | -1.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 346.35 | 323.06 | 323.01 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 09:15:00 | 319.25 | 326.62 | 326.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 11:15:00 | 316.70 | 326.45 | 326.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 314.95 | 314.81 | 319.46 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 15:15:00 | 307.40 | 313.92 | 318.29 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 314.00 | 313.86 | 318.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 318.20 | 313.94 | 318.14 | SL hit (close>ema400) qty=1.00 sl=318.14 alert=retest1 |

### Cycle 3 — BUY (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 14:15:00 | 332.80 | 320.53 | 320.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 13:15:00 | 334.65 | 321.08 | 320.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 359.95 | 360.29 | 347.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-22 10:00:00 | 359.95 | 360.29 | 347.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 344.60 | 359.32 | 347.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 344.60 | 359.32 | 347.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 344.25 | 359.17 | 347.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 344.30 | 359.17 | 347.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 353.25 | 356.05 | 347.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:00:00 | 358.70 | 355.46 | 347.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 12:15:00 | 339.25 | 353.74 | 347.37 | SL hit (close<static) qty=1.00 sl=340.05 alert=retest2 |

### Cycle 4 — SELL (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 09:15:00 | 317.60 | 342.83 | 342.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 13:15:00 | 315.00 | 341.80 | 342.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 292.95 | 291.02 | 302.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-30 09:30:00 | 293.90 | 291.02 | 302.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 299.25 | 291.55 | 302.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 13:30:00 | 298.95 | 291.55 | 302.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 302.10 | 291.66 | 302.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 15:00:00 | 302.10 | 291.66 | 302.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 15:15:00 | 302.40 | 291.76 | 302.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:15:00 | 310.20 | 291.76 | 302.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 311.20 | 291.96 | 302.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 310.00 | 291.96 | 302.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 315.40 | 292.19 | 302.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 10:45:00 | 312.00 | 292.19 | 302.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 311.50 | 303.42 | 306.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 11:00:00 | 311.50 | 303.42 | 306.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 310.95 | 304.11 | 306.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:30:00 | 308.75 | 304.11 | 306.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 310.35 | 304.17 | 306.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 14:30:00 | 307.20 | 304.33 | 306.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 15:15:00 | 306.20 | 304.33 | 306.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:00:00 | 307.30 | 304.48 | 306.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 291.84 | 304.20 | 306.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 290.89 | 304.20 | 306.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 291.94 | 304.20 | 306.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-18 14:15:00 | 305.25 | 303.90 | 306.38 | SL hit (close>ema200) qty=0.50 sl=303.90 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 329.05 | 303.95 | 303.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 12:15:00 | 333.50 | 304.74 | 304.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 316.50 | 316.97 | 312.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-09 10:00:00 | 316.50 | 316.97 | 312.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 309.30 | 327.90 | 320.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 13:15:00 | 333.25 | 324.11 | 319.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:30:00 | 335.80 | 322.03 | 320.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 334.65 | 322.03 | 320.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 334.80 | 336.91 | 330.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 331.55 | 336.68 | 330.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:45:00 | 339.10 | 336.67 | 330.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:15:00 | 339.05 | 336.67 | 330.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 13:15:00 | 323.35 | 336.54 | 330.76 | SL hit (close<static) qty=1.00 sl=328.20 alert=retest2 |

### Cycle 6 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 323.85 | 334.63 | 334.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 322.00 | 334.50 | 334.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 13:15:00 | 329.95 | 328.98 | 331.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 13:15:00 | 329.95 | 328.98 | 331.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 329.95 | 328.98 | 331.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:00:00 | 329.95 | 328.98 | 331.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 319.65 | 317.39 | 322.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:45:00 | 320.75 | 317.39 | 322.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 281.05 | 274.04 | 281.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 15:15:00 | 280.80 | 274.04 | 281.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 280.80 | 274.10 | 281.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 276.95 | 274.10 | 281.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 11:15:00 | 283.30 | 274.34 | 281.65 | SL hit (close>static) qty=1.00 sl=283.00 alert=retest2 |

### Cycle 7 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 265.15 | 243.03 | 243.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 266.15 | 243.65 | 243.32 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-08-27 15:15:00 | 307.40 | 2024-08-28 14:15:00 | 318.20 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2024-10-31 15:00:00 | 358.70 | 2024-11-06 12:15:00 | 339.25 | STOP_HIT | 1.00 | -5.42% |
| SELL | retest2 | 2025-02-12 14:30:00 | 307.20 | 2025-02-17 09:15:00 | 291.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 15:15:00 | 306.20 | 2025-02-17 09:15:00 | 290.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:00:00 | 307.30 | 2025-02-17 09:15:00 | 291.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 14:30:00 | 307.20 | 2025-02-18 14:15:00 | 305.25 | STOP_HIT | 0.50 | 0.63% |
| SELL | retest2 | 2025-02-12 15:15:00 | 306.20 | 2025-02-18 14:15:00 | 305.25 | STOP_HIT | 0.50 | 0.31% |
| SELL | retest2 | 2025-02-13 15:00:00 | 307.30 | 2025-02-18 14:15:00 | 305.25 | STOP_HIT | 0.50 | 0.67% |
| SELL | retest2 | 2025-03-03 09:15:00 | 304.00 | 2025-03-03 15:15:00 | 313.90 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-03-04 09:15:00 | 307.85 | 2025-03-17 11:15:00 | 292.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-04 09:15:00 | 307.85 | 2025-03-18 09:15:00 | 304.70 | STOP_HIT | 0.50 | 1.02% |
| SELL | retest2 | 2025-03-24 09:30:00 | 306.45 | 2025-03-28 15:15:00 | 292.46 | PARTIAL | 0.50 | 4.57% |
| SELL | retest2 | 2025-03-24 11:00:00 | 307.85 | 2025-04-02 09:15:00 | 291.13 | PARTIAL | 0.50 | 5.43% |
| SELL | retest2 | 2025-03-24 09:30:00 | 306.45 | 2025-04-02 14:15:00 | 301.45 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2025-03-24 11:00:00 | 307.85 | 2025-04-02 14:15:00 | 301.45 | STOP_HIT | 0.50 | 2.08% |
| BUY | retest2 | 2025-06-10 13:15:00 | 333.25 | 2025-07-31 13:15:00 | 323.35 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-07-08 13:30:00 | 335.80 | 2025-07-31 13:15:00 | 323.35 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2025-07-08 14:15:00 | 334.65 | 2025-08-28 09:15:00 | 329.95 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-30 09:15:00 | 334.80 | 2025-08-28 09:15:00 | 329.95 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-31 12:45:00 | 339.10 | 2025-08-28 09:15:00 | 329.95 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-07-31 13:15:00 | 339.05 | 2025-08-28 09:15:00 | 329.95 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-08-01 09:30:00 | 340.20 | 2025-08-28 11:15:00 | 325.35 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest2 | 2025-08-01 14:45:00 | 338.95 | 2025-08-28 11:15:00 | 325.35 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest2 | 2025-08-12 10:45:00 | 337.75 | 2025-09-12 13:15:00 | 332.55 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-08-12 11:45:00 | 335.95 | 2025-09-17 11:15:00 | 330.65 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-08-25 14:30:00 | 336.55 | 2025-09-23 14:15:00 | 323.85 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2025-08-26 10:00:00 | 336.95 | 2025-09-23 14:15:00 | 323.85 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-09-05 14:15:00 | 337.00 | 2025-09-23 14:15:00 | 323.85 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-09-15 09:15:00 | 336.95 | 2025-09-23 14:15:00 | 323.85 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2026-02-04 09:15:00 | 276.95 | 2026-02-04 11:15:00 | 283.30 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-02-04 14:15:00 | 274.85 | 2026-02-06 09:15:00 | 261.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 14:15:00 | 274.85 | 2026-02-13 09:15:00 | 247.37 | TARGET_HIT | 0.50 | 10.00% |
