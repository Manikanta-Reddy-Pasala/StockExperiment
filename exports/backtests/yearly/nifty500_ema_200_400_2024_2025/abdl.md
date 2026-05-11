# Allied Blenders and Distillers Ltd. (ABDL)

## Backtest Summary

- **Window:** 2024-07-02 09:15:00 → 2026-05-08 15:15:00 (3203 bars)
- **Last close:** 594.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 0 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 20 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 19
- **Target hits / Stop hits / Partials:** 2 / 22 / 5
- **Avg / median % per leg:** -0.47% / -2.29%
- **Sum % (uncompounded):** -13.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 0 | 0.0% | 0 | 10 | 0 | -2.62% | -26.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -2.62% | -26.2% |
| SELL (all) | 19 | 10 | 52.6% | 2 | 12 | 5 | 0.66% | 12.5% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -6.48% | -25.9% |
| SELL @ 3rd Alert (retest2) | 15 | 10 | 66.7% | 2 | 8 | 5 | 2.56% | 38.4% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -6.48% | -25.9% |
| retest2 (combined) | 25 | 10 | 40.0% | 2 | 18 | 5 | 0.49% | 12.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 12:15:00 | 317.00 | 326.14 | 326.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 14:15:00 | 312.10 | 324.64 | 325.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 10:15:00 | 323.10 | 322.45 | 324.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-26 10:30:00 | 321.60 | 322.45 | 324.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 327.75 | 322.29 | 323.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:45:00 | 329.75 | 322.29 | 323.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 325.00 | 322.31 | 323.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 11:15:00 | 323.70 | 322.31 | 323.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:15:00 | 323.55 | 322.33 | 323.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 09:15:00 | 331.05 | 322.43 | 323.96 | SL hit (close>static) qty=1.00 sl=328.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 13:15:00 | 346.00 | 325.48 | 325.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 12:15:00 | 347.55 | 326.63 | 325.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 11:15:00 | 396.00 | 400.67 | 376.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-13 12:00:00 | 396.00 | 400.67 | 376.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 377.30 | 400.52 | 383.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:00:00 | 377.30 | 400.52 | 383.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 374.25 | 400.26 | 383.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:30:00 | 374.70 | 400.26 | 383.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 379.90 | 399.11 | 383.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:15:00 | 371.40 | 399.11 | 383.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 357.50 | 398.34 | 382.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:00:00 | 357.50 | 398.34 | 382.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 386.15 | 396.78 | 382.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 12:45:00 | 389.35 | 396.57 | 382.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 13:45:00 | 388.50 | 396.49 | 382.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:30:00 | 393.50 | 396.53 | 383.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 14:30:00 | 390.35 | 396.28 | 383.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 385.40 | 396.06 | 383.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:30:00 | 391.10 | 395.79 | 383.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:30:00 | 389.80 | 396.85 | 386.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:45:00 | 389.85 | 396.78 | 386.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:15:00 | 389.80 | 396.71 | 386.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 379.40 | 396.28 | 386.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 379.40 | 396.28 | 386.15 | SL hit (close<static) qty=1.00 sl=382.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 10:15:00 | 325.55 | 378.07 | 378.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 316.10 | 362.55 | 369.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 326.10 | 321.46 | 335.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-15 09:45:00 | 325.45 | 321.46 | 335.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 329.35 | 320.94 | 333.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-23 10:15:00 | 327.85 | 320.94 | 333.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 13:30:00 | 328.30 | 321.93 | 332.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 14:30:00 | 327.75 | 321.98 | 332.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 09:15:00 | 311.46 | 321.14 | 331.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 09:15:00 | 311.88 | 321.14 | 331.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 09:15:00 | 311.36 | 321.14 | 331.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 09:15:00 | 320.60 | 318.75 | 328.62 | SL hit (close>ema200) qty=0.50 sl=318.75 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 10:15:00 | 401.00 | 335.89 | 335.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 405.00 | 342.92 | 339.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 498.35 | 498.73 | 473.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 10:45:00 | 498.15 | 498.73 | 473.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 603.20 | 627.03 | 599.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 597.45 | 627.03 | 599.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 603.00 | 624.34 | 604.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 603.00 | 624.34 | 604.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 607.75 | 624.17 | 604.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:15:00 | 611.10 | 624.17 | 604.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 601.25 | 623.17 | 607.50 | SL hit (close<static) qty=1.00 sl=602.50 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 507.60 | 597.29 | 597.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 10:15:00 | 504.45 | 596.37 | 597.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 510.80 | 509.40 | 541.40 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 11:45:00 | 503.30 | 509.41 | 540.00 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 12:45:00 | 503.75 | 509.35 | 539.82 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:30:00 | 503.30 | 509.31 | 539.20 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 10:00:00 | 504.55 | 509.45 | 537.24 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 536.35 | 509.87 | 534.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 536.35 | 509.87 | 534.69 | SL hit (close>ema400) qty=1.00 sl=534.69 alert=retest1 |

### Cycle 6 — BUY (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 13:15:00 | 564.80 | 489.93 | 489.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 572.00 | 514.31 | 503.91 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-28 11:15:00 | 323.70 | 2024-11-29 09:15:00 | 331.05 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2024-11-28 12:15:00 | 323.55 | 2024-11-29 09:15:00 | 331.05 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-01-29 12:45:00 | 389.35 | 2025-02-10 09:15:00 | 379.40 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-01-29 13:45:00 | 388.50 | 2025-02-10 09:15:00 | 379.40 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-01-31 09:30:00 | 393.50 | 2025-02-10 09:15:00 | 379.40 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-01-31 14:30:00 | 390.35 | 2025-02-10 09:15:00 | 379.40 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-02-01 12:30:00 | 391.10 | 2025-02-10 09:15:00 | 379.40 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-02-07 09:30:00 | 389.80 | 2025-02-10 09:15:00 | 379.40 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-02-07 10:45:00 | 389.85 | 2025-02-10 09:15:00 | 379.40 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-02-07 12:15:00 | 389.80 | 2025-02-10 09:15:00 | 379.40 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-04-23 10:15:00 | 327.85 | 2025-04-30 09:15:00 | 311.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-24 13:30:00 | 328.30 | 2025-04-30 09:15:00 | 311.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-24 14:30:00 | 327.75 | 2025-04-30 09:15:00 | 311.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-23 10:15:00 | 327.85 | 2025-05-07 09:15:00 | 320.60 | STOP_HIT | 0.50 | 2.21% |
| SELL | retest2 | 2025-04-24 13:30:00 | 328.30 | 2025-05-07 09:15:00 | 320.60 | STOP_HIT | 0.50 | 2.35% |
| SELL | retest2 | 2025-04-24 14:30:00 | 327.75 | 2025-05-07 09:15:00 | 320.60 | STOP_HIT | 0.50 | 2.18% |
| SELL | retest2 | 2025-05-09 14:30:00 | 327.80 | 2025-05-12 09:15:00 | 348.90 | STOP_HIT | 1.00 | -6.44% |
| BUY | retest2 | 2025-12-18 11:15:00 | 611.10 | 2025-12-29 09:15:00 | 601.25 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-12-31 15:00:00 | 614.70 | 2026-01-01 12:15:00 | 600.60 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest1 | 2026-02-04 11:45:00 | 503.30 | 2026-02-12 09:15:00 | 536.35 | STOP_HIT | 1.00 | -6.57% |
| SELL | retest1 | 2026-02-04 12:45:00 | 503.75 | 2026-02-12 09:15:00 | 536.35 | STOP_HIT | 1.00 | -6.47% |
| SELL | retest1 | 2026-02-05 09:30:00 | 503.30 | 2026-02-12 09:15:00 | 536.35 | STOP_HIT | 1.00 | -6.57% |
| SELL | retest1 | 2026-02-09 10:00:00 | 504.55 | 2026-02-12 09:15:00 | 536.35 | STOP_HIT | 1.00 | -6.30% |
| SELL | retest2 | 2026-02-13 09:15:00 | 529.30 | 2026-02-19 15:15:00 | 502.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 14:00:00 | 532.45 | 2026-02-19 15:15:00 | 505.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 529.30 | 2026-02-27 09:15:00 | 479.21 | TARGET_HIT | 0.50 | 9.46% |
| SELL | retest2 | 2026-02-13 14:00:00 | 532.45 | 2026-02-27 11:15:00 | 476.37 | TARGET_HIT | 0.50 | 10.53% |
| SELL | retest2 | 2026-04-16 12:15:00 | 533.00 | 2026-04-16 14:15:00 | 539.45 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-04-16 12:45:00 | 533.40 | 2026-04-16 14:15:00 | 539.45 | STOP_HIT | 1.00 | -1.13% |
