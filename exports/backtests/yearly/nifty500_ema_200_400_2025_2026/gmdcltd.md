# Gujarat Mineral Development Corporation Ltd. (GMDCLTD)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 685.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 20 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 19
- **Target hits / Stop hits / Partials:** 2 / 19 / 3
- **Avg / median % per leg:** -0.91% / -2.59%
- **Sum % (uncompounded):** -21.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 3 | 16.7% | 2 | 15 | 1 | -1.17% | -21.0% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| BUY @ 3rd Alert (retest2) | 16 | 1 | 6.2% | 1 | 15 | 0 | -2.25% | -36.0% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | -0.14% | -0.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 2 | 33.3% | 0 | 4 | 2 | -0.14% | -0.8% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest2 (combined) | 22 | 3 | 13.6% | 1 | 19 | 2 | -1.68% | -36.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-18 12:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 12:00:00 | 396.25 | 387.88 | 371.17 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 13:15:00 | 416.06 | 388.54 | 371.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-07-18 14:15:00 | 435.88 | 389.02 | 371.99 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 2 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 489.10 | 537.93 | 538.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 486.25 | 537.41 | 537.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 11:15:00 | 536.40 | 531.78 | 534.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 11:15:00 | 536.40 | 531.78 | 534.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 536.40 | 531.78 | 534.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 536.40 | 531.78 | 534.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 534.60 | 531.81 | 534.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 14:00:00 | 533.60 | 531.83 | 534.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 526.35 | 531.90 | 534.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 10:15:00 | 506.92 | 530.81 | 534.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 14:15:00 | 500.03 | 529.69 | 533.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 540.00 | 526.96 | 531.54 | SL hit (close>ema200) qty=0.50 sl=526.96 alert=retest2 |

### Cycle 3 — BUY (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 13:15:00 | 596.05 | 535.52 | 535.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 14:15:00 | 599.60 | 536.15 | 535.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 11:15:00 | 563.90 | 565.39 | 552.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-09 11:45:00 | 563.05 | 565.39 | 552.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 551.95 | 565.08 | 552.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 554.20 | 565.08 | 552.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 551.50 | 564.95 | 552.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 537.90 | 564.95 | 552.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 544.40 | 564.74 | 552.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 12:30:00 | 563.55 | 564.27 | 552.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 14:30:00 | 558.00 | 565.02 | 554.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 15:15:00 | 556.10 | 565.02 | 554.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 560.20 | 563.43 | 553.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 550.55 | 563.17 | 553.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:15:00 | 552.20 | 563.17 | 553.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 544.45 | 562.99 | 553.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 544.45 | 562.99 | 553.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 541.70 | 562.78 | 553.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:30:00 | 542.40 | 562.78 | 553.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-20 13:15:00 | 533.50 | 562.49 | 553.70 | SL hit (close<static) qty=1.00 sl=534.60 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 15:15:00 | 550.50 | 562.16 | 562.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 545.15 | 561.67 | 561.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 10:15:00 | 563.00 | 556.42 | 559.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 10:15:00 | 563.00 | 556.42 | 559.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 563.00 | 556.42 | 559.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 563.50 | 556.42 | 559.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 561.75 | 556.48 | 559.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:45:00 | 563.15 | 556.48 | 559.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 560.80 | 556.64 | 559.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 554.15 | 556.64 | 559.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 576.40 | 555.80 | 558.61 | SL hit (close>static) qty=1.00 sl=560.80 alert=retest2 |

### Cycle 5 — BUY (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 09:15:00 | 574.85 | 561.13 | 561.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 585.15 | 561.92 | 561.47 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-18 12:00:00 | 396.25 | 2025-07-18 13:15:00 | 416.06 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-07-18 12:00:00 | 396.25 | 2025-07-18 14:15:00 | 435.88 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-01 09:15:00 | 410.80 | 2025-09-03 09:15:00 | 451.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-15 14:00:00 | 533.60 | 2025-12-17 10:15:00 | 506.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-16 09:15:00 | 526.35 | 2025-12-17 14:15:00 | 500.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-15 14:00:00 | 533.60 | 2025-12-23 09:15:00 | 540.00 | STOP_HIT | 0.50 | -1.20% |
| SELL | retest2 | 2025-12-16 09:15:00 | 526.35 | 2025-12-23 09:15:00 | 540.00 | STOP_HIT | 0.50 | -2.59% |
| SELL | retest2 | 2025-12-23 11:30:00 | 532.40 | 2025-12-24 09:15:00 | 548.50 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2026-01-12 12:30:00 | 563.55 | 2026-01-20 13:15:00 | 533.50 | STOP_HIT | 1.00 | -5.33% |
| BUY | retest2 | 2026-01-16 14:30:00 | 558.00 | 2026-01-20 13:15:00 | 533.50 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2026-01-16 15:15:00 | 556.10 | 2026-01-20 13:15:00 | 533.50 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2026-01-20 09:15:00 | 560.20 | 2026-01-20 13:15:00 | 533.50 | STOP_HIT | 1.00 | -4.77% |
| BUY | retest2 | 2026-02-18 13:30:00 | 575.50 | 2026-02-24 12:15:00 | 555.65 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2026-02-19 11:15:00 | 572.90 | 2026-02-24 12:15:00 | 555.65 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2026-02-19 12:15:00 | 569.35 | 2026-02-24 12:15:00 | 555.65 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-02-19 13:00:00 | 570.10 | 2026-02-24 12:15:00 | 555.65 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2026-02-20 09:15:00 | 563.35 | 2026-03-02 10:15:00 | 559.05 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-02-23 13:15:00 | 564.65 | 2026-03-02 10:15:00 | 559.05 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-02-23 15:00:00 | 564.40 | 2026-03-02 10:15:00 | 559.05 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-02-24 15:15:00 | 563.70 | 2026-03-02 12:15:00 | 548.55 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2026-02-26 09:30:00 | 571.35 | 2026-03-02 12:15:00 | 548.55 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2026-02-27 12:45:00 | 567.45 | 2026-03-02 12:15:00 | 548.55 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2026-03-02 09:45:00 | 567.65 | 2026-03-02 12:15:00 | 548.55 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2026-03-19 09:15:00 | 554.15 | 2026-03-20 09:15:00 | 576.40 | STOP_HIT | 1.00 | -4.02% |
