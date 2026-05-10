# Berger Paints India Ltd. (BERGEPAINT)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 515.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 18
- **Target hits / Stop hits / Partials:** 0 / 19 / 1
- **Avg / median % per leg:** -1.06% / -1.16%
- **Sum % (uncompounded):** -21.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 20 | 2 | 10.0% | 0 | 19 | 1 | -1.06% | -21.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 2 | 10.0% | 0 | 19 | 1 | -1.06% | -21.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 2 | 10.0% | 0 | 19 | 1 | -1.06% | -21.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 536.50 | 560.20 | 560.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 530.65 | 556.78 | 558.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 13:15:00 | 544.70 | 543.98 | 549.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:45:00 | 544.65 | 543.98 | 549.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 550.55 | 544.04 | 549.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 550.55 | 544.04 | 549.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 552.00 | 544.12 | 549.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 550.35 | 544.12 | 549.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 548.25 | 544.23 | 549.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:15:00 | 546.80 | 544.23 | 549.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 551.40 | 544.32 | 549.72 | SL hit (close>static) qty=1.00 sl=550.45 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:30:00 | 547.30 | 544.32 | 549.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 14:15:00 | 546.65 | 544.36 | 549.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 551.25 | 544.23 | 549.38 | SL hit (close>static) qty=1.00 sl=550.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 551.25 | 544.23 | 549.38 | SL hit (close>static) qty=1.00 sl=550.45 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:45:00 | 546.80 | 544.36 | 549.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 15:15:00 | 519.46 | 539.16 | 545.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 12:15:00 | 539.15 | 534.73 | 542.10 | SL hit (close>ema200) qty=0.50 sl=534.73 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 543.00 | 534.85 | 542.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:45:00 | 543.10 | 534.85 | 542.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 537.90 | 534.88 | 542.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 535.20 | 534.88 | 542.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:45:00 | 536.70 | 533.71 | 539.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 12:45:00 | 536.65 | 533.87 | 539.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 13:15:00 | 536.95 | 533.87 | 539.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 541.95 | 533.95 | 539.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 541.95 | 533.95 | 539.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 545.05 | 534.06 | 539.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 545.05 | 534.06 | 539.45 | SL hit (close>static) qty=1.00 sl=543.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 545.05 | 534.06 | 539.45 | SL hit (close>static) qty=1.00 sl=543.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 545.05 | 534.06 | 539.45 | SL hit (close>static) qty=1.00 sl=543.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 545.05 | 534.06 | 539.45 | SL hit (close>static) qty=1.00 sl=543.25 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 541.40 | 534.32 | 539.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 11:45:00 | 541.50 | 534.39 | 539.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 12:30:00 | 541.35 | 534.47 | 539.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 13:15:00 | 541.40 | 534.47 | 539.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 539.80 | 534.65 | 539.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 540.90 | 534.70 | 539.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 540.10 | 534.75 | 539.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 540.10 | 534.75 | 539.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 537.25 | 534.78 | 539.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 546.00 | 537.44 | 540.10 | SL hit (close>static) qty=1.00 sl=545.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 546.00 | 537.44 | 540.10 | SL hit (close>static) qty=1.00 sl=545.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 546.00 | 537.44 | 540.10 | SL hit (close>static) qty=1.00 sl=545.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 546.00 | 537.44 | 540.10 | SL hit (close>static) qty=1.00 sl=545.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:30:00 | 536.00 | 537.81 | 540.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 550.80 | 538.01 | 540.13 | SL hit (close>static) qty=1.00 sl=541.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 10:00:00 | 535.80 | 538.39 | 540.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:30:00 | 536.00 | 537.86 | 539.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:00:00 | 535.00 | 537.86 | 539.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 549.85 | 537.91 | 539.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 549.85 | 537.91 | 539.87 | SL hit (close>static) qty=1.00 sl=541.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 549.85 | 537.91 | 539.87 | SL hit (close>static) qty=1.00 sl=541.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 549.85 | 537.91 | 539.87 | SL hit (close>static) qty=1.00 sl=541.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 555.15 | 537.91 | 539.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 546.00 | 537.99 | 539.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:45:00 | 544.55 | 538.07 | 539.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 545.00 | 538.35 | 540.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 551.30 | 538.54 | 540.13 | SL hit (close>static) qty=1.00 sl=550.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 551.30 | 538.54 | 540.13 | SL hit (close>static) qty=1.00 sl=550.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 15:15:00 | 581.00 | 541.82 | 541.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 10:15:00 | 587.00 | 547.28 | 544.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 10:15:00 | 560.30 | 560.92 | 553.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:45:00 | 560.30 | 560.92 | 553.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 553.20 | 560.67 | 553.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 554.40 | 560.67 | 553.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 552.05 | 560.58 | 553.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:15:00 | 551.10 | 560.58 | 553.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 551.20 | 560.49 | 553.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 551.20 | 560.49 | 553.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 553.50 | 560.32 | 553.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:30:00 | 551.40 | 560.32 | 553.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 558.50 | 560.31 | 553.47 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 10:15:00 | 537.80 | 549.63 | 549.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 535.25 | 548.00 | 548.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 431.00 | 430.76 | 454.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 15:00:00 | 431.00 | 430.76 | 454.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 452.15 | 431.78 | 452.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:00:00 | 452.15 | 431.78 | 452.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 452.00 | 431.98 | 452.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 452.00 | 431.98 | 452.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 454.50 | 432.20 | 452.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:00:00 | 454.50 | 432.20 | 452.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 452.80 | 432.41 | 452.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 445.70 | 433.03 | 452.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 10:15:00 | 455.15 | 433.40 | 452.23 | SL hit (close>static) qty=1.00 sl=454.70 alert=retest2 |

### Cycle 4 — BUY (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 12:15:00 | 517.60 | 461.55 | 461.51 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-12 11:15:00 | 546.80 | 2025-09-12 12:15:00 | 551.40 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-09-12 12:30:00 | 547.30 | 2025-09-16 09:15:00 | 551.25 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-09-12 14:15:00 | 546.65 | 2025-09-16 09:15:00 | 551.25 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-09-16 12:45:00 | 546.80 | 2025-09-26 15:15:00 | 519.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 12:45:00 | 546.80 | 2025-10-03 12:15:00 | 539.15 | STOP_HIT | 0.50 | 1.40% |
| SELL | retest2 | 2025-10-06 09:15:00 | 535.20 | 2025-10-17 14:15:00 | 545.05 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-10-17 09:45:00 | 536.70 | 2025-10-17 14:15:00 | 545.05 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-10-17 12:45:00 | 536.65 | 2025-10-17 14:15:00 | 545.05 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-10-17 13:15:00 | 536.95 | 2025-10-17 14:15:00 | 545.05 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-10-20 10:45:00 | 541.40 | 2025-10-30 15:15:00 | 546.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-10-20 11:45:00 | 541.50 | 2025-10-30 15:15:00 | 546.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-10-20 12:30:00 | 541.35 | 2025-10-30 15:15:00 | 546.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-10-20 13:15:00 | 541.40 | 2025-10-30 15:15:00 | 546.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-11-03 11:30:00 | 536.00 | 2025-11-06 09:15:00 | 550.80 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-11-07 10:00:00 | 535.80 | 2025-11-11 09:15:00 | 549.85 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-11-10 12:30:00 | 536.00 | 2025-11-11 09:15:00 | 549.85 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-11-10 13:00:00 | 535.00 | 2025-11-11 09:15:00 | 549.85 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-11-11 11:45:00 | 544.55 | 2025-11-12 09:15:00 | 551.30 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-11-11 15:15:00 | 545.00 | 2025-11-12 09:15:00 | 551.30 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-13 09:15:00 | 445.70 | 2026-04-13 10:15:00 | 455.15 | STOP_HIT | 1.00 | -2.12% |
