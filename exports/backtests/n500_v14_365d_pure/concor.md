# Container Corporation of India Ltd. (CONCOR)

## Backtest Summary

- **Window:** 2024-07-09 09:15:00 → 2026-05-08 15:15:00 (3168 bars)
- **Last close:** 533.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 8
- **Target hits / Stop hits / Partials:** 0 / 14 / 5
- **Avg / median % per leg:** 1.24% / 1.31%
- **Sum % (uncompounded):** 23.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 1 | 12.5% | 0 | 8 | 0 | -1.96% | -15.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 1 | 12.5% | 0 | 8 | 0 | -1.96% | -15.6% |
| SELL (all) | 11 | 10 | 90.9% | 0 | 6 | 5 | 3.56% | 39.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 10 | 90.9% | 0 | 6 | 5 | 3.56% | 39.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 11 | 57.9% | 0 | 14 | 5 | 1.24% | 23.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 585.24 | 561.70 | 561.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 589.92 | 561.98 | 561.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 604.40 | 604.91 | 588.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:45:00 | 602.56 | 604.91 | 588.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 589.36 | 604.56 | 590.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:45:00 | 590.44 | 604.56 | 590.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 585.72 | 604.37 | 590.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:45:00 | 585.56 | 604.37 | 590.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 588.28 | 603.06 | 589.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 588.28 | 603.06 | 589.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 586.24 | 602.90 | 589.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:30:00 | 584.92 | 602.90 | 589.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 588.00 | 602.47 | 589.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 587.60 | 602.47 | 589.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 589.00 | 602.33 | 589.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:30:00 | 591.36 | 601.85 | 589.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 595.60 | 601.58 | 589.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:45:00 | 590.70 | 600.34 | 592.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 12:30:00 | 591.60 | 600.31 | 592.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 594.00 | 600.24 | 592.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:45:00 | 594.25 | 600.24 | 592.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 598.70 | 600.23 | 592.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 600.25 | 600.21 | 592.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:45:00 | 604.00 | 600.29 | 592.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 600.85 | 608.09 | 600.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 12:30:00 | 600.10 | 607.92 | 600.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 600.05 | 607.84 | 600.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:30:00 | 598.45 | 607.84 | 600.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 599.70 | 607.76 | 600.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:45:00 | 598.55 | 607.76 | 600.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 598.10 | 607.66 | 600.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 600.20 | 607.66 | 600.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 597.55 | 607.49 | 600.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 597.55 | 607.49 | 600.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 593.70 | 607.35 | 600.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:00:00 | 593.70 | 607.35 | 600.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 591.30 | 607.19 | 600.26 | SL hit (close<static) qty=1.00 sl=592.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 591.30 | 607.19 | 600.26 | SL hit (close<static) qty=1.00 sl=592.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 591.30 | 607.19 | 600.26 | SL hit (close<static) qty=1.00 sl=592.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 591.30 | 607.19 | 600.26 | SL hit (close<static) qty=1.00 sl=592.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 578.80 | 604.47 | 599.44 | SL hit (close<static) qty=1.00 sl=583.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 578.80 | 604.47 | 599.44 | SL hit (close<static) qty=1.00 sl=583.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 578.80 | 604.47 | 599.44 | SL hit (close<static) qty=1.00 sl=583.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 578.80 | 604.47 | 599.44 | SL hit (close<static) qty=1.00 sl=583.40 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 537.50 | 595.03 | 595.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 534.00 | 594.42 | 594.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 561.90 | 553.67 | 565.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 561.90 | 553.67 | 565.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 566.45 | 554.00 | 565.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 568.60 | 554.00 | 565.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 565.25 | 554.11 | 565.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:00:00 | 563.85 | 554.21 | 565.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:30:00 | 564.05 | 554.30 | 565.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 14:30:00 | 564.05 | 554.48 | 565.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 562.50 | 554.66 | 565.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 535.66 | 554.37 | 563.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 535.85 | 554.37 | 563.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 535.85 | 554.37 | 563.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 534.38 | 554.37 | 563.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 544.10 | 541.61 | 552.95 | SL hit (close>ema200) qty=0.50 sl=541.61 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 544.10 | 541.61 | 552.95 | SL hit (close>ema200) qty=0.50 sl=541.61 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 544.10 | 541.61 | 552.95 | SL hit (close>ema200) qty=0.50 sl=541.61 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 544.10 | 541.61 | 552.95 | SL hit (close>ema200) qty=0.50 sl=541.61 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 548.00 | 538.44 | 547.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:45:00 | 547.10 | 538.44 | 547.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 553.40 | 538.58 | 547.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:00:00 | 553.40 | 538.58 | 547.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 544.30 | 539.88 | 547.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 546.10 | 539.88 | 547.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 547.00 | 539.96 | 547.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:45:00 | 547.60 | 539.96 | 547.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 547.90 | 540.03 | 547.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 547.90 | 540.03 | 547.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 546.15 | 540.10 | 547.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 15:00:00 | 544.25 | 540.14 | 547.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 549.90 | 540.30 | 547.36 | SL hit (close>static) qty=1.00 sl=548.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:00:00 | 544.65 | 540.93 | 547.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 517.42 | 539.47 | 546.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 537.50 | 536.16 | 543.76 | SL hit (close>ema200) qty=0.50 sl=536.16 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 518.40 | 486.83 | 486.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 521.20 | 493.03 | 490.04 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-23 13:30:00 | 591.36 | 2025-07-28 12:15:00 | 591.30 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-06-24 09:15:00 | 595.60 | 2025-07-28 12:15:00 | 591.30 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-08 11:45:00 | 590.70 | 2025-07-28 12:15:00 | 591.30 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-07-08 12:30:00 | 591.60 | 2025-07-28 12:15:00 | 591.30 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-07-09 09:15:00 | 600.25 | 2025-07-31 09:15:00 | 578.80 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-07-09 09:45:00 | 604.00 | 2025-07-31 09:15:00 | 578.80 | STOP_HIT | 1.00 | -4.17% |
| BUY | retest2 | 2025-07-25 11:15:00 | 600.85 | 2025-07-31 09:15:00 | 578.80 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2025-07-25 12:30:00 | 600.10 | 2025-07-31 09:15:00 | 578.80 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-09-16 12:00:00 | 563.85 | 2025-09-24 10:15:00 | 535.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 12:30:00 | 564.05 | 2025-09-24 10:15:00 | 535.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 14:30:00 | 564.05 | 2025-09-24 10:15:00 | 535.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 09:45:00 | 562.50 | 2025-09-24 10:15:00 | 534.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 12:00:00 | 563.85 | 2025-10-10 10:15:00 | 544.10 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2025-09-16 12:30:00 | 564.05 | 2025-10-10 10:15:00 | 544.10 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2025-09-16 14:30:00 | 564.05 | 2025-10-10 10:15:00 | 544.10 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2025-09-17 09:45:00 | 562.50 | 2025-10-10 10:15:00 | 544.10 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2025-10-31 15:00:00 | 544.25 | 2025-11-03 09:15:00 | 549.90 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-11-04 11:00:00 | 544.65 | 2025-11-07 09:15:00 | 517.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:00:00 | 544.65 | 2025-11-12 11:15:00 | 537.50 | STOP_HIT | 0.50 | 1.31% |
