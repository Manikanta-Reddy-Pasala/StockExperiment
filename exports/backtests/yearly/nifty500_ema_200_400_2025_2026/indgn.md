# Indegene Ltd. (INDGN)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 530.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 4 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 14
- **Target hits / Stop hits / Partials:** 2 / 14 / 2
- **Avg / median % per leg:** -0.09% / -1.44%
- **Sum % (uncompounded):** -1.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 4 | 22.2% | 2 | 14 | 2 | -0.09% | -1.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 4 | 22.2% | 2 | 14 | 2 | -0.09% | -1.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 4 | 22.2% | 2 | 14 | 2 | -0.09% | -1.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 555.50 | 570.77 | 570.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 553.50 | 570.60 | 570.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 572.25 | 564.10 | 567.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 572.25 | 564.10 | 567.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 572.25 | 564.10 | 567.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:15:00 | 576.00 | 564.10 | 567.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 571.70 | 564.18 | 567.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:30:00 | 576.45 | 564.18 | 567.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 568.20 | 566.29 | 568.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 568.20 | 566.29 | 568.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 568.10 | 566.30 | 568.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 568.10 | 566.30 | 568.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 568.50 | 566.33 | 568.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:45:00 | 569.30 | 566.33 | 568.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 567.20 | 566.33 | 568.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 564.55 | 566.33 | 568.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:00:00 | 565.85 | 566.33 | 568.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 565.30 | 566.32 | 568.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:30:00 | 565.80 | 566.35 | 568.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 572.70 | 566.42 | 568.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 572.70 | 566.42 | 568.11 | SL hit (close>static) qty=1.00 sl=569.95 alert=retest2 |

### Cycle 2 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 589.00 | 569.34 | 569.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 12:15:00 | 589.35 | 570.49 | 569.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 10:15:00 | 571.25 | 572.09 | 570.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 10:15:00 | 571.25 | 572.09 | 570.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 571.25 | 572.09 | 570.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 571.25 | 572.09 | 570.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 572.25 | 572.10 | 570.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:30:00 | 572.85 | 572.10 | 570.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 572.95 | 572.10 | 570.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:30:00 | 571.50 | 572.10 | 570.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 572.00 | 572.13 | 570.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 574.00 | 572.13 | 570.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 567.50 | 572.10 | 570.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 567.50 | 572.10 | 570.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 567.55 | 572.06 | 570.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 561.50 | 572.06 | 570.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 551.05 | 569.70 | 569.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 13:15:00 | 541.00 | 569.03 | 569.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 564.30 | 561.87 | 565.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 564.30 | 561.87 | 565.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 564.30 | 561.87 | 565.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 564.30 | 561.87 | 565.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 564.60 | 561.90 | 565.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 565.80 | 561.90 | 565.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 567.55 | 561.96 | 565.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:45:00 | 567.35 | 561.96 | 565.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 567.50 | 562.01 | 565.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:30:00 | 568.80 | 562.01 | 565.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 570.55 | 562.41 | 565.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:30:00 | 566.55 | 562.49 | 565.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 15:15:00 | 576.00 | 562.91 | 565.44 | SL hit (close>static) qty=1.00 sl=575.20 alert=retest2 |

### Cycle 4 — BUY (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 10:15:00 | 590.20 | 567.73 | 567.63 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 543.85 | 568.01 | 568.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 536.65 | 563.96 | 565.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 14:15:00 | 557.85 | 557.51 | 562.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 14:15:00 | 557.85 | 557.51 | 562.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 557.85 | 557.51 | 562.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:30:00 | 560.40 | 557.51 | 562.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 536.85 | 527.95 | 537.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 536.85 | 527.95 | 537.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 532.95 | 528.44 | 536.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 14:30:00 | 525.70 | 528.62 | 536.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 527.05 | 528.33 | 535.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 10:30:00 | 528.45 | 528.31 | 535.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 538.10 | 528.44 | 535.10 | SL hit (close>static) qty=1.00 sl=537.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 09:15:00 | 499.15 | 480.90 | 480.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 525.30 | 482.50 | 481.64 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-06 09:15:00 | 564.55 | 2025-08-06 12:15:00 | 572.70 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-08-06 10:00:00 | 565.85 | 2025-08-06 12:15:00 | 572.70 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-08-06 10:30:00 | 565.30 | 2025-08-06 12:15:00 | 572.70 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-08-06 11:30:00 | 565.80 | 2025-08-06 12:15:00 | 572.70 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-08-08 09:30:00 | 570.40 | 2025-08-18 09:15:00 | 590.65 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-08-11 09:15:00 | 569.25 | 2025-08-18 09:15:00 | 590.65 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2025-08-11 11:45:00 | 570.35 | 2025-08-18 09:15:00 | 590.65 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-08-13 10:30:00 | 570.10 | 2025-08-18 09:15:00 | 590.65 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-09-12 11:30:00 | 566.55 | 2025-09-12 15:15:00 | 576.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-09-15 14:45:00 | 569.75 | 2025-09-15 15:15:00 | 576.55 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-12 14:30:00 | 525.70 | 2025-12-18 15:15:00 | 538.10 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-12-18 09:30:00 | 527.05 | 2025-12-18 15:15:00 | 538.10 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-12-18 10:30:00 | 528.45 | 2025-12-18 15:15:00 | 538.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-19 11:00:00 | 529.20 | 2025-12-22 15:15:00 | 543.80 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-12-23 09:15:00 | 531.30 | 2026-01-09 15:15:00 | 507.92 | PARTIAL | 0.50 | 4.40% |
| SELL | retest2 | 2025-12-23 10:15:00 | 534.65 | 2026-01-12 09:15:00 | 504.73 | PARTIAL | 0.50 | 5.60% |
| SELL | retest2 | 2025-12-23 09:15:00 | 531.30 | 2026-01-19 11:15:00 | 481.19 | TARGET_HIT | 0.50 | 9.43% |
| SELL | retest2 | 2025-12-23 10:15:00 | 534.65 | 2026-01-19 14:15:00 | 478.17 | TARGET_HIT | 0.50 | 10.56% |
