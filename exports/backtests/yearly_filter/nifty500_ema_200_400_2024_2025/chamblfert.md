# Chambal Fertilizers & Chemicals Ltd. (CHAMBLFERT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 455.85
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
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 31 |
| PARTIAL | 11 |
| TARGET_HIT | 8 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 22 / 15
- **Target hits / Stop hits / Partials:** 8 / 18 / 11
- **Avg / median % per leg:** 2.82% / 3.20%
- **Sum % (uncompounded):** 104.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 1 | 3 | 0 | 0.77% | 3.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 1 | 25.0% | 1 | 3 | 0 | 0.77% | 3.1% |
| SELL (all) | 33 | 21 | 63.6% | 7 | 15 | 11 | 3.07% | 101.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 21 | 63.6% | 7 | 15 | 11 | 3.07% | 101.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 37 | 22 | 59.5% | 8 | 18 | 11 | 2.82% | 104.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 14:15:00 | 472.30 | 498.54 | 498.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 461.70 | 497.93 | 498.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 495.00 | 486.98 | 491.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 495.00 | 486.98 | 491.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 495.00 | 486.98 | 491.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 495.00 | 486.98 | 491.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 494.25 | 487.05 | 491.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:15:00 | 495.65 | 487.05 | 491.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 495.70 | 487.14 | 491.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:30:00 | 494.40 | 487.14 | 491.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 491.65 | 490.10 | 493.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 11:15:00 | 489.00 | 490.10 | 493.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:30:00 | 488.60 | 489.36 | 492.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 464.55 | 487.96 | 491.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 464.17 | 487.96 | 491.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-25 11:15:00 | 481.05 | 480.97 | 487.21 | SL hit (close>ema200) qty=0.50 sl=480.97 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 13:15:00 | 536.70 | 491.71 | 491.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 09:15:00 | 540.45 | 493.03 | 492.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 508.25 | 513.36 | 504.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 15:00:00 | 508.25 | 513.36 | 504.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 512.45 | 513.30 | 505.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 11:15:00 | 513.75 | 513.30 | 505.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-24 10:15:00 | 513.10 | 513.05 | 505.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 14:15:00 | 503.25 | 512.71 | 505.15 | SL hit (close<static) qty=1.00 sl=504.30 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 454.35 | 501.11 | 501.12 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 15:15:00 | 520.50 | 499.14 | 499.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 09:15:00 | 527.00 | 499.42 | 499.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 663.30 | 664.92 | 626.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 11:00:00 | 663.30 | 664.92 | 626.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 625.05 | 659.47 | 628.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 621.10 | 659.47 | 628.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 624.70 | 659.13 | 628.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:00:00 | 624.70 | 659.13 | 628.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 623.00 | 658.77 | 628.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:30:00 | 624.50 | 658.77 | 628.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 619.10 | 656.67 | 628.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 619.10 | 656.67 | 628.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 614.85 | 656.26 | 627.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 614.50 | 656.26 | 627.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 11:15:00 | 551.20 | 610.34 | 610.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 12:15:00 | 549.40 | 609.73 | 610.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 558.95 | 558.33 | 570.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 10:45:00 | 560.80 | 558.33 | 570.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 565.35 | 547.61 | 561.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:45:00 | 567.80 | 547.61 | 561.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 564.50 | 547.77 | 561.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:45:00 | 565.30 | 547.77 | 561.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 567.10 | 548.65 | 561.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:00:00 | 567.10 | 548.65 | 561.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 567.85 | 548.84 | 561.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:45:00 | 567.50 | 548.84 | 561.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 560.10 | 549.66 | 561.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:30:00 | 562.05 | 549.66 | 561.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 563.50 | 549.80 | 561.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:00:00 | 563.50 | 549.80 | 561.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 557.50 | 549.88 | 561.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 556.30 | 549.88 | 561.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 546.05 | 550.18 | 561.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 15:15:00 | 528.48 | 548.97 | 560.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:15:00 | 518.75 | 548.21 | 560.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-19 09:15:00 | 550.15 | 543.05 | 555.40 | SL hit (close>ema200) qty=0.50 sl=543.05 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 454.85 | 444.81 | 444.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 455.65 | 446.11 | 445.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 444.75 | 446.25 | 445.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 444.75 | 446.25 | 445.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 444.75 | 446.25 | 445.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 444.75 | 446.25 | 445.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 436.75 | 446.15 | 445.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 436.75 | 446.15 | 445.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 440.55 | 446.10 | 445.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:15:00 | 440.85 | 446.10 | 445.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:45:00 | 441.10 | 446.03 | 445.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:45:00 | 442.25 | 445.98 | 445.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 14:45:00 | 440.80 | 445.90 | 445.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 444.75 | 445.79 | 445.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 448.95 | 445.30 | 445.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-04 13:00:00 | 362.85 | 2024-06-04 14:15:00 | 352.05 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2024-06-05 10:00:00 | 363.40 | 2024-06-06 09:15:00 | 399.74 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-08 11:15:00 | 489.00 | 2024-11-13 14:15:00 | 464.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:30:00 | 488.60 | 2024-11-13 14:15:00 | 464.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 11:15:00 | 489.00 | 2024-11-25 11:15:00 | 481.05 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2024-11-12 12:30:00 | 488.60 | 2024-11-25 11:15:00 | 481.05 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2024-11-27 14:45:00 | 489.40 | 2024-11-28 09:15:00 | 502.00 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2024-12-23 11:15:00 | 513.75 | 2024-12-24 14:15:00 | 503.25 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-12-24 10:15:00 | 513.10 | 2024-12-24 14:15:00 | 503.25 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-08-07 13:15:00 | 556.30 | 2025-08-08 15:15:00 | 528.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 09:15:00 | 546.05 | 2025-08-11 11:15:00 | 518.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-07 13:15:00 | 556.30 | 2025-08-19 09:15:00 | 550.15 | STOP_HIT | 0.50 | 1.11% |
| SELL | retest2 | 2025-08-08 09:15:00 | 546.05 | 2025-08-19 09:15:00 | 550.15 | STOP_HIT | 0.50 | -0.75% |
| SELL | retest2 | 2025-08-22 09:15:00 | 555.05 | 2025-09-02 09:15:00 | 561.60 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-08-22 09:45:00 | 556.70 | 2025-09-02 09:15:00 | 561.60 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-08-26 14:45:00 | 552.90 | 2025-09-02 10:15:00 | 567.70 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-08-28 09:15:00 | 544.50 | 2025-09-02 10:15:00 | 567.70 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2025-09-05 12:30:00 | 552.50 | 2025-09-17 11:15:00 | 557.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-05 14:00:00 | 552.85 | 2025-09-17 11:15:00 | 557.90 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-09-09 09:15:00 | 543.75 | 2025-09-17 11:15:00 | 557.90 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-09-10 13:45:00 | 544.45 | 2025-09-17 11:15:00 | 557.90 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-09-10 15:15:00 | 544.30 | 2025-09-26 09:15:00 | 524.88 | PARTIAL | 0.50 | 3.57% |
| SELL | retest2 | 2025-09-11 09:30:00 | 542.55 | 2025-09-26 09:15:00 | 525.21 | PARTIAL | 0.50 | 3.20% |
| SELL | retest2 | 2025-09-18 11:45:00 | 547.50 | 2025-09-26 09:15:00 | 520.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 15:00:00 | 546.90 | 2025-09-26 09:15:00 | 519.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 12:15:00 | 548.00 | 2025-09-26 09:15:00 | 520.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 12:45:00 | 547.65 | 2025-09-26 09:15:00 | 520.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 15:15:00 | 544.30 | 2025-10-10 14:15:00 | 497.25 | TARGET_HIT | 0.50 | 8.64% |
| SELL | retest2 | 2025-09-11 09:30:00 | 542.55 | 2025-10-10 14:15:00 | 497.57 | TARGET_HIT | 0.50 | 8.29% |
| SELL | retest2 | 2025-09-18 11:45:00 | 547.50 | 2025-10-14 09:15:00 | 492.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-18 15:00:00 | 546.90 | 2025-10-14 09:15:00 | 492.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-19 12:15:00 | 548.00 | 2025-10-14 09:15:00 | 493.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-19 12:45:00 | 547.65 | 2025-10-14 09:15:00 | 492.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-23 12:45:00 | 469.85 | 2025-12-24 13:15:00 | 475.60 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-12-26 11:15:00 | 469.50 | 2025-12-29 10:15:00 | 475.40 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-01-07 09:15:00 | 468.95 | 2026-01-09 09:15:00 | 445.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 09:15:00 | 468.95 | 2026-01-21 10:15:00 | 422.06 | TARGET_HIT | 0.50 | 10.00% |
