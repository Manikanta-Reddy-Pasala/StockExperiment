# Indegene Ltd. (INDGN)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-11 15:15:00 (3451 bars)
- **Last close:** 542.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 55 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 44 |
| PARTIAL | 9 |
| TARGET_HIT | 7 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 31
- **Target hits / Stop hits / Partials:** 7 / 37 / 9
- **Avg / median % per leg:** 1.15% / -1.08%
- **Sum % (uncompounded):** 60.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.97% | -15.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.97% | -15.8% |
| SELL (all) | 45 | 22 | 48.9% | 7 | 29 | 9 | 1.70% | 76.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 22 | 48.9% | 7 | 29 | 9 | 1.70% | 76.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 22 | 41.5% | 7 | 37 | 9 | 1.15% | 60.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 15:15:00 | 589.00 | 649.03 | 649.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 12:15:00 | 588.50 | 646.74 | 647.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 616.80 | 612.67 | 624.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 10:45:00 | 618.45 | 612.67 | 624.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 629.50 | 613.25 | 624.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:00:00 | 623.25 | 613.35 | 624.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:30:00 | 620.95 | 613.41 | 624.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 634.60 | 614.17 | 624.99 | SL hit (close>static) qty=1.00 sl=632.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 15:15:00 | 662.20 | 631.06 | 631.04 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 11:15:00 | 597.95 | 631.12 | 631.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 576.35 | 617.12 | 622.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 573.45 | 560.42 | 584.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 15:00:00 | 573.45 | 560.42 | 584.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 572.80 | 560.54 | 584.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 541.00 | 560.54 | 584.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 14:15:00 | 513.95 | 556.03 | 580.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-24 12:15:00 | 551.15 | 548.30 | 572.15 | SL hit (close>ema200) qty=0.50 sl=548.30 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 12:15:00 | 595.65 | 569.35 | 569.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 603.70 | 570.50 | 569.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 581.65 | 588.11 | 580.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 581.65 | 588.11 | 580.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 581.65 | 588.11 | 580.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 583.05 | 588.11 | 580.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 585.00 | 588.08 | 580.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:30:00 | 586.25 | 588.08 | 580.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 580.60 | 587.91 | 580.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:00:00 | 580.60 | 587.91 | 580.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 580.70 | 587.84 | 580.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 580.70 | 587.84 | 580.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 583.00 | 587.79 | 580.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 586.35 | 587.79 | 580.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 11:00:00 | 584.85 | 587.91 | 581.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 12:00:00 | 584.45 | 587.88 | 581.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 591.50 | 587.67 | 581.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 586.30 | 589.80 | 583.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:30:00 | 587.35 | 589.80 | 583.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 593.00 | 589.79 | 583.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 588.75 | 589.79 | 583.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 580.65 | 589.67 | 583.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 580.65 | 589.67 | 583.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 580.90 | 589.58 | 583.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 588.85 | 589.58 | 583.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 10:15:00 | 580.00 | 589.40 | 583.51 | SL hit (close<static) qty=1.00 sl=580.40 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 15:15:00 | 567.20 | 579.75 | 579.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 15:15:00 | 565.85 | 578.97 | 579.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 13:15:00 | 577.35 | 574.76 | 576.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 13:15:00 | 577.35 | 574.76 | 576.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 577.35 | 574.76 | 576.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:30:00 | 573.80 | 574.76 | 576.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 577.25 | 574.78 | 576.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 12:15:00 | 575.30 | 574.92 | 577.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 12:45:00 | 574.40 | 574.93 | 577.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 14:00:00 | 575.05 | 575.24 | 577.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:00:00 | 574.10 | 575.26 | 577.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 576.05 | 575.20 | 577.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:45:00 | 576.35 | 575.20 | 577.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 575.10 | 575.20 | 577.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:30:00 | 577.55 | 575.20 | 577.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 14:15:00 | 546.53 | 568.75 | 573.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 14:15:00 | 546.30 | 568.75 | 573.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 545.68 | 568.33 | 572.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 545.39 | 568.33 | 572.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 572.25 | 564.17 | 570.22 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 572.25 | 564.17 | 570.22 | SL hit (close>ema200) qty=0.50 sl=564.17 alert=retest2 |

### Cycle 6 — BUY (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 13:15:00 | 578.55 | 569.42 | 569.40 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 563.40 | 569.45 | 569.47 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 579.40 | 569.54 | 569.51 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 556.05 | 569.46 | 569.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 13:15:00 | 554.90 | 569.07 | 569.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 14:15:00 | 557.85 | 557.51 | 562.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 14:15:00 | 557.85 | 557.51 | 562.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 557.85 | 557.51 | 562.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:30:00 | 560.40 | 557.51 | 562.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 536.85 | 527.95 | 537.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 536.85 | 527.95 | 537.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 532.95 | 528.44 | 536.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 14:30:00 | 525.70 | 528.62 | 536.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 527.05 | 528.33 | 535.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 10:30:00 | 528.45 | 528.31 | 535.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 538.10 | 528.44 | 535.21 | SL hit (close>static) qty=1.00 sl=537.00 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 09:15:00 | 499.15 | 480.90 | 480.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 525.30 | 482.50 | 481.64 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-01-08 11:00:00 | 623.25 | 2025-01-09 09:15:00 | 634.60 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-01-08 11:30:00 | 620.95 | 2025-01-09 09:15:00 | 634.60 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-01-10 09:15:00 | 614.40 | 2025-01-14 15:15:00 | 634.50 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2025-01-10 12:00:00 | 614.00 | 2025-01-14 15:15:00 | 634.50 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2025-03-12 09:15:00 | 541.00 | 2025-03-13 14:15:00 | 513.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-12 09:15:00 | 541.00 | 2025-03-24 12:15:00 | 551.15 | STOP_HIT | 0.50 | -1.88% |
| SELL | retest2 | 2025-03-28 09:45:00 | 571.45 | 2025-04-02 14:15:00 | 577.95 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-04-01 09:15:00 | 564.15 | 2025-04-02 14:15:00 | 577.95 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-04-02 09:15:00 | 565.80 | 2025-04-07 09:15:00 | 514.31 | TARGET_HIT | 1.00 | 9.10% |
| SELL | retest2 | 2025-04-02 12:15:00 | 572.20 | 2025-04-07 09:15:00 | 507.74 | TARGET_HIT | 1.00 | 11.27% |
| SELL | retest2 | 2025-04-02 12:45:00 | 572.10 | 2025-04-07 09:15:00 | 509.22 | TARGET_HIT | 1.00 | 10.99% |
| SELL | retest2 | 2025-04-03 09:45:00 | 571.70 | 2025-04-07 09:15:00 | 514.53 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-03 15:15:00 | 571.50 | 2025-04-07 09:15:00 | 514.35 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-21 09:15:00 | 566.95 | 2025-04-21 12:15:00 | 573.80 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-04-21 10:15:00 | 567.95 | 2025-04-21 12:15:00 | 573.80 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-04-21 15:00:00 | 568.20 | 2025-04-22 09:15:00 | 571.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-04-25 09:30:00 | 564.00 | 2025-04-28 13:15:00 | 571.75 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-04-29 11:45:00 | 567.00 | 2025-05-02 09:15:00 | 538.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 11:45:00 | 567.00 | 2025-05-05 15:15:00 | 560.00 | STOP_HIT | 0.50 | 1.23% |
| SELL | retest2 | 2025-05-13 13:15:00 | 570.00 | 2025-05-15 09:15:00 | 582.00 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-05-13 13:45:00 | 570.00 | 2025-05-15 09:15:00 | 582.00 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-05-14 09:45:00 | 568.65 | 2025-05-15 09:15:00 | 582.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-06-09 09:15:00 | 586.35 | 2025-06-19 10:15:00 | 580.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-06-10 11:00:00 | 584.85 | 2025-06-19 12:15:00 | 570.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-06-10 12:00:00 | 584.45 | 2025-06-19 12:15:00 | 570.00 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-06-11 09:15:00 | 591.50 | 2025-06-19 12:15:00 | 570.00 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2025-06-18 09:15:00 | 588.85 | 2025-06-19 12:15:00 | 570.00 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2025-06-25 09:45:00 | 582.80 | 2025-06-25 11:15:00 | 578.60 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-06-25 10:45:00 | 583.40 | 2025-06-25 11:15:00 | 578.60 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-06-26 09:15:00 | 583.50 | 2025-06-26 09:15:00 | 576.05 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-07-16 12:15:00 | 575.30 | 2025-07-28 14:15:00 | 546.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 12:45:00 | 574.40 | 2025-07-28 14:15:00 | 546.30 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-07-17 14:00:00 | 575.05 | 2025-07-29 09:15:00 | 545.68 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-07-18 10:00:00 | 574.10 | 2025-07-29 09:15:00 | 545.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 12:15:00 | 575.30 | 2025-08-01 09:15:00 | 572.25 | STOP_HIT | 0.50 | 0.53% |
| SELL | retest2 | 2025-07-16 12:45:00 | 574.40 | 2025-08-01 09:15:00 | 572.25 | STOP_HIT | 0.50 | 0.37% |
| SELL | retest2 | 2025-07-17 14:00:00 | 575.05 | 2025-08-01 09:15:00 | 572.25 | STOP_HIT | 0.50 | 0.49% |
| SELL | retest2 | 2025-07-18 10:00:00 | 574.10 | 2025-08-01 09:15:00 | 572.25 | STOP_HIT | 0.50 | 0.32% |
| SELL | retest2 | 2025-08-11 09:30:00 | 566.65 | 2025-08-12 09:15:00 | 573.45 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-08-25 15:00:00 | 567.50 | 2025-09-01 11:15:00 | 539.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 15:00:00 | 567.50 | 2025-09-11 09:15:00 | 564.30 | STOP_HIT | 0.50 | 0.56% |
| SELL | retest2 | 2025-08-26 09:15:00 | 561.50 | 2025-09-12 14:15:00 | 574.05 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-09-11 11:45:00 | 567.35 | 2025-09-12 14:15:00 | 574.05 | STOP_HIT | 1.00 | -1.18% |
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
