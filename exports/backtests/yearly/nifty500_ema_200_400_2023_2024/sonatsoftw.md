# Sonata Software Ltd. (SONATSOFTW)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 296.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 1 |
| ALERT3 | 48 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 51 |
| PARTIAL | 10 |
| TARGET_HIT | 2 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 41
- **Target hits / Stop hits / Partials:** 2 / 49 / 10
- **Avg / median % per leg:** -0.31% / -1.84%
- **Sum % (uncompounded):** -18.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.70% | -13.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.70% | -13.6% |
| SELL (all) | 53 | 20 | 37.7% | 2 | 41 | 10 | -0.10% | -5.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 53 | 20 | 37.7% | 2 | 41 | 10 | -0.10% | -5.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 61 | 20 | 32.8% | 2 | 49 | 10 | -0.31% | -18.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 10:15:00 | 733.20 | 755.23 | 755.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 730.35 | 754.75 | 755.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 09:15:00 | 572.65 | 562.11 | 616.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-07 10:00:00 | 572.65 | 562.11 | 616.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 614.90 | 568.29 | 605.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:00:00 | 605.15 | 570.49 | 605.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 09:15:00 | 574.89 | 572.66 | 604.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 581.80 | 572.66 | 604.15 | SL hit (close>static) qty=0.50 sl=572.66 alert=retest2 |

### Cycle 2 — BUY (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 14:15:00 | 692.50 | 618.21 | 618.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 701.05 | 629.58 | 624.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 649.35 | 663.59 | 644.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 649.35 | 663.59 | 644.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 649.35 | 663.59 | 644.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:45:00 | 646.30 | 663.59 | 644.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 646.65 | 662.97 | 644.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:30:00 | 636.55 | 662.97 | 644.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 628.15 | 662.62 | 644.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 628.15 | 662.62 | 644.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 629.80 | 662.29 | 644.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 13:15:00 | 635.85 | 661.96 | 644.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 14:45:00 | 632.50 | 661.32 | 644.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 09:15:00 | 635.85 | 660.98 | 644.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 11:45:00 | 633.15 | 658.62 | 644.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 13:15:00 | 620.30 | 657.92 | 644.24 | SL hit (close<static) qty=1.00 sl=623.70 alert=retest2 |

### Cycle 3 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 626.60 | 634.58 | 634.59 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 09:15:00 | 648.55 | 634.72 | 634.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 680.50 | 644.31 | 639.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 11:15:00 | 655.00 | 659.32 | 649.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 12:15:00 | 652.00 | 659.32 | 649.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 646.75 | 659.19 | 649.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:00:00 | 646.75 | 659.19 | 649.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 648.80 | 659.09 | 649.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 14:15:00 | 650.45 | 659.09 | 649.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 10:15:00 | 643.25 | 658.75 | 649.80 | SL hit (close<static) qty=1.00 sl=646.25 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 601.00 | 644.15 | 644.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 587.60 | 643.59 | 644.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 12:15:00 | 625.00 | 624.72 | 633.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 13:00:00 | 625.00 | 624.72 | 633.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 627.45 | 624.77 | 632.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:30:00 | 629.25 | 624.77 | 632.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 628.70 | 624.81 | 632.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:45:00 | 630.45 | 624.81 | 632.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 617.05 | 612.80 | 623.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 604.30 | 612.80 | 623.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 11:00:00 | 604.60 | 612.59 | 622.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 11:30:00 | 605.00 | 612.55 | 622.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 12:45:00 | 605.90 | 612.49 | 622.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 623.50 | 612.01 | 621.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:45:00 | 623.00 | 612.01 | 621.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 626.30 | 612.16 | 621.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 626.30 | 612.16 | 621.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 627.05 | 612.31 | 621.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-06 15:15:00 | 627.05 | 612.31 | 621.89 | SL hit (close>static) qty=1.00 sl=626.45 alert=retest2 |

### Cycle 6 — BUY (started 2024-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 13:15:00 | 670.55 | 616.35 | 616.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 673.05 | 617.96 | 617.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 633.80 | 640.70 | 630.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 15:00:00 | 633.80 | 640.70 | 630.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 638.20 | 640.67 | 630.66 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 13:15:00 | 610.05 | 624.27 | 624.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 15:15:00 | 608.00 | 623.95 | 624.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 11:15:00 | 368.20 | 349.18 | 396.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-24 12:00:00 | 368.20 | 349.18 | 396.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 397.35 | 351.78 | 392.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:45:00 | 395.75 | 351.78 | 392.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 400.80 | 352.27 | 392.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 14:15:00 | 391.70 | 366.60 | 395.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 372.11 | 367.25 | 395.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 09:15:00 | 393.65 | 367.25 | 395.46 | SL hit (close>static) qty=0.50 sl=367.25 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 12:15:00 | 423.70 | 401.58 | 401.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 10:15:00 | 434.45 | 407.61 | 405.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 421.75 | 422.54 | 415.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 09:45:00 | 422.70 | 422.54 | 415.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 414.80 | 422.38 | 415.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:00:00 | 414.80 | 422.38 | 415.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 411.45 | 422.28 | 415.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 411.45 | 422.28 | 415.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 414.20 | 419.54 | 414.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 400.65 | 419.54 | 414.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 356.95 | 410.35 | 410.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 354.00 | 408.25 | 409.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 384.20 | 380.61 | 391.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 09:45:00 | 382.55 | 380.61 | 391.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 392.85 | 380.73 | 391.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:45:00 | 395.75 | 380.73 | 391.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 387.30 | 380.79 | 391.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:00:00 | 384.30 | 380.87 | 391.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 13:15:00 | 365.08 | 380.31 | 391.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 377.50 | 370.47 | 382.23 | SL hit (close>ema200) qty=0.50 sl=370.47 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-21 15:00:00 | 605.15 | 2024-06-26 09:15:00 | 574.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-21 15:00:00 | 605.15 | 2024-06-26 09:15:00 | 581.80 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2024-07-01 10:15:00 | 609.00 | 2024-07-03 09:15:00 | 636.05 | STOP_HIT | 1.00 | -4.44% |
| BUY | retest2 | 2024-08-05 13:15:00 | 635.85 | 2024-08-08 13:15:00 | 620.30 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-08-05 14:45:00 | 632.50 | 2024-08-08 13:15:00 | 620.30 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-08-06 09:15:00 | 635.85 | 2024-08-08 13:15:00 | 620.30 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-08-08 11:45:00 | 633.15 | 2024-08-08 13:15:00 | 620.30 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-08-28 10:30:00 | 637.00 | 2024-08-28 13:15:00 | 629.25 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-09-18 14:15:00 | 650.45 | 2024-09-19 10:15:00 | 643.25 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-09-20 10:45:00 | 652.00 | 2024-09-20 13:15:00 | 642.40 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-09-23 09:30:00 | 650.45 | 2024-09-23 11:15:00 | 644.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-11-04 09:15:00 | 604.30 | 2024-11-06 15:15:00 | 627.05 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2024-11-04 11:00:00 | 604.60 | 2024-11-06 15:15:00 | 627.05 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2024-11-04 11:30:00 | 605.00 | 2024-11-06 15:15:00 | 627.05 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2024-11-04 12:45:00 | 605.90 | 2024-11-06 15:15:00 | 627.05 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2024-11-08 09:15:00 | 618.05 | 2024-11-11 13:15:00 | 587.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 14:15:00 | 620.35 | 2024-11-11 13:15:00 | 589.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 618.05 | 2024-11-14 09:15:00 | 556.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-08 14:15:00 | 620.35 | 2024-11-14 09:15:00 | 558.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-06 14:15:00 | 391.70 | 2025-05-07 09:15:00 | 372.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 14:15:00 | 391.70 | 2025-05-07 09:15:00 | 393.65 | STOP_HIT | 0.50 | -0.50% |
| SELL | retest2 | 2025-05-07 10:00:00 | 393.65 | 2025-05-09 10:15:00 | 374.49 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2025-05-07 10:00:00 | 393.65 | 2025-05-09 10:15:00 | 374.85 | STOP_HIT | 0.50 | 4.78% |
| SELL | retest2 | 2025-05-07 14:45:00 | 394.20 | 2025-05-09 11:15:00 | 373.97 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2025-05-07 14:45:00 | 394.20 | 2025-05-09 11:15:00 | 372.45 | STOP_HIT | 0.50 | 5.52% |
| SELL | retest2 | 2025-05-08 10:15:00 | 392.70 | 2025-05-09 11:15:00 | 373.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 10:15:00 | 392.70 | 2025-05-09 11:15:00 | 372.45 | STOP_HIT | 0.50 | 5.16% |
| SELL | retest2 | 2025-05-08 14:45:00 | 390.90 | 2025-05-09 11:15:00 | 371.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 14:45:00 | 390.90 | 2025-05-09 11:15:00 | 372.45 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2025-05-12 10:45:00 | 395.75 | 2025-05-13 09:15:00 | 403.05 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-05-12 12:15:00 | 395.40 | 2025-05-13 09:15:00 | 403.05 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-05-12 13:00:00 | 396.30 | 2025-05-13 09:15:00 | 403.05 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-05-13 12:30:00 | 400.40 | 2025-05-23 10:15:00 | 400.40 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-05-13 13:15:00 | 400.30 | 2025-05-23 10:15:00 | 400.40 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-05-15 09:45:00 | 400.75 | 2025-05-23 10:15:00 | 400.40 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-05-16 09:45:00 | 399.60 | 2025-05-23 10:15:00 | 400.40 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-05-19 11:15:00 | 394.30 | 2025-05-26 10:15:00 | 405.25 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-05-19 11:45:00 | 393.65 | 2025-05-26 10:15:00 | 405.25 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-05-20 10:15:00 | 393.30 | 2025-05-26 10:15:00 | 405.25 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-05-20 10:45:00 | 393.75 | 2025-05-26 10:15:00 | 405.25 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-05-21 12:00:00 | 390.30 | 2025-05-30 15:15:00 | 409.35 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2025-05-22 09:15:00 | 391.10 | 2025-05-30 15:15:00 | 409.35 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2025-05-22 11:30:00 | 391.20 | 2025-05-30 15:15:00 | 409.35 | STOP_HIT | 1.00 | -4.64% |
| SELL | retest2 | 2025-05-22 12:00:00 | 391.15 | 2025-05-30 15:15:00 | 409.35 | STOP_HIT | 1.00 | -4.65% |
| SELL | retest2 | 2025-08-25 14:00:00 | 384.30 | 2025-08-26 13:15:00 | 365.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 14:00:00 | 384.30 | 2025-09-10 09:15:00 | 377.50 | STOP_HIT | 0.50 | 1.77% |
| SELL | retest2 | 2025-09-12 10:15:00 | 385.00 | 2025-09-18 09:15:00 | 413.20 | STOP_HIT | 1.00 | -7.32% |
| SELL | retest2 | 2025-09-17 10:15:00 | 384.45 | 2025-09-18 09:15:00 | 413.20 | STOP_HIT | 1.00 | -7.48% |
| SELL | retest2 | 2025-09-22 10:00:00 | 385.05 | 2025-09-22 15:15:00 | 395.00 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-10-09 09:30:00 | 367.20 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-10-09 10:00:00 | 367.55 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-10-10 13:30:00 | 368.85 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-10-10 14:30:00 | 369.15 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-10-16 10:00:00 | 366.55 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-10-16 11:00:00 | 366.60 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-10-17 09:30:00 | 366.55 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-10-20 09:15:00 | 364.10 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2025-10-24 13:30:00 | 374.25 | 2025-11-12 09:15:00 | 383.35 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-11-10 13:00:00 | 376.75 | 2025-11-12 09:15:00 | 383.35 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-11-14 09:15:00 | 369.30 | 2025-12-02 13:15:00 | 350.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 09:15:00 | 369.30 | 2025-12-03 10:15:00 | 368.40 | STOP_HIT | 0.50 | 0.24% |
