# Birlasoft Ltd. (BSOFT)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 362.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 8 |
| ALERT2_SKIP | 1 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 9 |
| TARGET_HIT | 6 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 19 / 24
- **Target hits / Stop hits / Partials:** 6 / 28 / 9
- **Avg / median % per leg:** 0.54% / -1.49%
- **Sum % (uncompounded):** 23.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 1 | 4.8% | 1 | 20 | 0 | -2.76% | -57.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 1 | 4.8% | 1 | 20 | 0 | -2.76% | -57.9% |
| SELL (all) | 22 | 18 | 81.8% | 5 | 8 | 9 | 3.68% | 81.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 18 | 81.8% | 5 | 8 | 9 | 3.68% | 81.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 43 | 19 | 44.2% | 6 | 28 | 9 | 0.54% | 23.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 14:15:00 | 707.25 | 755.00 | 755.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 11:15:00 | 704.50 | 753.11 | 754.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 15:15:00 | 631.10 | 630.98 | 662.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-06 09:15:00 | 649.65 | 630.98 | 662.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 679.40 | 632.44 | 662.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:00:00 | 679.40 | 632.44 | 662.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 685.15 | 632.97 | 662.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:00:00 | 685.15 | 632.97 | 662.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 15:15:00 | 710.60 | 676.13 | 676.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 720.15 | 676.57 | 676.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 10:15:00 | 702.85 | 706.13 | 695.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 11:00:00 | 702.85 | 706.13 | 695.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 684.55 | 705.89 | 695.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 684.55 | 705.89 | 695.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 676.35 | 705.60 | 695.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:45:00 | 677.35 | 705.60 | 695.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 09:15:00 | 602.00 | 686.50 | 686.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 586.00 | 682.78 | 684.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 10:15:00 | 633.25 | 632.19 | 652.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-26 11:00:00 | 633.25 | 632.19 | 652.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 652.50 | 633.03 | 651.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:00:00 | 652.50 | 633.03 | 651.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 663.45 | 633.33 | 651.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:00:00 | 663.45 | 633.33 | 651.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 662.90 | 633.63 | 651.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:45:00 | 664.95 | 633.63 | 651.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 661.15 | 644.24 | 654.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:30:00 | 657.40 | 645.54 | 655.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 12:15:00 | 656.80 | 645.54 | 655.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 10:00:00 | 656.85 | 646.19 | 655.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 12:45:00 | 656.40 | 646.58 | 655.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 656.75 | 646.68 | 655.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:30:00 | 657.20 | 646.68 | 655.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 14:15:00 | 624.53 | 645.94 | 654.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 14:15:00 | 623.96 | 645.94 | 654.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 14:15:00 | 624.01 | 645.94 | 654.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 14:15:00 | 623.58 | 645.94 | 654.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-10 13:15:00 | 645.80 | 645.31 | 653.97 | SL hit (close>ema200) qty=0.50 sl=645.31 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 10:15:00 | 434.45 | 420.57 | 420.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 11:15:00 | 436.55 | 421.63 | 421.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 426.25 | 426.96 | 424.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:45:00 | 426.50 | 426.96 | 424.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 423.30 | 426.93 | 424.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 423.30 | 426.93 | 424.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 425.65 | 426.91 | 424.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:30:00 | 428.50 | 426.91 | 424.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 10:00:00 | 427.40 | 426.93 | 424.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 422.10 | 426.88 | 424.28 | SL hit (close<static) qty=1.00 sl=423.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 398.75 | 422.67 | 422.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 396.80 | 422.17 | 422.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 11:15:00 | 420.10 | 417.05 | 419.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 11:15:00 | 420.10 | 417.05 | 419.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 420.10 | 417.05 | 419.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:45:00 | 420.70 | 417.05 | 419.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 413.40 | 417.01 | 419.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:45:00 | 412.50 | 416.97 | 419.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 404.30 | 416.92 | 419.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 14:15:00 | 391.88 | 415.79 | 418.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 384.08 | 410.43 | 415.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-08 09:15:00 | 371.25 | 408.31 | 414.41 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 388.10 | 376.52 | 376.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 399.70 | 379.36 | 378.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 11:15:00 | 422.60 | 424.15 | 409.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 12:00:00 | 422.60 | 424.15 | 409.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 414.00 | 425.60 | 412.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 10:15:00 | 421.30 | 424.73 | 412.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 11:15:00 | 420.75 | 424.69 | 412.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 420.85 | 424.28 | 412.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 434.10 | 424.23 | 412.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 410.55 | 424.92 | 414.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 409.95 | 424.92 | 414.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 405.95 | 424.74 | 414.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 405.95 | 424.74 | 414.12 | SL hit (close<static) qty=1.00 sl=409.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 12:15:00 | 379.15 | 414.33 | 414.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 13:15:00 | 378.40 | 413.97 | 414.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 370.50 | 368.73 | 383.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:45:00 | 370.25 | 368.73 | 383.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 374.25 | 368.70 | 382.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:00:00 | 373.15 | 368.94 | 382.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:45:00 | 371.85 | 368.95 | 382.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 368.95 | 368.96 | 382.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 370.65 | 369.17 | 382.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 380.95 | 369.26 | 380.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:00:00 | 380.95 | 369.26 | 380.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 379.30 | 369.36 | 380.94 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 390.40 | 369.68 | 380.99 | SL hit (close>static) qty=1.00 sl=383.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-02-29 15:00:00 | 780.50 | 2024-03-05 09:15:00 | 757.00 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2024-03-01 14:45:00 | 774.40 | 2024-03-05 09:15:00 | 757.00 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-03-01 15:15:00 | 776.00 | 2024-03-05 09:15:00 | 757.00 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-03-07 10:45:00 | 774.40 | 2024-03-13 10:15:00 | 750.30 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2024-03-11 11:15:00 | 771.85 | 2024-03-13 10:15:00 | 750.30 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-03-13 09:15:00 | 773.00 | 2024-03-13 10:15:00 | 750.30 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2024-04-02 10:15:00 | 773.80 | 2024-04-05 13:15:00 | 749.50 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2024-04-03 11:15:00 | 772.40 | 2024-04-05 13:15:00 | 749.50 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2024-09-05 11:30:00 | 657.40 | 2024-09-09 14:15:00 | 624.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 12:15:00 | 656.80 | 2024-09-09 14:15:00 | 623.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-06 10:00:00 | 656.85 | 2024-09-09 14:15:00 | 624.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-06 12:45:00 | 656.40 | 2024-09-09 14:15:00 | 623.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 11:30:00 | 657.40 | 2024-09-10 13:15:00 | 645.80 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2024-09-05 12:15:00 | 656.80 | 2024-09-10 13:15:00 | 645.80 | STOP_HIT | 0.50 | 1.67% |
| SELL | retest2 | 2024-09-06 10:00:00 | 656.85 | 2024-09-10 13:15:00 | 645.80 | STOP_HIT | 0.50 | 1.68% |
| SELL | retest2 | 2024-09-06 12:45:00 | 656.40 | 2024-09-10 13:15:00 | 645.80 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2024-09-16 13:45:00 | 644.70 | 2024-09-19 11:15:00 | 612.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-18 09:15:00 | 639.50 | 2024-09-19 11:15:00 | 607.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-19 09:30:00 | 637.80 | 2024-09-19 11:15:00 | 605.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-16 13:45:00 | 644.70 | 2024-10-04 14:15:00 | 580.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-18 09:15:00 | 639.50 | 2024-10-07 10:15:00 | 575.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-19 09:30:00 | 637.80 | 2024-10-07 10:15:00 | 574.02 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-10 14:30:00 | 428.50 | 2025-07-11 10:15:00 | 422.10 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-07-11 10:00:00 | 427.40 | 2025-07-11 10:15:00 | 422.10 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-07-15 11:00:00 | 427.40 | 2025-07-18 12:15:00 | 422.35 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-07-30 13:45:00 | 412.50 | 2025-07-31 14:15:00 | 391.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 404.30 | 2025-08-07 09:15:00 | 384.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 13:45:00 | 412.50 | 2025-08-08 09:15:00 | 371.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 404.30 | 2025-08-28 09:15:00 | 363.87 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-13 10:15:00 | 421.30 | 2026-01-20 10:15:00 | 405.95 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2026-01-13 11:15:00 | 420.75 | 2026-01-20 10:15:00 | 405.95 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest2 | 2026-01-14 15:00:00 | 420.85 | 2026-01-20 10:15:00 | 405.95 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2026-01-16 09:15:00 | 434.10 | 2026-01-20 10:15:00 | 405.95 | STOP_HIT | 1.00 | -6.48% |
| BUY | retest2 | 2026-01-22 14:30:00 | 416.35 | 2026-01-23 13:15:00 | 408.65 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-01-23 09:15:00 | 424.50 | 2026-01-23 13:15:00 | 408.65 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2026-01-29 09:15:00 | 423.60 | 2026-01-29 09:15:00 | 408.20 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2026-01-30 09:15:00 | 418.90 | 2026-02-01 10:15:00 | 412.05 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-02-01 15:00:00 | 427.45 | 2026-02-10 09:15:00 | 470.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-03 09:15:00 | 437.75 | 2026-02-13 09:15:00 | 379.35 | STOP_HIT | 1.00 | -13.34% |
| SELL | retest2 | 2026-04-08 14:00:00 | 373.15 | 2026-04-16 09:15:00 | 390.40 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2026-04-08 14:45:00 | 371.85 | 2026-04-16 09:15:00 | 390.40 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2026-04-09 09:30:00 | 368.95 | 2026-04-16 09:15:00 | 390.40 | STOP_HIT | 1.00 | -5.81% |
| SELL | retest2 | 2026-04-10 09:30:00 | 370.65 | 2026-04-16 09:15:00 | 390.40 | STOP_HIT | 1.00 | -5.33% |
