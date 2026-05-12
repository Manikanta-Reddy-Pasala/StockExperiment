# Jubilant Foodworks Ltd. (JUBLFOOD)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 473.00
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
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 13
- **Target hits / Stop hits / Partials:** 0 / 16 / 3
- **Avg / median % per leg:** 0.00% / -1.06%
- **Sum % (uncompounded):** 0.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 0 | 0.0% | 0 | 13 | 0 | -1.40% | -18.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 0 | 0.0% | 0 | 13 | 0 | -1.40% | -18.2% |
| SELL (all) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.05% | 18.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.05% | 18.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 6 | 31.6% | 0 | 16 | 3 | 0.00% | 0.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 14:15:00 | 660.20 | 677.25 | 677.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 656.05 | 676.52 | 676.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 12:15:00 | 678.00 | 675.51 | 676.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 12:15:00 | 678.00 | 675.51 | 676.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 678.00 | 675.51 | 676.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 678.00 | 675.51 | 676.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 678.90 | 675.55 | 676.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:45:00 | 679.80 | 675.55 | 676.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 686.45 | 675.65 | 676.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 686.45 | 675.65 | 676.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 695.75 | 677.42 | 677.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 12:15:00 | 698.45 | 678.51 | 677.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 677.45 | 680.71 | 679.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 677.45 | 680.71 | 679.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 677.45 | 680.71 | 679.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 677.45 | 680.71 | 679.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 671.80 | 680.62 | 679.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 671.80 | 680.62 | 679.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 677.45 | 680.04 | 678.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:00:00 | 677.45 | 680.04 | 678.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 681.25 | 680.05 | 678.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 685.70 | 680.12 | 678.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:00:00 | 685.50 | 680.17 | 678.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:30:00 | 685.65 | 680.80 | 679.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 10:15:00 | 687.10 | 680.80 | 679.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 677.75 | 680.83 | 679.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:30:00 | 679.00 | 680.83 | 679.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 677.00 | 680.80 | 679.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 682.40 | 680.80 | 679.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 681.50 | 680.83 | 679.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 682.80 | 680.83 | 679.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 679.90 | 680.82 | 679.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:45:00 | 679.10 | 680.82 | 679.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 680.60 | 680.82 | 679.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 679.60 | 680.82 | 679.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 688.35 | 694.08 | 687.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:45:00 | 687.75 | 694.08 | 687.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 682.90 | 693.97 | 687.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 682.90 | 693.97 | 687.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 685.90 | 693.89 | 687.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 688.20 | 693.47 | 687.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 15:00:00 | 689.15 | 693.26 | 687.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 688.65 | 693.20 | 687.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 680.75 | 692.49 | 687.52 | SL hit (close<static) qty=1.00 sl=682.15 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 656.45 | 684.49 | 684.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 15:15:00 | 655.50 | 683.67 | 684.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 649.00 | 647.44 | 660.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 12:00:00 | 649.00 | 647.44 | 660.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 654.55 | 643.42 | 655.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:15:00 | 639.00 | 648.24 | 655.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 639.00 | 648.15 | 655.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:00:00 | 639.40 | 648.06 | 655.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 607.05 | 636.41 | 646.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 607.05 | 636.41 | 646.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 607.43 | 636.41 | 646.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 632.15 | 631.69 | 642.57 | SL hit (close>ema200) qty=0.50 sl=631.69 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 15:15:00 | 684.90 | 2025-05-16 09:15:00 | 673.15 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-05-19 10:45:00 | 685.20 | 2025-05-20 13:15:00 | 675.60 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-05-19 11:45:00 | 685.45 | 2025-05-20 13:15:00 | 675.60 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-06-16 10:15:00 | 685.70 | 2025-07-10 09:15:00 | 680.75 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-06-16 11:00:00 | 685.50 | 2025-07-10 09:15:00 | 680.75 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-06-17 09:30:00 | 685.65 | 2025-07-10 09:15:00 | 680.75 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-06-17 10:15:00 | 687.10 | 2025-07-11 09:15:00 | 679.80 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-08 11:00:00 | 688.20 | 2025-07-11 15:15:00 | 675.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-07-08 15:00:00 | 689.15 | 2025-07-11 15:15:00 | 675.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-07-09 09:15:00 | 688.65 | 2025-07-11 15:15:00 | 675.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-07-10 14:15:00 | 688.10 | 2025-07-11 15:15:00 | 675.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-07-16 09:15:00 | 692.10 | 2025-07-18 09:15:00 | 683.25 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-16 12:45:00 | 692.50 | 2025-07-18 09:15:00 | 683.25 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-09-12 10:15:00 | 639.00 | 2025-09-26 14:15:00 | 607.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-12 10:45:00 | 639.00 | 2025-09-26 14:15:00 | 607.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-12 12:00:00 | 639.40 | 2025-09-26 14:15:00 | 607.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-12 10:15:00 | 639.00 | 2025-10-06 09:15:00 | 632.15 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2025-09-12 10:45:00 | 639.00 | 2025-10-06 09:15:00 | 632.15 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2025-09-12 12:00:00 | 639.40 | 2025-10-06 09:15:00 | 632.15 | STOP_HIT | 0.50 | 1.13% |
