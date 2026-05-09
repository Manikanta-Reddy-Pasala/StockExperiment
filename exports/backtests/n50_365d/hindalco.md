# HINDALCO (HINDALCO)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1044.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 0 |
| TARGET_HIT | 9 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 1
- **Target hits / Stop hits / Partials:** 9 / 1 / 0
- **Avg / median % per leg:** 8.73% / 10.00%
- **Sum % (uncompounded):** 87.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 9 | 100.0% | 9 | 0 | 0 | 10.00% | 90.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 9 | 100.0% | 9 | 0 | 0 | 10.00% | 90.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.71% | -2.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.71% | -2.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 9 | 90.0% | 9 | 1 | 0 | 8.73% | 87.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 657.50 | 636.11 | 636.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 12:15:00 | 659.50 | 637.15 | 636.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 637.20 | 645.55 | 641.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 637.55 | 645.47 | 641.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 636.70 | 645.47 | 641.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 637.05 | 643.08 | 640.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 636.60 | 643.08 | 640.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 635.70 | 642.58 | 640.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 635.70 | 642.58 | 640.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 642.95 | 642.45 | 640.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 12:30:00 | 645.15 | 642.49 | 640.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 13:00:00 | 645.40 | 642.49 | 640.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:45:00 | 645.30 | 645.47 | 642.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 645.05 | 645.47 | 642.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 638.65 | 645.57 | 642.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:00:00 | 638.65 | 645.57 | 642.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 640.40 | 645.52 | 642.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 14:15:00 | 641.50 | 645.52 | 642.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 15:00:00 | 642.20 | 645.49 | 642.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 11:00:00 | 642.60 | 645.37 | 642.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 14:15:00 | 641.50 | 645.24 | 642.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 639.50 | 645.15 | 642.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 640.60 | 645.15 | 642.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 644.20 | 645.38 | 643.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:30:00 | 650.95 | 645.49 | 643.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-02 11:15:00 | 705.65 | 660.85 | 652.07 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-07-02 11:15:00 | 705.65 | 660.85 | 652.07 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-07-03 09:15:00 | 706.42 | 662.76 | 653.26 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-07-03 09:15:00 | 706.86 | 662.76 | 653.26 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-18 09:15:00 | 709.67 | 681.57 | 673.37 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-18 09:15:00 | 709.94 | 681.57 | 673.37 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-18 09:15:00 | 709.83 | 681.57 | 673.37 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-18 09:15:00 | 709.56 | 681.57 | 673.37 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-18 13:15:00 | 716.05 | 682.79 | 674.14 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-13 13:45:00 | 632.85 | 2025-05-14 09:15:00 | 650.00 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-06-06 12:30:00 | 645.15 | 2025-07-02 11:15:00 | 705.65 | TARGET_HIT | 1.00 | 9.38% |
| BUY | retest2 | 2025-06-06 13:00:00 | 645.40 | 2025-07-02 11:15:00 | 705.65 | TARGET_HIT | 1.00 | 9.34% |
| BUY | retest2 | 2025-06-16 10:45:00 | 645.30 | 2025-07-03 09:15:00 | 706.42 | TARGET_HIT | 1.00 | 9.47% |
| BUY | retest2 | 2025-06-16 11:15:00 | 645.05 | 2025-07-03 09:15:00 | 706.86 | TARGET_HIT | 1.00 | 9.58% |
| BUY | retest2 | 2025-06-17 14:15:00 | 641.50 | 2025-08-18 09:15:00 | 709.67 | TARGET_HIT | 1.00 | 10.63% |
| BUY | retest2 | 2025-06-17 15:00:00 | 642.20 | 2025-08-18 09:15:00 | 709.94 | TARGET_HIT | 1.00 | 10.55% |
| BUY | retest2 | 2025-06-19 11:00:00 | 642.60 | 2025-08-18 09:15:00 | 709.83 | TARGET_HIT | 1.00 | 10.46% |
| BUY | retest2 | 2025-06-19 14:15:00 | 641.50 | 2025-08-18 09:15:00 | 709.56 | TARGET_HIT | 1.00 | 10.61% |
| BUY | retest2 | 2025-06-23 11:30:00 | 650.95 | 2025-08-18 13:15:00 | 716.05 | TARGET_HIT | 1.00 | 10.00% |
