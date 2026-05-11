# Hindalco Industries Ltd. (HINDALCO)

## Backtest Summary

- **Window:** 2024-10-14 09:15:00 → 2026-05-08 15:15:00 (2706 bars)
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
| ALERT3 | 8 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 9 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 0
- **Target hits / Stop hits / Partials:** 9 / 0 / 0
- **Avg / median % per leg:** 10.00% / 10.00%
- **Sum % (uncompounded):** 90.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 9 | 100.0% | 9 | 0 | 0 | 10.00% | 90.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 9 | 100.0% | 9 | 0 | 0 | 10.00% | 90.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 9 | 100.0% | 9 | 0 | 0 | 10.00% | 90.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 659.50 | 637.15 | 637.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 15:15:00 | 661.45 | 637.80 | 637.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 637.20 | 645.55 | 642.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 637.20 | 645.55 | 642.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 637.20 | 645.55 | 642.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 637.20 | 645.55 | 642.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 637.55 | 645.47 | 642.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 636.70 | 645.47 | 642.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 635.70 | 642.58 | 640.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 635.70 | 642.58 | 640.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 642.95 | 642.45 | 640.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 12:30:00 | 645.15 | 642.49 | 640.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 13:00:00 | 645.40 | 642.49 | 640.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:45:00 | 645.30 | 645.47 | 642.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 645.05 | 645.47 | 642.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 638.65 | 645.57 | 643.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:00:00 | 638.65 | 645.57 | 643.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 640.40 | 645.52 | 643.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 14:15:00 | 641.50 | 645.52 | 643.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 15:00:00 | 642.20 | 645.49 | 643.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 11:00:00 | 642.60 | 645.37 | 643.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 14:15:00 | 641.50 | 645.24 | 643.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 639.50 | 645.15 | 643.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 640.60 | 645.15 | 643.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 644.20 | 645.38 | 643.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:30:00 | 650.95 | 645.49 | 643.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-02 11:15:00 | 705.65 | 660.85 | 652.24 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-06 12:30:00 | 645.15 | 2025-07-02 11:15:00 | 705.65 | TARGET_HIT | 1.00 | 9.38% |
| BUY | retest2 | 2025-06-06 13:00:00 | 645.40 | 2025-07-02 11:15:00 | 705.65 | TARGET_HIT | 1.00 | 9.34% |
| BUY | retest2 | 2025-06-16 10:45:00 | 645.30 | 2025-07-03 09:15:00 | 706.42 | TARGET_HIT | 1.00 | 9.47% |
| BUY | retest2 | 2025-06-16 11:15:00 | 645.05 | 2025-07-03 09:15:00 | 706.86 | TARGET_HIT | 1.00 | 9.58% |
| BUY | retest2 | 2025-06-17 14:15:00 | 641.50 | 2025-08-18 09:15:00 | 709.67 | TARGET_HIT | 1.00 | 10.63% |
| BUY | retest2 | 2025-06-17 15:00:00 | 642.20 | 2025-08-18 09:15:00 | 709.94 | TARGET_HIT | 1.00 | 10.55% |
| BUY | retest2 | 2025-06-19 11:00:00 | 642.60 | 2025-08-18 09:15:00 | 709.83 | TARGET_HIT | 1.00 | 10.46% |
| BUY | retest2 | 2025-06-19 14:15:00 | 641.50 | 2025-08-18 09:15:00 | 709.56 | TARGET_HIT | 1.00 | 10.61% |
| BUY | retest2 | 2025-06-23 11:30:00 | 650.95 | 2025-08-18 13:15:00 | 716.05 | TARGET_HIT | 1.00 | 10.00% |
