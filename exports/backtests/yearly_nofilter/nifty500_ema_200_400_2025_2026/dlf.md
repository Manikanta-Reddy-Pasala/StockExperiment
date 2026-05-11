# DLF Ltd. (DLF)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 606.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 12
- **Target hits / Stop hits / Partials:** 3 / 14 / 5
- **Avg / median % per leg:** 1.90% / -0.21%
- **Sum % (uncompounded):** 41.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 22 | 10 | 45.5% | 3 | 14 | 5 | 1.90% | 41.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 10 | 45.5% | 3 | 14 | 5 | 1.90% | 41.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 10 | 45.5% | 3 | 14 | 5 | 1.90% | 41.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 771.45 | 693.21 | 692.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 780.75 | 703.14 | 698.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 824.10 | 824.36 | 790.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 10:00:00 | 824.10 | 824.36 | 790.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 805.35 | 831.50 | 808.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 801.85 | 831.50 | 808.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 800.80 | 831.20 | 808.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 800.80 | 831.20 | 808.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 796.50 | 830.85 | 808.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:00:00 | 796.50 | 830.85 | 808.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 754.00 | 794.89 | 795.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 752.00 | 794.46 | 794.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 11:15:00 | 770.65 | 767.23 | 776.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 12:00:00 | 770.65 | 767.23 | 776.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 777.30 | 767.33 | 776.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 777.30 | 767.33 | 776.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 774.15 | 767.40 | 776.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 779.55 | 767.40 | 776.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 775.80 | 767.48 | 776.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 774.80 | 767.48 | 776.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 776.10 | 767.57 | 776.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 775.95 | 767.57 | 776.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 775.45 | 767.65 | 776.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 10:45:00 | 773.80 | 767.74 | 776.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 13:15:00 | 780.80 | 768.03 | 776.73 | SL hit (close>static) qty=1.00 sl=780.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-20 10:45:00 | 769.80 | 2025-05-22 10:15:00 | 771.45 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-05-21 12:00:00 | 768.75 | 2025-05-22 10:15:00 | 771.45 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-09-16 10:45:00 | 773.80 | 2025-09-16 13:15:00 | 780.80 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-09-22 14:45:00 | 771.75 | 2025-09-24 14:15:00 | 733.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 09:15:00 | 766.05 | 2025-09-25 10:15:00 | 727.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 14:45:00 | 771.75 | 2025-10-15 09:15:00 | 757.05 | STOP_HIT | 0.50 | 1.90% |
| SELL | retest2 | 2025-09-23 09:15:00 | 766.05 | 2025-10-15 09:15:00 | 757.05 | STOP_HIT | 0.50 | 1.17% |
| SELL | retest2 | 2025-10-20 15:15:00 | 773.50 | 2025-10-23 11:15:00 | 780.25 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-10-31 11:15:00 | 762.45 | 2025-11-03 10:15:00 | 777.80 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-10-31 12:00:00 | 761.55 | 2025-11-03 10:15:00 | 777.80 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-11-06 10:00:00 | 761.80 | 2025-11-13 10:15:00 | 769.75 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-11-06 11:15:00 | 762.95 | 2025-11-13 10:15:00 | 769.75 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-11-07 09:15:00 | 751.00 | 2025-11-13 10:15:00 | 769.75 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-11-07 12:45:00 | 756.50 | 2025-11-13 10:15:00 | 769.75 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-11-07 13:30:00 | 756.50 | 2025-11-17 09:15:00 | 770.60 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-11-11 09:45:00 | 755.85 | 2025-11-17 09:15:00 | 770.60 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-11-14 09:30:00 | 762.60 | 2025-11-21 14:15:00 | 724.80 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2025-11-14 15:15:00 | 763.00 | 2025-11-21 15:15:00 | 723.71 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2025-11-18 09:15:00 | 762.45 | 2025-11-21 15:15:00 | 724.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 09:30:00 | 762.60 | 2025-12-08 13:15:00 | 686.66 | TARGET_HIT | 0.50 | 9.96% |
| SELL | retest2 | 2025-11-14 15:15:00 | 763.00 | 2025-12-08 14:15:00 | 685.62 | TARGET_HIT | 0.50 | 10.14% |
| SELL | retest2 | 2025-11-18 09:15:00 | 762.45 | 2025-12-08 14:15:00 | 686.21 | TARGET_HIT | 0.50 | 10.00% |
