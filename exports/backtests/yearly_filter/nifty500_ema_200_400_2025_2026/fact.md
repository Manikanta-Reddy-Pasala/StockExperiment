# Fertilisers and Chemicals Travancore Ltd. (FACT)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 902.80
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
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 11 |
| TARGET_HIT | 4 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 12
- **Target hits / Stop hits / Partials:** 4 / 18 / 11
- **Avg / median % per leg:** 2.25% / 3.19%
- **Sum % (uncompounded):** 74.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 1 | 5 | 0 | 0.14% | 0.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 1 | 16.7% | 1 | 5 | 0 | 0.14% | 0.9% |
| SELL (all) | 27 | 20 | 74.1% | 3 | 13 | 11 | 2.72% | 73.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 20 | 74.1% | 3 | 13 | 11 | 2.72% | 73.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 33 | 21 | 63.6% | 4 | 18 | 11 | 2.25% | 74.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 830.00 | 745.50 | 745.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 834.00 | 747.97 | 746.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 14:15:00 | 967.40 | 971.67 | 906.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 967.40 | 971.67 | 906.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 916.70 | 957.71 | 918.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 916.70 | 957.71 | 918.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 917.85 | 957.31 | 918.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 946.55 | 952.37 | 918.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:00:00 | 923.95 | 952.23 | 934.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 925.40 | 950.32 | 934.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 911.70 | 949.94 | 934.13 | SL hit (close<static) qty=1.00 sl=915.45 alert=retest2 |

### Cycle 2 — SELL (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 14:15:00 | 898.00 | 955.29 | 955.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 15:15:00 | 897.55 | 954.71 | 955.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 14:15:00 | 903.00 | 902.78 | 918.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 903.00 | 902.78 | 918.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 913.90 | 902.91 | 918.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 896.40 | 904.36 | 916.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 14:15:00 | 851.58 | 891.98 | 906.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 13:15:00 | 806.76 | 879.16 | 898.00 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 870.00 | 807.96 | 807.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 931.50 | 809.19 | 808.50 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-15 09:15:00 | 946.55 | 2025-08-11 09:15:00 | 911.70 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest2 | 2025-08-07 14:00:00 | 923.95 | 2025-08-11 09:15:00 | 911.70 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-08-11 09:15:00 | 925.40 | 2025-08-11 09:15:00 | 911.70 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-08-12 09:15:00 | 930.90 | 2025-08-21 09:15:00 | 1023.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 15:15:00 | 954.00 | 2025-09-26 11:15:00 | 942.05 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-09-02 09:30:00 | 955.40 | 2025-09-26 11:15:00 | 942.05 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-11-21 09:15:00 | 896.40 | 2025-12-02 14:15:00 | 851.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 896.40 | 2025-12-08 13:15:00 | 806.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-16 13:30:00 | 899.15 | 2025-12-18 09:15:00 | 854.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-16 14:15:00 | 897.60 | 2025-12-18 09:15:00 | 852.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-16 15:00:00 | 897.60 | 2025-12-18 09:15:00 | 852.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-16 13:30:00 | 899.15 | 2025-12-19 15:15:00 | 869.00 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2025-12-16 14:15:00 | 897.60 | 2025-12-19 15:15:00 | 869.00 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2025-12-16 15:00:00 | 897.60 | 2025-12-19 15:15:00 | 869.00 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2025-12-24 14:15:00 | 876.80 | 2025-12-26 09:15:00 | 894.75 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-01-06 15:15:00 | 875.10 | 2026-01-08 09:15:00 | 890.75 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-01-09 09:15:00 | 873.55 | 2026-01-16 15:15:00 | 831.87 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2026-01-09 11:00:00 | 875.65 | 2026-01-19 09:15:00 | 829.87 | PARTIAL | 0.50 | 5.23% |
| SELL | retest2 | 2026-01-09 09:15:00 | 873.55 | 2026-01-20 15:15:00 | 788.09 | TARGET_HIT | 0.50 | 9.78% |
| SELL | retest2 | 2026-01-09 11:00:00 | 875.65 | 2026-01-21 09:15:00 | 786.19 | TARGET_HIT | 0.50 | 10.22% |
| SELL | retest2 | 2026-03-17 09:15:00 | 788.85 | 2026-03-17 10:15:00 | 826.20 | STOP_HIT | 1.00 | -4.73% |
| SELL | retest2 | 2026-03-19 13:15:00 | 797.35 | 2026-03-23 09:15:00 | 758.95 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2026-03-20 11:30:00 | 798.90 | 2026-03-23 10:15:00 | 757.48 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2026-03-20 12:15:00 | 798.00 | 2026-03-23 10:15:00 | 758.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 13:15:00 | 797.35 | 2026-03-24 12:15:00 | 789.20 | STOP_HIT | 0.50 | 1.02% |
| SELL | retest2 | 2026-03-20 11:30:00 | 798.90 | 2026-03-24 12:15:00 | 789.20 | STOP_HIT | 0.50 | 1.21% |
| SELL | retest2 | 2026-03-20 12:15:00 | 798.00 | 2026-03-24 12:15:00 | 789.20 | STOP_HIT | 0.50 | 1.10% |
| SELL | retest2 | 2026-03-27 10:30:00 | 781.75 | 2026-03-30 15:15:00 | 742.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 11:00:00 | 781.45 | 2026-03-30 15:15:00 | 742.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 10:30:00 | 781.75 | 2026-04-01 11:15:00 | 792.95 | STOP_HIT | 0.50 | -1.43% |
| SELL | retest2 | 2026-03-27 11:00:00 | 781.45 | 2026-04-01 11:15:00 | 792.95 | STOP_HIT | 0.50 | -1.47% |
| SELL | retest2 | 2026-03-30 09:15:00 | 774.55 | 2026-04-08 12:15:00 | 829.15 | STOP_HIT | 1.00 | -7.05% |
| SELL | retest2 | 2026-04-01 14:30:00 | 780.80 | 2026-04-08 12:15:00 | 829.15 | STOP_HIT | 1.00 | -6.19% |
