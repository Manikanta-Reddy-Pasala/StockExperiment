# Newgen Software Technologies Ltd. (NEWGEN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 506.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 0 / 6 / 1
- **Avg / median % per leg:** -0.56% / -2.05%
- **Sum % (uncompounded):** -3.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.66% | -13.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.66% | -13.3% |
| SELL (all) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.68% | 9.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.68% | 9.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 2 | 28.6% | 0 | 6 | 1 | -0.56% | -3.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 1166.50 | 1053.74 | 1053.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 1218.00 | 1060.17 | 1056.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 1173.80 | 1182.30 | 1139.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 1173.80 | 1182.30 | 1139.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1145.90 | 1181.30 | 1140.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 1139.30 | 1181.30 | 1140.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1152.70 | 1187.51 | 1150.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 1152.70 | 1187.51 | 1150.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1154.70 | 1187.19 | 1150.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 1160.10 | 1187.19 | 1150.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1153.40 | 1186.85 | 1150.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 1189.50 | 1178.72 | 1149.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:30:00 | 1165.80 | 1178.91 | 1151.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 10:15:00 | 1138.80 | 1176.04 | 1152.24 | SL hit (close<static) qty=1.00 sl=1140.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 1027.90 | 1137.35 | 1137.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 972.70 | 1132.42 | 1135.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 933.45 | 923.06 | 985.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 10:45:00 | 933.00 | 923.06 | 985.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 932.65 | 900.51 | 943.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 936.00 | 900.51 | 943.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 917.50 | 900.06 | 936.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 917.00 | 900.06 | 936.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 912.80 | 885.74 | 911.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 912.80 | 885.74 | 911.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 910.80 | 885.99 | 911.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 910.80 | 885.99 | 911.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 910.40 | 886.23 | 911.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 910.40 | 886.23 | 911.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 909.00 | 886.46 | 911.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 911.25 | 886.46 | 911.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 909.40 | 886.69 | 911.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:45:00 | 908.45 | 886.69 | 911.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 911.45 | 887.94 | 909.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 910.80 | 887.94 | 909.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 910.45 | 888.16 | 909.74 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 959.30 | 925.94 | 925.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 971.00 | 926.71 | 926.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 09:15:00 | 929.90 | 933.30 | 929.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 929.90 | 933.30 | 929.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 929.90 | 933.30 | 929.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:45:00 | 927.55 | 933.30 | 929.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 923.90 | 933.21 | 929.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 924.30 | 933.21 | 929.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 924.70 | 933.13 | 929.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 922.20 | 933.13 | 929.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 920.50 | 932.34 | 929.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:30:00 | 917.25 | 932.34 | 929.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 927.35 | 931.78 | 929.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:30:00 | 928.35 | 931.78 | 929.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 930.00 | 931.76 | 929.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 930.00 | 931.76 | 929.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 929.00 | 931.73 | 929.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 929.00 | 931.73 | 929.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 929.15 | 931.71 | 929.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:30:00 | 928.00 | 931.71 | 929.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 918.35 | 931.57 | 929.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 918.35 | 931.57 | 929.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 916.00 | 931.42 | 929.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 09:45:00 | 920.70 | 931.31 | 929.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 14:30:00 | 920.60 | 930.71 | 928.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 901.80 | 930.28 | 928.72 | SL hit (close<static) qty=1.00 sl=915.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 891.55 | 926.99 | 927.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 09:15:00 | 877.40 | 925.73 | 926.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 10:15:00 | 610.75 | 605.51 | 692.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-19 11:00:00 | 610.75 | 605.51 | 692.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 515.55 | 473.68 | 512.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:45:00 | 524.60 | 473.68 | 512.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 512.35 | 474.06 | 512.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:15:00 | 509.35 | 474.06 | 512.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:45:00 | 505.80 | 474.34 | 512.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 12:15:00 | 483.88 | 476.63 | 511.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 487.15 | 476.63 | 511.75 | SL hit (close>static) qty=0.50 sl=476.63 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-26 09:15:00 | 1189.50 | 2025-07-02 10:15:00 | 1138.80 | STOP_HIT | 1.00 | -4.26% |
| BUY | retest2 | 2025-06-30 09:30:00 | 1165.80 | 2025-07-02 10:15:00 | 1138.80 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-11-20 09:45:00 | 920.70 | 2025-11-21 09:15:00 | 901.80 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-11-20 14:30:00 | 920.60 | 2025-11-21 09:15:00 | 901.80 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-11-24 09:15:00 | 923.80 | 2025-11-24 09:15:00 | 899.55 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2026-04-30 14:15:00 | 509.35 | 2026-05-05 12:15:00 | 483.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-30 14:15:00 | 509.35 | 2026-05-05 12:15:00 | 487.15 | STOP_HIT | 0.50 | 4.36% |
