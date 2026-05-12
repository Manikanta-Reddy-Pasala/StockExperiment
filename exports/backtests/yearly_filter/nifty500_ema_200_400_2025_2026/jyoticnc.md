# Jyoti CNC Automation Ltd. (JYOTICNC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-11 15:15:00 (3605 bars)
- **Last close:** 749.00
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
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 0
- **Avg / median % per leg:** -1.99% / -2.14%
- **Sum % (uncompounded):** -15.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.99% | -15.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.99% | -15.9% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.99% | -15.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 1052.50 | 1145.52 | 1145.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 1042.80 | 1143.58 | 1144.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 13:15:00 | 1076.60 | 1076.54 | 1103.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:45:00 | 1080.80 | 1076.54 | 1103.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 983.10 | 917.61 | 949.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 983.10 | 917.61 | 949.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1005.75 | 918.48 | 949.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 1005.75 | 918.48 | 949.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 929.45 | 906.49 | 930.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:45:00 | 931.30 | 906.49 | 930.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 942.80 | 906.86 | 930.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 942.80 | 906.86 | 930.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 941.75 | 907.20 | 930.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 943.15 | 907.20 | 930.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 924.95 | 906.56 | 927.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:15:00 | 949.55 | 906.56 | 927.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 954.25 | 907.04 | 927.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 951.25 | 907.04 | 927.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 967.20 | 907.63 | 927.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:45:00 | 963.25 | 907.63 | 927.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 952.00 | 916.47 | 930.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 952.00 | 916.47 | 930.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 943.00 | 916.73 | 930.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 969.60 | 916.73 | 930.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 1008.55 | 942.12 | 941.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 1037.85 | 943.85 | 942.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 964.20 | 969.90 | 957.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 10:00:00 | 964.20 | 969.90 | 957.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 962.00 | 969.75 | 957.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 967.00 | 969.45 | 957.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:30:00 | 971.60 | 969.46 | 957.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 14:15:00 | 946.30 | 968.29 | 958.03 | SL hit (close<static) qty=1.00 sl=956.20 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 916.15 | 957.24 | 957.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 910.50 | 956.77 | 957.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 10:15:00 | 871.65 | 866.65 | 899.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-10 11:00:00 | 871.65 | 866.65 | 899.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 811.10 | 776.42 | 811.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:00:00 | 811.10 | 776.42 | 811.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 13:15:00 | 806.85 | 776.72 | 811.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:45:00 | 806.05 | 776.72 | 811.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 15:15:00 | 810.00 | 777.38 | 811.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:15:00 | 825.60 | 777.38 | 811.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 815.15 | 777.75 | 811.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 699.60 | 783.16 | 811.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-12-03 15:15:00 | 967.00 | 2025-12-05 14:15:00 | 946.30 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-12-04 09:30:00 | 971.60 | 2025-12-05 14:15:00 | 946.30 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-12-10 09:15:00 | 974.80 | 2025-12-10 15:15:00 | 940.30 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2025-12-11 14:15:00 | 975.90 | 2025-12-12 15:15:00 | 954.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-12-22 14:30:00 | 962.30 | 2026-01-08 09:15:00 | 954.10 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-12-23 09:15:00 | 966.30 | 2026-01-08 09:15:00 | 954.10 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-12-24 09:15:00 | 977.30 | 2026-01-08 09:15:00 | 954.10 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-01-07 14:15:00 | 962.65 | 2026-01-08 09:15:00 | 954.10 | STOP_HIT | 1.00 | -0.89% |
