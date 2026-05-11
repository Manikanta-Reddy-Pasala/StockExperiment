# Amara Raja Energy & Mobility Ltd. (ARE&M)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 890.85
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
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 22
- **Target hits / Stop hits / Partials:** 1 / 26 / 5
- **Avg / median % per leg:** -0.25% / -2.02%
- **Sum % (uncompounded):** -8.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.99% | -8.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.99% | -8.0% |
| SELL (all) | 28 | 10 | 35.7% | 1 | 22 | 5 | -0.00% | -0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 10 | 35.7% | 1 | 22 | 5 | -0.00% | -0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 10 | 31.2% | 1 | 26 | 5 | -0.25% | -8.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 15:15:00 | 974.70 | 1004.64 | 1004.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 972.40 | 1003.22 | 1004.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 987.80 | 983.18 | 991.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:45:00 | 988.00 | 983.18 | 991.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 988.65 | 983.24 | 991.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 12:00:00 | 985.95 | 983.27 | 991.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:30:00 | 986.15 | 983.19 | 991.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:45:00 | 984.40 | 982.95 | 990.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 11:15:00 | 986.20 | 983.12 | 990.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 990.65 | 983.24 | 990.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:30:00 | 988.50 | 983.24 | 990.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 988.70 | 983.30 | 990.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 14:45:00 | 983.45 | 983.32 | 990.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 15:00:00 | 987.60 | 983.27 | 990.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 985.90 | 983.35 | 990.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:00:00 | 986.30 | 983.38 | 990.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 978.00 | 983.36 | 990.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 974.10 | 983.36 | 990.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 13:00:00 | 977.45 | 983.21 | 990.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 15:00:00 | 977.00 | 983.08 | 989.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 976.65 | 983.02 | 989.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 990.10 | 983.12 | 989.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 990.10 | 983.12 | 989.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1008.00 | 983.37 | 989.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-21 11:15:00 | 1008.00 | 983.37 | 989.92 | SL hit (close>static) qty=1.00 sl=996.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 11:15:00 | 1016.70 | 983.22 | 983.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 1021.50 | 984.58 | 983.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 12:15:00 | 1010.00 | 1010.55 | 999.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:30:00 | 1010.00 | 1010.55 | 999.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1008.00 | 1010.38 | 999.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 12:15:00 | 1014.95 | 1010.34 | 999.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 996.00 | 1010.30 | 1000.20 | SL hit (close<static) qty=1.00 sl=997.95 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 958.00 | 996.91 | 997.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 956.90 | 986.85 | 991.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 937.60 | 933.12 | 950.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:45:00 | 936.40 | 933.12 | 950.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 897.50 | 873.02 | 899.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:00:00 | 897.50 | 873.02 | 899.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 907.00 | 873.59 | 899.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 873.00 | 877.56 | 900.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 829.35 | 862.60 | 883.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 785.70 | 849.23 | 873.49 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 887.90 | 821.89 | 821.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 890.25 | 826.48 | 823.94 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-07-10 12:00:00 | 985.95 | 2025-07-21 11:15:00 | 1008.00 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-07-11 09:30:00 | 986.15 | 2025-07-21 11:15:00 | 1008.00 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-07-14 13:45:00 | 984.40 | 2025-07-21 11:15:00 | 1008.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-07-15 11:15:00 | 986.20 | 2025-07-21 11:15:00 | 1008.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-07-15 14:45:00 | 983.45 | 2025-07-21 11:15:00 | 1008.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-07-16 15:00:00 | 987.60 | 2025-07-21 11:15:00 | 1008.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-07-17 09:30:00 | 985.90 | 2025-07-21 11:15:00 | 1008.00 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-07-17 11:00:00 | 986.30 | 2025-07-21 11:15:00 | 1008.00 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-07-18 10:15:00 | 974.10 | 2025-07-21 11:15:00 | 1008.00 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2025-07-18 13:00:00 | 977.45 | 2025-07-21 11:15:00 | 1008.00 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-07-18 15:00:00 | 977.00 | 2025-07-21 11:15:00 | 1008.00 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2025-07-21 09:15:00 | 976.65 | 2025-07-21 11:15:00 | 1008.00 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-07-25 14:00:00 | 993.45 | 2025-07-25 15:15:00 | 998.95 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-07-25 14:45:00 | 993.30 | 2025-07-25 15:15:00 | 998.95 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-28 09:45:00 | 991.70 | 2025-08-06 09:15:00 | 942.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-29 15:00:00 | 993.90 | 2025-08-06 09:15:00 | 944.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 10:15:00 | 985.80 | 2025-08-06 11:15:00 | 936.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 11:30:00 | 985.25 | 2025-08-06 11:15:00 | 935.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 09:45:00 | 991.70 | 2025-08-19 09:15:00 | 978.20 | STOP_HIT | 0.50 | 1.36% |
| SELL | retest2 | 2025-07-29 15:00:00 | 993.90 | 2025-08-19 09:15:00 | 978.20 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2025-07-30 10:15:00 | 985.80 | 2025-08-19 09:15:00 | 978.20 | STOP_HIT | 0.50 | 0.77% |
| SELL | retest2 | 2025-07-30 11:30:00 | 985.25 | 2025-08-19 09:15:00 | 978.20 | STOP_HIT | 0.50 | 0.72% |
| SELL | retest2 | 2025-08-19 12:30:00 | 983.00 | 2025-08-29 13:15:00 | 990.45 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-08-19 13:15:00 | 983.55 | 2025-08-29 13:15:00 | 990.45 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-08-26 09:15:00 | 980.00 | 2025-09-01 09:15:00 | 1010.95 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-08-26 13:45:00 | 984.60 | 2025-09-01 09:15:00 | 1010.95 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-09-23 12:15:00 | 1014.95 | 2025-09-24 10:15:00 | 996.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-10-30 09:15:00 | 1015.50 | 2025-11-04 12:15:00 | 995.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-10-30 14:45:00 | 1015.80 | 2025-11-04 12:15:00 | 995.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-10-31 14:00:00 | 1015.60 | 2025-11-04 12:15:00 | 995.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-02-12 09:15:00 | 873.00 | 2026-03-02 09:15:00 | 829.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 873.00 | 2026-03-09 09:15:00 | 785.70 | TARGET_HIT | 0.50 | 10.00% |
