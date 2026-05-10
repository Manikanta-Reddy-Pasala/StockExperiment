# HDFC Bank Ltd. (HDFCBANK)

## Backtest Summary

- **Window:** 2025-01-15 09:15:00 → 2026-05-08 15:15:00 (2263 bars)
- **Last close:** 781.25
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
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 26 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 22
- **Target hits / Stop hits / Partials:** 2 / 24 / 2
- **Avg / median % per leg:** 0.21% / -0.38%
- **Sum % (uncompounded):** 5.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 2 | 8.7% | 0 | 23 | 0 | -1.01% | -23.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 2 | 8.7% | 0 | 23 | 0 | -1.01% | -23.2% |
| SELL (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 5.79% | 28.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 4 | 80.0% | 2 | 1 | 2 | 5.79% | 28.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 6 | 21.4% | 2 | 24 | 2 | 0.21% | 5.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 962.60 | 979.74 | 979.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 960.40 | 979.55 | 979.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 976.05 | 973.34 | 976.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 976.05 | 973.34 | 976.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 976.05 | 973.34 | 976.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:30:00 | 970.70 | 973.38 | 975.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 981.05 | 965.76 | 970.49 | SL hit (close>static) qty=1.00 sl=979.65 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 14:15:00 | 1002.55 | 973.74 | 973.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1008.40 | 974.35 | 974.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 14:15:00 | 984.05 | 986.98 | 981.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 15:00:00 | 984.05 | 986.98 | 981.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 979.85 | 986.89 | 981.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:30:00 | 976.80 | 986.89 | 981.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 981.75 | 986.84 | 981.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 11:15:00 | 983.00 | 986.84 | 981.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 976.50 | 986.75 | 981.81 | SL hit (close<static) qty=1.00 sl=979.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 12:45:00 | 982.50 | 986.55 | 981.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 13:15:00 | 983.65 | 986.55 | 981.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 14:15:00 | 983.35 | 986.49 | 981.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 982.35 | 986.45 | 981.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 982.35 | 986.45 | 981.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 986.35 | 986.43 | 981.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 10:15:00 | 990.05 | 986.43 | 981.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 13:15:00 | 988.25 | 986.51 | 981.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 979.95 | 986.35 | 981.96 | SL hit (close<static) qty=1.00 sl=980.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 979.95 | 986.35 | 981.96 | SL hit (close<static) qty=1.00 sl=980.20 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:15:00 | 987.95 | 986.33 | 981.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 14:00:00 | 988.20 | 986.35 | 982.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 980.05 | 986.82 | 982.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 980.05 | 986.82 | 982.63 | SL hit (close<static) qty=1.00 sl=980.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 980.05 | 986.82 | 982.63 | SL hit (close<static) qty=1.00 sl=980.20 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 990.45 | 986.75 | 982.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:00:00 | 986.90 | 987.68 | 983.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 11:45:00 | 986.65 | 987.65 | 983.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 13:15:00 | 989.70 | 995.89 | 990.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 989.40 | 995.75 | 990.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 15:15:00 | 991.70 | 995.75 | 990.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:00:00 | 994.40 | 995.70 | 990.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 987.20 | 995.93 | 991.34 | SL hit (close<static) qty=1.00 sl=987.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 987.20 | 995.93 | 991.34 | SL hit (close<static) qty=1.00 sl=987.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 992.60 | 993.78 | 990.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 987.30 | 993.66 | 990.60 | SL hit (close<static) qty=1.00 sl=987.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 991.40 | 993.33 | 990.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 991.30 | 993.66 | 990.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 991.30 | 993.66 | 990.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 990.20 | 993.63 | 990.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:45:00 | 990.40 | 993.63 | 990.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 990.10 | 993.59 | 990.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 990.10 | 993.59 | 990.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 989.40 | 993.53 | 990.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 989.40 | 993.53 | 990.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 988.30 | 993.47 | 990.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:00:00 | 988.30 | 993.47 | 990.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 990.50 | 993.40 | 990.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 14:15:00 | 991.90 | 993.40 | 990.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 985.00 | 993.21 | 990.86 | SL hit (close<static) qty=1.00 sl=987.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 985.00 | 993.21 | 990.86 | SL hit (close<static) qty=1.00 sl=987.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 15:15:00 | 995.00 | 993.03 | 990.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:15:00 | 991.60 | 992.97 | 990.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 15:15:00 | 994.00 | 993.01 | 990.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 994.00 | 993.02 | 990.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 995.65 | 993.02 | 990.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 10:30:00 | 994.10 | 993.03 | 990.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 11:15:00 | 990.00 | 993.00 | 990.91 | SL hit (close<static) qty=1.00 sl=990.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-01 11:15:00 | 990.00 | 993.00 | 990.91 | SL hit (close<static) qty=1.00 sl=990.40 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 994.60 | 992.95 | 990.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 990.15 | 993.33 | 991.20 | SL hit (close<static) qty=1.00 sl=990.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 983.85 | 993.19 | 991.15 | SL hit (close<static) qty=1.00 sl=987.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 983.85 | 993.19 | 991.15 | SL hit (close<static) qty=1.00 sl=987.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 983.85 | 993.19 | 991.15 | SL hit (close<static) qty=1.00 sl=987.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 978.15 | 992.93 | 991.04 | SL hit (close<static) qty=1.00 sl=979.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 978.15 | 992.93 | 991.04 | SL hit (close<static) qty=1.00 sl=979.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 978.15 | 992.93 | 991.04 | SL hit (close<static) qty=1.00 sl=979.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.34 | 990.77 | SL hit (close<static) qty=1.00 sl=976.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.34 | 990.77 | SL hit (close<static) qty=1.00 sl=976.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.34 | 990.77 | SL hit (close<static) qty=1.00 sl=976.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.34 | 990.77 | SL hit (close<static) qty=1.00 sl=976.05 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 948.60 | 989.05 | 989.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 945.70 | 987.46 | 988.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 944.90 | 947.58 | 960.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:00:00 | 944.15 | 947.54 | 960.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 15:15:00 | 897.65 | 929.26 | 944.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 15:15:00 | 896.94 | 929.26 | 944.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 850.41 | 922.93 | 940.13 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 849.74 | 922.93 | 940.13 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-19 09:30:00 | 970.70 | 2025-10-07 09:15:00 | 981.05 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-11-06 11:15:00 | 983.00 | 2025-11-07 09:15:00 | 976.50 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-11-07 12:45:00 | 982.50 | 2025-11-11 10:15:00 | 979.95 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-11-07 13:15:00 | 983.65 | 2025-11-11 10:15:00 | 979.95 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-11-07 14:15:00 | 983.35 | 2025-11-14 09:15:00 | 980.05 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-11-10 10:15:00 | 990.05 | 2025-11-14 09:15:00 | 980.05 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-11-10 13:15:00 | 988.25 | 2025-12-17 10:15:00 | 987.20 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-11-11 13:15:00 | 987.95 | 2025-12-17 10:15:00 | 987.20 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-11-11 14:00:00 | 988.20 | 2025-12-22 10:15:00 | 987.30 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-11-14 15:00:00 | 990.45 | 2025-12-30 11:15:00 | 985.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-11-19 10:00:00 | 986.90 | 2025-12-30 11:15:00 | 985.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-11-19 11:45:00 | 986.65 | 2026-01-01 11:15:00 | 990.00 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2025-12-10 13:15:00 | 989.70 | 2026-01-01 11:15:00 | 990.00 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-12-10 15:15:00 | 991.70 | 2026-01-05 09:15:00 | 990.15 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-12-11 10:00:00 | 994.40 | 2026-01-05 11:15:00 | 983.85 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-22 09:15:00 | 992.60 | 2026-01-05 11:15:00 | 983.85 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-12-23 09:45:00 | 991.40 | 2026-01-05 11:15:00 | 983.85 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-12-29 14:15:00 | 991.90 | 2026-01-05 13:15:00 | 978.15 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-12-30 15:15:00 | 995.00 | 2026-01-05 13:15:00 | 978.15 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-12-31 12:15:00 | 991.60 | 2026-01-05 13:15:00 | 978.15 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-12-31 15:15:00 | 994.00 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2026-01-01 09:15:00 | 995.65 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2026-01-01 10:30:00 | 994.10 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2026-01-02 09:15:00 | 994.60 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2026-02-06 09:15:00 | 944.90 | 2026-02-26 15:15:00 | 897.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 10:00:00 | 944.15 | 2026-02-26 15:15:00 | 896.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 09:15:00 | 944.90 | 2026-03-04 09:15:00 | 850.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-06 10:00:00 | 944.15 | 2026-03-04 09:15:00 | 849.74 | TARGET_HIT | 0.50 | 10.00% |
