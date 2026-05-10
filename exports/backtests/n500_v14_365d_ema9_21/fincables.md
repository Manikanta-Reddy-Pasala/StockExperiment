# Finolex Cables Ltd. (FINCABLES)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1144.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 65 |
| ALERT1 | 41 |
| ALERT2 | 40 |
| ALERT2_SKIP | 20 |
| ALERT3 | 92 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 52 |
| PARTIAL | 12 |
| TARGET_HIT | 3 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 34
- **Target hits / Stop hits / Partials:** 3 / 51 / 12
- **Avg / median % per leg:** 1.35% / -0.03%
- **Sum % (uncompounded):** 89.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 6 | 27.3% | 3 | 18 | 1 | 0.51% | 11.2% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.21% | 9.6% |
| BUY @ 3rd Alert (retest2) | 19 | 3 | 15.8% | 3 | 16 | 0 | 0.08% | 1.6% |
| SELL (all) | 44 | 26 | 59.1% | 0 | 33 | 11 | 1.77% | 77.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 44 | 26 | 59.1% | 0 | 33 | 11 | 1.77% | 77.9% |
| retest1 (combined) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.21% | 9.6% |
| retest2 (combined) | 63 | 29 | 46.0% | 3 | 49 | 11 | 1.26% | 79.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 913.55 | 891.33 | 890.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 917.65 | 900.24 | 894.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 917.90 | 918.41 | 909.96 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:15:00 | 932.60 | 918.82 | 910.92 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 11:00:00 | 922.20 | 921.12 | 913.44 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 923.80 | 921.24 | 914.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 914.95 | 921.24 | 914.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-19 09:15:00 | 968.31 | 953.97 | 942.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 957.95 | 961.63 | 950.41 | SL hit (close<ema200) qty=0.50 sl=961.63 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 960.00 | 961.98 | 953.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 955.00 | 961.98 | 953.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 957.75 | 960.82 | 954.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 954.55 | 960.82 | 954.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 950.50 | 958.76 | 954.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-20 12:15:00 | 950.50 | 958.76 | 954.01 | SL hit (close<ema400) qty=1.00 sl=954.01 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-05-20 12:30:00 | 951.60 | 958.76 | 954.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 942.20 | 955.44 | 952.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 942.20 | 955.44 | 952.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 938.75 | 949.37 | 950.44 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 955.15 | 949.96 | 949.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 959.55 | 954.17 | 951.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 14:15:00 | 967.05 | 968.58 | 961.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:00:00 | 967.05 | 968.58 | 961.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 978.45 | 984.53 | 981.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 978.45 | 984.53 | 981.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 972.00 | 982.03 | 980.52 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 968.20 | 979.26 | 979.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 10:15:00 | 959.35 | 975.28 | 977.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 966.00 | 964.80 | 970.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 966.00 | 964.80 | 970.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 966.00 | 964.80 | 970.21 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 978.55 | 972.30 | 971.97 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 966.75 | 971.60 | 971.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 12:15:00 | 964.80 | 970.24 | 971.10 | Break + close below crossover candle low |

### Cycle 7 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 995.00 | 971.13 | 970.75 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 961.75 | 970.71 | 971.15 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 969.80 | 968.88 | 968.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 980.25 | 971.15 | 969.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 985.90 | 992.19 | 987.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 985.90 | 992.19 | 987.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 985.90 | 992.19 | 987.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 985.80 | 992.19 | 987.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 991.70 | 992.09 | 987.86 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 970.95 | 985.52 | 986.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 963.45 | 977.98 | 982.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 963.00 | 961.83 | 969.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 963.00 | 961.83 | 969.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 964.70 | 958.13 | 963.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 969.00 | 958.13 | 963.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 963.95 | 959.29 | 963.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 968.65 | 959.29 | 963.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 959.25 | 959.28 | 963.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:45:00 | 961.20 | 959.28 | 963.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 957.70 | 958.18 | 961.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:45:00 | 962.10 | 958.18 | 961.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 965.85 | 959.20 | 961.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 965.85 | 959.20 | 961.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 964.70 | 960.30 | 961.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:45:00 | 957.00 | 961.03 | 962.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:30:00 | 961.00 | 961.59 | 962.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 960.15 | 961.74 | 962.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 961.00 | 961.82 | 962.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 961.00 | 961.66 | 962.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 958.40 | 961.66 | 962.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 956.95 | 960.72 | 961.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 952.25 | 960.72 | 961.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 912.95 | 929.30 | 940.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 912.14 | 929.30 | 940.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 912.95 | 929.30 | 940.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 15:15:00 | 909.15 | 927.44 | 938.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 932.80 | 928.51 | 938.01 | SL hit (close>ema200) qty=0.50 sl=928.51 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 932.80 | 928.51 | 938.01 | SL hit (close>ema200) qty=0.50 sl=928.51 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 932.80 | 928.51 | 938.01 | SL hit (close>ema200) qty=0.50 sl=928.51 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 932.80 | 928.51 | 938.01 | SL hit (close>ema200) qty=0.50 sl=928.51 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 939.05 | 937.22 | 937.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 939.05 | 937.22 | 937.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 10:15:00 | 955.50 | 942.75 | 940.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 13:15:00 | 980.00 | 980.49 | 971.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 14:00:00 | 980.00 | 980.49 | 971.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 973.55 | 978.56 | 973.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 975.00 | 978.56 | 973.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 971.50 | 977.15 | 973.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:30:00 | 972.30 | 977.15 | 973.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 972.00 | 976.12 | 972.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 971.95 | 976.12 | 972.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 978.40 | 974.96 | 973.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:30:00 | 973.30 | 974.96 | 973.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 979.20 | 980.73 | 977.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 977.90 | 980.73 | 977.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 976.80 | 979.95 | 977.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:00:00 | 976.80 | 979.95 | 977.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 971.35 | 978.23 | 976.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 970.00 | 978.23 | 976.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 973.35 | 977.25 | 976.60 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 971.40 | 976.08 | 976.13 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 978.65 | 976.59 | 976.36 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 967.80 | 975.20 | 975.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 964.20 | 970.47 | 973.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 963.35 | 962.20 | 965.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 963.35 | 962.20 | 965.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 963.35 | 962.20 | 965.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 11:15:00 | 955.25 | 961.13 | 963.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 13:30:00 | 958.00 | 959.22 | 961.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:15:00 | 957.65 | 960.51 | 961.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 15:15:00 | 953.00 | 957.31 | 959.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 953.00 | 956.45 | 958.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 951.95 | 956.45 | 958.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:00:00 | 951.50 | 955.07 | 957.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:30:00 | 950.50 | 953.57 | 956.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 951.20 | 951.50 | 954.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 954.00 | 952.06 | 954.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 954.00 | 952.06 | 954.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 950.50 | 951.75 | 953.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 14:15:00 | 949.40 | 951.75 | 953.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 15:15:00 | 948.00 | 951.54 | 953.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:45:00 | 948.25 | 950.71 | 952.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 10:15:00 | 955.50 | 951.67 | 952.97 | SL hit (close>static) qty=1.00 sl=954.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 10:15:00 | 955.50 | 951.67 | 952.97 | SL hit (close>static) qty=1.00 sl=954.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 10:15:00 | 955.50 | 951.67 | 952.97 | SL hit (close>static) qty=1.00 sl=954.25 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:45:00 | 950.00 | 951.67 | 952.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 952.40 | 951.82 | 952.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-16 12:15:00 | 956.90 | 952.83 | 953.28 | SL hit (close>static) qty=1.00 sl=954.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 959.45 | 954.16 | 953.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 959.45 | 954.16 | 953.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 959.45 | 954.16 | 953.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 959.45 | 954.16 | 953.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 959.45 | 954.16 | 953.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 959.45 | 954.16 | 953.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 959.45 | 954.16 | 953.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 959.45 | 954.16 | 953.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 959.45 | 954.16 | 953.84 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 948.10 | 953.78 | 954.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 942.20 | 949.36 | 951.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 15:15:00 | 923.00 | 922.76 | 928.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 09:15:00 | 922.70 | 922.76 | 928.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 923.20 | 922.85 | 927.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:15:00 | 919.60 | 922.85 | 927.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 15:00:00 | 921.85 | 921.08 | 924.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 921.65 | 921.33 | 924.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 15:15:00 | 875.76 | 883.24 | 887.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 15:15:00 | 875.57 | 883.24 | 887.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 873.62 | 880.53 | 886.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 12:15:00 | 877.95 | 877.75 | 883.32 | SL hit (close>ema200) qty=0.50 sl=877.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 12:15:00 | 877.95 | 877.75 | 883.32 | SL hit (close>ema200) qty=0.50 sl=877.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 12:15:00 | 877.95 | 877.75 | 883.32 | SL hit (close>ema200) qty=0.50 sl=877.75 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 13:15:00 | 832.20 | 829.50 | 829.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 833.75 | 830.35 | 829.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 841.55 | 843.04 | 839.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:45:00 | 841.60 | 843.04 | 839.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 850.80 | 844.27 | 840.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 872.15 | 845.74 | 841.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 13:15:00 | 841.25 | 845.02 | 845.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 841.25 | 845.02 | 845.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 838.45 | 843.71 | 844.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 820.80 | 820.63 | 826.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:15:00 | 821.85 | 820.63 | 826.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 825.10 | 821.95 | 826.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 825.90 | 821.95 | 826.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 830.65 | 823.30 | 825.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 833.65 | 823.30 | 825.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 835.15 | 825.67 | 826.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:30:00 | 835.55 | 825.67 | 826.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 838.95 | 828.33 | 827.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 840.50 | 830.76 | 828.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 855.00 | 858.22 | 850.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 855.00 | 858.22 | 850.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 851.50 | 856.39 | 851.77 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 841.35 | 849.77 | 850.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 12:15:00 | 838.55 | 843.76 | 846.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 840.15 | 839.18 | 842.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 840.15 | 839.18 | 842.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 840.15 | 839.18 | 842.77 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 844.50 | 842.61 | 842.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 10:15:00 | 846.75 | 844.27 | 843.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 843.10 | 844.18 | 843.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 12:15:00 | 843.10 | 844.18 | 843.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 843.10 | 844.18 | 843.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 843.10 | 844.18 | 843.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 841.00 | 843.54 | 843.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:45:00 | 840.70 | 843.54 | 843.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 844.50 | 843.79 | 843.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 848.20 | 843.79 | 843.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 850.00 | 845.03 | 844.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 854.95 | 846.27 | 845.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:30:00 | 854.00 | 850.96 | 848.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 845.00 | 849.47 | 849.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 845.00 | 849.47 | 849.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 845.00 | 849.47 | 849.63 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 850.85 | 849.65 | 849.56 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 847.60 | 849.24 | 849.38 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 851.95 | 849.78 | 849.62 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 09:15:00 | 848.65 | 849.43 | 849.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 843.05 | 847.45 | 848.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 840.70 | 837.39 | 841.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 840.70 | 837.39 | 841.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 840.70 | 837.39 | 841.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:30:00 | 842.85 | 837.39 | 841.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 838.90 | 837.69 | 841.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 835.75 | 837.60 | 840.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 822.00 | 815.81 | 814.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 822.00 | 815.81 | 814.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 11:15:00 | 823.85 | 819.86 | 817.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 825.00 | 825.04 | 821.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:45:00 | 824.00 | 825.04 | 821.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 821.00 | 824.23 | 821.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 821.00 | 824.23 | 821.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 823.20 | 824.02 | 821.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:30:00 | 824.65 | 824.34 | 822.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 11:15:00 | 810.00 | 822.63 | 822.47 | SL hit (close<static) qty=1.00 sl=820.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 12:15:00 | 817.05 | 821.51 | 821.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 11:15:00 | 808.90 | 813.60 | 816.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 15:15:00 | 801.00 | 799.15 | 803.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 09:15:00 | 802.50 | 799.15 | 803.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 798.00 | 798.92 | 802.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:15:00 | 795.70 | 798.92 | 802.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:00:00 | 793.40 | 792.34 | 793.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 798.50 | 788.81 | 788.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 798.50 | 788.81 | 788.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 798.50 | 788.81 | 788.64 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 788.00 | 790.21 | 790.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 786.50 | 789.46 | 789.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 794.10 | 789.04 | 789.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 794.10 | 789.04 | 789.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 794.10 | 789.04 | 789.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:45:00 | 795.50 | 789.04 | 789.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 796.00 | 790.43 | 790.02 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 786.25 | 790.73 | 790.99 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 795.40 | 790.87 | 790.44 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 786.50 | 790.57 | 790.83 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 804.10 | 791.09 | 790.47 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 786.30 | 791.67 | 792.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 774.35 | 785.46 | 788.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 792.10 | 773.51 | 775.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 792.10 | 773.51 | 775.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 792.10 | 773.51 | 775.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 794.10 | 773.51 | 775.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 795.20 | 777.85 | 777.68 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 11:15:00 | 765.85 | 780.39 | 781.17 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 789.00 | 781.16 | 781.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 10:15:00 | 795.55 | 784.89 | 782.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 789.95 | 791.53 | 788.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 789.95 | 791.53 | 788.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 789.95 | 791.22 | 788.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 786.65 | 791.22 | 788.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 781.85 | 789.34 | 788.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 781.85 | 789.34 | 788.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 787.20 | 788.91 | 788.08 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 782.25 | 787.41 | 787.53 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 792.55 | 787.60 | 787.31 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 786.95 | 788.57 | 788.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 785.00 | 787.73 | 788.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 776.90 | 776.46 | 781.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 15:00:00 | 776.90 | 776.46 | 781.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 772.80 | 775.34 | 779.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:15:00 | 771.10 | 774.09 | 778.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:45:00 | 768.20 | 772.49 | 776.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:15:00 | 732.54 | 735.84 | 739.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:15:00 | 729.79 | 735.84 | 739.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 723.85 | 721.31 | 727.73 | SL hit (close>ema200) qty=0.50 sl=721.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 723.85 | 721.31 | 727.73 | SL hit (close>ema200) qty=0.50 sl=721.31 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 739.45 | 730.38 | 729.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 745.00 | 738.86 | 736.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 12:15:00 | 767.50 | 770.44 | 762.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 13:00:00 | 767.50 | 770.44 | 762.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 770.00 | 769.29 | 763.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:30:00 | 761.45 | 769.30 | 764.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 784.15 | 775.54 | 770.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 15:15:00 | 790.05 | 780.91 | 775.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 772.00 | 779.05 | 779.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 772.00 | 779.05 | 779.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 09:15:00 | 767.00 | 775.07 | 777.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 11:15:00 | 776.05 | 774.06 | 776.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 12:00:00 | 776.05 | 774.06 | 776.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 775.00 | 774.25 | 776.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 15:00:00 | 772.00 | 773.97 | 775.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:15:00 | 733.40 | 744.43 | 755.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 737.60 | 736.75 | 745.80 | SL hit (close>ema200) qty=0.50 sl=736.75 alert=retest2 |

### Cycle 45 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 11:15:00 | 770.85 | 751.00 | 748.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 785.45 | 757.89 | 751.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 779.45 | 781.70 | 773.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 781.40 | 782.05 | 777.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 781.40 | 782.05 | 777.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 785.85 | 782.68 | 781.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 14:15:00 | 770.70 | 779.24 | 780.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 770.70 | 779.24 | 780.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 768.65 | 774.72 | 777.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 761.50 | 759.69 | 765.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 761.50 | 759.69 | 765.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 761.50 | 759.69 | 765.44 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 779.35 | 767.36 | 765.77 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 759.50 | 769.21 | 770.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 757.80 | 766.92 | 769.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 728.25 | 720.07 | 732.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 728.25 | 720.07 | 732.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 728.25 | 720.07 | 732.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 728.25 | 720.07 | 732.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 729.20 | 721.89 | 728.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 729.20 | 721.89 | 728.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 729.40 | 723.39 | 728.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 734.50 | 723.39 | 728.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 722.60 | 723.24 | 727.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 720.25 | 722.38 | 727.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 706.30 | 720.32 | 723.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 726.00 | 717.92 | 717.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 726.00 | 717.92 | 717.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 726.00 | 717.92 | 717.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 727.00 | 719.74 | 718.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 719.30 | 719.65 | 718.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 719.30 | 719.65 | 718.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 719.30 | 719.65 | 718.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 719.30 | 719.65 | 718.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 716.80 | 719.08 | 718.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 716.40 | 719.08 | 718.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 717.95 | 718.85 | 718.55 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 717.00 | 718.18 | 718.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 14:15:00 | 710.50 | 716.64 | 717.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 10:15:00 | 721.90 | 717.10 | 717.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 10:15:00 | 721.90 | 717.10 | 717.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 721.90 | 717.10 | 717.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 721.70 | 717.10 | 717.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 726.00 | 718.88 | 718.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 736.80 | 724.13 | 721.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 724.95 | 726.02 | 723.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 14:00:00 | 724.95 | 726.02 | 723.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 720.65 | 724.95 | 722.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 720.65 | 724.95 | 722.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 723.00 | 724.56 | 722.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 711.65 | 724.56 | 722.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 713.00 | 722.25 | 722.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:15:00 | 710.45 | 722.25 | 722.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 704.40 | 718.68 | 720.47 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 739.45 | 720.72 | 719.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 751.05 | 746.10 | 741.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 14:15:00 | 818.20 | 818.29 | 805.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 15:00:00 | 818.20 | 818.29 | 805.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 821.35 | 818.85 | 807.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:15:00 | 827.85 | 819.47 | 809.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 824.40 | 819.78 | 810.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:45:00 | 825.45 | 820.44 | 811.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 824.25 | 820.44 | 811.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 801.60 | 816.60 | 812.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 801.60 | 816.60 | 812.55 | SL hit (close<static) qty=1.00 sl=805.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 801.60 | 816.60 | 812.55 | SL hit (close<static) qty=1.00 sl=805.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 801.60 | 816.60 | 812.55 | SL hit (close<static) qty=1.00 sl=805.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 801.60 | 816.60 | 812.55 | SL hit (close<static) qty=1.00 sl=805.50 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 801.60 | 816.60 | 812.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 805.05 | 814.29 | 811.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 808.70 | 814.57 | 812.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 11:15:00 | 807.75 | 811.59 | 811.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 11:15:00 | 807.75 | 811.59 | 811.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 14:15:00 | 804.80 | 808.86 | 810.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 814.50 | 809.21 | 810.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 814.50 | 809.21 | 810.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 814.50 | 809.21 | 810.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 814.35 | 809.21 | 810.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 809.45 | 809.26 | 810.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:15:00 | 808.15 | 809.26 | 810.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 816.85 | 811.21 | 811.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 816.85 | 811.21 | 811.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 819.90 | 814.76 | 812.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 815.05 | 817.37 | 815.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 815.05 | 817.37 | 815.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 815.05 | 817.37 | 815.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 812.90 | 817.37 | 815.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 813.65 | 816.62 | 815.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 813.85 | 816.62 | 815.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 809.00 | 813.99 | 814.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 807.85 | 812.76 | 813.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 807.45 | 799.86 | 804.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 807.45 | 799.86 | 804.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 807.45 | 799.86 | 804.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 807.45 | 799.86 | 804.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 808.20 | 801.53 | 804.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:30:00 | 804.90 | 802.42 | 804.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 823.00 | 807.68 | 806.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 823.00 | 807.68 | 806.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 842.20 | 825.58 | 819.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 900.50 | 902.31 | 877.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 900.50 | 902.31 | 877.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 900.50 | 902.31 | 877.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 10:15:00 | 925.70 | 902.31 | 877.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 924.00 | 901.64 | 900.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 12:30:00 | 911.70 | 907.74 | 904.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 13:30:00 | 911.80 | 907.37 | 904.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 913.35 | 908.57 | 905.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 15:15:00 | 921.00 | 908.57 | 905.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 909.60 | 939.76 | 943.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 909.60 | 939.76 | 943.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 909.60 | 939.76 | 943.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 909.60 | 939.76 | 943.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 909.60 | 939.76 | 943.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 909.60 | 939.76 | 943.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 902.25 | 932.25 | 939.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 902.25 | 900.11 | 915.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 902.25 | 900.11 | 915.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 853.35 | 849.19 | 863.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 844.35 | 848.77 | 861.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:45:00 | 844.15 | 848.61 | 859.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 12:15:00 | 873.95 | 863.56 | 862.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 12:15:00 | 873.95 | 863.56 | 862.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 873.95 | 863.56 | 862.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 877.60 | 866.37 | 864.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 851.00 | 866.84 | 865.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 851.00 | 866.84 | 865.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 851.00 | 866.84 | 865.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 851.95 | 866.84 | 865.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 848.10 | 863.09 | 863.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 845.55 | 854.93 | 859.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-19 15:15:00 | 856.00 | 855.14 | 858.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 09:15:00 | 880.55 | 855.14 | 858.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 872.85 | 858.68 | 860.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 880.00 | 858.68 | 860.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 872.90 | 861.53 | 861.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 12:15:00 | 886.00 | 867.95 | 864.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 842.35 | 869.06 | 866.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 842.35 | 869.06 | 866.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 842.35 | 869.06 | 866.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 842.35 | 869.06 | 866.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 836.45 | 862.54 | 864.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 824.80 | 854.99 | 860.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 826.95 | 826.45 | 839.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 826.95 | 826.45 | 839.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 814.95 | 826.68 | 836.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 820.50 | 826.68 | 836.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 834.70 | 826.41 | 834.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 833.45 | 826.41 | 834.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 834.30 | 827.98 | 834.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 821.50 | 831.75 | 834.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 14:15:00 | 780.42 | 794.48 | 807.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 813.40 | 794.99 | 805.34 | SL hit (close>ema200) qty=0.50 sl=794.99 alert=retest2 |

### Cycle 63 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 804.70 | 799.46 | 799.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 810.50 | 801.67 | 800.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 802.05 | 803.08 | 801.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 802.05 | 803.08 | 801.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 802.05 | 803.08 | 801.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:45:00 | 802.60 | 803.08 | 801.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 848.00 | 850.87 | 843.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:30:00 | 844.85 | 850.87 | 843.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 831.70 | 846.90 | 843.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 841.15 | 846.90 | 843.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 925.27 | 879.92 | 865.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 12:15:00 | 990.30 | 997.61 | 997.67 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 1010.50 | 997.59 | 997.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 12:15:00 | 1013.40 | 1002.98 | 999.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 1009.00 | 1009.42 | 1004.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:00:00 | 1009.00 | 1009.42 | 1004.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1039.00 | 1015.34 | 1007.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:15:00 | 1049.40 | 1015.34 | 1007.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 1049.55 | 1039.32 | 1031.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 09:15:00 | 1154.34 | 1091.02 | 1066.72 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-05-08 09:15:00 | 1154.51 | 1091.02 | 1066.72 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:15:00 | 932.60 | 2025-05-19 09:15:00 | 968.31 | PARTIAL | 0.50 | 3.83% |
| BUY | retest1 | 2025-05-14 09:15:00 | 932.60 | 2025-05-19 13:15:00 | 957.95 | STOP_HIT | 0.50 | 2.72% |
| BUY | retest1 | 2025-05-14 11:00:00 | 922.20 | 2025-05-20 12:15:00 | 950.50 | STOP_HIT | 1.00 | 3.07% |
| SELL | retest2 | 2025-06-18 11:45:00 | 957.00 | 2025-06-20 14:15:00 | 912.95 | PARTIAL | 0.50 | 4.60% |
| SELL | retest2 | 2025-06-18 12:30:00 | 961.00 | 2025-06-20 14:15:00 | 912.14 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2025-06-18 14:15:00 | 960.15 | 2025-06-20 14:15:00 | 912.95 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-06-18 15:15:00 | 961.00 | 2025-06-20 15:15:00 | 909.15 | PARTIAL | 0.50 | 5.40% |
| SELL | retest2 | 2025-06-18 11:45:00 | 957.00 | 2025-06-23 09:15:00 | 932.80 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2025-06-18 12:30:00 | 961.00 | 2025-06-23 09:15:00 | 932.80 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2025-06-18 14:15:00 | 960.15 | 2025-06-23 09:15:00 | 932.80 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2025-06-18 15:15:00 | 961.00 | 2025-06-23 09:15:00 | 932.80 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2025-06-19 10:15:00 | 952.25 | 2025-06-25 10:15:00 | 939.05 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2025-07-10 11:15:00 | 955.25 | 2025-07-16 10:15:00 | 955.50 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-07-10 13:30:00 | 958.00 | 2025-07-16 10:15:00 | 955.50 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-07-11 10:15:00 | 957.65 | 2025-07-16 10:15:00 | 955.50 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-07-11 15:15:00 | 953.00 | 2025-07-16 12:15:00 | 956.90 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-07-14 09:15:00 | 951.95 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-07-14 11:00:00 | 951.50 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-07-14 12:30:00 | 950.50 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-15 09:30:00 | 951.20 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-07-15 14:15:00 | 949.40 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-07-15 15:15:00 | 948.00 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-07-16 09:45:00 | 948.25 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-07-16 10:45:00 | 950.00 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-07-23 10:15:00 | 919.60 | 2025-08-01 15:15:00 | 875.76 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2025-07-23 15:00:00 | 921.85 | 2025-08-01 15:15:00 | 875.57 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-07-24 09:15:00 | 921.65 | 2025-08-04 09:15:00 | 873.62 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2025-07-23 10:15:00 | 919.60 | 2025-08-04 12:15:00 | 877.95 | STOP_HIT | 0.50 | 4.53% |
| SELL | retest2 | 2025-07-23 15:00:00 | 921.85 | 2025-08-04 12:15:00 | 877.95 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2025-07-24 09:15:00 | 921.65 | 2025-08-04 12:15:00 | 877.95 | STOP_HIT | 0.50 | 4.74% |
| BUY | retest2 | 2025-08-22 09:15:00 | 872.15 | 2025-08-25 13:15:00 | 841.25 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2025-09-15 09:15:00 | 854.95 | 2025-09-16 14:15:00 | 845.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-09-15 12:30:00 | 854.00 | 2025-09-16 14:15:00 | 845.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-09-22 13:45:00 | 835.75 | 2025-10-01 09:15:00 | 822.00 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2025-10-06 13:30:00 | 824.65 | 2025-10-07 11:15:00 | 810.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-10-14 10:15:00 | 795.70 | 2025-10-21 13:15:00 | 798.50 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-10-16 11:00:00 | 793.40 | 2025-10-21 13:15:00 | 798.50 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-20 13:15:00 | 771.10 | 2025-12-08 09:15:00 | 732.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 14:45:00 | 768.20 | 2025-12-08 09:15:00 | 729.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 13:15:00 | 771.10 | 2025-12-09 11:15:00 | 723.85 | STOP_HIT | 0.50 | 6.13% |
| SELL | retest2 | 2025-11-20 14:45:00 | 768.20 | 2025-12-09 11:15:00 | 723.85 | STOP_HIT | 0.50 | 5.77% |
| BUY | retest2 | 2025-12-19 15:15:00 | 790.05 | 2025-12-23 13:15:00 | 772.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-12-24 15:00:00 | 772.00 | 2025-12-30 10:15:00 | 733.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 15:00:00 | 772.00 | 2025-12-31 09:15:00 | 737.60 | STOP_HIT | 0.50 | 4.46% |
| BUY | retest2 | 2026-01-08 09:30:00 | 785.85 | 2026-01-08 14:15:00 | 770.70 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-01-23 10:45:00 | 720.25 | 2026-01-28 14:15:00 | 726.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-27 09:15:00 | 706.30 | 2026-01-28 14:15:00 | 726.00 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2026-02-12 11:15:00 | 827.85 | 2026-02-13 09:15:00 | 801.60 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2026-02-12 12:15:00 | 824.40 | 2026-02-13 09:15:00 | 801.60 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2026-02-12 12:45:00 | 825.45 | 2026-02-13 09:15:00 | 801.60 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2026-02-12 13:15:00 | 824.25 | 2026-02-13 09:15:00 | 801.60 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2026-02-13 11:45:00 | 808.70 | 2026-02-16 11:15:00 | 807.75 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2026-02-17 11:15:00 | 808.15 | 2026-02-17 12:15:00 | 816.85 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-02-23 11:30:00 | 804.90 | 2026-02-23 13:15:00 | 823.00 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-03-02 10:15:00 | 925.70 | 2026-03-11 09:15:00 | 909.60 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-03-05 09:15:00 | 924.00 | 2026-03-11 09:15:00 | 909.60 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-03-05 12:30:00 | 911.70 | 2026-03-11 09:15:00 | 909.60 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2026-03-05 13:30:00 | 911.80 | 2026-03-11 09:15:00 | 909.60 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2026-03-05 15:15:00 | 921.00 | 2026-03-11 09:15:00 | 909.60 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-03-17 12:15:00 | 844.35 | 2026-03-18 12:15:00 | 873.95 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2026-03-17 12:45:00 | 844.15 | 2026-03-18 12:15:00 | 873.95 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-03-27 09:15:00 | 821.50 | 2026-03-30 14:15:00 | 780.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:15:00 | 821.50 | 2026-04-01 09:15:00 | 813.40 | STOP_HIT | 0.50 | 0.99% |
| BUY | retest2 | 2026-04-13 10:15:00 | 841.15 | 2026-04-16 09:15:00 | 925.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-05 11:15:00 | 1049.40 | 2026-05-08 09:15:00 | 1154.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-06 14:15:00 | 1049.55 | 2026-05-08 09:15:00 | 1154.51 | TARGET_HIT | 1.00 | 10.00% |
