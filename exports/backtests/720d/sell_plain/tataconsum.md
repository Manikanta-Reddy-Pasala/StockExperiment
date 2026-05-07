# TATACONSUM (TATACONSUM)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1156.40
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 4 |
| ALERT3 | 4 |
| PENDING | 12 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / Stop hits / Partials:** 0 / 7 / 0
- **Avg / median % per leg:** -3.41% / -3.60%
- **Sum % (uncompounded):** -23.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.41% | -23.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.41% | -23.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.41% | -23.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 12:15:00 | 1111.05 | 1173.05 | 1173.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 1103.95 | 1161.93 | 1167.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 934.30 | 932.79 | 975.12 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 10:15:00 | 981.15 | 937.73 | 972.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 981.15 | 937.73 | 972.19 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-13 09:15:00 | 961.90 | 941.72 | 972.08 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-13 10:15:00 | 966.25 | 941.97 | 972.05 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-15 09:15:00 | 946.40 | 944.77 | 971.61 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 11:15:00 | 949.65 | 944.90 | 971.41 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-20 13:15:00 | 962.25 | 945.59 | 968.87 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 15:15:00 | 960.00 | 945.88 | 968.78 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-22 09:15:00 | 961.55 | 947.79 | 968.86 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-22 10:15:00 | 966.30 | 947.98 | 968.85 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 983.85 | 949.22 | 968.86 | SL hit (close>static) qty=1.00 sl=981.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 983.85 | 949.22 | 968.86 | SL hit (close>static) qty=1.00 sl=981.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-28 13:15:00 | 961.85 | 956.28 | 970.34 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 15:15:00 | 960.75 | 956.38 | 970.25 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-30 11:15:00 | 958.20 | 956.58 | 969.67 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-30 12:15:00 | 966.10 | 956.68 | 969.66 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-30 13:15:00 | 961.05 | 956.72 | 969.61 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-30 14:15:00 | 968.45 | 956.84 | 969.61 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 1006.35 | 957.46 | 969.79 | SL hit (close>static) qty=1.00 sl=981.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-03 10:15:00 | 957.25 | 1000.42 | 993.97 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 12:15:00 | 955.15 | 999.52 | 993.58 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-03-07 14:15:00 | 962.05 | 988.42 | 988.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 14:15:00 | 962.05 | 988.42 | 988.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 957.75 | 986.70 | 987.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 11:15:00 | 973.60 | 971.81 | 978.66 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 969.40 | 971.84 | 978.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 969.40 | 971.84 | 978.51 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-25 10:15:00 | 963.75 | 971.76 | 978.43 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 12:15:00 | 965.85 | 971.58 | 978.27 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-03-26 10:15:00 | 958.05 | 971.37 | 978.01 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 12:15:00 | 964.00 | 971.21 | 977.86 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 1009.10 | 971.33 | 977.56 | SL hit (close>static) qty=1.00 sl=979.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 1009.10 | 971.33 | 977.56 | SL hit (close>static) qty=1.00 sl=979.90 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 15:15:00 | 1064.00 | 1093.61 | 1093.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1061.20 | 1092.08 | 1092.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1081.30 | 1070.87 | 1079.53 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1081.30 | 1070.87 | 1079.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1081.30 | 1070.87 | 1079.53 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-29 09:15:00 | 1056.60 | 1075.64 | 1080.21 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-29 10:15:00 | 1064.80 | 1075.53 | 1080.14 | ENTRY2 sustain failed after 60m |

### Cycle 4 — SELL (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 14:15:00 | 1124.50 | 1161.68 | 1161.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 1115.20 | 1156.69 | 1158.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 11:15:00 | 1082.30 | 1076.96 | 1104.10 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1102.40 | 1078.71 | 1102.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1102.40 | 1078.71 | 1102.54 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-15 15:15:00 | 1092.00 | 1079.91 | 1102.45 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:15:00 | 1090.30 | 1080.01 | 1102.39 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 1124.50 | 1081.55 | 1102.40 | SL hit (close>static) qty=1.00 sl=1108.30 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-01-15 11:15:00 | 949.65 | 2025-01-23 09:15:00 | 983.85 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-01-20 15:15:00 | 960.00 | 2025-01-23 09:15:00 | 983.85 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-01-28 15:15:00 | 960.75 | 2025-01-31 09:15:00 | 1006.35 | STOP_HIT | 1.00 | -4.75% |
| SELL | retest2 | 2025-03-03 12:15:00 | 955.15 | 2025-03-07 14:15:00 | 962.05 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-03-25 12:15:00 | 965.85 | 2025-03-28 09:15:00 | 1009.10 | STOP_HIT | 1.00 | -4.48% |
| SELL | retest2 | 2025-03-26 12:15:00 | 964.00 | 2025-03-28 09:15:00 | 1009.10 | STOP_HIT | 1.00 | -4.68% |
| SELL | retest2 | 2026-04-16 09:15:00 | 1090.30 | 2026-04-17 09:15:00 | 1124.50 | STOP_HIT | 1.00 | -3.14% |
