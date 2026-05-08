# TATACONSUM (TATACONSUM)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1176.60
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
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 8 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 1 |
| ENTRY2 | 4 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** -1.60% / -3.14%
- **Sum % (uncompounded):** -9.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | -1.29% | -6.5% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.01% | 6.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.16% | -12.5% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.14% | -3.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.14% | -3.1% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.01% | 6.0% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.90% | -15.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 1127.00 | 1091.09 | 1091.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 15:15:00 | 1129.10 | 1092.14 | 1091.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 1114.90 | 1114.96 | 1106.43 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-15 13:15:00 | 1119.00 | 1115.14 | 1106.93 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-15 14:15:00 | 1115.50 | 1115.14 | 1106.98 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-16 11:15:00 | 1122.40 | 1115.17 | 1107.15 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 12:15:00 | 1133.10 | 1115.35 | 1107.28 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 09:15:00 | 1189.75 | 1124.82 | 1112.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-10 12:15:00 | 1144.70 | 1150.23 | 1132.03 | SL hit (close<ema200) qty=0.50 sl=1150.23 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1149.30 | 1162.92 | 1147.87 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-05 14:15:00 | 1164.90 | 1160.18 | 1147.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-05 15:15:00 | 1161.90 | 1160.19 | 1147.85 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-15 10:15:00 | 1165.80 | 1156.16 | 1147.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-15 11:15:00 | 1158.20 | 1156.18 | 1147.78 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-16 09:15:00 | 1168.30 | 1156.39 | 1148.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 1174.40 | 1156.57 | 1148.23 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-21 15:15:00 | 1165.00 | 1179.82 | 1169.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 09:15:00 | 1183.90 | 1179.86 | 1169.32 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-01-27 11:15:00 | 1168.90 | 1178.31 | 1169.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 12:15:00 | 1175.70 | 1178.29 | 1169.37 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1129.00 | 1177.86 | 1169.34 | SL hit (close<static) qty=1.00 sl=1144.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1129.00 | 1177.86 | 1169.34 | SL hit (close<static) qty=1.00 sl=1144.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1129.00 | 1177.86 | 1169.34 | SL hit (close<static) qty=1.00 sl=1144.30 alert=retest2 |

### Cycle 2 — SELL (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 13:15:00 | 1121.60 | 1162.05 | 1162.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 1115.20 | 1156.69 | 1158.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 11:15:00 | 1082.30 | 1076.96 | 1104.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1102.40 | 1078.71 | 1102.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1102.40 | 1078.71 | 1102.58 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-15 15:15:00 | 1092.00 | 1079.91 | 1102.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:15:00 | 1090.30 | 1080.01 | 1102.42 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 1124.50 | 1081.55 | 1102.43 | SL hit (close>static) qty=1.00 sl=1108.30 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 1148.60 | 1117.10 | 1117.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 1163.00 | 1118.63 | 1117.78 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-16 12:15:00 | 1133.10 | 2025-10-23 09:15:00 | 1189.75 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-10-16 12:15:00 | 1133.10 | 2025-11-10 12:15:00 | 1144.70 | STOP_HIT | 0.50 | 1.02% |
| BUY | retest2 | 2025-12-16 10:15:00 | 1174.40 | 2026-01-28 09:15:00 | 1129.00 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2026-01-22 09:15:00 | 1183.90 | 2026-01-28 09:15:00 | 1129.00 | STOP_HIT | 1.00 | -4.64% |
| BUY | retest2 | 2026-01-27 12:15:00 | 1175.70 | 2026-01-28 09:15:00 | 1129.00 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2026-04-16 09:15:00 | 1090.30 | 2026-04-17 09:15:00 | 1124.50 | STOP_HIT | 1.00 | -3.14% |
