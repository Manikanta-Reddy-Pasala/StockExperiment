# NESTLEIND (NESTLEIND)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1475.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 3 |
| PENDING | 4 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 3 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 0
- **Avg / median % per leg:** 1.88% / -1.33%
- **Sum % (uncompounded):** 5.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.18% | -4.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.18% | -4.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 3 | 1 | 33.3% | 1 | 2 | 0 | 1.88% | 5.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 12:15:00 | 1220.60 | 1175.48 | 1175.30 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 15:15:00 | 1167.00 | 1178.29 | 1178.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 09:15:00 | 1163.50 | 1178.14 | 1178.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 12:15:00 | 1180.60 | 1177.31 | 1177.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 12:15:00 | 1180.60 | 1177.31 | 1177.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 1180.60 | 1177.31 | 1177.80 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-10-08 09:15:00 | 1166.40 | 1177.60 | 1177.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 1169.40 | 1177.52 | 1177.88 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-09 13:15:00 | 1185.00 | 1177.66 | 1177.94 | SL hit (close>static) qty=1.00 sl=1180.90 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 1193.80 | 1178.28 | 1178.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 1195.00 | 1178.44 | 1178.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 1175.40 | 1179.67 | 1178.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 11:15:00 | 1175.40 | 1179.67 | 1178.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 1175.40 | 1179.67 | 1178.97 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-15 09:15:00 | 1197.90 | 1179.65 | 1178.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:15:00 | 1209.90 | 1179.95 | 1179.14 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2026-01-07 09:15:00 | 1330.89 | 1265.75 | 1251.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 1234.50 | 1278.90 | 1279.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1221.30 | 1277.41 | 1278.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1230.70 | 1222.56 | 1243.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1230.70 | 1222.56 | 1243.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1230.70 | 1222.56 | 1243.20 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-08 11:15:00 | 1210.30 | 1222.43 | 1242.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 12:15:00 | 1216.70 | 1222.37 | 1242.80 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-09 13:15:00 | 1218.00 | 1221.93 | 1241.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-09 14:15:00 | 1229.50 | 1222.00 | 1241.71 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-04-15 13:15:00 | 1253.60 | 1224.69 | 1241.24 | SL hit (close>static) qty=1.00 sl=1250.60 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1408.00 | 1253.92 | 1253.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 1416.30 | 1264.47 | 1259.25 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-08 10:15:00 | 1169.40 | 2025-10-09 13:15:00 | 1185.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-10-15 10:15:00 | 1209.90 | 2026-01-07 09:15:00 | 1330.89 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-08 12:15:00 | 1216.70 | 2026-04-15 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -3.03% |
