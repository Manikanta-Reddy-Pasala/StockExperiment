# JSWSTEEL (JSWSTEEL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1272.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 0 |
| TARGET_HIT | 11 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 10
- **Target hits / Stop hits / Partials:** 7 / 11 / 0
- **Avg / median % per leg:** 1.44% / -1.46%
- **Sum % (uncompounded):** 25.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 8 | 44.4% | 7 | 11 | 0 | 1.44% | 25.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 8 | 44.4% | 7 | 11 | 0 | 1.44% | 25.9% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 8 | 44.4% | 7 | 11 | 0 | 1.44% | 25.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1076.90 | 1134.81 | 1135.00 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 1187.00 | 1131.48 | 1131.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 1188.50 | 1132.05 | 1131.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1225.60 | 1231.96 | 1204.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:00:00 | 1225.60 | 1231.96 | 1204.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 1202.80 | 1231.53 | 1204.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:00:00 | 1202.80 | 1231.53 | 1204.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 1212.80 | 1231.34 | 1204.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 13:15:00 | 1215.70 | 1231.34 | 1204.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 14:00:00 | 1217.90 | 1231.21 | 1204.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 1220.10 | 1230.79 | 1204.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 1186.60 | 1231.31 | 1206.86 | SL hit (close<static) qty=1.00 sl=1200.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 1186.60 | 1231.31 | 1206.86 | SL hit (close<static) qty=1.00 sl=1200.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 1186.60 | 1231.31 | 1206.86 | SL hit (close<static) qty=1.00 sl=1200.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 12:45:00 | 1213.00 | 1227.99 | 1206.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1195.30 | 1227.33 | 1206.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 1195.30 | 1227.33 | 1206.44 | SL hit (close<static) qty=1.00 sl=1200.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-03-11 10:00:00 | 1195.30 | 1227.33 | 1206.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 1200.80 | 1227.07 | 1206.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:45:00 | 1198.90 | 1227.07 | 1206.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 1120.50 | 1191.41 | 1191.63 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 09:15:00 | 1250.30 | 1186.69 | 1186.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 1253.30 | 1187.35 | 1186.77 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 09:45:00 | 1014.00 | 2025-05-22 09:15:00 | 993.60 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-05-23 09:15:00 | 1014.10 | 2025-05-30 10:15:00 | 997.50 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-05-23 12:30:00 | 1012.60 | 2025-05-30 14:15:00 | 992.20 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-05-26 09:15:00 | 1014.60 | 2025-05-30 14:15:00 | 992.20 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-05-29 10:00:00 | 1018.10 | 2025-05-30 14:15:00 | 992.20 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-06-20 11:00:00 | 1022.00 | 2025-06-23 09:15:00 | 987.90 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2025-06-24 11:15:00 | 1020.80 | 2025-07-02 11:15:00 | 1061.50 | TARGET_HIT | 1.00 | 3.99% |
| BUY | retest2 | 2025-06-25 12:30:00 | 1017.95 | 2025-07-02 11:15:00 | 1057.54 | TARGET_HIT | 1.00 | 3.89% |
| BUY | retest2 | 2025-07-28 13:45:00 | 1029.40 | 2025-08-07 15:15:00 | 1074.10 | TARGET_HIT | 1.00 | 4.34% |
| BUY | retest2 | 2025-08-01 10:00:00 | 1028.80 | 2025-09-01 15:15:00 | 1033.70 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-08-01 15:15:00 | 1028.00 | 2025-09-17 09:15:00 | 1119.75 | TARGET_HIT | 1.00 | 8.92% |
| BUY | retest2 | 2025-08-29 14:15:00 | 1029.20 | 2025-09-18 13:15:00 | 1122.88 | TARGET_HIT | 1.00 | 9.10% |
| BUY | retest2 | 2025-09-01 14:45:00 | 1038.00 | 2025-09-23 12:15:00 | 1132.34 | TARGET_HIT | 1.00 | 9.09% |
| BUY | retest2 | 2025-09-02 09:30:00 | 1038.50 | 2025-09-23 12:15:00 | 1131.68 | TARGET_HIT | 1.00 | 8.97% |
| BUY | retest2 | 2026-03-04 13:15:00 | 1215.70 | 2026-03-09 09:15:00 | 1186.60 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2026-03-04 14:00:00 | 1217.90 | 2026-03-09 09:15:00 | 1186.60 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-03-05 09:15:00 | 1220.10 | 2026-03-09 09:15:00 | 1186.60 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2026-03-10 12:45:00 | 1213.00 | 2026-03-11 09:15:00 | 1195.30 | STOP_HIT | 1.00 | -1.46% |
