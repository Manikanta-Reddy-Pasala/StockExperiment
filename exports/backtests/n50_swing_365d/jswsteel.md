# JSWSTEEL (JSWSTEEL)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1277.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / Stop hits / Partials:** 1 / 7 / 1
- **Avg / median % per leg:** 0.20% / -0.20%
- **Sum % (uncompounded):** 1.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 1 | 7 | 1 | 0.20% | 1.8% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 7 | 1 | 0.20% | 1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 4 | 44.4% | 1 | 7 | 1 | 0.20% | 1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 05:30:00 | 1032.55 | 981.17 | 1003.90 | Stage2 pullback-breakout RSI=60 vol=1.9x ATR=22.26 |
| Stop hit — per-position SL triggered | 2025-07-10 05:30:00 | 1043.30 | 986.76 | 1027.38 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-08-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 05:30:00 | 1079.80 | 999.66 | 1047.73 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=22.15 |
| Stop hit — per-position SL triggered | 2025-08-26 05:30:00 | 1046.58 | 1003.43 | 1054.48 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2025-09-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 05:30:00 | 1072.20 | 1005.36 | 1050.62 | Stage2 pullback-breakout RSI=58 vol=2.0x ATR=20.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 05:30:00 | 1113.88 | 1009.52 | 1067.02 | T1 booked 50% @ 1113.88 |
| Target hit | 2025-10-20 05:30:00 | 1145.20 | 1040.76 | 1146.36 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2025-10-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 05:30:00 | 1184.20 | 1046.26 | 1149.76 | Stage2 pullback-breakout RSI=63 vol=2.3x ATR=22.90 |
| Stop hit — per-position SL triggered | 2025-11-12 05:30:00 | 1181.80 | 1059.85 | 1173.15 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2025-12-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 05:30:00 | 1164.80 | 1079.11 | 1114.37 | Stage2 pullback-breakout RSI=63 vol=3.0x ATR=25.15 |
| Stop hit — per-position SL triggered | 2026-01-14 05:30:00 | 1181.80 | 1088.34 | 1153.42 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2026-01-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 05:30:00 | 1222.00 | 1094.76 | 1169.25 | Stage2 pullback-breakout RSI=65 vol=2.0x ATR=27.47 |
| Stop hit — per-position SL triggered | 2026-02-01 05:30:00 | 1180.80 | 1099.49 | 1184.07 | SL hit (bars_held=4) |

### Cycle 7 — BUY (started 2026-02-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 05:30:00 | 1275.00 | 1122.80 | 1234.92 | Stage2 pullback-breakout RSI=66 vol=1.6x ATR=25.71 |
| Stop hit — per-position SL triggered | 2026-03-04 05:30:00 | 1236.44 | 1128.03 | 1241.17 | SL hit (bars_held=4) |

### Cycle 8 — BUY (started 2026-04-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 05:30:00 | 1274.50 | 1140.66 | 1196.20 | Stage2 pullback-breakout RSI=65 vol=2.0x ATR=35.58 |
| Stop hit — per-position SL triggered | 2026-05-05 05:30:00 | 1252.40 | 1152.81 | 1241.36 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-26 05:30:00 | 1032.55 | 2025-07-10 05:30:00 | 1043.30 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest1 | 2025-08-18 05:30:00 | 1079.80 | 2025-08-26 05:30:00 | 1046.58 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest1 | 2025-09-03 05:30:00 | 1072.20 | 2025-09-10 05:30:00 | 1113.88 | PARTIAL | 0.50 | 3.89% |
| BUY | retest1 | 2025-09-03 05:30:00 | 1072.20 | 2025-10-20 05:30:00 | 1145.20 | TARGET_HIT | 0.50 | 6.81% |
| BUY | retest1 | 2025-10-28 05:30:00 | 1184.20 | 2025-11-12 05:30:00 | 1181.80 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-31 05:30:00 | 1164.80 | 2026-01-14 05:30:00 | 1181.80 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest1 | 2026-01-27 05:30:00 | 1222.00 | 2026-02-01 05:30:00 | 1180.80 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest1 | 2026-02-25 05:30:00 | 1275.00 | 2026-03-04 05:30:00 | 1236.44 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest1 | 2026-04-20 05:30:00 | 1274.50 | 2026-05-05 05:30:00 | 1252.40 | STOP_HIT | 1.00 | -1.73% |
