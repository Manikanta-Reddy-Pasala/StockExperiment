# TATACONSUM (TATACONSUM)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1176.20
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 0 / 7 / 0
- **Avg / median % per leg:** -1.44% / -2.95%
- **Sum % (uncompounded):** -10.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 7 | 0 | -1.44% | -10.1% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 7 | 0 | -1.44% | -10.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 0 | 7 | 0 | -1.44% | -10.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 05:30:00 | 1145.40 | 1075.98 | 1108.97 | Stage2 pullback-breakout RSI=62 vol=2.0x ATR=22.51 |
| Stop hit — per-position SL triggered | 2025-06-30 05:30:00 | 1111.63 | 1076.68 | 1109.33 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2025-09-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 05:30:00 | 1100.80 | 1076.50 | 1075.85 | Stage2 pullback-breakout RSI=59 vol=2.1x ATR=21.93 |
| Stop hit — per-position SL triggered | 2025-09-04 05:30:00 | 1067.91 | 1076.72 | 1077.86 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2025-09-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 05:30:00 | 1136.30 | 1078.48 | 1091.39 | Stage2 pullback-breakout RSI=66 vol=3.0x ATR=22.74 |
| Stop hit — per-position SL triggered | 2025-10-01 05:30:00 | 1144.80 | 1083.44 | 1116.46 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-10-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 05:30:00 | 1149.30 | 1087.52 | 1122.62 | Stage2 pullback-breakout RSI=62 vol=2.1x ATR=21.77 |
| Stop hit — per-position SL triggered | 2025-10-31 05:30:00 | 1165.00 | 1095.28 | 1152.17 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2025-11-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 05:30:00 | 1197.50 | 1096.30 | 1156.48 | Stage2 pullback-breakout RSI=68 vol=2.1x ATR=23.71 |
| Stop hit — per-position SL triggered | 2025-11-04 05:30:00 | 1161.93 | 1097.13 | 1158.66 | SL hit (bars_held=1) |

### Cycle 6 — BUY (started 2025-11-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 05:30:00 | 1183.10 | 1104.84 | 1164.03 | Stage2 pullback-breakout RSI=57 vol=1.9x ATR=24.26 |
| Stop hit — per-position SL triggered | 2025-12-03 05:30:00 | 1146.71 | 1109.88 | 1166.01 | SL hit (bars_held=8) |

### Cycle 7 — BUY (started 2025-12-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 05:30:00 | 1195.20 | 1118.67 | 1171.20 | Stage2 pullback-breakout RSI=61 vol=2.3x ATR=20.45 |
| Stop hit — per-position SL triggered | 2026-01-12 05:30:00 | 1192.30 | 1125.37 | 1183.14 | Time-stop (10d <3%) |

### Cycle 8 — BUY (started 2026-05-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 05:30:00 | 1176.20 | 1123.70 | 1139.51 | Stage2 pullback-breakout RSI=63 vol=2.2x ATR=28.56 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-26 05:30:00 | 1145.40 | 2025-06-30 05:30:00 | 1111.63 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest1 | 2025-09-02 05:30:00 | 1100.80 | 2025-09-04 05:30:00 | 1067.91 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest1 | 2025-09-17 05:30:00 | 1136.30 | 2025-10-01 05:30:00 | 1144.80 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest1 | 2025-10-16 05:30:00 | 1149.30 | 2025-10-31 05:30:00 | 1165.00 | STOP_HIT | 1.00 | 1.37% |
| BUY | retest1 | 2025-11-03 05:30:00 | 1197.50 | 2025-11-04 05:30:00 | 1161.93 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest1 | 2025-11-21 05:30:00 | 1183.10 | 2025-12-03 05:30:00 | 1146.71 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest1 | 2025-12-29 05:30:00 | 1195.20 | 2026-01-12 05:30:00 | 1192.30 | STOP_HIT | 1.00 | -0.24% |
