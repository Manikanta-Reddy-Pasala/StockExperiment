# Kalpataru Projects International Ltd. (KPIL)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 1275.70
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 3
- **Avg / median % per leg:** 3.12% / 2.83%
- **Sum % (uncompounded):** 21.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 1 | 3 | 3 | 3.12% | 21.8% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 1 | 3 | 3 | 3.12% | 21.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 1 | 3 | 3 | 3.12% | 21.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 00:00:00 | 1272.65 | 982.69 | 1207.15 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=48.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 00:00:00 | 1369.96 | 989.47 | 1228.64 | T1 booked 50% @ 1369.96 |
| Stop hit — per-position SL triggered | 2024-07-23 00:00:00 | 1272.65 | 1015.78 | 1282.47 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2024-08-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 00:00:00 | 1298.25 | 1067.30 | 1262.32 | Stage2 pullback-breakout RSI=56 vol=2.5x ATR=50.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 00:00:00 | 1399.25 | 1083.20 | 1299.00 | T1 booked 50% @ 1399.25 |
| Target hit | 2024-09-20 00:00:00 | 1335.05 | 1122.07 | 1357.91 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-09-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 00:00:00 | 1407.55 | 1132.64 | 1351.73 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=50.74 |
| Stop hit — per-position SL triggered | 2024-10-03 00:00:00 | 1331.44 | 1138.87 | 1349.11 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2024-12-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 00:00:00 | 1249.15 | 1169.63 | 1185.48 | Stage2 pullback-breakout RSI=59 vol=7.0x ATR=42.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 00:00:00 | 1334.39 | 1178.07 | 1241.70 | T1 booked 50% @ 1334.39 |
| Stop hit — per-position SL triggered | 2024-12-20 00:00:00 | 1275.95 | 1183.13 | 1262.49 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-08 00:00:00 | 1272.65 | 2024-07-10 00:00:00 | 1369.96 | PARTIAL | 0.50 | 7.65% |
| BUY | retest1 | 2024-07-08 00:00:00 | 1272.65 | 2024-07-23 00:00:00 | 1272.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-23 00:00:00 | 1298.25 | 2024-09-02 00:00:00 | 1399.25 | PARTIAL | 0.50 | 7.78% |
| BUY | retest1 | 2024-08-23 00:00:00 | 1298.25 | 2024-09-20 00:00:00 | 1335.05 | TARGET_HIT | 0.50 | 2.83% |
| BUY | retest1 | 2024-09-27 00:00:00 | 1407.55 | 2024-10-03 00:00:00 | 1331.44 | STOP_HIT | 1.00 | -5.41% |
| BUY | retest1 | 2024-12-05 00:00:00 | 1249.15 | 2024-12-16 00:00:00 | 1334.39 | PARTIAL | 0.50 | 6.82% |
| BUY | retest1 | 2024-12-05 00:00:00 | 1249.15 | 2024-12-20 00:00:00 | 1275.95 | STOP_HIT | 0.50 | 2.15% |
