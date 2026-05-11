# Nippon Life India Asset Management Ltd. (NAM-INDIA)

## Backtest Summary

- **Window:** 2026-03-09 09:15:00 → 2026-05-08 15:25:00 (3000 bars)
- **Last close:** 1100.20
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 2 / 4 / 2
- **Avg / median % per leg:** 0.50% / 0.61%
- **Sum % (uncompounded):** 4.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.62% | 4.4% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.62% | 4.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.35% | -0.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.35% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 2 | 4 | 2 | 0.50% | 4.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-03-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 11:10:00 | 829.65 | 827.44 | 0.00 | ORB-long ORB[815.10,827.55] vol=1.6x ATR=3.57 |
| Stop hit — per-position SL triggered | 2026-03-16 11:50:00 | 826.08 | 827.84 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-03-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:00:00 | 840.35 | 835.85 | 0.00 | ORB-long ORB[829.85,836.90] vol=3.0x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:15:00 | 845.64 | 837.59 | 0.00 | T1 1.5R @ 845.64 |
| Target hit | 2026-03-17 15:20:00 | 846.85 | 845.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-03-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:00:00 | 822.75 | 826.90 | 0.00 | ORB-short ORB[827.25,835.90] vol=1.5x ATR=2.87 |
| Stop hit — per-position SL triggered | 2026-03-24 11:05:00 | 825.62 | 826.77 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:15:00 | 970.00 | 965.26 | 0.00 | ORB-long ORB[958.65,966.70] vol=3.5x ATR=2.92 |
| Stop hit — per-position SL triggered | 2026-04-16 11:20:00 | 967.08 | 965.33 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:45:00 | 981.55 | 973.76 | 0.00 | ORB-long ORB[962.00,976.60] vol=1.8x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 09:55:00 | 987.55 | 976.41 | 0.00 | T1 1.5R @ 987.55 |
| Target hit | 2026-04-17 15:20:00 | 1015.65 | 998.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:15:00 | 1073.80 | 1065.93 | 0.00 | ORB-long ORB[1058.40,1071.80] vol=1.9x ATR=4.31 |
| Stop hit — per-position SL triggered | 2026-05-06 10:45:00 | 1069.49 | 1067.88 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-03-16 11:10:00 | 829.65 | 2026-03-16 11:50:00 | 826.08 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-17 10:00:00 | 840.35 | 2026-03-17 10:15:00 | 845.64 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-03-17 10:00:00 | 840.35 | 2026-03-17 15:20:00 | 846.85 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2026-03-24 11:00:00 | 822.75 | 2026-03-24 11:05:00 | 825.62 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-16 11:15:00 | 970.00 | 2026-04-16 11:20:00 | 967.08 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-17 09:45:00 | 981.55 | 2026-04-17 09:55:00 | 987.55 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-04-17 09:45:00 | 981.55 | 2026-04-17 15:20:00 | 1015.65 | TARGET_HIT | 0.50 | 3.47% |
| BUY | retest1 | 2026-05-06 10:15:00 | 1073.80 | 2026-05-06 10:45:00 | 1069.49 | STOP_HIT | 1.00 | -0.40% |
