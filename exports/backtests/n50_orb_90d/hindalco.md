# HINDALCO (HINDALCO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1044.50
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
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 3 / 5 / 3
- **Avg / median % per leg:** 0.30% / 0.40%
- **Sum % (uncompounded):** 3.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.35% | 2.5% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.35% | 2.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.20% | 0.8% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.20% | 0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 6 | 54.5% | 3 | 5 | 3 | 0.30% | 3.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:30:00 | 921.30 | 913.06 | 0.00 | ORB-long ORB[901.10,912.55] vol=2.0x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:00:00 | 925.36 | 916.08 | 0.00 | T1 1.5R @ 925.36 |
| Target hit | 2026-02-20 15:20:00 | 936.20 | 928.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:10:00 | 924.80 | 935.96 | 0.00 | ORB-short ORB[938.20,946.90] vol=2.3x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:20:00 | 921.15 | 934.97 | 0.00 | T1 1.5R @ 921.15 |
| Target hit | 2026-02-23 15:20:00 | 916.00 | 920.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-03-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:55:00 | 961.00 | 942.49 | 0.00 | ORB-long ORB[926.40,935.00] vol=2.4x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:05:00 | 968.04 | 951.42 | 0.00 | T1 1.5R @ 968.04 |
| Target hit | 2026-03-05 12:00:00 | 967.50 | 969.05 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-04-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-01 11:10:00 | 921.50 | 912.77 | 0.00 | ORB-long ORB[901.05,912.00] vol=1.8x ATR=3.26 |
| Stop hit — per-position SL triggered | 2026-04-01 11:20:00 | 918.24 | 912.97 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 10:50:00 | 974.65 | 974.86 | 0.00 | ORB-short ORB[974.80,988.60] vol=1.6x ATR=3.07 |
| Stop hit — per-position SL triggered | 2026-04-13 11:00:00 | 977.72 | 974.96 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:50:00 | 1064.45 | 1058.97 | 0.00 | ORB-long ORB[1054.10,1059.95] vol=1.6x ATR=3.11 |
| Stop hit — per-position SL triggered | 2026-04-27 10:25:00 | 1061.34 | 1060.02 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-05-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:55:00 | 1045.80 | 1037.66 | 0.00 | ORB-long ORB[1031.90,1044.60] vol=1.7x ATR=3.78 |
| Stop hit — per-position SL triggered | 2026-05-05 10:10:00 | 1042.02 | 1038.70 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:10:00 | 1055.10 | 1062.38 | 0.00 | ORB-short ORB[1062.00,1073.70] vol=2.0x ATR=2.52 |
| Stop hit — per-position SL triggered | 2026-05-06 11:40:00 | 1057.62 | 1061.70 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-20 10:30:00 | 921.30 | 2026-02-20 11:00:00 | 925.36 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-02-20 10:30:00 | 921.30 | 2026-02-20 15:20:00 | 936.20 | TARGET_HIT | 0.50 | 1.62% |
| SELL | retest1 | 2026-02-23 11:10:00 | 924.80 | 2026-02-23 11:20:00 | 921.15 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-23 11:10:00 | 924.80 | 2026-02-23 15:20:00 | 916.00 | TARGET_HIT | 0.50 | 0.95% |
| BUY | retest1 | 2026-03-05 09:55:00 | 961.00 | 2026-03-05 10:05:00 | 968.04 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-03-05 09:55:00 | 961.00 | 2026-03-05 12:00:00 | 967.50 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2026-04-01 11:10:00 | 921.50 | 2026-04-01 11:20:00 | 918.24 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-13 10:50:00 | 974.65 | 2026-04-13 11:00:00 | 977.72 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-27 09:50:00 | 1064.45 | 2026-04-27 10:25:00 | 1061.34 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-05 09:55:00 | 1045.80 | 2026-05-05 10:10:00 | 1042.02 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-05-06 11:10:00 | 1055.10 | 2026-05-06 11:40:00 | 1057.62 | STOP_HIT | 1.00 | -0.24% |
