# Firstsource Solutions Ltd. (FSL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 272.05
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 2
- **Avg / median % per leg:** 0.12% / -0.32%
- **Sum % (uncompounded):** 1.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.40% | -1.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.40% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.38% | 2.3% |
| SELL @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.38% | 2.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.12% | 1.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 290.00 | 291.20 | 0.00 | ORB-short ORB[290.20,294.20] vol=2.1x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:40:00 | 288.97 | 290.65 | 0.00 | T1 1.5R @ 288.97 |
| Target hit | 2026-02-11 14:55:00 | 284.00 | 283.81 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2026-03-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 11:10:00 | 212.98 | 214.73 | 0.00 | ORB-short ORB[214.13,217.22] vol=1.8x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:25:00 | 211.78 | 214.41 | 0.00 | T1 1.5R @ 211.78 |
| Stop hit — per-position SL triggered | 2026-03-10 14:55:00 | 212.98 | 212.78 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-04-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:05:00 | 219.90 | 221.30 | 0.00 | ORB-short ORB[220.00,222.25] vol=1.9x ATR=0.70 |
| Stop hit — per-position SL triggered | 2026-04-10 10:15:00 | 220.60 | 221.21 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 215.98 | 214.61 | 0.00 | ORB-long ORB[212.50,214.65] vol=3.5x ATR=0.81 |
| Stop hit — per-position SL triggered | 2026-04-29 10:10:00 | 215.17 | 214.84 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:30:00 | 216.19 | 214.46 | 0.00 | ORB-long ORB[213.00,215.50] vol=2.4x ATR=0.93 |
| Stop hit — per-position SL triggered | 2026-04-30 10:00:00 | 215.26 | 215.52 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-05-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:30:00 | 218.20 | 216.53 | 0.00 | ORB-long ORB[214.24,217.22] vol=1.9x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-05-04 10:50:00 | 217.33 | 216.59 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-05-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:35:00 | 219.60 | 220.63 | 0.00 | ORB-short ORB[220.16,222.59] vol=1.6x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-05-06 10:45:00 | 220.47 | 220.64 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:30:00 | 290.00 | 2026-02-11 09:40:00 | 288.97 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-11 09:30:00 | 290.00 | 2026-02-11 14:55:00 | 284.00 | TARGET_HIT | 0.50 | 2.07% |
| SELL | retest1 | 2026-03-10 11:10:00 | 212.98 | 2026-03-10 11:25:00 | 211.78 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-10 11:10:00 | 212.98 | 2026-03-10 14:55:00 | 212.98 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-10 10:05:00 | 219.90 | 2026-04-10 10:15:00 | 220.60 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-29 09:55:00 | 215.98 | 2026-04-29 10:10:00 | 215.17 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-30 09:30:00 | 216.19 | 2026-04-30 10:00:00 | 215.26 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-05-04 10:30:00 | 218.20 | 2026-05-04 10:50:00 | 217.33 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-05-06 10:35:00 | 219.60 | 2026-05-06 10:45:00 | 220.47 | STOP_HIT | 1.00 | -0.40% |
