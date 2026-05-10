# Angel One Ltd. (ANGELONE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 326.00
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 1
- **Avg / median % per leg:** 0.14% / -0.27%
- **Sum % (uncompounded):** 0.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.44% | -0.9% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.44% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.43% | 1.7% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.43% | 1.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.14% | 0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 259.30 | 256.39 | 0.00 | ORB-long ORB[254.14,257.16] vol=2.7x ATR=1.17 |
| Stop hit — per-position SL triggered | 2026-02-17 10:45:00 | 258.13 | 257.22 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:30:00 | 223.31 | 224.39 | 0.00 | ORB-short ORB[223.75,225.88] vol=1.8x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-03-05 10:15:00 | 224.18 | 223.89 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:45:00 | 222.32 | 224.51 | 0.00 | ORB-short ORB[224.61,227.70] vol=4.2x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:20:00 | 221.03 | 224.20 | 0.00 | T1 1.5R @ 221.03 |
| Target hit | 2026-03-11 15:20:00 | 218.30 | 221.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 325.70 | 322.43 | 0.00 | ORB-long ORB[318.73,322.65] vol=2.3x ATR=1.40 |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 324.30 | 322.92 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:00:00 | 318.01 | 320.19 | 0.00 | ORB-short ORB[319.27,323.40] vol=1.7x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-04-29 11:05:00 | 318.88 | 320.16 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 10:25:00 | 259.30 | 2026-02-17 10:45:00 | 258.13 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-03-05 09:30:00 | 223.31 | 2026-03-05 10:15:00 | 224.18 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-03-11 10:45:00 | 222.32 | 2026-03-11 11:20:00 | 221.03 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-03-11 10:45:00 | 222.32 | 2026-03-11 15:20:00 | 218.30 | TARGET_HIT | 0.50 | 1.81% |
| BUY | retest1 | 2026-04-21 10:00:00 | 325.70 | 2026-04-21 10:15:00 | 324.30 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-29 11:00:00 | 318.01 | 2026-04-29 11:05:00 | 318.88 | STOP_HIT | 1.00 | -0.27% |
