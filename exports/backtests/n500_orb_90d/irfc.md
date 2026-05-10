# Indian Railway Finance Corporation Ltd. (IRFC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 106.02
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 2
- **Avg / median % per leg:** -0.11% / -0.25%
- **Sum % (uncompounded):** -1.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.30% | -1.5% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.30% | -1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.08% | 0.4% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.08% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.11% | -1.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 114.25 | 114.79 | 0.00 | ORB-short ORB[114.43,115.80] vol=1.5x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:05:00 | 113.84 | 114.46 | 0.00 | T1 1.5R @ 113.84 |
| Stop hit — per-position SL triggered | 2026-02-11 10:10:00 | 114.25 | 114.44 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:35:00 | 111.34 | 111.80 | 0.00 | ORB-short ORB[111.51,112.85] vol=1.8x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 111.66 | 111.80 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 111.60 | 111.18 | 0.00 | ORB-long ORB[110.57,111.50] vol=1.5x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-02-16 09:35:00 | 111.31 | 111.21 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:00:00 | 112.61 | 113.16 | 0.00 | ORB-short ORB[112.85,113.75] vol=1.8x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:15:00 | 112.23 | 113.01 | 0.00 | T1 1.5R @ 112.23 |
| Stop hit — per-position SL triggered | 2026-02-18 11:10:00 | 112.61 | 112.86 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:00:00 | 104.30 | 103.19 | 0.00 | ORB-long ORB[102.41,103.64] vol=4.0x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-04-16 11:35:00 | 103.95 | 103.48 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:00:00 | 105.96 | 105.37 | 0.00 | ORB-long ORB[104.41,105.90] vol=2.4x ATR=0.27 |
| Stop hit — per-position SL triggered | 2026-04-27 11:25:00 | 105.69 | 105.45 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 106.31 | 106.04 | 0.00 | ORB-long ORB[105.25,106.30] vol=2.0x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 105.99 | 106.17 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 105.19 | 104.67 | 0.00 | ORB-long ORB[104.00,104.87] vol=3.5x ATR=0.39 |
| Stop hit — per-position SL triggered | 2026-05-05 09:45:00 | 104.80 | 104.75 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:30:00 | 114.25 | 2026-02-11 10:05:00 | 113.84 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-11 09:30:00 | 114.25 | 2026-02-11 10:10:00 | 114.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 09:35:00 | 111.34 | 2026-02-13 09:40:00 | 111.66 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-16 09:30:00 | 111.60 | 2026-02-16 09:35:00 | 111.31 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-18 10:00:00 | 112.61 | 2026-02-18 10:15:00 | 112.23 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-18 10:00:00 | 112.61 | 2026-02-18 11:10:00 | 112.61 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-16 11:00:00 | 104.30 | 2026-04-16 11:35:00 | 103.95 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-27 11:00:00 | 105.96 | 2026-04-27 11:25:00 | 105.69 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-28 09:45:00 | 106.31 | 2026-04-28 11:05:00 | 105.99 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-05 09:35:00 | 105.19 | 2026-05-05 09:45:00 | 104.80 | STOP_HIT | 1.00 | -0.37% |
