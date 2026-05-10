# Bajaj Housing Finance Ltd. (BAJAJHFL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 87.60
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 12
- **Target hits / Stop hits / Partials:** 3 / 12 / 7
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 2.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.01% | 0.1% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.01% | 0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.16% | 2.4% |
| SELL @ 2nd Alert (retest1) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.16% | 2.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 10 | 45.5% | 3 | 12 | 7 | 0.11% | 2.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 09:30:00 | 90.08 | 90.30 | 0.00 | ORB-short ORB[90.11,91.25] vol=4.0x ATR=0.19 |
| Stop hit — per-position SL triggered | 2026-02-12 09:55:00 | 90.27 | 90.24 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:00:00 | 88.66 | 89.10 | 0.00 | ORB-short ORB[88.83,89.25] vol=1.7x ATR=0.18 |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 88.84 | 89.06 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:35:00 | 88.00 | 88.41 | 0.00 | ORB-short ORB[88.37,88.90] vol=2.4x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:45:00 | 87.77 | 88.26 | 0.00 | T1 1.5R @ 87.77 |
| Target hit | 2026-02-23 15:20:00 | 87.27 | 87.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:50:00 | 86.96 | 87.32 | 0.00 | ORB-short ORB[87.25,87.68] vol=2.0x ATR=0.17 |
| Stop hit — per-position SL triggered | 2026-02-25 09:55:00 | 87.13 | 87.31 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:50:00 | 82.78 | 82.35 | 0.00 | ORB-long ORB[82.02,82.48] vol=2.8x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:15:00 | 83.08 | 82.48 | 0.00 | T1 1.5R @ 83.08 |
| Target hit | 2026-03-20 12:55:00 | 82.88 | 82.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2026-03-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 09:50:00 | 78.18 | 78.96 | 0.00 | ORB-short ORB[78.73,79.69] vol=1.9x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 10:35:00 | 77.64 | 78.61 | 0.00 | T1 1.5R @ 77.64 |
| Stop hit — per-position SL triggered | 2026-03-24 12:40:00 | 78.18 | 77.81 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 09:30:00 | 82.76 | 83.26 | 0.00 | ORB-short ORB[82.82,83.90] vol=1.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2026-04-08 09:35:00 | 83.17 | 83.25 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:50:00 | 90.34 | 89.78 | 0.00 | ORB-long ORB[88.93,89.80] vol=2.8x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:10:00 | 90.75 | 90.06 | 0.00 | T1 1.5R @ 90.75 |
| Stop hit — per-position SL triggered | 2026-04-17 11:25:00 | 90.34 | 90.34 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:40:00 | 90.70 | 91.12 | 0.00 | ORB-short ORB[90.75,91.69] vol=1.6x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 12:40:00 | 90.39 | 90.96 | 0.00 | T1 1.5R @ 90.39 |
| Stop hit — per-position SL triggered | 2026-04-23 13:20:00 | 90.70 | 90.93 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:15:00 | 88.88 | 89.69 | 0.00 | ORB-short ORB[89.82,90.89] vol=2.5x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:50:00 | 88.57 | 89.58 | 0.00 | T1 1.5R @ 88.57 |
| Stop hit — per-position SL triggered | 2026-04-24 12:10:00 | 88.88 | 89.52 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:50:00 | 91.13 | 90.72 | 0.00 | ORB-long ORB[90.07,90.99] vol=1.8x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-04-27 10:30:00 | 90.81 | 90.78 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:05:00 | 87.55 | 88.03 | 0.00 | ORB-short ORB[87.60,88.25] vol=1.6x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:15:00 | 87.19 | 87.82 | 0.00 | T1 1.5R @ 87.19 |
| Target hit | 2026-05-04 15:20:00 | 86.87 | 87.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:50:00 | 87.47 | 86.92 | 0.00 | ORB-long ORB[86.29,87.17] vol=2.1x ATR=0.24 |
| Stop hit — per-position SL triggered | 2026-05-05 10:05:00 | 87.23 | 86.98 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:00:00 | 86.86 | 87.07 | 0.00 | ORB-short ORB[86.88,87.55] vol=1.7x ATR=0.12 |
| Stop hit — per-position SL triggered | 2026-05-06 11:10:00 | 86.98 | 87.07 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:45:00 | 88.55 | 87.95 | 0.00 | ORB-long ORB[87.20,88.00] vol=1.6x ATR=0.23 |
| Stop hit — per-position SL triggered | 2026-05-07 11:05:00 | 88.32 | 87.99 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 09:30:00 | 90.08 | 2026-02-12 09:55:00 | 90.27 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-17 11:00:00 | 88.66 | 2026-02-17 11:15:00 | 88.84 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-23 10:35:00 | 88.00 | 2026-02-23 11:45:00 | 87.77 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-02-23 10:35:00 | 88.00 | 2026-02-23 15:20:00 | 87.27 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2026-02-25 09:50:00 | 86.96 | 2026-02-25 09:55:00 | 87.13 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-03-20 10:50:00 | 82.78 | 2026-03-20 11:15:00 | 83.08 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-03-20 10:50:00 | 82.78 | 2026-03-20 12:55:00 | 82.88 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2026-03-24 09:50:00 | 78.18 | 2026-03-24 10:35:00 | 77.64 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2026-03-24 09:50:00 | 78.18 | 2026-03-24 12:40:00 | 78.18 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-08 09:30:00 | 82.76 | 2026-04-08 09:35:00 | 83.17 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-04-17 09:50:00 | 90.34 | 2026-04-17 10:10:00 | 90.75 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-17 09:50:00 | 90.34 | 2026-04-17 11:25:00 | 90.34 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 10:40:00 | 90.70 | 2026-04-23 12:40:00 | 90.39 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-04-23 10:40:00 | 90.70 | 2026-04-23 13:20:00 | 90.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 11:15:00 | 88.88 | 2026-04-24 11:50:00 | 88.57 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-24 11:15:00 | 88.88 | 2026-04-24 12:10:00 | 88.88 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:50:00 | 91.13 | 2026-04-27 10:30:00 | 90.81 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-05-04 10:05:00 | 87.55 | 2026-05-04 10:15:00 | 87.19 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-05-04 10:05:00 | 87.55 | 2026-05-04 15:20:00 | 86.87 | TARGET_HIT | 0.50 | 0.78% |
| BUY | retest1 | 2026-05-05 09:50:00 | 87.47 | 2026-05-05 10:05:00 | 87.23 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-05-06 11:00:00 | 86.86 | 2026-05-06 11:10:00 | 86.98 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2026-05-07 10:45:00 | 88.55 | 2026-05-07 11:05:00 | 88.32 | STOP_HIT | 1.00 | -0.26% |
