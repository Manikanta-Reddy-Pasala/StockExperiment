# Yes Bank Ltd. (YESBANK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 22.90
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 11
- **Target hits / Stop hits / Partials:** 3 / 11 / 6
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 0.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.15% | -1.4% |
| BUY @ 2nd Alert (retest1) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.15% | -1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 8 | 72.7% | 3 | 3 | 5 | 0.19% | 2.1% |
| SELL @ 2nd Alert (retest1) | 11 | 8 | 72.7% | 3 | 3 | 5 | 0.19% | 2.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 9 | 45.0% | 3 | 11 | 6 | 0.04% | 0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 09:40:00 | 21.14 | 21.20 | 0.00 | ORB-short ORB[21.16,21.32] vol=1.6x ATR=0.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:35:00 | 21.08 | 21.17 | 0.00 | T1 1.5R @ 21.08 |
| Target hit | 2026-02-12 13:35:00 | 21.11 | 21.11 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 20.81 | 20.92 | 0.00 | ORB-short ORB[20.92,21.07] vol=2.2x ATR=0.05 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 20.86 | 20.89 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:30:00 | 21.10 | 20.99 | 0.00 | ORB-long ORB[20.88,20.96] vol=2.1x ATR=0.04 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 21.06 | 21.00 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:40:00 | 20.98 | 21.05 | 0.00 | ORB-short ORB[21.01,21.14] vol=1.6x ATR=0.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:40:00 | 20.92 | 21.00 | 0.00 | T1 1.5R @ 20.92 |
| Target hit | 2026-02-23 15:10:00 | 20.92 | 20.90 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — BUY (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 20.93 | 20.87 | 0.00 | ORB-long ORB[20.75,20.89] vol=1.5x ATR=0.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:30:00 | 20.99 | 20.89 | 0.00 | T1 1.5R @ 20.99 |
| Stop hit — per-position SL triggered | 2026-02-24 11:40:00 | 20.93 | 20.89 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:45:00 | 20.73 | 20.83 | 0.00 | ORB-short ORB[20.80,20.94] vol=2.7x ATR=0.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:50:00 | 20.67 | 20.80 | 0.00 | T1 1.5R @ 20.67 |
| Stop hit — per-position SL triggered | 2026-02-25 10:55:00 | 20.73 | 20.80 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:40:00 | 19.11 | 19.16 | 0.00 | ORB-short ORB[19.12,19.25] vol=1.9x ATR=0.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:50:00 | 19.03 | 19.13 | 0.00 | T1 1.5R @ 19.03 |
| Target hit | 2026-03-13 11:30:00 | 19.06 | 19.05 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — BUY (started 2026-04-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:55:00 | 18.99 | 18.82 | 0.00 | ORB-long ORB[18.68,18.88] vol=2.2x ATR=0.06 |
| Stop hit — per-position SL triggered | 2026-04-08 11:40:00 | 18.93 | 18.84 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 19.60 | 19.52 | 0.00 | ORB-long ORB[19.37,19.57] vol=3.1x ATR=0.05 |
| Stop hit — per-position SL triggered | 2026-04-16 09:35:00 | 19.55 | 19.53 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 19.99 | 19.86 | 0.00 | ORB-long ORB[19.67,19.91] vol=1.7x ATR=0.05 |
| Stop hit — per-position SL triggered | 2026-04-22 09:55:00 | 19.94 | 19.87 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:50:00 | 20.01 | 19.92 | 0.00 | ORB-long ORB[19.86,20.00] vol=1.6x ATR=0.05 |
| Stop hit — per-position SL triggered | 2026-04-23 10:45:00 | 19.96 | 19.95 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:50:00 | 19.91 | 19.98 | 0.00 | ORB-short ORB[19.96,20.15] vol=1.6x ATR=0.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 13:10:00 | 19.84 | 19.95 | 0.00 | T1 1.5R @ 19.84 |
| Stop hit — per-position SL triggered | 2026-04-24 14:40:00 | 19.91 | 19.93 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:00:00 | 20.02 | 19.94 | 0.00 | ORB-long ORB[19.88,19.98] vol=1.7x ATR=0.03 |
| Stop hit — per-position SL triggered | 2026-04-28 10:20:00 | 19.99 | 19.95 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:55:00 | 20.19 | 20.08 | 0.00 | ORB-long ORB[20.00,20.10] vol=4.0x ATR=0.04 |
| Stop hit — per-position SL triggered | 2026-04-29 11:00:00 | 20.15 | 20.09 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 09:40:00 | 21.14 | 2026-02-12 10:35:00 | 21.08 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-12 09:40:00 | 21.14 | 2026-02-12 13:35:00 | 21.11 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2026-02-13 09:30:00 | 20.81 | 2026-02-13 09:40:00 | 20.86 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-17 10:30:00 | 21.10 | 2026-02-17 10:40:00 | 21.06 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-23 09:40:00 | 20.98 | 2026-02-23 10:40:00 | 20.92 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-02-23 09:40:00 | 20.98 | 2026-02-23 15:10:00 | 20.92 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2026-02-24 11:10:00 | 20.93 | 2026-02-24 11:30:00 | 20.99 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-02-24 11:10:00 | 20.93 | 2026-02-24 11:40:00 | 20.93 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 10:45:00 | 20.73 | 2026-02-25 10:50:00 | 20.67 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-25 10:45:00 | 20.73 | 2026-02-25 10:55:00 | 20.73 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 09:40:00 | 19.11 | 2026-03-13 09:50:00 | 19.03 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-13 09:40:00 | 19.11 | 2026-03-13 11:30:00 | 19.06 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2026-04-08 10:55:00 | 18.99 | 2026-04-08 11:40:00 | 18.93 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-16 09:30:00 | 19.60 | 2026-04-16 09:35:00 | 19.55 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-22 09:45:00 | 19.99 | 2026-04-22 09:55:00 | 19.94 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-23 09:50:00 | 20.01 | 2026-04-23 10:45:00 | 19.96 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-04-24 10:50:00 | 19.91 | 2026-04-24 13:10:00 | 19.84 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-04-24 10:50:00 | 19.91 | 2026-04-24 14:40:00 | 19.91 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 10:00:00 | 20.02 | 2026-04-28 10:20:00 | 19.99 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-04-29 10:55:00 | 20.19 | 2026-04-29 11:00:00 | 20.15 | STOP_HIT | 1.00 | -0.21% |
