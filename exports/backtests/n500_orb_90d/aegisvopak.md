# Aegis Vopak Terminals Ltd. (AEGISVOPAK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 211.15
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 5
- **Target hits / Stop hits / Partials:** 2 / 5 / 3
- **Avg / median % per leg:** 0.17% / 0.31%
- **Sum % (uncompounded):** 1.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.01% | -0.0% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.01% | -0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.29% | 1.8% |
| SELL @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.29% | 1.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.17% | 1.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:50:00 | 218.25 | 218.96 | 0.00 | ORB-short ORB[219.40,222.13] vol=3.6x ATR=0.77 |
| Stop hit — per-position SL triggered | 2026-02-11 10:10:00 | 219.02 | 218.90 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:55:00 | 216.00 | 217.20 | 0.00 | ORB-short ORB[216.75,218.95] vol=2.5x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:15:00 | 215.19 | 216.77 | 0.00 | T1 1.5R @ 215.19 |
| Target hit | 2026-02-12 15:20:00 | 212.77 | 214.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:35:00 | 207.80 | 209.35 | 0.00 | ORB-short ORB[209.00,212.00] vol=1.5x ATR=0.89 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 208.69 | 209.23 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 10:45:00 | 208.33 | 209.68 | 0.00 | ORB-short ORB[209.61,212.52] vol=1.7x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 10:55:00 | 206.95 | 209.26 | 0.00 | T1 1.5R @ 206.95 |
| Stop hit — per-position SL triggered | 2026-02-16 14:45:00 | 208.33 | 207.63 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 232.64 | 229.04 | 0.00 | ORB-long ORB[224.52,227.95] vol=3.0x ATR=1.43 |
| Stop hit — per-position SL triggered | 2026-02-25 09:45:00 | 231.21 | 230.04 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 191.13 | 190.71 | 0.00 | ORB-long ORB[189.25,190.95] vol=2.7x ATR=0.73 |
| Stop hit — per-position SL triggered | 2026-03-17 11:25:00 | 190.40 | 190.74 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:45:00 | 197.96 | 196.80 | 0.00 | ORB-long ORB[195.45,197.79] vol=2.1x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:15:00 | 199.26 | 197.19 | 0.00 | T1 1.5R @ 199.26 |
| Target hit | 2026-04-28 12:25:00 | 198.58 | 198.98 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:50:00 | 218.25 | 2026-02-11 10:10:00 | 219.02 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-12 10:55:00 | 216.00 | 2026-02-12 11:15:00 | 215.19 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-12 10:55:00 | 216.00 | 2026-02-12 15:20:00 | 212.77 | TARGET_HIT | 0.50 | 1.50% |
| SELL | retest1 | 2026-02-13 09:35:00 | 207.80 | 2026-02-13 09:40:00 | 208.69 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-02-16 10:45:00 | 208.33 | 2026-02-16 10:55:00 | 206.95 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-02-16 10:45:00 | 208.33 | 2026-02-16 14:45:00 | 208.33 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:35:00 | 232.64 | 2026-02-25 09:45:00 | 231.21 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2026-03-17 11:15:00 | 191.13 | 2026-03-17 11:25:00 | 190.40 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-28 10:45:00 | 197.96 | 2026-04-28 11:15:00 | 199.26 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-04-28 10:45:00 | 197.96 | 2026-04-28 12:25:00 | 198.58 | TARGET_HIT | 0.50 | 0.31% |
