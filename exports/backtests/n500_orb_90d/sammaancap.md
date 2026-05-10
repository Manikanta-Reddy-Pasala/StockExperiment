# Sammaan Capital Ltd. (SAMMAANCAP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 148.78
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 2 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 16
- **Target hits / Stop hits / Partials:** 2 / 15 / 11
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 4.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 7 | 43.8% | 1 | 8 | 7 | 0.19% | 3.0% |
| BUY @ 2nd Alert (retest1) | 16 | 7 | 43.8% | 1 | 8 | 7 | 0.19% | 3.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.10% | 1.2% |
| SELL @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.10% | 1.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 12 | 42.9% | 2 | 15 | 11 | 0.15% | 4.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 11:05:00 | 148.48 | 147.50 | 0.00 | ORB-long ORB[146.85,147.95] vol=7.5x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:25:00 | 149.15 | 147.74 | 0.00 | T1 1.5R @ 149.15 |
| Stop hit — per-position SL triggered | 2026-02-11 12:10:00 | 148.48 | 147.93 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:55:00 | 146.70 | 147.21 | 0.00 | ORB-short ORB[147.11,148.59] vol=3.6x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:15:00 | 146.16 | 147.12 | 0.00 | T1 1.5R @ 146.16 |
| Stop hit — per-position SL triggered | 2026-02-12 13:15:00 | 146.70 | 146.91 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:20:00 | 144.62 | 145.03 | 0.00 | ORB-short ORB[144.86,146.50] vol=1.8x ATR=0.48 |
| Stop hit — per-position SL triggered | 2026-02-13 10:30:00 | 145.10 | 145.04 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 148.70 | 148.55 | 0.00 | ORB-long ORB[147.52,148.69] vol=2.0x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:05:00 | 149.38 | 148.69 | 0.00 | T1 1.5R @ 149.38 |
| Stop hit — per-position SL triggered | 2026-02-18 10:20:00 | 148.70 | 148.74 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:15:00 | 157.00 | 155.45 | 0.00 | ORB-long ORB[153.85,155.78] vol=2.0x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:20:00 | 158.31 | 155.97 | 0.00 | T1 1.5R @ 158.31 |
| Stop hit — per-position SL triggered | 2026-02-25 10:45:00 | 157.00 | 156.54 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:40:00 | 152.42 | 153.20 | 0.00 | ORB-short ORB[153.02,154.90] vol=1.6x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:35:00 | 151.65 | 152.94 | 0.00 | T1 1.5R @ 151.65 |
| Stop hit — per-position SL triggered | 2026-02-27 12:10:00 | 152.42 | 152.63 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:15:00 | 140.03 | 140.94 | 0.00 | ORB-short ORB[141.00,143.00] vol=1.5x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:25:00 | 139.36 | 140.77 | 0.00 | T1 1.5R @ 139.36 |
| Stop hit — per-position SL triggered | 2026-03-13 10:30:00 | 140.03 | 140.74 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:30:00 | 140.59 | 139.54 | 0.00 | ORB-long ORB[138.25,139.65] vol=2.0x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:40:00 | 141.29 | 139.84 | 0.00 | T1 1.5R @ 141.29 |
| Stop hit — per-position SL triggered | 2026-03-17 11:25:00 | 140.59 | 140.42 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 142.30 | 141.47 | 0.00 | ORB-long ORB[140.00,141.80] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-03-18 10:10:00 | 141.80 | 142.03 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:10:00 | 150.03 | 148.57 | 0.00 | ORB-long ORB[147.00,148.99] vol=4.5x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 10:30:00 | 151.07 | 148.98 | 0.00 | T1 1.5R @ 151.07 |
| Stop hit — per-position SL triggered | 2026-04-08 11:15:00 | 150.03 | 149.19 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:35:00 | 145.18 | 146.66 | 0.00 | ORB-short ORB[146.56,148.11] vol=2.0x ATR=0.56 |
| Stop hit — per-position SL triggered | 2026-04-23 11:10:00 | 145.74 | 146.21 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 143.63 | 144.57 | 0.00 | ORB-short ORB[144.01,145.48] vol=1.8x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 144.07 | 144.05 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 144.92 | 144.64 | 0.00 | ORB-long ORB[143.00,144.91] vol=4.7x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:45:00 | 145.76 | 144.69 | 0.00 | T1 1.5R @ 145.76 |
| Target hit | 2026-04-28 09:45:00 | 144.66 | 144.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2026-04-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 11:05:00 | 145.64 | 141.95 | 0.00 | ORB-long ORB[138.93,141.09] vol=2.6x ATR=0.99 |
| Stop hit — per-position SL triggered | 2026-04-30 11:10:00 | 144.65 | 142.16 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:50:00 | 145.02 | 145.95 | 0.00 | ORB-short ORB[145.75,147.69] vol=2.1x ATR=0.46 |
| Stop hit — per-position SL triggered | 2026-05-05 11:00:00 | 145.48 | 145.93 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:45:00 | 148.73 | 146.95 | 0.00 | ORB-long ORB[145.76,147.27] vol=7.5x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:50:00 | 149.73 | 147.39 | 0.00 | T1 1.5R @ 149.73 |
| Stop hit — per-position SL triggered | 2026-05-06 10:55:00 | 148.73 | 147.56 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:00:00 | 149.77 | 151.35 | 0.00 | ORB-short ORB[151.50,153.00] vol=1.6x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:55:00 | 149.03 | 150.65 | 0.00 | T1 1.5R @ 149.03 |
| Target hit | 2026-05-08 15:20:00 | 148.66 | 149.66 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 11:05:00 | 148.48 | 2026-02-11 11:25:00 | 149.15 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-11 11:05:00 | 148.48 | 2026-02-11 12:10:00 | 148.48 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 10:55:00 | 146.70 | 2026-02-12 11:15:00 | 146.16 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-12 10:55:00 | 146.70 | 2026-02-12 13:15:00 | 146.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 10:20:00 | 144.62 | 2026-02-13 10:30:00 | 145.10 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-18 09:55:00 | 148.70 | 2026-02-18 10:05:00 | 149.38 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-18 09:55:00 | 148.70 | 2026-02-18 10:20:00 | 148.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:15:00 | 157.00 | 2026-02-25 10:20:00 | 158.31 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2026-02-25 10:15:00 | 157.00 | 2026-02-25 10:45:00 | 157.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:40:00 | 152.42 | 2026-02-27 11:35:00 | 151.65 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-27 10:40:00 | 152.42 | 2026-02-27 12:10:00 | 152.42 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 10:15:00 | 140.03 | 2026-03-13 10:25:00 | 139.36 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-03-13 10:15:00 | 140.03 | 2026-03-13 10:30:00 | 140.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:30:00 | 140.59 | 2026-03-17 10:40:00 | 141.29 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-03-17 10:30:00 | 140.59 | 2026-03-17 11:25:00 | 140.59 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 09:30:00 | 142.30 | 2026-03-18 10:10:00 | 141.80 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-08 10:10:00 | 150.03 | 2026-04-08 10:30:00 | 151.07 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-04-08 10:10:00 | 150.03 | 2026-04-08 11:15:00 | 150.03 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 10:35:00 | 145.18 | 2026-04-23 11:10:00 | 145.74 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-04-24 09:30:00 | 143.63 | 2026-04-24 10:00:00 | 144.07 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-28 09:40:00 | 144.92 | 2026-04-28 09:45:00 | 145.76 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-28 09:40:00 | 144.92 | 2026-04-28 09:45:00 | 144.66 | TARGET_HIT | 0.50 | -0.18% |
| BUY | retest1 | 2026-04-30 11:05:00 | 145.64 | 2026-04-30 11:10:00 | 144.65 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest1 | 2026-05-05 10:50:00 | 145.02 | 2026-05-05 11:00:00 | 145.48 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-05-06 10:45:00 | 148.73 | 2026-05-06 10:50:00 | 149.73 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-05-06 10:45:00 | 148.73 | 2026-05-06 10:55:00 | 148.73 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 10:00:00 | 149.77 | 2026-05-08 10:55:00 | 149.03 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-05-08 10:00:00 | 149.77 | 2026-05-08 15:20:00 | 148.66 | TARGET_HIT | 0.50 | 0.74% |
