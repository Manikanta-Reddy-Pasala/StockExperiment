# NCC Ltd. (NCC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 170.10
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 11
- **Target hits / Stop hits / Partials:** 1 / 11 / 4
- **Avg / median % per leg:** -0.06% / 0.00%
- **Sum % (uncompounded):** -0.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.02% | 0.1% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.02% | 0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.12% | -1.0% |
| SELL @ 2nd Alert (retest1) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.12% | -1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 5 | 31.2% | 1 | 11 | 4 | -0.06% | -0.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 152.10 | 152.97 | 0.00 | ORB-short ORB[152.45,154.12] vol=1.6x ATR=0.54 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 152.64 | 152.77 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:55:00 | 150.64 | 149.66 | 0.00 | ORB-long ORB[149.00,150.31] vol=2.4x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:05:00 | 151.37 | 150.28 | 0.00 | T1 1.5R @ 151.37 |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 150.64 | 150.39 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:35:00 | 152.00 | 152.71 | 0.00 | ORB-short ORB[152.50,153.79] vol=4.9x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:40:00 | 151.35 | 152.37 | 0.00 | T1 1.5R @ 151.35 |
| Stop hit — per-position SL triggered | 2026-02-27 11:05:00 | 152.00 | 152.25 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:05:00 | 144.26 | 142.69 | 0.00 | ORB-long ORB[141.68,143.64] vol=2.4x ATR=0.57 |
| Stop hit — per-position SL triggered | 2026-03-10 11:50:00 | 143.69 | 142.94 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:05:00 | 141.22 | 142.18 | 0.00 | ORB-short ORB[141.35,143.22] vol=3.4x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:25:00 | 140.32 | 141.93 | 0.00 | T1 1.5R @ 140.32 |
| Stop hit — per-position SL triggered | 2026-03-17 13:00:00 | 141.22 | 141.57 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 09:30:00 | 140.40 | 141.37 | 0.00 | ORB-short ORB[140.55,142.50] vol=5.8x ATR=0.75 |
| Stop hit — per-position SL triggered | 2026-04-07 10:45:00 | 141.15 | 140.91 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 151.18 | 149.92 | 0.00 | ORB-long ORB[148.51,150.30] vol=2.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2026-04-10 09:35:00 | 150.62 | 150.07 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:35:00 | 159.84 | 161.73 | 0.00 | ORB-short ORB[161.26,163.56] vol=2.9x ATR=0.69 |
| Stop hit — per-position SL triggered | 2026-04-17 10:45:00 | 160.53 | 161.47 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 159.20 | 159.84 | 0.00 | ORB-short ORB[159.27,160.70] vol=2.2x ATR=0.56 |
| Stop hit — per-position SL triggered | 2026-04-22 11:50:00 | 159.76 | 159.44 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:40:00 | 164.28 | 163.75 | 0.00 | ORB-long ORB[162.22,164.00] vol=1.8x ATR=0.43 |
| Stop hit — per-position SL triggered | 2026-04-27 11:25:00 | 163.85 | 163.80 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 09:45:00 | 166.20 | 166.84 | 0.00 | ORB-short ORB[166.40,168.10] vol=1.7x ATR=0.73 |
| Stop hit — per-position SL triggered | 2026-05-04 11:15:00 | 166.93 | 166.58 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 166.80 | 166.00 | 0.00 | ORB-long ORB[164.95,166.20] vol=1.6x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:10:00 | 167.74 | 166.93 | 0.00 | T1 1.5R @ 167.74 |
| Target hit | 2026-05-05 10:40:00 | 166.95 | 167.07 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 09:30:00 | 152.10 | 2026-02-13 09:40:00 | 152.64 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-25 09:55:00 | 150.64 | 2026-02-25 10:05:00 | 151.37 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-25 09:55:00 | 150.64 | 2026-02-25 10:15:00 | 150.64 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:35:00 | 152.00 | 2026-02-27 10:40:00 | 151.35 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-27 10:35:00 | 152.00 | 2026-02-27 11:05:00 | 152.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 11:05:00 | 144.26 | 2026-03-10 11:50:00 | 143.69 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-03-17 11:05:00 | 141.22 | 2026-03-17 11:25:00 | 140.32 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2026-03-17 11:05:00 | 141.22 | 2026-03-17 13:00:00 | 141.22 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-07 09:30:00 | 140.40 | 2026-04-07 10:45:00 | 141.15 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-04-10 09:30:00 | 151.18 | 2026-04-10 09:35:00 | 150.62 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-17 10:35:00 | 159.84 | 2026-04-17 10:45:00 | 160.53 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-22 09:40:00 | 159.20 | 2026-04-22 11:50:00 | 159.76 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-27 10:40:00 | 164.28 | 2026-04-27 11:25:00 | 163.85 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-05-04 09:45:00 | 166.20 | 2026-05-04 11:15:00 | 166.93 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-05-05 09:30:00 | 166.80 | 2026-05-05 10:10:00 | 167.74 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-05-05 09:30:00 | 166.80 | 2026-05-05 10:40:00 | 166.95 | TARGET_HIT | 0.50 | 0.09% |
