# Swan Corp Ltd. (SWANCORP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 353.15
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
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 14
- **Target hits / Stop hits / Partials:** 0 / 14 / 3
- **Avg / median % per leg:** -0.16% / -0.33%
- **Sum % (uncompounded):** -2.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.09% | -0.8% |
| BUY @ 2nd Alert (retest1) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.09% | -0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 1 | 12.5% | 0 | 7 | 1 | -0.24% | -1.9% |
| SELL @ 2nd Alert (retest1) | 8 | 1 | 12.5% | 0 | 7 | 1 | -0.24% | -1.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 3 | 17.6% | 0 | 14 | 3 | -0.16% | -2.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:45:00 | 419.50 | 423.04 | 0.00 | ORB-short ORB[420.00,425.00] vol=1.6x ATR=1.44 |
| Stop hit — per-position SL triggered | 2026-02-10 11:00:00 | 420.94 | 422.85 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 412.35 | 414.45 | 0.00 | ORB-short ORB[413.35,417.35] vol=1.7x ATR=1.36 |
| Stop hit — per-position SL triggered | 2026-02-11 10:45:00 | 413.71 | 413.46 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 408.15 | 411.78 | 0.00 | ORB-short ORB[410.50,415.50] vol=1.9x ATR=1.43 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 409.58 | 411.13 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:10:00 | 409.00 | 407.57 | 0.00 | ORB-long ORB[405.30,408.70] vol=1.5x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-02-16 10:25:00 | 407.89 | 407.67 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:35:00 | 412.40 | 409.20 | 0.00 | ORB-long ORB[405.50,410.90] vol=1.9x ATR=1.22 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 411.18 | 409.35 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:55:00 | 406.20 | 407.93 | 0.00 | ORB-short ORB[408.20,411.30] vol=1.9x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:50:00 | 404.97 | 407.26 | 0.00 | T1 1.5R @ 404.97 |
| Stop hit — per-position SL triggered | 2026-02-19 13:05:00 | 406.20 | 406.57 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:45:00 | 400.95 | 404.49 | 0.00 | ORB-short ORB[405.20,410.45] vol=1.8x ATR=1.42 |
| Stop hit — per-position SL triggered | 2026-02-23 11:05:00 | 402.37 | 404.04 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:45:00 | 347.45 | 353.01 | 0.00 | ORB-short ORB[356.40,361.50] vol=2.6x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 349.12 | 352.89 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 352.10 | 349.14 | 0.00 | ORB-long ORB[347.10,349.90] vol=1.6x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:45:00 | 354.24 | 350.03 | 0.00 | T1 1.5R @ 354.24 |
| Stop hit — per-position SL triggered | 2026-04-21 09:50:00 | 352.10 | 350.12 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:50:00 | 353.20 | 348.21 | 0.00 | ORB-long ORB[344.50,348.50] vol=3.1x ATR=1.82 |
| Stop hit — per-position SL triggered | 2026-04-23 09:55:00 | 351.38 | 349.35 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 335.75 | 333.44 | 0.00 | ORB-long ORB[330.75,334.70] vol=2.0x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:50:00 | 338.02 | 335.12 | 0.00 | T1 1.5R @ 338.02 |
| Stop hit — per-position SL triggered | 2026-04-27 13:00:00 | 335.75 | 336.74 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:15:00 | 342.20 | 339.19 | 0.00 | ORB-long ORB[336.60,340.85] vol=1.8x ATR=1.27 |
| Stop hit — per-position SL triggered | 2026-04-28 10:25:00 | 340.93 | 339.39 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 350.90 | 346.94 | 0.00 | ORB-long ORB[342.65,345.75] vol=1.9x ATR=2.21 |
| Stop hit — per-position SL triggered | 2026-05-05 09:50:00 | 348.69 | 348.84 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 352.00 | 353.76 | 0.00 | ORB-short ORB[353.05,356.00] vol=2.0x ATR=1.22 |
| Stop hit — per-position SL triggered | 2026-05-08 09:50:00 | 353.22 | 353.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:45:00 | 419.50 | 2026-02-10 11:00:00 | 420.94 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-11 09:35:00 | 412.35 | 2026-02-11 10:45:00 | 413.71 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-13 09:30:00 | 408.15 | 2026-02-13 09:40:00 | 409.58 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-16 10:10:00 | 409.00 | 2026-02-16 10:25:00 | 407.89 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-17 10:35:00 | 412.40 | 2026-02-17 10:40:00 | 411.18 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-19 10:55:00 | 406.20 | 2026-02-19 11:50:00 | 404.97 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-19 10:55:00 | 406.20 | 2026-02-19 13:05:00 | 406.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 10:45:00 | 400.95 | 2026-02-23 11:05:00 | 402.37 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-13 10:45:00 | 347.45 | 2026-03-13 10:50:00 | 349.12 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-04-21 09:40:00 | 352.10 | 2026-04-21 09:45:00 | 354.24 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-04-21 09:40:00 | 352.10 | 2026-04-21 09:50:00 | 352.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 09:50:00 | 353.20 | 2026-04-23 09:55:00 | 351.38 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-04-27 09:30:00 | 335.75 | 2026-04-27 09:50:00 | 338.02 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-04-27 09:30:00 | 335.75 | 2026-04-27 13:00:00 | 335.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 10:15:00 | 342.20 | 2026-04-28 10:25:00 | 340.93 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-05-05 09:35:00 | 350.90 | 2026-05-05 09:50:00 | 348.69 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest1 | 2026-05-08 09:40:00 | 352.00 | 2026-05-08 09:50:00 | 353.22 | STOP_HIT | 1.00 | -0.35% |
