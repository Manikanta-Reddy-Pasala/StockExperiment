# Apollo Tyres Ltd. (APOLLOTYRE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 408.65
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 15
- **Target hits / Stop hits / Partials:** 3 / 15 / 7
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 2.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 3 | 25.0% | 1 | 9 | 2 | -0.11% | -1.4% |
| BUY @ 2nd Alert (retest1) | 12 | 3 | 25.0% | 1 | 9 | 2 | -0.11% | -1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 7 | 53.8% | 2 | 6 | 5 | 0.29% | 3.8% |
| SELL @ 2nd Alert (retest1) | 13 | 7 | 53.8% | 2 | 6 | 5 | 0.29% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 10 | 40.0% | 3 | 15 | 7 | 0.10% | 2.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:15:00 | 511.50 | 510.37 | 0.00 | ORB-long ORB[504.30,509.50] vol=2.9x ATR=2.45 |
| Stop hit — per-position SL triggered | 2026-02-09 11:20:00 | 509.05 | 510.43 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:15:00 | 493.15 | 496.66 | 0.00 | ORB-short ORB[496.75,500.50] vol=7.3x ATR=1.46 |
| Stop hit — per-position SL triggered | 2026-02-13 11:45:00 | 494.61 | 495.87 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 09:40:00 | 484.60 | 487.82 | 0.00 | ORB-short ORB[488.00,492.00] vol=1.6x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:50:00 | 482.63 | 486.92 | 0.00 | T1 1.5R @ 482.63 |
| Stop hit — per-position SL triggered | 2026-02-16 09:55:00 | 484.60 | 486.77 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:10:00 | 481.65 | 480.07 | 0.00 | ORB-long ORB[476.10,481.50] vol=1.5x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 480.59 | 480.15 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 458.65 | 457.62 | 0.00 | ORB-long ORB[453.95,456.60] vol=1.9x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:10:00 | 460.39 | 457.84 | 0.00 | T1 1.5R @ 460.39 |
| Target hit | 2026-02-25 12:30:00 | 460.40 | 460.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-03-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:00:00 | 438.85 | 436.47 | 0.00 | ORB-long ORB[434.00,438.00] vol=2.9x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-03-10 11:50:00 | 437.32 | 436.99 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 09:30:00 | 409.50 | 406.96 | 0.00 | ORB-long ORB[402.25,408.10] vol=2.2x ATR=1.88 |
| Stop hit — per-position SL triggered | 2026-03-30 09:50:00 | 407.62 | 407.22 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:35:00 | 445.50 | 442.92 | 0.00 | ORB-long ORB[439.15,444.00] vol=1.6x ATR=1.54 |
| Stop hit — per-position SL triggered | 2026-04-10 09:45:00 | 443.96 | 443.27 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:10:00 | 441.00 | 442.91 | 0.00 | ORB-short ORB[442.10,445.25] vol=1.7x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:25:00 | 439.16 | 442.26 | 0.00 | T1 1.5R @ 439.16 |
| Stop hit — per-position SL triggered | 2026-04-16 11:45:00 | 441.00 | 441.16 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 446.75 | 445.12 | 0.00 | ORB-long ORB[441.40,445.00] vol=2.4x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-04-17 09:35:00 | 445.22 | 445.40 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 11:05:00 | 438.15 | 440.31 | 0.00 | ORB-short ORB[440.75,444.55] vol=4.5x ATR=0.68 |
| Stop hit — per-position SL triggered | 2026-04-22 11:50:00 | 438.83 | 439.83 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:45:00 | 426.15 | 430.07 | 0.00 | ORB-short ORB[429.75,434.20] vol=2.5x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:50:00 | 424.75 | 428.77 | 0.00 | T1 1.5R @ 424.75 |
| Stop hit — per-position SL triggered | 2026-04-24 12:10:00 | 426.15 | 428.65 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:10:00 | 429.90 | 428.49 | 0.00 | ORB-long ORB[424.70,429.25] vol=3.2x ATR=0.84 |
| Stop hit — per-position SL triggered | 2026-04-27 11:35:00 | 429.06 | 428.55 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:00:00 | 431.95 | 430.64 | 0.00 | ORB-long ORB[427.50,431.60] vol=2.3x ATR=0.95 |
| Stop hit — per-position SL triggered | 2026-04-28 11:20:00 | 431.00 | 430.57 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 09:35:00 | 415.45 | 417.48 | 0.00 | ORB-short ORB[416.50,420.80] vol=2.0x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:55:00 | 413.27 | 416.52 | 0.00 | T1 1.5R @ 413.27 |
| Target hit | 2026-04-30 15:20:00 | 408.00 | 411.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2026-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:45:00 | 405.35 | 406.06 | 0.00 | ORB-short ORB[406.15,410.00] vol=17.3x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:40:00 | 403.28 | 405.20 | 0.00 | T1 1.5R @ 403.28 |
| Target hit | 2026-05-05 15:20:00 | 403.20 | 404.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 410.00 | 407.14 | 0.00 | ORB-long ORB[403.70,408.40] vol=1.8x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:45:00 | 411.99 | 407.77 | 0.00 | T1 1.5R @ 411.99 |
| Stop hit — per-position SL triggered | 2026-05-06 09:50:00 | 410.00 | 407.87 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:05:00 | 411.00 | 414.38 | 0.00 | ORB-short ORB[412.60,415.70] vol=2.7x ATR=1.27 |
| Stop hit — per-position SL triggered | 2026-05-07 12:10:00 | 412.27 | 413.48 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:15:00 | 511.50 | 2026-02-09 11:20:00 | 509.05 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-02-13 11:15:00 | 493.15 | 2026-02-13 11:45:00 | 494.61 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-16 09:40:00 | 484.60 | 2026-02-16 09:50:00 | 482.63 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-16 09:40:00 | 484.60 | 2026-02-16 09:55:00 | 484.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 11:10:00 | 481.65 | 2026-02-18 11:15:00 | 480.59 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-25 11:00:00 | 458.65 | 2026-02-25 11:10:00 | 460.39 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-25 11:00:00 | 458.65 | 2026-02-25 12:30:00 | 460.40 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2026-03-10 11:00:00 | 438.85 | 2026-03-10 11:50:00 | 437.32 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-03-30 09:30:00 | 409.50 | 2026-03-30 09:50:00 | 407.62 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-04-10 09:35:00 | 445.50 | 2026-04-10 09:45:00 | 443.96 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-16 10:10:00 | 441.00 | 2026-04-16 10:25:00 | 439.16 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-16 10:10:00 | 441.00 | 2026-04-16 11:45:00 | 441.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 09:30:00 | 446.75 | 2026-04-17 09:35:00 | 445.22 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-22 11:05:00 | 438.15 | 2026-04-22 11:50:00 | 438.83 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2026-04-24 10:45:00 | 426.15 | 2026-04-24 11:50:00 | 424.75 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-04-24 10:45:00 | 426.15 | 2026-04-24 12:10:00 | 426.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 11:10:00 | 429.90 | 2026-04-27 11:35:00 | 429.06 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-04-28 11:00:00 | 431.95 | 2026-04-28 11:20:00 | 431.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-30 09:35:00 | 415.45 | 2026-04-30 09:55:00 | 413.27 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-04-30 09:35:00 | 415.45 | 2026-04-30 15:20:00 | 408.00 | TARGET_HIT | 0.50 | 1.79% |
| SELL | retest1 | 2026-05-05 09:45:00 | 405.35 | 2026-05-05 10:40:00 | 403.28 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-05-05 09:45:00 | 405.35 | 2026-05-05 15:20:00 | 403.20 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2026-05-06 09:40:00 | 410.00 | 2026-05-06 09:45:00 | 411.99 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-05-06 09:40:00 | 410.00 | 2026-05-06 09:50:00 | 410.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 11:05:00 | 411.00 | 2026-05-07 12:10:00 | 412.27 | STOP_HIT | 1.00 | -0.31% |
