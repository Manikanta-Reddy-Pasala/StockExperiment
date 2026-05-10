# LT Foods Ltd. (LTFOODS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 427.00
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 12
- **Target hits / Stop hits / Partials:** 4 / 12 / 5
- **Avg / median % per leg:** 0.03% / -0.26%
- **Sum % (uncompounded):** 0.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 7 | 53.8% | 3 | 6 | 4 | 0.15% | 1.9% |
| BUY @ 2nd Alert (retest1) | 13 | 7 | 53.8% | 3 | 6 | 4 | 0.15% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 2 | 25.0% | 1 | 6 | 1 | -0.16% | -1.3% |
| SELL @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 1 | 6 | 1 | -0.16% | -1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 9 | 42.9% | 4 | 12 | 5 | 0.03% | 0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:15:00 | 429.30 | 427.50 | 0.00 | ORB-long ORB[423.35,429.00] vol=1.6x ATR=1.19 |
| Stop hit — per-position SL triggered | 2026-02-17 13:55:00 | 428.11 | 428.18 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 424.45 | 426.54 | 0.00 | ORB-short ORB[426.35,429.40] vol=2.3x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:05:00 | 422.90 | 425.57 | 0.00 | T1 1.5R @ 422.90 |
| Target hit | 2026-02-18 15:05:00 | 423.00 | 422.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 417.20 | 418.94 | 0.00 | ORB-short ORB[418.20,422.05] vol=1.6x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-02-19 09:35:00 | 418.30 | 418.76 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 415.35 | 413.48 | 0.00 | ORB-long ORB[410.15,415.10] vol=1.9x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 13:30:00 | 417.54 | 414.95 | 0.00 | T1 1.5R @ 417.54 |
| Stop hit — per-position SL triggered | 2026-02-20 15:05:00 | 415.35 | 415.24 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 405.00 | 406.86 | 0.00 | ORB-short ORB[405.65,411.65] vol=1.6x ATR=1.43 |
| Stop hit — per-position SL triggered | 2026-02-25 09:45:00 | 406.43 | 406.57 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:55:00 | 390.10 | 386.27 | 0.00 | ORB-long ORB[383.15,388.00] vol=1.9x ATR=2.29 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 387.81 | 387.53 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:35:00 | 390.50 | 385.90 | 0.00 | ORB-long ORB[379.95,385.25] vol=1.5x ATR=1.70 |
| Stop hit — per-position SL triggered | 2026-03-25 10:50:00 | 388.80 | 386.34 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 418.75 | 415.50 | 0.00 | ORB-long ORB[410.50,416.15] vol=8.4x ATR=2.00 |
| Stop hit — per-position SL triggered | 2026-04-15 10:00:00 | 416.75 | 416.29 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:05:00 | 417.50 | 414.92 | 0.00 | ORB-long ORB[414.00,417.45] vol=1.5x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-04-16 11:25:00 | 416.32 | 415.06 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 420.45 | 419.04 | 0.00 | ORB-long ORB[415.65,419.75] vol=1.6x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:45:00 | 422.62 | 420.87 | 0.00 | T1 1.5R @ 422.62 |
| Target hit | 2026-04-21 11:15:00 | 421.80 | 422.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2026-04-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:40:00 | 413.80 | 416.33 | 0.00 | ORB-short ORB[415.30,421.00] vol=1.8x ATR=1.59 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 415.39 | 415.21 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 425.40 | 422.70 | 0.00 | ORB-long ORB[418.40,423.55] vol=3.0x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:35:00 | 427.93 | 423.63 | 0.00 | T1 1.5R @ 427.93 |
| Target hit | 2026-04-27 15:20:00 | 429.05 | 428.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:35:00 | 430.50 | 429.02 | 0.00 | ORB-long ORB[424.00,429.35] vol=2.9x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 09:45:00 | 432.94 | 430.19 | 0.00 | T1 1.5R @ 432.94 |
| Target hit | 2026-04-29 10:30:00 | 433.05 | 434.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2026-05-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:10:00 | 426.00 | 427.80 | 0.00 | ORB-short ORB[428.00,431.00] vol=2.1x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-05-05 12:50:00 | 427.48 | 426.69 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:00:00 | 421.85 | 424.97 | 0.00 | ORB-short ORB[425.25,429.80] vol=1.9x ATR=1.17 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 423.02 | 424.76 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:45:00 | 424.95 | 426.95 | 0.00 | ORB-short ORB[425.90,432.00] vol=1.8x ATR=1.41 |
| Stop hit — per-position SL triggered | 2026-05-07 10:00:00 | 426.36 | 426.68 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 11:15:00 | 429.30 | 2026-02-17 13:55:00 | 428.11 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-18 10:55:00 | 424.45 | 2026-02-18 11:05:00 | 422.90 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-18 10:55:00 | 424.45 | 2026-02-18 15:05:00 | 423.00 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-19 09:30:00 | 417.20 | 2026-02-19 09:35:00 | 418.30 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-20 10:35:00 | 415.35 | 2026-02-20 13:30:00 | 417.54 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-20 10:35:00 | 415.35 | 2026-02-20 15:05:00 | 415.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 09:30:00 | 405.00 | 2026-02-25 09:45:00 | 406.43 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-03-17 09:55:00 | 390.10 | 2026-03-17 10:30:00 | 387.81 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2026-03-25 10:35:00 | 390.50 | 2026-03-25 10:50:00 | 388.80 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-15 09:40:00 | 418.75 | 2026-04-15 10:00:00 | 416.75 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-04-16 11:05:00 | 417.50 | 2026-04-16 11:25:00 | 416.32 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-21 09:35:00 | 420.45 | 2026-04-21 09:45:00 | 422.62 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-21 09:35:00 | 420.45 | 2026-04-21 11:15:00 | 421.80 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2026-04-24 09:40:00 | 413.80 | 2026-04-24 10:00:00 | 415.39 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-27 09:30:00 | 425.40 | 2026-04-27 09:35:00 | 427.93 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-27 09:30:00 | 425.40 | 2026-04-27 15:20:00 | 429.05 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2026-04-29 09:35:00 | 430.50 | 2026-04-29 09:45:00 | 432.94 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-29 09:35:00 | 430.50 | 2026-04-29 10:30:00 | 433.05 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2026-05-05 10:10:00 | 426.00 | 2026-05-05 12:50:00 | 427.48 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-05-06 11:00:00 | 421.85 | 2026-05-06 11:15:00 | 423.02 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-05-07 09:45:00 | 424.95 | 2026-05-07 10:00:00 | 426.36 | STOP_HIT | 1.00 | -0.33% |
