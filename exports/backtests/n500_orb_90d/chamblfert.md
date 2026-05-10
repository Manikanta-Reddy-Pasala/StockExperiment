# Chambal Fertilizers & Chemicals Ltd. (CHAMBLFERT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 455.85
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
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 14
- **Target hits / Stop hits / Partials:** 3 / 14 / 5
- **Avg / median % per leg:** 0.04% / -0.14%
- **Sum % (uncompounded):** 0.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.15% | 2.1% |
| BUY @ 2nd Alert (retest1) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.15% | 2.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 2 | 25.0% | 1 | 6 | 1 | -0.14% | -1.1% |
| SELL @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 1 | 6 | 1 | -0.14% | -1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 8 | 36.4% | 3 | 14 | 5 | 0.04% | 1.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:25:00 | 435.00 | 433.18 | 0.00 | ORB-long ORB[430.00,434.80] vol=3.4x ATR=2.49 |
| Stop hit — per-position SL triggered | 2026-02-09 10:45:00 | 432.51 | 433.23 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:10:00 | 441.90 | 444.29 | 0.00 | ORB-short ORB[442.95,448.00] vol=1.5x ATR=1.73 |
| Stop hit — per-position SL triggered | 2026-02-10 10:20:00 | 443.63 | 443.87 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 451.80 | 453.74 | 0.00 | ORB-short ORB[452.70,457.75] vol=2.6x ATR=1.94 |
| Stop hit — per-position SL triggered | 2026-02-13 10:40:00 | 453.74 | 452.80 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:05:00 | 448.15 | 450.27 | 0.00 | ORB-short ORB[448.30,453.65] vol=5.8x ATR=1.16 |
| Stop hit — per-position SL triggered | 2026-02-16 11:25:00 | 449.31 | 450.22 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 449.00 | 447.14 | 0.00 | ORB-long ORB[443.50,448.40] vol=1.5x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:50:00 | 450.92 | 448.03 | 0.00 | T1 1.5R @ 450.92 |
| Target hit | 2026-02-17 14:15:00 | 449.25 | 450.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2026-02-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:35:00 | 454.80 | 457.52 | 0.00 | ORB-short ORB[454.85,461.40] vol=2.0x ATR=1.46 |
| Stop hit — per-position SL triggered | 2026-02-23 10:20:00 | 456.26 | 456.08 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:10:00 | 459.85 | 457.99 | 0.00 | ORB-long ORB[455.85,459.40] vol=2.7x ATR=0.64 |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 459.21 | 458.01 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:10:00 | 435.20 | 432.83 | 0.00 | ORB-long ORB[429.05,434.05] vol=2.4x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-03-06 10:30:00 | 433.57 | 433.05 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:20:00 | 425.10 | 418.56 | 0.00 | ORB-long ORB[415.10,421.00] vol=2.7x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 10:25:00 | 427.70 | 423.17 | 0.00 | T1 1.5R @ 427.70 |
| Target hit | 2026-03-10 12:20:00 | 433.00 | 433.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-03-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:55:00 | 435.65 | 432.71 | 0.00 | ORB-long ORB[428.70,434.80] vol=2.6x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:05:00 | 439.48 | 434.33 | 0.00 | T1 1.5R @ 439.48 |
| Stop hit — per-position SL triggered | 2026-03-11 10:40:00 | 435.65 | 435.24 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:35:00 | 437.50 | 433.82 | 0.00 | ORB-long ORB[427.00,431.85] vol=3.1x ATR=2.14 |
| Stop hit — per-position SL triggered | 2026-03-18 10:05:00 | 435.36 | 436.47 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:10:00 | 413.65 | 418.33 | 0.00 | ORB-short ORB[417.40,423.00] vol=3.9x ATR=1.62 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 415.27 | 418.11 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-03-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:40:00 | 429.95 | 427.75 | 0.00 | ORB-long ORB[422.80,429.00] vol=1.6x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-03-25 10:55:00 | 428.37 | 427.81 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:55:00 | 455.45 | 456.78 | 0.00 | ORB-short ORB[456.50,461.00] vol=2.4x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:20:00 | 453.17 | 456.31 | 0.00 | T1 1.5R @ 453.17 |
| Target hit | 2026-04-16 14:55:00 | 453.10 | 453.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 455.80 | 453.75 | 0.00 | ORB-long ORB[450.30,455.25] vol=2.3x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-04-21 09:55:00 | 454.52 | 454.02 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 443.50 | 445.79 | 0.00 | ORB-short ORB[444.00,450.20] vol=1.5x ATR=1.41 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 444.91 | 444.81 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 445.90 | 444.02 | 0.00 | ORB-long ORB[441.05,444.50] vol=1.7x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:40:00 | 447.91 | 445.02 | 0.00 | T1 1.5R @ 447.91 |
| Stop hit — per-position SL triggered | 2026-04-27 11:05:00 | 445.90 | 445.21 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:25:00 | 435.00 | 2026-02-09 10:45:00 | 432.51 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2026-02-10 10:10:00 | 441.90 | 2026-02-10 10:20:00 | 443.63 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-02-13 09:30:00 | 451.80 | 2026-02-13 10:40:00 | 453.74 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-02-16 11:05:00 | 448.15 | 2026-02-16 11:25:00 | 449.31 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-17 09:35:00 | 449.00 | 2026-02-17 09:50:00 | 450.92 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-17 09:35:00 | 449.00 | 2026-02-17 14:15:00 | 449.25 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2026-02-23 09:35:00 | 454.80 | 2026-02-23 10:20:00 | 456.26 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-25 11:10:00 | 459.85 | 2026-02-25 11:15:00 | 459.21 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2026-03-06 10:10:00 | 435.20 | 2026-03-06 10:30:00 | 433.57 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-10 10:20:00 | 425.10 | 2026-03-10 10:25:00 | 427.70 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-03-10 10:20:00 | 425.10 | 2026-03-10 12:20:00 | 433.00 | TARGET_HIT | 0.50 | 1.86% |
| BUY | retest1 | 2026-03-11 09:55:00 | 435.65 | 2026-03-11 10:05:00 | 439.48 | PARTIAL | 0.50 | 0.88% |
| BUY | retest1 | 2026-03-11 09:55:00 | 435.65 | 2026-03-11 10:40:00 | 435.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 09:35:00 | 437.50 | 2026-03-18 10:05:00 | 435.36 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-03-24 11:10:00 | 413.65 | 2026-03-24 11:15:00 | 415.27 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-25 10:40:00 | 429.95 | 2026-03-25 10:55:00 | 428.37 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-16 09:55:00 | 455.45 | 2026-04-16 10:20:00 | 453.17 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-16 09:55:00 | 455.45 | 2026-04-16 14:55:00 | 453.10 | TARGET_HIT | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-21 09:40:00 | 455.80 | 2026-04-21 09:55:00 | 454.52 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-24 09:35:00 | 443.50 | 2026-04-24 10:00:00 | 444.91 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-27 09:45:00 | 445.90 | 2026-04-27 10:40:00 | 447.91 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-27 09:45:00 | 445.90 | 2026-04-27 11:05:00 | 445.90 | STOP_HIT | 0.50 | 0.00% |
