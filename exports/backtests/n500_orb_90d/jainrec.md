# Jain Resource Recycling Ltd. (JAINREC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 565.30
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 5
- **Target hits / Stop hits / Partials:** 3 / 5 / 4
- **Avg / median % per leg:** 0.40% / 0.30%
- **Sum % (uncompounded):** 4.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.45% | 2.2% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.45% | 2.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.37% | 2.6% |
| SELL @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.37% | 2.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 7 | 58.3% | 3 | 5 | 4 | 0.40% | 4.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 402.15 | 405.68 | 0.00 | ORB-short ORB[404.90,410.00] vol=2.3x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:40:00 | 399.23 | 405.24 | 0.00 | T1 1.5R @ 399.23 |
| Target hit | 2026-02-11 14:45:00 | 391.70 | 391.11 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2026-02-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:50:00 | 372.50 | 375.47 | 0.00 | ORB-short ORB[376.55,380.00] vol=1.8x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-02-24 10:35:00 | 373.84 | 374.49 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:30:00 | 441.60 | 444.87 | 0.00 | ORB-short ORB[443.20,449.60] vol=2.1x ATR=1.97 |
| Stop hit — per-position SL triggered | 2026-04-09 09:35:00 | 443.57 | 444.74 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:05:00 | 420.15 | 421.38 | 0.00 | ORB-short ORB[420.55,423.65] vol=2.5x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:30:00 | 418.91 | 421.22 | 0.00 | T1 1.5R @ 418.91 |
| Target hit | 2026-04-16 15:20:00 | 419.50 | 419.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 424.85 | 423.01 | 0.00 | ORB-long ORB[419.00,423.20] vol=8.0x ATR=1.84 |
| Stop hit — per-position SL triggered | 2026-04-22 10:30:00 | 423.01 | 423.50 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 412.15 | 413.52 | 0.00 | ORB-short ORB[412.50,417.00] vol=1.5x ATR=1.70 |
| Stop hit — per-position SL triggered | 2026-04-23 09:40:00 | 413.85 | 413.41 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:05:00 | 429.00 | 424.44 | 0.00 | ORB-long ORB[419.05,424.75] vol=7.9x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:15:00 | 432.02 | 425.80 | 0.00 | T1 1.5R @ 432.02 |
| Target hit | 2026-04-28 11:25:00 | 434.75 | 435.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2026-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:35:00 | 464.00 | 461.09 | 0.00 | ORB-long ORB[457.20,462.80] vol=3.2x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:40:00 | 466.91 | 462.32 | 0.00 | T1 1.5R @ 466.91 |
| Stop hit — per-position SL triggered | 2026-05-06 09:45:00 | 464.00 | 462.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:35:00 | 402.15 | 2026-02-11 09:40:00 | 399.23 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2026-02-11 09:35:00 | 402.15 | 2026-02-11 14:45:00 | 391.70 | TARGET_HIT | 0.50 | 2.60% |
| SELL | retest1 | 2026-02-24 09:50:00 | 372.50 | 2026-02-24 10:35:00 | 373.84 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-09 09:30:00 | 441.60 | 2026-04-09 09:35:00 | 443.57 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-16 11:05:00 | 420.15 | 2026-04-16 11:30:00 | 418.91 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-04-16 11:05:00 | 420.15 | 2026-04-16 15:20:00 | 419.50 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2026-04-22 09:35:00 | 424.85 | 2026-04-22 10:30:00 | 423.01 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-23 09:35:00 | 412.15 | 2026-04-23 09:40:00 | 413.85 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-28 10:05:00 | 429.00 | 2026-04-28 10:15:00 | 432.02 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-04-28 10:05:00 | 429.00 | 2026-04-28 11:25:00 | 434.75 | TARGET_HIT | 0.50 | 1.34% |
| BUY | retest1 | 2026-05-06 09:35:00 | 464.00 | 2026-05-06 09:40:00 | 466.91 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-05-06 09:35:00 | 464.00 | 2026-05-06 09:45:00 | 464.00 | STOP_HIT | 0.50 | 0.00% |
