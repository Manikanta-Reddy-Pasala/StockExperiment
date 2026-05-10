# Hindustan Petroleum Corporation Ltd. (HINDPETRO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 387.50
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 12
- **Target hits / Stop hits / Partials:** 1 / 12 / 3
- **Avg / median % per leg:** -0.06% / -0.25%
- **Sum % (uncompounded):** -0.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.18% | -1.1% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.18% | -1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 3 | 30.0% | 1 | 7 | 2 | 0.02% | 0.2% |
| SELL @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 1 | 7 | 2 | 0.02% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 4 | 25.0% | 1 | 12 | 3 | -0.06% | -0.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:25:00 | 465.10 | 464.01 | 0.00 | ORB-long ORB[460.50,464.85] vol=2.4x ATR=2.45 |
| Stop hit — per-position SL triggered | 2026-02-09 14:10:00 | 462.65 | 464.73 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 464.35 | 460.84 | 0.00 | ORB-long ORB[457.30,463.95] vol=2.2x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-02-10 09:35:00 | 462.77 | 461.13 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:30:00 | 466.95 | 464.75 | 0.00 | ORB-long ORB[460.50,466.00] vol=1.5x ATR=1.29 |
| Stop hit — per-position SL triggered | 2026-02-11 10:35:00 | 465.66 | 464.96 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:20:00 | 458.10 | 459.61 | 0.00 | ORB-short ORB[459.30,464.60] vol=5.5x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:30:00 | 456.32 | 459.42 | 0.00 | T1 1.5R @ 456.32 |
| Target hit | 2026-02-12 15:20:00 | 452.45 | 455.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-02-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:35:00 | 445.50 | 447.03 | 0.00 | ORB-short ORB[446.10,451.60] vol=2.9x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-02-17 11:35:00 | 446.62 | 446.46 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:45:00 | 454.45 | 456.44 | 0.00 | ORB-short ORB[455.05,458.70] vol=2.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-02-18 10:20:00 | 455.55 | 456.25 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:50:00 | 437.45 | 435.86 | 0.00 | ORB-long ORB[432.60,437.40] vol=2.0x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:55:00 | 439.32 | 436.25 | 0.00 | T1 1.5R @ 439.32 |
| Stop hit — per-position SL triggered | 2026-02-23 12:30:00 | 437.45 | 437.69 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:05:00 | 444.30 | 449.83 | 0.00 | ORB-short ORB[448.00,454.70] vol=1.5x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:10:00 | 441.62 | 447.57 | 0.00 | T1 1.5R @ 441.62 |
| Stop hit — per-position SL triggered | 2026-02-25 10:30:00 | 444.30 | 445.77 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-02-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:50:00 | 436.30 | 439.99 | 0.00 | ORB-short ORB[440.15,444.05] vol=1.6x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-02-27 11:00:00 | 437.51 | 439.85 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 09:35:00 | 321.55 | 323.53 | 0.00 | ORB-short ORB[322.10,325.85] vol=2.5x ATR=1.83 |
| Stop hit — per-position SL triggered | 2026-04-07 10:20:00 | 323.38 | 322.72 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 384.20 | 381.64 | 0.00 | ORB-long ORB[377.55,382.40] vol=1.7x ATR=1.31 |
| Stop hit — per-position SL triggered | 2026-04-21 10:25:00 | 382.89 | 381.93 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:05:00 | 372.35 | 375.51 | 0.00 | ORB-short ORB[373.10,377.35] vol=3.0x ATR=1.36 |
| Stop hit — per-position SL triggered | 2026-04-23 11:25:00 | 373.71 | 375.31 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 370.45 | 373.22 | 0.00 | ORB-short ORB[373.00,376.40] vol=1.9x ATR=1.30 |
| Stop hit — per-position SL triggered | 2026-04-24 09:55:00 | 371.75 | 372.26 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:25:00 | 465.10 | 2026-02-09 14:10:00 | 462.65 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-02-10 09:30:00 | 464.35 | 2026-02-10 09:35:00 | 462.77 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-11 10:30:00 | 466.95 | 2026-02-11 10:35:00 | 465.66 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-12 10:20:00 | 458.10 | 2026-02-12 10:30:00 | 456.32 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-12 10:20:00 | 458.10 | 2026-02-12 15:20:00 | 452.45 | TARGET_HIT | 0.50 | 1.23% |
| SELL | retest1 | 2026-02-17 10:35:00 | 445.50 | 2026-02-17 11:35:00 | 446.62 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-18 09:45:00 | 454.45 | 2026-02-18 10:20:00 | 455.55 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-23 10:50:00 | 437.45 | 2026-02-23 10:55:00 | 439.32 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-23 10:50:00 | 437.45 | 2026-02-23 12:30:00 | 437.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 10:05:00 | 444.30 | 2026-02-25 10:10:00 | 441.62 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-02-25 10:05:00 | 444.30 | 2026-02-25 10:30:00 | 444.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:50:00 | 436.30 | 2026-02-27 11:00:00 | 437.51 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-07 09:35:00 | 321.55 | 2026-04-07 10:20:00 | 323.38 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2026-04-21 10:10:00 | 384.20 | 2026-04-21 10:25:00 | 382.89 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-23 11:05:00 | 372.35 | 2026-04-23 11:25:00 | 373.71 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-24 09:35:00 | 370.45 | 2026-04-24 09:55:00 | 371.75 | STOP_HIT | 1.00 | -0.35% |
