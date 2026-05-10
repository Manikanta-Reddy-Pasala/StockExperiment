# BEL (BEL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 439.50
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
| TARGET_HIT | 0 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 13
- **Target hits / Stop hits / Partials:** 0 / 13 / 3
- **Avg / median % per leg:** -0.12% / -0.25%
- **Sum % (uncompounded):** -1.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.10% | -0.9% |
| BUY @ 2nd Alert (retest1) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.10% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.15% | -1.0% |
| SELL @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.15% | -1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 3 | 18.8% | 0 | 13 | 3 | -0.12% | -1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:55:00 | 435.10 | 438.62 | 0.00 | ORB-short ORB[435.15,441.30] vol=1.5x ATR=1.08 |
| Stop hit — per-position SL triggered | 2026-02-10 11:30:00 | 436.18 | 438.24 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:40:00 | 441.40 | 438.79 | 0.00 | ORB-long ORB[434.60,438.75] vol=2.2x ATR=1.09 |
| Stop hit — per-position SL triggered | 2026-02-12 11:50:00 | 440.31 | 440.92 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:30:00 | 445.65 | 442.62 | 0.00 | ORB-long ORB[440.05,444.90] vol=1.7x ATR=1.42 |
| Stop hit — per-position SL triggered | 2026-02-13 10:40:00 | 444.23 | 442.89 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 433.50 | 434.98 | 0.00 | ORB-short ORB[434.75,437.50] vol=2.1x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-02-24 11:35:00 | 434.71 | 433.70 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:25:00 | 442.95 | 441.53 | 0.00 | ORB-long ORB[440.15,442.55] vol=2.1x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:50:00 | 444.28 | 442.13 | 0.00 | T1 1.5R @ 444.28 |
| Stop hit — per-position SL triggered | 2026-02-26 10:55:00 | 442.95 | 442.21 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:30:00 | 420.55 | 418.61 | 0.00 | ORB-long ORB[416.30,419.90] vol=1.7x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-03-25 10:50:00 | 419.10 | 419.52 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:45:00 | 459.85 | 457.14 | 0.00 | ORB-long ORB[454.45,458.95] vol=2.4x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:05:00 | 461.58 | 457.76 | 0.00 | T1 1.5R @ 461.58 |
| Stop hit — per-position SL triggered | 2026-04-17 11:30:00 | 459.85 | 458.16 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 460.80 | 457.60 | 0.00 | ORB-long ORB[454.15,459.80] vol=2.9x ATR=1.47 |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 459.33 | 458.72 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 446.05 | 447.74 | 0.00 | ORB-short ORB[446.10,451.70] vol=1.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-04-23 10:00:00 | 447.15 | 447.33 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:15:00 | 440.35 | 444.72 | 0.00 | ORB-short ORB[445.80,451.20] vol=2.0x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:50:00 | 438.82 | 444.03 | 0.00 | T1 1.5R @ 438.82 |
| Stop hit — per-position SL triggered | 2026-04-24 12:10:00 | 440.35 | 443.73 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 441.50 | 444.74 | 0.00 | ORB-short ORB[442.70,448.30] vol=1.6x ATR=1.30 |
| Stop hit — per-position SL triggered | 2026-04-27 09:50:00 | 442.80 | 443.93 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 09:30:00 | 429.95 | 431.59 | 0.00 | ORB-short ORB[430.05,436.45] vol=1.7x ATR=1.36 |
| Stop hit — per-position SL triggered | 2026-04-30 09:45:00 | 431.31 | 431.17 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 434.70 | 432.61 | 0.00 | ORB-long ORB[429.30,433.50] vol=1.6x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-05-05 09:45:00 | 433.17 | 433.22 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:55:00 | 435.10 | 2026-02-10 11:30:00 | 436.18 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-12 09:40:00 | 441.40 | 2026-02-12 11:50:00 | 440.31 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-13 10:30:00 | 445.65 | 2026-02-13 10:40:00 | 444.23 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-24 09:35:00 | 433.50 | 2026-02-24 11:35:00 | 434.71 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-26 10:25:00 | 442.95 | 2026-02-26 10:50:00 | 444.28 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-26 10:25:00 | 442.95 | 2026-02-26 10:55:00 | 442.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 09:30:00 | 420.55 | 2026-03-25 10:50:00 | 419.10 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-17 10:45:00 | 459.85 | 2026-04-17 11:05:00 | 461.58 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-04-17 10:45:00 | 459.85 | 2026-04-17 11:30:00 | 459.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:35:00 | 460.80 | 2026-04-21 10:15:00 | 459.33 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-23 09:30:00 | 446.05 | 2026-04-23 10:00:00 | 447.15 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-24 11:15:00 | 440.35 | 2026-04-24 11:50:00 | 438.82 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-24 11:15:00 | 440.35 | 2026-04-24 12:10:00 | 440.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-27 09:30:00 | 441.50 | 2026-04-27 09:50:00 | 442.80 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-30 09:30:00 | 429.95 | 2026-04-30 09:45:00 | 431.31 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-05-05 09:30:00 | 434.70 | 2026-05-05 09:45:00 | 433.17 | STOP_HIT | 1.00 | -0.35% |
