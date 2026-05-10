# Five-Star Business Finance Ltd. (FIVESTAR)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 462.60
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 4
- **Target hits / Stop hits / Partials:** 3 / 4 / 4
- **Avg / median % per leg:** 0.87% / 0.48%
- **Sum % (uncompounded):** 9.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 6 | 85.7% | 3 | 1 | 3 | 1.38% | 9.7% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 3 | 1 | 3 | 1.38% | 9.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.02% | -0.1% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.02% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 7 | 63.6% | 3 | 4 | 4 | 0.87% | 9.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 442.90 | 439.77 | 0.00 | ORB-long ORB[437.20,441.50] vol=1.6x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 12:55:00 | 445.09 | 441.32 | 0.00 | T1 1.5R @ 445.09 |
| Target hit | 2026-02-17 15:20:00 | 447.00 | 444.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 435.65 | 437.93 | 0.00 | ORB-short ORB[436.55,442.00] vol=1.9x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-02-19 10:05:00 | 437.02 | 437.32 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:30:00 | 446.10 | 441.64 | 0.00 | ORB-long ORB[437.25,443.00] vol=3.0x ATR=2.05 |
| Stop hit — per-position SL triggered | 2026-02-20 10:00:00 | 444.05 | 443.22 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:45:00 | 356.15 | 359.72 | 0.00 | ORB-short ORB[363.05,368.00] vol=2.6x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:55:00 | 353.55 | 358.84 | 0.00 | T1 1.5R @ 353.55 |
| Stop hit — per-position SL triggered | 2026-03-13 11:20:00 | 356.15 | 357.45 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 466.45 | 463.05 | 0.00 | ORB-long ORB[457.05,463.95] vol=2.2x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 09:40:00 | 469.10 | 464.92 | 0.00 | T1 1.5R @ 469.10 |
| Target hit | 2026-04-17 10:40:00 | 467.40 | 467.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 474.90 | 471.91 | 0.00 | ORB-long ORB[469.00,473.70] vol=2.3x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:00:00 | 477.20 | 473.94 | 0.00 | T1 1.5R @ 477.20 |
| Target hit | 2026-04-21 12:20:00 | 510.30 | 511.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 465.10 | 470.89 | 0.00 | ORB-short ORB[471.05,476.65] vol=2.6x ATR=2.34 |
| Stop hit — per-position SL triggered | 2026-05-08 09:55:00 | 467.44 | 469.21 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 10:25:00 | 442.90 | 2026-02-17 12:55:00 | 445.09 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-17 10:25:00 | 442.90 | 2026-02-17 15:20:00 | 447.00 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2026-02-19 09:30:00 | 435.65 | 2026-02-19 10:05:00 | 437.02 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-20 09:30:00 | 446.10 | 2026-02-20 10:00:00 | 444.05 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-03-13 10:45:00 | 356.15 | 2026-03-13 10:55:00 | 353.55 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2026-03-13 10:45:00 | 356.15 | 2026-03-13 11:20:00 | 356.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 09:35:00 | 466.45 | 2026-04-17 09:40:00 | 469.10 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-17 09:35:00 | 466.45 | 2026-04-17 10:40:00 | 467.40 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2026-04-21 09:40:00 | 474.90 | 2026-04-21 10:00:00 | 477.20 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-21 09:40:00 | 474.90 | 2026-04-21 12:20:00 | 510.30 | TARGET_HIT | 0.50 | 7.45% |
| SELL | retest1 | 2026-05-08 09:40:00 | 465.10 | 2026-05-08 09:55:00 | 467.44 | STOP_HIT | 1.00 | -0.50% |
