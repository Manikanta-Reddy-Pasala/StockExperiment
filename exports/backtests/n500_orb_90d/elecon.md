# Elecon Engineering Co. Ltd. (ELECON)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 562.40
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
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 5
- **Target hits / Stop hits / Partials:** 3 / 5 / 5
- **Avg / median % per leg:** 0.41% / 0.56%
- **Sum % (uncompounded):** 5.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.19% | 0.9% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.19% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.55% | 4.4% |
| SELL @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.55% | 4.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 8 | 61.5% | 3 | 5 | 5 | 0.41% | 5.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 435.80 | 431.60 | 0.00 | ORB-long ORB[429.25,434.20] vol=2.1x ATR=1.86 |
| Stop hit — per-position SL triggered | 2026-02-16 09:35:00 | 433.94 | 432.26 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 437.65 | 440.06 | 0.00 | ORB-short ORB[439.20,444.20] vol=3.8x ATR=2.09 |
| Stop hit — per-position SL triggered | 2026-02-18 10:45:00 | 439.74 | 439.27 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:05:00 | 420.30 | 423.58 | 0.00 | ORB-short ORB[421.60,426.70] vol=1.9x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:20:00 | 417.88 | 422.92 | 0.00 | T1 1.5R @ 417.88 |
| Target hit | 2026-02-27 15:20:00 | 415.70 | 418.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-03-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:45:00 | 391.80 | 394.26 | 0.00 | ORB-short ORB[394.15,399.15] vol=1.8x ATR=1.95 |
| Stop hit — per-position SL triggered | 2026-03-05 09:50:00 | 393.75 | 394.21 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 09:50:00 | 428.60 | 432.09 | 0.00 | ORB-short ORB[429.85,434.85] vol=1.9x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:40:00 | 425.48 | 430.81 | 0.00 | T1 1.5R @ 425.48 |
| Target hit | 2026-04-10 15:20:00 | 419.40 | 423.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 412.55 | 409.06 | 0.00 | ORB-long ORB[405.40,410.65] vol=3.7x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:10:00 | 414.85 | 410.00 | 0.00 | T1 1.5R @ 414.85 |
| Stop hit — per-position SL triggered | 2026-04-17 10:30:00 | 412.55 | 410.50 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 490.15 | 496.26 | 0.00 | ORB-short ORB[497.00,501.55] vol=2.3x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:35:00 | 486.42 | 493.45 | 0.00 | T1 1.5R @ 486.42 |
| Target hit | 2026-04-28 10:40:00 | 490.00 | 489.88 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — BUY (started 2026-05-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:30:00 | 569.80 | 565.22 | 0.00 | ORB-long ORB[560.00,567.70] vol=1.8x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:50:00 | 574.45 | 566.34 | 0.00 | T1 1.5R @ 574.45 |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 569.80 | 567.17 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 09:30:00 | 435.80 | 2026-02-16 09:35:00 | 433.94 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-02-18 09:55:00 | 437.65 | 2026-02-18 10:45:00 | 439.74 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-02-27 10:05:00 | 420.30 | 2026-02-27 10:20:00 | 417.88 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-02-27 10:05:00 | 420.30 | 2026-02-27 15:20:00 | 415.70 | TARGET_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2026-03-05 09:45:00 | 391.80 | 2026-03-05 09:50:00 | 393.75 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-04-10 09:50:00 | 428.60 | 2026-04-10 10:40:00 | 425.48 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2026-04-10 09:50:00 | 428.60 | 2026-04-10 15:20:00 | 419.40 | TARGET_HIT | 0.50 | 2.15% |
| BUY | retest1 | 2026-04-17 10:05:00 | 412.55 | 2026-04-17 10:10:00 | 414.85 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-17 10:05:00 | 412.55 | 2026-04-17 10:30:00 | 412.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 09:30:00 | 490.15 | 2026-04-28 09:35:00 | 486.42 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2026-04-28 09:30:00 | 490.15 | 2026-04-28 10:40:00 | 490.00 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2026-05-07 10:30:00 | 569.80 | 2026-05-07 10:50:00 | 574.45 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2026-05-07 10:30:00 | 569.80 | 2026-05-07 11:15:00 | 569.80 | STOP_HIT | 0.50 | 0.00% |
