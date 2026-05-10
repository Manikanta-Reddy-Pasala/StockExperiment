# Graphite India Ltd. (GRAPHITE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 752.00
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 1
- **Avg / median % per leg:** -0.11% / -0.44%
- **Sum % (uncompounded):** -0.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 1 | 6 | 1 | -0.11% | -0.9% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 1 | 6 | 1 | -0.11% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 2 | 25.0% | 1 | 6 | 1 | -0.11% | -0.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:30:00 | 672.45 | 663.77 | 0.00 | ORB-long ORB[663.35,670.75] vol=1.8x ATR=3.61 |
| Stop hit — per-position SL triggered | 2026-02-13 10:35:00 | 668.84 | 665.72 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:35:00 | 700.85 | 694.45 | 0.00 | ORB-long ORB[687.00,697.00] vol=2.7x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:45:00 | 705.36 | 700.59 | 0.00 | T1 1.5R @ 705.36 |
| Target hit | 2026-02-25 13:00:00 | 711.55 | 712.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 734.00 | 727.49 | 0.00 | ORB-long ORB[719.40,727.00] vol=3.8x ATR=4.52 |
| Stop hit — per-position SL triggered | 2026-02-26 09:45:00 | 729.48 | 730.53 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:20:00 | 691.80 | 685.43 | 0.00 | ORB-long ORB[678.55,688.75] vol=4.7x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-03-06 10:30:00 | 688.66 | 686.19 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 631.80 | 627.86 | 0.00 | ORB-long ORB[620.80,630.00] vol=2.2x ATR=2.76 |
| Stop hit — per-position SL triggered | 2026-03-18 09:35:00 | 629.04 | 627.96 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 656.00 | 653.13 | 0.00 | ORB-long ORB[645.50,654.40] vol=7.9x ATR=3.78 |
| Stop hit — per-position SL triggered | 2026-04-15 09:35:00 | 652.22 | 653.31 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-05-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:00:00 | 729.25 | 720.48 | 0.00 | ORB-long ORB[716.45,724.00] vol=2.1x ATR=3.16 |
| Stop hit — per-position SL triggered | 2026-05-07 11:05:00 | 726.09 | 721.46 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-13 10:30:00 | 672.45 | 2026-02-13 10:35:00 | 668.84 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2026-02-25 10:35:00 | 700.85 | 2026-02-25 10:45:00 | 705.36 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-02-25 10:35:00 | 700.85 | 2026-02-25 13:00:00 | 711.55 | TARGET_HIT | 0.50 | 1.53% |
| BUY | retest1 | 2026-02-26 09:30:00 | 734.00 | 2026-02-26 09:45:00 | 729.48 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2026-03-06 10:20:00 | 691.80 | 2026-03-06 10:30:00 | 688.66 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-03-18 09:30:00 | 631.80 | 2026-03-18 09:35:00 | 629.04 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-15 09:30:00 | 656.00 | 2026-04-15 09:35:00 | 652.22 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2026-05-07 11:00:00 | 729.25 | 2026-05-07 11:05:00 | 726.09 | STOP_HIT | 1.00 | -0.43% |
