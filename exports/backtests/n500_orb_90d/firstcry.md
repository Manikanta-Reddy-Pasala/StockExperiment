# Brainbees Solutions Ltd. (FIRSTCRY)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 234.91
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / Stop hits / Partials:** 1 / 7 / 2
- **Avg / median % per leg:** -0.02% / -0.29%
- **Sum % (uncompounded):** -0.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.55% | -1.7% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.55% | -1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.21% | 1.5% |
| SELL @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.21% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 3 | 30.0% | 1 | 7 | 2 | -0.02% | -0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 248.30 | 245.48 | 0.00 | ORB-long ORB[242.27,245.81] vol=4.9x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-04-10 09:45:00 | 246.77 | 245.72 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:15:00 | 247.51 | 248.84 | 0.00 | ORB-short ORB[249.39,252.42] vol=1.8x ATR=0.72 |
| Stop hit — per-position SL triggered | 2026-04-16 11:30:00 | 248.23 | 248.80 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-04-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:25:00 | 257.45 | 253.67 | 0.00 | ORB-long ORB[251.03,254.78] vol=3.0x ATR=1.38 |
| Stop hit — per-position SL triggered | 2026-04-21 10:30:00 | 256.07 | 254.52 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:40:00 | 248.65 | 251.54 | 0.00 | ORB-short ORB[250.66,254.19] vol=2.4x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 12:40:00 | 247.14 | 250.49 | 0.00 | T1 1.5R @ 247.14 |
| Target hit | 2026-04-28 15:20:00 | 245.81 | 249.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-05-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:55:00 | 238.72 | 240.67 | 0.00 | ORB-short ORB[239.90,242.99] vol=1.5x ATR=0.94 |
| Stop hit — per-position SL triggered | 2026-05-04 11:00:00 | 239.66 | 240.65 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-05-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:05:00 | 233.85 | 235.90 | 0.00 | ORB-short ORB[235.29,238.25] vol=1.8x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:20:00 | 232.22 | 235.45 | 0.00 | T1 1.5R @ 232.22 |
| Stop hit — per-position SL triggered | 2026-05-05 14:55:00 | 233.85 | 233.68 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:50:00 | 239.11 | 236.86 | 0.00 | ORB-long ORB[234.50,236.82] vol=2.0x ATR=1.19 |
| Stop hit — per-position SL triggered | 2026-05-06 10:05:00 | 237.92 | 237.55 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:50:00 | 236.18 | 237.74 | 0.00 | ORB-short ORB[237.25,239.50] vol=2.7x ATR=0.73 |
| Stop hit — per-position SL triggered | 2026-05-08 11:00:00 | 236.91 | 237.65 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-04-10 09:40:00 | 248.30 | 2026-04-10 09:45:00 | 246.77 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2026-04-16 11:15:00 | 247.51 | 2026-04-16 11:30:00 | 248.23 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-21 10:25:00 | 257.45 | 2026-04-21 10:30:00 | 256.07 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2026-04-28 10:40:00 | 248.65 | 2026-04-28 12:40:00 | 247.14 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-04-28 10:40:00 | 248.65 | 2026-04-28 15:20:00 | 245.81 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2026-05-04 10:55:00 | 238.72 | 2026-05-04 11:00:00 | 239.66 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-05-05 10:05:00 | 233.85 | 2026-05-05 10:20:00 | 232.22 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-05-05 10:05:00 | 233.85 | 2026-05-05 14:55:00 | 233.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 09:50:00 | 239.11 | 2026-05-06 10:05:00 | 237.92 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-05-08 10:50:00 | 236.18 | 2026-05-08 11:00:00 | 236.91 | STOP_HIT | 1.00 | -0.31% |
