# ZF Commercial Vehicle Control Systems India Ltd. (ZFCVINDIA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 14571.00
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 1
- **Avg / median % per leg:** -1.49% / -3.24%
- **Sum % (uncompounded):** -10.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | -1.49% | -10.4% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | -1.49% | -10.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 1 | 5 | 1 | -1.49% | -10.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 05:30:00 | 13587.00 | 13116.14 | 13247.22 | Stage2 pullback-breakout RSI=60 vol=4.2x ATR=355.39 |
| Stop hit — per-position SL triggered | 2025-07-28 05:30:00 | 13053.92 | 13121.74 | 13273.49 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2025-08-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 05:30:00 | 13604.00 | 13131.33 | 13284.94 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=374.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 05:30:00 | 14352.86 | 13163.67 | 13455.68 | T1 booked 50% @ 14352.86 |
| Target hit | 2025-09-03 05:30:00 | 13802.00 | 13291.90 | 14010.10 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-10-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 05:30:00 | 13569.00 | 13293.67 | 13268.97 | Stage2 pullback-breakout RSI=54 vol=4.2x ATR=396.47 |
| Stop hit — per-position SL triggered | 2025-10-14 05:30:00 | 12974.30 | 13295.70 | 13281.58 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2025-12-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 05:30:00 | 15030.00 | 13342.26 | 14081.82 | Stage2 pullback-breakout RSI=69 vol=1.8x ATR=500.98 |
| Stop hit — per-position SL triggered | 2025-12-31 05:30:00 | 14878.00 | 13496.35 | 14620.97 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2026-02-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 05:30:00 | 15950.00 | 13781.65 | 14874.30 | Stage2 pullback-breakout RSI=68 vol=2.9x ATR=513.13 |
| Stop hit — per-position SL triggered | 2026-02-20 05:30:00 | 15180.30 | 13908.49 | 15246.54 | SL hit (bars_held=7) |

### Cycle 6 — BUY (started 2026-04-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 05:30:00 | 15032.00 | 13989.60 | 14198.81 | Stage2 pullback-breakout RSI=62 vol=2.5x ATR=574.65 |
| Stop hit — per-position SL triggered | 2026-05-05 05:30:00 | 14545.00 | 14077.48 | 14612.35 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-24 05:30:00 | 13587.00 | 2025-07-28 05:30:00 | 13053.92 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest1 | 2025-08-05 05:30:00 | 13604.00 | 2025-08-14 05:30:00 | 14352.86 | PARTIAL | 0.50 | 5.50% |
| BUY | retest1 | 2025-08-05 05:30:00 | 13604.00 | 2025-09-03 05:30:00 | 13802.00 | TARGET_HIT | 0.50 | 1.46% |
| BUY | retest1 | 2025-10-03 05:30:00 | 13569.00 | 2025-10-14 05:30:00 | 12974.30 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest1 | 2025-12-16 05:30:00 | 15030.00 | 2025-12-31 05:30:00 | 14878.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest1 | 2026-02-11 05:30:00 | 15950.00 | 2026-02-20 05:30:00 | 15180.30 | STOP_HIT | 1.00 | -4.83% |
| BUY | retest1 | 2026-04-20 05:30:00 | 15032.00 | 2026-05-05 05:30:00 | 14545.00 | STOP_HIT | 1.00 | -3.24% |
