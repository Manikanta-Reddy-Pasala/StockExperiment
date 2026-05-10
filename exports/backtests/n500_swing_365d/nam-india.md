# Nippon Life India Asset Management Ltd. (NAM-INDIA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1103.40
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
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 7 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 5
- **Avg / median % per leg:** 3.69% / 5.90%
- **Sum % (uncompounded):** 36.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 7 | 70.0% | 1 | 4 | 5 | 3.69% | 36.9% |
| BUY @ 2nd Alert (retest1) | 10 | 7 | 70.0% | 1 | 4 | 5 | 3.69% | 36.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 7 | 70.0% | 1 | 4 | 5 | 3.69% | 36.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 05:30:00 | 815.10 | 680.78 | 782.75 | Stage2 pullback-breakout RSI=63 vol=2.6x ATR=24.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 05:30:00 | 863.27 | 686.74 | 800.08 | T1 booked 50% @ 863.27 |
| Stop hit — per-position SL triggered | 2025-07-24 05:30:00 | 828.70 | 696.11 | 821.21 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-08-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 05:30:00 | 828.65 | 712.78 | 811.48 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=23.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 05:30:00 | 874.83 | 715.77 | 821.01 | T1 booked 50% @ 874.83 |
| Stop hit — per-position SL triggered | 2025-08-26 05:30:00 | 828.65 | 720.99 | 829.73 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2025-10-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 05:30:00 | 913.30 | 759.40 | 870.78 | Stage2 pullback-breakout RSI=67 vol=1.5x ATR=26.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-21 05:30:00 | 967.22 | 763.36 | 886.98 | T1 booked 50% @ 967.22 |
| Stop hit — per-position SL triggered | 2025-10-27 05:30:00 | 913.30 | 768.18 | 896.98 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2025-12-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 05:30:00 | 915.65 | 796.92 | 858.18 | Stage2 pullback-breakout RSI=65 vol=8.3x ATR=26.53 |
| Stop hit — per-position SL triggered | 2025-12-26 05:30:00 | 875.85 | 801.33 | 869.15 | SL hit (bars_held=5) |

### Cycle 5 — BUY (started 2026-01-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 05:30:00 | 878.10 | 814.69 | 858.94 | Stage2 pullback-breakout RSI=54 vol=4.5x ATR=31.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-03 05:30:00 | 940.80 | 816.84 | 866.78 | T1 booked 50% @ 940.80 |
| Target hit | 2026-02-27 05:30:00 | 931.20 | 837.85 | 936.01 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2026-04-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 05:30:00 | 903.20 | 840.45 | 856.56 | Stage2 pullback-breakout RSI=56 vol=2.9x ATR=40.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 05:30:00 | 984.35 | 846.96 | 901.85 | T1 booked 50% @ 984.35 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-10 05:30:00 | 815.10 | 2025-07-16 05:30:00 | 863.27 | PARTIAL | 0.50 | 5.91% |
| BUY | retest1 | 2025-07-10 05:30:00 | 815.10 | 2025-07-24 05:30:00 | 828.70 | STOP_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2025-08-18 05:30:00 | 828.65 | 2025-08-20 05:30:00 | 874.83 | PARTIAL | 0.50 | 5.57% |
| BUY | retest1 | 2025-08-18 05:30:00 | 828.65 | 2025-08-26 05:30:00 | 828.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-17 05:30:00 | 913.30 | 2025-10-21 05:30:00 | 967.22 | PARTIAL | 0.50 | 5.90% |
| BUY | retest1 | 2025-10-17 05:30:00 | 913.30 | 2025-10-27 05:30:00 | 913.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-18 05:30:00 | 915.65 | 2025-12-26 05:30:00 | 875.85 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest1 | 2026-01-30 05:30:00 | 878.10 | 2026-02-03 05:30:00 | 940.80 | PARTIAL | 0.50 | 7.14% |
| BUY | retest1 | 2026-01-30 05:30:00 | 878.10 | 2026-02-27 05:30:00 | 931.20 | TARGET_HIT | 0.50 | 6.05% |
| BUY | retest1 | 2026-04-08 05:30:00 | 903.20 | 2026-04-17 05:30:00 | 984.35 | PARTIAL | 0.50 | 8.98% |
