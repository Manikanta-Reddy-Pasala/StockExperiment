# Aegis Logistics Ltd. (AEGISLOG)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 722.60
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
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 0 / 6 / 1
- **Avg / median % per leg:** -0.94% / -0.15%
- **Sum % (uncompounded):** -6.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 0 | 6 | 1 | -0.94% | -6.6% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 0 | 6 | 1 | -0.94% | -6.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 0 | 6 | 1 | -0.94% | -6.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 00:00:00 | 817.55 | 601.35 | 787.48 | Stage2 pullback-breakout RSI=55 vol=4.6x ATR=36.42 |
| Stop hit — per-position SL triggered | 2024-08-27 00:00:00 | 762.92 | 610.42 | 786.52 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2024-09-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 00:00:00 | 803.60 | 616.54 | 780.43 | Stage2 pullback-breakout RSI=54 vol=2.4x ATR=34.30 |
| Stop hit — per-position SL triggered | 2024-09-16 00:00:00 | 802.40 | 636.98 | 810.90 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-10-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-29 00:00:00 | 773.80 | 662.37 | 738.88 | Stage2 pullback-breakout RSI=59 vol=2.2x ATR=31.37 |
| Stop hit — per-position SL triggered | 2024-11-12 00:00:00 | 774.75 | 674.89 | 771.63 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-11-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-18 00:00:00 | 842.90 | 678.79 | 781.32 | Stage2 pullback-breakout RSI=66 vol=2.6x ATR=30.27 |
| Stop hit — per-position SL triggered | 2024-11-21 00:00:00 | 797.49 | 681.58 | 788.16 | SL hit (bars_held=2) |

### Cycle 5 — BUY (started 2024-11-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 00:00:00 | 872.20 | 688.94 | 806.42 | Stage2 pullback-breakout RSI=64 vol=2.9x ATR=34.47 |
| Stop hit — per-position SL triggered | 2024-12-03 00:00:00 | 820.50 | 694.01 | 819.93 | SL hit (bars_held=3) |

### Cycle 6 — BUY (started 2024-12-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 00:00:00 | 825.80 | 706.03 | 797.56 | Stage2 pullback-breakout RSI=55 vol=11.1x ATR=37.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 00:00:00 | 899.99 | 716.48 | 810.59 | T1 booked 50% @ 899.99 |
| Stop hit — per-position SL triggered | 2025-01-06 00:00:00 | 846.30 | 716.48 | 810.59 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-20 00:00:00 | 817.55 | 2024-08-27 00:00:00 | 762.92 | STOP_HIT | 1.00 | -6.68% |
| BUY | retest1 | 2024-09-02 00:00:00 | 803.60 | 2024-09-16 00:00:00 | 802.40 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2024-10-29 00:00:00 | 773.80 | 2024-11-12 00:00:00 | 774.75 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest1 | 2024-11-18 00:00:00 | 842.90 | 2024-11-21 00:00:00 | 797.49 | STOP_HIT | 1.00 | -5.39% |
| BUY | retest1 | 2024-11-28 00:00:00 | 872.20 | 2024-12-03 00:00:00 | 820.50 | STOP_HIT | 1.00 | -5.93% |
| BUY | retest1 | 2024-12-20 00:00:00 | 825.80 | 2025-01-06 00:00:00 | 899.99 | PARTIAL | 0.50 | 8.98% |
| BUY | retest1 | 2024-12-20 00:00:00 | 825.80 | 2025-01-06 00:00:00 | 846.30 | STOP_HIT | 0.50 | 2.48% |
