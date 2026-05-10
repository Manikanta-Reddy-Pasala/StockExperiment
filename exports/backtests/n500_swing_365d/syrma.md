# Syrma SGS Technology Ltd. (SYRMA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1100.15
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
| TARGET_HIT | 3 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 8 / 2
- **Target hits / Stop hits / Partials:** 3 / 2 / 5
- **Avg / median % per leg:** 6.95% / 8.05%
- **Sum % (uncompounded):** 69.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 8 | 80.0% | 3 | 2 | 5 | 6.95% | 69.5% |
| BUY @ 2nd Alert (retest1) | 10 | 8 | 80.0% | 3 | 2 | 5 | 6.95% | 69.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 8 | 80.0% | 3 | 2 | 5 | 6.95% | 69.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 05:30:00 | 556.95 | 495.61 | 529.87 | Stage2 pullback-breakout RSI=63 vol=2.4x ATR=19.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 05:30:00 | 596.85 | 497.23 | 538.52 | T1 booked 50% @ 596.85 |
| Target hit | 2025-08-08 05:30:00 | 702.95 | 546.04 | 705.47 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-09-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 05:30:00 | 791.50 | 573.67 | 735.47 | Stage2 pullback-breakout RSI=70 vol=3.6x ATR=31.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 05:30:00 | 853.63 | 578.81 | 753.27 | T1 booked 50% @ 853.63 |
| Stop hit — per-position SL triggered | 2025-09-17 05:30:00 | 791.50 | 598.14 | 793.28 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2025-10-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 05:30:00 | 851.95 | 624.95 | 811.56 | Stage2 pullback-breakout RSI=60 vol=2.7x ATR=35.96 |
| Stop hit — per-position SL triggered | 2025-10-14 05:30:00 | 798.01 | 634.88 | 817.68 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2025-11-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 05:30:00 | 831.05 | 661.28 | 802.38 | Stage2 pullback-breakout RSI=57 vol=8.5x ATR=35.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 05:30:00 | 901.92 | 665.81 | 818.24 | T1 booked 50% @ 901.92 |
| Target hit | 2025-11-21 05:30:00 | 835.90 | 678.04 | 842.91 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2026-01-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 05:30:00 | 761.00 | 698.54 | 708.12 | Stage2 pullback-breakout RSI=61 vol=8.9x ATR=31.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-03 05:30:00 | 823.90 | 701.12 | 728.67 | T1 booked 50% @ 823.90 |
| Target hit | 2026-02-27 05:30:00 | 822.25 | 726.36 | 827.52 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2026-03-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 05:30:00 | 784.45 | 729.49 | 776.21 | Stage2 pullback-breakout RSI=51 vol=1.9x ATR=36.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 05:30:00 | 856.78 | 738.10 | 799.02 | T1 booked 50% @ 856.78 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-27 05:30:00 | 556.95 | 2025-07-01 05:30:00 | 596.85 | PARTIAL | 0.50 | 7.16% |
| BUY | retest1 | 2025-06-27 05:30:00 | 556.95 | 2025-08-08 05:30:00 | 702.95 | TARGET_HIT | 0.50 | 26.21% |
| BUY | retest1 | 2025-09-03 05:30:00 | 791.50 | 2025-09-05 05:30:00 | 853.63 | PARTIAL | 0.50 | 7.85% |
| BUY | retest1 | 2025-09-03 05:30:00 | 791.50 | 2025-09-17 05:30:00 | 791.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-07 05:30:00 | 851.95 | 2025-10-14 05:30:00 | 798.01 | STOP_HIT | 1.00 | -6.33% |
| BUY | retest1 | 2025-11-11 05:30:00 | 831.05 | 2025-11-13 05:30:00 | 901.92 | PARTIAL | 0.50 | 8.53% |
| BUY | retest1 | 2025-11-11 05:30:00 | 831.05 | 2025-11-21 05:30:00 | 835.90 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2026-01-30 05:30:00 | 761.00 | 2026-02-03 05:30:00 | 823.90 | PARTIAL | 0.50 | 8.27% |
| BUY | retest1 | 2026-01-30 05:30:00 | 761.00 | 2026-02-27 05:30:00 | 822.25 | TARGET_HIT | 0.50 | 8.05% |
| BUY | retest1 | 2026-03-18 05:30:00 | 784.45 | 2026-04-09 05:30:00 | 856.78 | PARTIAL | 0.50 | 9.22% |
