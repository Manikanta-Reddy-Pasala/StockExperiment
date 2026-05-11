# Godawari Power & Ispat Ltd. (GPIL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 294.35
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 7 / 6
- **Target hits / Stop hits / Partials:** 0 / 8 / 5
- **Avg / median % per leg:** 1.95% / 1.52%
- **Sum % (uncompounded):** 25.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 7 | 53.8% | 0 | 8 | 5 | 1.95% | 25.3% |
| BUY @ 2nd Alert (retest1) | 13 | 7 | 53.8% | 0 | 8 | 5 | 1.95% | 25.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 7 | 53.8% | 0 | 8 | 5 | 1.95% | 25.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 123.86 | 86.59 | 116.54 | Stage2 pullback-breakout RSI=67 vol=2.1x ATR=4.07 |
| Stop hit — per-position SL triggered | 2023-09-07 00:00:00 | 117.76 | 88.43 | 119.55 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-09-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 00:00:00 | 126.71 | 93.18 | 122.06 | Stage2 pullback-breakout RSI=62 vol=2.4x ATR=4.38 |
| Stop hit — per-position SL triggered | 2023-10-06 00:00:00 | 120.14 | 94.33 | 122.17 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-10-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 00:00:00 | 125.82 | 96.22 | 122.46 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 00:00:00 | 133.69 | 96.55 | 123.11 | T1 booked 50% @ 133.69 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 125.82 | 97.14 | 123.67 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2023-11-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 00:00:00 | 128.51 | 100.93 | 123.66 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 00:00:00 | 136.83 | 101.25 | 124.59 | T1 booked 50% @ 136.83 |
| Stop hit — per-position SL triggered | 2023-11-29 00:00:00 | 130.46 | 104.00 | 129.50 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 137.37 | 104.34 | 130.25 | Stage2 pullback-breakout RSI=67 vol=1.9x ATR=4.46 |
| Stop hit — per-position SL triggered | 2023-12-14 00:00:00 | 136.90 | 107.45 | 134.60 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2023-12-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 00:00:00 | 144.23 | 107.82 | 135.51 | Stage2 pullback-breakout RSI=68 vol=3.3x ATR=4.51 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 137.46 | 108.77 | 136.63 | SL hit (bars_held=3) |

### Cycle 7 — BUY (started 2023-12-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 00:00:00 | 147.16 | 109.76 | 138.14 | Stage2 pullback-breakout RSI=66 vol=2.1x ATR=5.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 00:00:00 | 157.25 | 111.80 | 143.67 | T1 booked 50% @ 157.25 |
| Stop hit — per-position SL triggered | 2024-01-16 00:00:00 | 151.33 | 115.83 | 150.14 | Time-stop (10d <3%) |

### Cycle 8 — BUY (started 2024-01-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 00:00:00 | 151.93 | 117.92 | 148.52 | Stage2 pullback-breakout RSI=56 vol=1.8x ATR=5.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 00:00:00 | 163.90 | 120.55 | 150.86 | T1 booked 50% @ 163.90 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 151.93 | 121.27 | 151.98 | SL hit (bars_held=10) |

### Cycle 9 — BUY (started 2024-03-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 00:00:00 | 146.54 | 128.21 | 142.19 | Stage2 pullback-breakout RSI=54 vol=3.0x ATR=6.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 00:00:00 | 159.21 | 130.22 | 148.09 | T1 booked 50% @ 159.21 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-31 00:00:00 | 123.86 | 2023-09-07 00:00:00 | 117.76 | STOP_HIT | 1.00 | -4.92% |
| BUY | retest1 | 2023-09-29 00:00:00 | 126.71 | 2023-10-06 00:00:00 | 120.14 | STOP_HIT | 1.00 | -5.18% |
| BUY | retest1 | 2023-10-17 00:00:00 | 125.82 | 2023-10-18 00:00:00 | 133.69 | PARTIAL | 0.50 | 6.25% |
| BUY | retest1 | 2023-10-17 00:00:00 | 125.82 | 2023-10-20 00:00:00 | 125.82 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-13 00:00:00 | 128.51 | 2023-11-15 00:00:00 | 136.83 | PARTIAL | 0.50 | 6.48% |
| BUY | retest1 | 2023-11-13 00:00:00 | 128.51 | 2023-11-29 00:00:00 | 130.46 | STOP_HIT | 0.50 | 1.52% |
| BUY | retest1 | 2023-11-30 00:00:00 | 137.37 | 2023-12-14 00:00:00 | 136.90 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-12-15 00:00:00 | 144.23 | 2023-12-20 00:00:00 | 137.46 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest1 | 2023-12-26 00:00:00 | 147.16 | 2024-01-02 00:00:00 | 157.25 | PARTIAL | 0.50 | 6.86% |
| BUY | retest1 | 2023-12-26 00:00:00 | 147.16 | 2024-01-16 00:00:00 | 151.33 | STOP_HIT | 0.50 | 2.83% |
| BUY | retest1 | 2024-01-25 00:00:00 | 151.93 | 2024-02-07 00:00:00 | 163.90 | PARTIAL | 0.50 | 7.88% |
| BUY | retest1 | 2024-01-25 00:00:00 | 151.93 | 2024-02-09 00:00:00 | 151.93 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-27 00:00:00 | 146.54 | 2024-04-10 00:00:00 | 159.21 | PARTIAL | 0.50 | 8.64% |
