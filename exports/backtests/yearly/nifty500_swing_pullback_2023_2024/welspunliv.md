# Welspun Living Ltd. (WELSPUNLIV)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (903 bars)
- **Last close:** 133.80
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 8
- **Target hits / Stop hits / Partials:** 0 / 9 / 1
- **Avg / median % per leg:** -3.18% / -5.09%
- **Sum % (uncompounded):** -31.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 2 | 20.0% | 0 | 9 | 1 | -3.18% | -31.8% |
| BUY @ 2nd Alert (retest1) | 10 | 2 | 20.0% | 0 | 9 | 1 | -3.18% | -31.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 2 | 20.0% | 0 | 9 | 1 | -3.18% | -31.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 05:30:00 | 97.00 | 81.41 | 93.84 | Stage2 pullback-breakout RSI=61 vol=1.8x ATR=2.72 |
| Stop hit — per-position SL triggered | 2023-07-07 05:30:00 | 92.92 | 81.68 | 94.07 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-07-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 05:30:00 | 103.15 | 82.47 | 95.66 | Stage2 pullback-breakout RSI=66 vol=7.9x ATR=3.50 |
| Stop hit — per-position SL triggered | 2023-07-20 05:30:00 | 97.90 | 83.17 | 97.16 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-10-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 05:30:00 | 124.35 | 98.38 | 120.60 | Stage2 pullback-breakout RSI=58 vol=4.8x ATR=4.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 05:30:00 | 132.62 | 99.59 | 123.53 | T1 booked 50% @ 132.62 |
| Stop hit — per-position SL triggered | 2023-10-25 05:30:00 | 124.35 | 101.36 | 128.46 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2023-10-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 05:30:00 | 145.70 | 102.10 | 130.39 | Stage2 pullback-breakout RSI=64 vol=4.7x ATR=7.76 |
| Stop hit — per-position SL triggered | 2023-11-10 05:30:00 | 148.85 | 106.62 | 142.41 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-01-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 05:30:00 | 157.00 | 119.66 | 149.68 | Stage2 pullback-breakout RSI=61 vol=3.8x ATR=5.58 |
| Stop hit — per-position SL triggered | 2024-01-17 05:30:00 | 148.63 | 121.28 | 150.79 | SL hit (bars_held=5) |

### Cycle 6 — BUY (started 2024-01-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 05:30:00 | 163.80 | 123.58 | 149.65 | Stage2 pullback-breakout RSI=64 vol=5.7x ATR=8.03 |
| Stop hit — per-position SL triggered | 2024-02-05 05:30:00 | 151.76 | 124.58 | 151.52 | SL hit (bars_held=3) |

### Cycle 7 — BUY (started 2024-02-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 05:30:00 | 161.20 | 128.39 | 153.76 | Stage2 pullback-breakout RSI=60 vol=4.2x ATR=7.44 |
| Stop hit — per-position SL triggered | 2024-02-29 05:30:00 | 150.04 | 129.46 | 154.34 | SL hit (bars_held=4) |

### Cycle 8 — BUY (started 2024-04-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 05:30:00 | 154.60 | 132.51 | 146.00 | Stage2 pullback-breakout RSI=57 vol=4.0x ATR=8.11 |
| Stop hit — per-position SL triggered | 2024-04-16 05:30:00 | 146.40 | 134.42 | 149.74 | Time-stop (10d <3%) |

### Cycle 9 — BUY (started 2024-04-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 05:30:00 | 161.10 | 135.06 | 150.27 | Stage2 pullback-breakout RSI=63 vol=3.8x ATR=6.84 |
| Stop hit — per-position SL triggered | 2024-04-30 05:30:00 | 150.84 | 135.98 | 151.53 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-05 05:30:00 | 97.00 | 2023-07-07 05:30:00 | 92.92 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest1 | 2023-07-14 05:30:00 | 103.15 | 2023-07-20 05:30:00 | 97.90 | STOP_HIT | 1.00 | -5.09% |
| BUY | retest1 | 2023-10-11 05:30:00 | 124.35 | 2023-10-17 05:30:00 | 132.62 | PARTIAL | 0.50 | 6.65% |
| BUY | retest1 | 2023-10-11 05:30:00 | 124.35 | 2023-10-25 05:30:00 | 124.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-27 05:30:00 | 145.70 | 2023-11-10 05:30:00 | 148.85 | STOP_HIT | 1.00 | 2.16% |
| BUY | retest1 | 2024-01-10 05:30:00 | 157.00 | 2024-01-17 05:30:00 | 148.63 | STOP_HIT | 1.00 | -5.33% |
| BUY | retest1 | 2024-01-31 05:30:00 | 163.80 | 2024-02-05 05:30:00 | 151.76 | STOP_HIT | 1.00 | -7.35% |
| BUY | retest1 | 2024-02-23 05:30:00 | 161.20 | 2024-02-29 05:30:00 | 150.04 | STOP_HIT | 1.00 | -6.92% |
| BUY | retest1 | 2024-04-01 05:30:00 | 154.60 | 2024-04-16 05:30:00 | 146.40 | STOP_HIT | 1.00 | -5.30% |
| BUY | retest1 | 2024-04-23 05:30:00 | 161.10 | 2024-04-30 05:30:00 | 150.84 | STOP_HIT | 1.00 | -6.37% |
