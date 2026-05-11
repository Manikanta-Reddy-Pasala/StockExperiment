# Rail Vikas Nigam Ltd. (RVNL)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 305.00
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
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 0 / 6 / 1
- **Avg / median % per leg:** -1.18% / -0.12%
- **Sum % (uncompounded):** -8.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 6 | 1 | -1.18% | -8.3% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 6 | 1 | -1.18% | -8.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 0 | 6 | 1 | -1.18% | -8.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 05:30:00 | 128.70 | 86.95 | 121.73 | Stage2 pullback-breakout RSI=67 vol=5.8x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 05:30:00 | 136.72 | 87.44 | 123.15 | T1 booked 50% @ 136.72 |
| Stop hit — per-position SL triggered | 2023-07-25 05:30:00 | 128.70 | 88.34 | 124.84 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2023-11-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 05:30:00 | 166.85 | 123.68 | 159.40 | Stage2 pullback-breakout RSI=59 vol=3.5x ATR=5.74 |
| Stop hit — per-position SL triggered | 2023-12-05 05:30:00 | 170.00 | 128.17 | 164.57 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-12-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 05:30:00 | 178.35 | 129.91 | 167.32 | Stage2 pullback-breakout RSI=69 vol=3.1x ATR=5.62 |
| Stop hit — per-position SL triggered | 2023-12-21 05:30:00 | 169.92 | 133.72 | 173.91 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2024-02-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 05:30:00 | 266.25 | 170.02 | 261.06 | Stage2 pullback-breakout RSI=54 vol=1.7x ATR=19.78 |
| Stop hit — per-position SL triggered | 2024-03-02 05:30:00 | 250.95 | 178.45 | 258.48 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-03-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 05:30:00 | 260.45 | 187.45 | 247.80 | Stage2 pullback-breakout RSI=57 vol=2.4x ATR=14.50 |
| Stop hit — per-position SL triggered | 2024-04-10 05:30:00 | 260.15 | 194.50 | 256.73 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-04-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 05:30:00 | 277.35 | 199.02 | 259.61 | Stage2 pullback-breakout RSI=65 vol=2.5x ATR=10.69 |
| Stop hit — per-position SL triggered | 2024-05-08 05:30:00 | 261.31 | 206.93 | 272.75 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-20 05:30:00 | 128.70 | 2023-07-21 05:30:00 | 136.72 | PARTIAL | 0.50 | 6.23% |
| BUY | retest1 | 2023-07-20 05:30:00 | 128.70 | 2023-07-25 05:30:00 | 128.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-17 05:30:00 | 166.85 | 2023-12-05 05:30:00 | 170.00 | STOP_HIT | 1.00 | 1.89% |
| BUY | retest1 | 2023-12-11 05:30:00 | 178.35 | 2023-12-21 05:30:00 | 169.92 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest1 | 2024-02-19 05:30:00 | 266.25 | 2024-03-02 05:30:00 | 250.95 | STOP_HIT | 1.00 | -5.75% |
| BUY | retest1 | 2024-03-26 05:30:00 | 260.45 | 2024-04-10 05:30:00 | 260.15 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2024-04-23 05:30:00 | 277.35 | 2024-05-08 05:30:00 | 261.31 | STOP_HIT | 1.00 | -5.78% |
