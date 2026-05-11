# Biocon Ltd. (BIOCON)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 381.40
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** -0.89% / -3.81%
- **Sum % (uncompounded):** -6.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.89% | -6.3% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.89% | -6.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.89% | -6.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 00:00:00 | 273.15 | 251.99 | 257.48 | Stage2 pullback-breakout RSI=67 vol=3.1x ATR=6.93 |
| Stop hit — per-position SL triggered | 2023-08-10 00:00:00 | 262.75 | 252.26 | 258.93 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2024-01-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 00:00:00 | 287.60 | 252.17 | 269.88 | Stage2 pullback-breakout RSI=68 vol=1.9x ATR=8.77 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 274.44 | 252.72 | 271.72 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-02-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-05 00:00:00 | 284.65 | 254.37 | 270.24 | Stage2 pullback-breakout RSI=63 vol=2.0x ATR=9.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-06 00:00:00 | 303.21 | 254.81 | 272.94 | T1 booked 50% @ 303.21 |
| Stop hit — per-position SL triggered | 2024-02-07 00:00:00 | 284.65 | 255.16 | 274.59 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-02-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 00:00:00 | 291.05 | 256.85 | 276.47 | Stage2 pullback-breakout RSI=61 vol=3.4x ATR=12.05 |
| Stop hit — per-position SL triggered | 2024-02-21 00:00:00 | 272.97 | 257.26 | 276.65 | SL hit (bars_held=2) |

### Cycle 5 — BUY (started 2024-04-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 00:00:00 | 278.30 | 260.07 | 267.71 | Stage2 pullback-breakout RSI=59 vol=2.7x ATR=8.80 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 265.10 | 260.26 | 268.02 | SL hit (bars_held=2) |

### Cycle 6 — BUY (started 2024-04-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 00:00:00 | 280.15 | 260.68 | 268.62 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=9.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 00:00:00 | 298.54 | 261.72 | 275.84 | T1 booked 50% @ 298.54 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-08 00:00:00 | 273.15 | 2023-08-10 00:00:00 | 262.75 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest1 | 2024-01-16 00:00:00 | 287.60 | 2024-01-18 00:00:00 | 274.44 | STOP_HIT | 1.00 | -4.57% |
| BUY | retest1 | 2024-02-05 00:00:00 | 284.65 | 2024-02-06 00:00:00 | 303.21 | PARTIAL | 0.50 | 6.52% |
| BUY | retest1 | 2024-02-05 00:00:00 | 284.65 | 2024-02-07 00:00:00 | 284.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-19 00:00:00 | 291.05 | 2024-02-21 00:00:00 | 272.97 | STOP_HIT | 1.00 | -6.21% |
| BUY | retest1 | 2024-04-10 00:00:00 | 278.30 | 2024-04-15 00:00:00 | 265.10 | STOP_HIT | 1.00 | -4.74% |
| BUY | retest1 | 2024-04-23 00:00:00 | 280.15 | 2024-04-26 00:00:00 | 298.54 | PARTIAL | 0.50 | 6.56% |
