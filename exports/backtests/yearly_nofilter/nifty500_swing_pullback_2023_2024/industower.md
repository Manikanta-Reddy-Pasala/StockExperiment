# Indus Towers Ltd. (INDUSTOWER)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 404.30
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 3
- **Target hits / Stop hits / Partials:** 2 / 4 / 3
- **Avg / median % per leg:** 5.13% / 7.39%
- **Sum % (uncompounded):** 46.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 6 | 66.7% | 2 | 4 | 3 | 5.13% | 46.2% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 2 | 4 | 3 | 5.13% | 46.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 6 | 66.7% | 2 | 4 | 3 | 5.13% | 46.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 00:00:00 | 188.00 | 169.39 | 175.51 | Stage2 pullback-breakout RSI=67 vol=1.7x ATR=5.96 |
| Stop hit — per-position SL triggered | 2023-09-21 00:00:00 | 179.06 | 170.54 | 180.11 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 190.15 | 173.87 | 181.26 | Stage2 pullback-breakout RSI=58 vol=2.2x ATR=7.35 |
| Stop hit — per-position SL triggered | 2023-11-16 00:00:00 | 187.00 | 175.01 | 184.28 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-12-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 00:00:00 | 193.80 | 176.59 | 186.51 | Stage2 pullback-breakout RSI=63 vol=3.5x ATR=5.08 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 186.18 | 177.97 | 190.14 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2023-12-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 00:00:00 | 199.05 | 178.56 | 189.47 | Stage2 pullback-breakout RSI=60 vol=4.4x ATR=7.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-04 00:00:00 | 213.75 | 179.69 | 195.53 | T1 booked 50% @ 213.75 |
| Target hit | 2024-02-02 00:00:00 | 216.15 | 186.90 | 217.40 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-02-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 00:00:00 | 241.15 | 191.76 | 221.81 | Stage2 pullback-breakout RSI=66 vol=2.9x ATR=9.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-01 00:00:00 | 260.54 | 194.58 | 233.06 | T1 booked 50% @ 260.54 |
| Stop hit — per-position SL triggered | 2024-03-11 00:00:00 | 247.45 | 198.29 | 244.03 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-03-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 00:00:00 | 271.30 | 202.63 | 247.67 | Stage2 pullback-breakout RSI=65 vol=1.5x ATR=12.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 00:00:00 | 295.80 | 206.19 | 263.19 | T1 booked 50% @ 295.80 |
| Target hit | 2024-05-09 00:00:00 | 327.85 | 235.33 | 336.02 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-08 00:00:00 | 188.00 | 2023-09-21 00:00:00 | 179.06 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest1 | 2023-11-02 00:00:00 | 190.15 | 2023-11-16 00:00:00 | 187.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest1 | 2023-12-07 00:00:00 | 193.80 | 2023-12-20 00:00:00 | 186.18 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest1 | 2023-12-29 00:00:00 | 199.05 | 2024-01-04 00:00:00 | 213.75 | PARTIAL | 0.50 | 7.39% |
| BUY | retest1 | 2023-12-29 00:00:00 | 199.05 | 2024-02-02 00:00:00 | 216.15 | TARGET_HIT | 0.50 | 8.59% |
| BUY | retest1 | 2024-02-23 00:00:00 | 241.15 | 2024-03-01 00:00:00 | 260.54 | PARTIAL | 0.50 | 8.04% |
| BUY | retest1 | 2024-02-23 00:00:00 | 241.15 | 2024-03-11 00:00:00 | 247.45 | STOP_HIT | 0.50 | 2.61% |
| BUY | retest1 | 2024-03-22 00:00:00 | 271.30 | 2024-04-01 00:00:00 | 295.80 | PARTIAL | 0.50 | 9.03% |
| BUY | retest1 | 2024-03-22 00:00:00 | 271.30 | 2024-05-09 00:00:00 | 327.85 | TARGET_HIT | 0.50 | 20.84% |
