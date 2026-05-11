# Wockhardt Ltd. (WOCKPHARMA)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 1606.50
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 2
- **Avg / median % per leg:** 17.84% / 11.36%
- **Sum % (uncompounded):** 107.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 17.84% | 107.0% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 17.84% | 107.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 17.84% | 107.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 05:30:00 | 248.10 | 217.20 | 240.15 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=8.73 |
| Stop hit — per-position SL triggered | 2023-09-21 05:30:00 | 235.00 | 220.07 | 243.67 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2023-10-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 05:30:00 | 245.10 | 221.68 | 235.65 | Stage2 pullback-breakout RSI=57 vol=4.3x ATR=8.23 |
| Stop hit — per-position SL triggered | 2023-10-23 05:30:00 | 232.76 | 223.20 | 239.36 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2023-11-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 05:30:00 | 238.70 | 223.42 | 232.44 | Stage2 pullback-breakout RSI=54 vol=1.7x ATR=9.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-12 05:30:00 | 257.16 | 224.59 | 237.99 | T1 booked 50% @ 257.16 |
| Target hit | 2024-01-16 05:30:00 | 441.80 | 287.47 | 443.53 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 05:30:00 | 487.20 | 319.58 | 451.41 | Stage2 pullback-breakout RSI=62 vol=4.5x ATR=27.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-19 05:30:00 | 542.56 | 321.89 | 460.96 | T1 booked 50% @ 542.56 |
| Target hit | 2024-03-12 05:30:00 | 551.30 | 360.25 | 554.33 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-06 05:30:00 | 248.10 | 2023-09-21 05:30:00 | 235.00 | STOP_HIT | 1.00 | -5.28% |
| BUY | retest1 | 2023-10-12 05:30:00 | 245.10 | 2023-10-23 05:30:00 | 232.76 | STOP_HIT | 1.00 | -5.04% |
| BUY | retest1 | 2023-11-03 05:30:00 | 238.70 | 2023-11-12 05:30:00 | 257.16 | PARTIAL | 0.50 | 7.74% |
| BUY | retest1 | 2023-11-03 05:30:00 | 238.70 | 2024-01-16 05:30:00 | 441.80 | TARGET_HIT | 0.50 | 85.09% |
| BUY | retest1 | 2024-02-16 05:30:00 | 487.20 | 2024-02-19 05:30:00 | 542.56 | PARTIAL | 0.50 | 11.36% |
| BUY | retest1 | 2024-02-16 05:30:00 | 487.20 | 2024-03-12 05:30:00 | 551.30 | TARGET_HIT | 0.50 | 13.16% |
