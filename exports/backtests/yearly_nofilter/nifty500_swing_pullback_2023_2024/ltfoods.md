# LT Foods Ltd. (LTFOODS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 429.90
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
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 3.16% / 9.68%
- **Sum % (uncompounded):** 18.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 3.16% | 19.0% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 3.16% | 19.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 3.16% | 19.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 00:00:00 | 176.10 | 130.09 | 169.22 | Stage2 pullback-breakout RSI=60 vol=3.8x ATR=7.00 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 165.60 | 131.73 | 169.94 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 189.85 | 142.34 | 169.72 | Stage2 pullback-breakout RSI=69 vol=6.1x ATR=9.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 00:00:00 | 208.55 | 148.61 | 190.44 | T1 booked 50% @ 208.55 |
| Target hit | 2023-12-18 00:00:00 | 210.50 | 160.26 | 212.00 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 00:00:00 | 204.45 | 169.75 | 202.14 | Stage2 pullback-breakout RSI=52 vol=1.9x ATR=7.46 |
| Stop hit — per-position SL triggered | 2024-02-05 00:00:00 | 193.26 | 171.48 | 198.40 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2024-03-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 00:00:00 | 189.95 | 173.85 | 177.18 | Stage2 pullback-breakout RSI=57 vol=3.1x ATR=9.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 00:00:00 | 208.35 | 175.62 | 188.77 | T1 booked 50% @ 208.35 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 189.95 | 177.57 | 197.83 | SL hit (bars_held=14) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-06 00:00:00 | 176.10 | 2023-09-12 00:00:00 | 165.60 | STOP_HIT | 1.00 | -5.96% |
| BUY | retest1 | 2023-11-02 00:00:00 | 189.85 | 2023-11-20 00:00:00 | 208.55 | PARTIAL | 0.50 | 9.85% |
| BUY | retest1 | 2023-11-02 00:00:00 | 189.85 | 2023-12-18 00:00:00 | 210.50 | TARGET_HIT | 0.50 | 10.88% |
| BUY | retest1 | 2024-01-24 00:00:00 | 204.45 | 2024-02-05 00:00:00 | 193.26 | STOP_HIT | 1.00 | -5.47% |
| BUY | retest1 | 2024-03-21 00:00:00 | 189.95 | 2024-04-04 00:00:00 | 208.35 | PARTIAL | 0.50 | 9.68% |
| BUY | retest1 | 2024-03-21 00:00:00 | 189.95 | 2024-04-15 00:00:00 | 189.95 | STOP_HIT | 0.50 | 0.00% |
