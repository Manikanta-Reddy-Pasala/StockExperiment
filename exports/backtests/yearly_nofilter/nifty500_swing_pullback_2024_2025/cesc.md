# CESC Ltd. (CESC)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 184.41
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
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 0.74% / 1.73%
- **Sum % (uncompounded):** 4.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 0 | 4 | 2 | 0.74% | 4.4% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 4 | 2 | 0.74% | 4.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 4 | 2 | 0.74% | 4.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 00:00:00 | 181.84 | 136.49 | 170.65 | Stage2 pullback-breakout RSI=64 vol=2.4x ATR=8.11 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 169.67 | 137.25 | 171.41 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2024-08-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 00:00:00 | 187.75 | 141.26 | 174.65 | Stage2 pullback-breakout RSI=65 vol=4.3x ATR=7.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 00:00:00 | 203.70 | 143.36 | 181.51 | T1 booked 50% @ 203.70 |
| Stop hit — per-position SL triggered | 2024-09-06 00:00:00 | 187.75 | 147.38 | 188.87 | SL hit (bars_held=12) |

### Cycle 3 — BUY (started 2024-09-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 00:00:00 | 200.83 | 152.65 | 192.46 | Stage2 pullback-breakout RSI=60 vol=1.7x ATR=7.85 |
| Stop hit — per-position SL triggered | 2024-10-04 00:00:00 | 189.05 | 155.85 | 195.87 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2024-12-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 00:00:00 | 184.28 | 164.50 | 178.11 | Stage2 pullback-breakout RSI=57 vol=3.8x ATR=6.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 00:00:00 | 196.71 | 165.32 | 181.82 | T1 booked 50% @ 196.71 |
| Stop hit — per-position SL triggered | 2024-12-17 00:00:00 | 187.46 | 167.09 | 186.41 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-01 00:00:00 | 181.84 | 2024-08-05 00:00:00 | 169.67 | STOP_HIT | 1.00 | -6.69% |
| BUY | retest1 | 2024-08-21 00:00:00 | 187.75 | 2024-08-27 00:00:00 | 203.70 | PARTIAL | 0.50 | 8.50% |
| BUY | retest1 | 2024-08-21 00:00:00 | 187.75 | 2024-09-06 00:00:00 | 187.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-24 00:00:00 | 200.83 | 2024-10-04 00:00:00 | 189.05 | STOP_HIT | 1.00 | -5.86% |
| BUY | retest1 | 2024-12-03 00:00:00 | 184.28 | 2024-12-06 00:00:00 | 196.71 | PARTIAL | 0.50 | 6.74% |
| BUY | retest1 | 2024-12-03 00:00:00 | 184.28 | 2024-12-17 00:00:00 | 187.46 | STOP_HIT | 0.50 | 1.73% |
