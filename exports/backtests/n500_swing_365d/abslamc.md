# Aditya Birla Sun Life AMC Ltd. (ABSLAMC)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1072.40
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 0.32% / 0.00%
- **Sum % (uncompounded):** 2.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.32% | 2.3% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.32% | 2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.32% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 05:30:00 | 799.85 | 718.89 | 762.37 | Stage2 pullback-breakout RSI=64 vol=3.4x ATR=24.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 05:30:00 | 849.20 | 726.10 | 791.72 | T1 booked 50% @ 849.20 |
| Target hit | 2025-08-01 05:30:00 | 834.60 | 746.88 | 850.89 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-09-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 05:30:00 | 866.55 | 771.39 | 849.77 | Stage2 pullback-breakout RSI=56 vol=2.2x ATR=24.66 |
| Stop hit — per-position SL triggered | 2025-09-17 05:30:00 | 829.56 | 774.72 | 845.20 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2025-10-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 05:30:00 | 857.95 | 780.44 | 820.45 | Stage2 pullback-breakout RSI=62 vol=6.3x ATR=22.33 |
| Stop hit — per-position SL triggered | 2025-10-27 05:30:00 | 824.45 | 786.60 | 837.60 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2026-02-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 05:30:00 | 860.85 | 783.46 | 807.13 | Stage2 pullback-breakout RSI=67 vol=2.0x ATR=27.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 05:30:00 | 916.82 | 789.32 | 842.71 | T1 booked 50% @ 916.82 |
| Stop hit — per-position SL triggered | 2026-03-02 05:30:00 | 860.85 | 797.35 | 874.35 | SL hit (bars_held=13) |

### Cycle 5 — BUY (started 2026-03-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 05:30:00 | 975.75 | 802.69 | 888.59 | Stage2 pullback-breakout RSI=66 vol=7.9x ATR=42.74 |
| Stop hit — per-position SL triggered | 2026-03-23 05:30:00 | 911.64 | 815.70 | 923.59 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-30 05:30:00 | 799.85 | 2025-07-10 05:30:00 | 849.20 | PARTIAL | 0.50 | 6.17% |
| BUY | retest1 | 2025-06-30 05:30:00 | 799.85 | 2025-08-01 05:30:00 | 834.60 | TARGET_HIT | 0.50 | 4.34% |
| BUY | retest1 | 2025-09-10 05:30:00 | 866.55 | 2025-09-17 05:30:00 | 829.56 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest1 | 2025-10-13 05:30:00 | 857.95 | 2025-10-27 05:30:00 | 824.45 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest1 | 2026-02-11 05:30:00 | 860.85 | 2026-02-19 05:30:00 | 916.82 | PARTIAL | 0.50 | 6.50% |
| BUY | retest1 | 2026-02-11 05:30:00 | 860.85 | 2026-03-02 05:30:00 | 860.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 05:30:00 | 975.75 | 2026-03-23 05:30:00 | 911.64 | STOP_HIT | 1.00 | -6.57% |
