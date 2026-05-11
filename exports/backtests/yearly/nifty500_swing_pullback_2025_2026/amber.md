# Amber Enterprises India Ltd. (AMBER)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 8824.50
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
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 2.18% / 6.60%
- **Sum % (uncompounded):** 13.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.18% | 13.1% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.18% | 13.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.18% | 13.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-02 05:30:00 | 7224.50 | 6161.08 | 6712.74 | Stage2 pullback-breakout RSI=69 vol=2.0x ATR=242.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 05:30:00 | 7708.51 | 6224.78 | 7015.96 | T1 booked 50% @ 7708.51 |
| Stop hit — per-position SL triggered | 2025-07-22 05:30:00 | 7369.00 | 6342.21 | 7344.76 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-07-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 05:30:00 | 7810.50 | 6396.22 | 7394.04 | Stage2 pullback-breakout RSI=65 vol=3.5x ATR=267.94 |
| Stop hit — per-position SL triggered | 2025-08-08 05:30:00 | 7408.59 | 6504.19 | 7596.47 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2025-09-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 05:30:00 | 7631.00 | 6598.67 | 7358.59 | Stage2 pullback-breakout RSI=58 vol=1.6x ATR=251.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 05:30:00 | 8134.89 | 6727.71 | 7714.31 | T1 booked 50% @ 8134.89 |
| Target hit | 2025-10-15 05:30:00 | 8148.50 | 7005.91 | 8178.42 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2025-10-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 05:30:00 | 8476.00 | 7093.26 | 8244.82 | Stage2 pullback-breakout RSI=64 vol=1.5x ATR=218.90 |
| Stop hit — per-position SL triggered | 2025-10-30 05:30:00 | 8147.66 | 7129.12 | 8257.44 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-02 05:30:00 | 7224.50 | 2025-07-09 05:30:00 | 7708.51 | PARTIAL | 0.50 | 6.70% |
| BUY | retest1 | 2025-07-02 05:30:00 | 7224.50 | 2025-07-22 05:30:00 | 7369.00 | STOP_HIT | 0.50 | 2.00% |
| BUY | retest1 | 2025-07-29 05:30:00 | 7810.50 | 2025-08-08 05:30:00 | 7408.59 | STOP_HIT | 1.00 | -5.15% |
| BUY | retest1 | 2025-09-01 05:30:00 | 7631.00 | 2025-09-16 05:30:00 | 8134.89 | PARTIAL | 0.50 | 6.60% |
| BUY | retest1 | 2025-09-01 05:30:00 | 7631.00 | 2025-10-15 05:30:00 | 8148.50 | TARGET_HIT | 0.50 | 6.78% |
| BUY | retest1 | 2025-10-27 05:30:00 | 8476.00 | 2025-10-30 05:30:00 | 8147.66 | STOP_HIT | 1.00 | -3.87% |
