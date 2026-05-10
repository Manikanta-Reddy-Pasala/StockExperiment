# Karur Vysya Bank Ltd. (KARURVYSYA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 304.65
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 3
- **Avg / median % per leg:** 2.41% / 5.08%
- **Sum % (uncompounded):** 16.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 3 | 3 | 2.41% | 16.9% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 3 | 3 | 2.41% | 16.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 3 | 3 | 2.41% | 16.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 05:30:00 | 225.11 | 201.23 | 213.83 | Stage2 pullback-breakout RSI=64 vol=5.0x ATR=6.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 05:30:00 | 237.16 | 203.20 | 221.46 | T1 booked 50% @ 237.16 |
| Target hit | 2025-11-24 05:30:00 | 243.52 | 212.49 | 246.05 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-12-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 05:30:00 | 253.83 | 214.35 | 247.90 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=6.30 |
| Stop hit — per-position SL triggered | 2025-12-03 05:30:00 | 244.38 | 215.05 | 248.28 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2025-12-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 05:30:00 | 262.81 | 219.93 | 249.47 | Stage2 pullback-breakout RSI=68 vol=7.2x ATR=6.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 05:30:00 | 276.17 | 222.63 | 257.60 | T1 booked 50% @ 276.17 |
| Stop hit — per-position SL triggered | 2026-01-09 05:30:00 | 262.81 | 224.52 | 261.84 | SL hit (bars_held=10) |

### Cycle 4 — BUY (started 2026-03-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 05:30:00 | 295.75 | 254.59 | 290.57 | Stage2 pullback-breakout RSI=51 vol=2.9x ATR=12.67 |
| Stop hit — per-position SL triggered | 2026-04-02 05:30:00 | 276.74 | 255.48 | 288.88 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2026-04-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 05:30:00 | 290.70 | 257.82 | 282.38 | Stage2 pullback-breakout RSI=54 vol=2.0x ATR=12.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 05:30:00 | 315.11 | 262.13 | 294.31 | T1 booked 50% @ 315.11 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-08 05:30:00 | 225.11 | 2025-10-20 05:30:00 | 237.16 | PARTIAL | 0.50 | 5.36% |
| BUY | retest1 | 2025-10-08 05:30:00 | 225.11 | 2025-11-24 05:30:00 | 243.52 | TARGET_HIT | 0.50 | 8.18% |
| BUY | retest1 | 2025-12-01 05:30:00 | 253.83 | 2025-12-03 05:30:00 | 244.38 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest1 | 2025-12-26 05:30:00 | 262.81 | 2026-01-05 05:30:00 | 276.17 | PARTIAL | 0.50 | 5.08% |
| BUY | retest1 | 2025-12-26 05:30:00 | 262.81 | 2026-01-09 05:30:00 | 262.81 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-27 05:30:00 | 295.75 | 2026-04-02 05:30:00 | 276.74 | STOP_HIT | 1.00 | -6.43% |
| BUY | retest1 | 2026-04-21 05:30:00 | 290.70 | 2026-05-07 05:30:00 | 315.11 | PARTIAL | 0.50 | 8.40% |
