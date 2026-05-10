# Godawari Power & Ispat Ltd. (GPIL)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 294.35
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
| TARGET_HIT | 3 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / Stop hits / Partials:** 3 / 2 / 3
- **Avg / median % per leg:** 5.54% / 6.35%
- **Sum % (uncompounded):** 44.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 5.54% | 44.3% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 3 | 2 | 3 | 5.54% | 44.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 6 | 75.0% | 3 | 2 | 3 | 5.54% | 44.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 05:30:00 | 199.97 | 188.71 | 191.57 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=5.42 |
| Stop hit — per-position SL triggered | 2025-08-06 05:30:00 | 191.84 | 188.73 | 191.47 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2025-08-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 05:30:00 | 204.29 | 189.07 | 193.39 | Stage2 pullback-breakout RSI=63 vol=1.9x ATR=6.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 05:30:00 | 217.26 | 189.61 | 197.11 | T1 booked 50% @ 217.26 |
| Target hit | 2025-09-29 05:30:00 | 249.33 | 203.98 | 251.38 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-12-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 05:30:00 | 247.21 | 223.27 | 241.73 | Stage2 pullback-breakout RSI=53 vol=1.9x ATR=8.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 05:30:00 | 264.09 | 225.64 | 247.11 | T1 booked 50% @ 264.09 |
| Target hit | 2026-01-09 05:30:00 | 258.40 | 229.02 | 259.37 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2026-03-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 05:30:00 | 264.70 | 238.67 | 257.98 | Stage2 pullback-breakout RSI=55 vol=1.6x ATR=10.79 |
| Stop hit — per-position SL triggered | 2026-03-16 05:30:00 | 248.52 | 238.90 | 256.59 | SL hit (bars_held=2) |

### Cycle 5 — BUY (started 2026-03-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 05:30:00 | 277.30 | 240.39 | 259.22 | Stage2 pullback-breakout RSI=60 vol=2.6x ATR=11.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 05:30:00 | 301.05 | 244.81 | 276.62 | T1 booked 50% @ 301.05 |
| Target hit | 2026-05-08 05:30:00 | 294.35 | 252.98 | 295.62 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-05 05:30:00 | 199.97 | 2025-08-06 05:30:00 | 191.84 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest1 | 2025-08-13 05:30:00 | 204.29 | 2025-08-19 05:30:00 | 217.26 | PARTIAL | 0.50 | 6.35% |
| BUY | retest1 | 2025-08-13 05:30:00 | 204.29 | 2025-09-29 05:30:00 | 249.33 | TARGET_HIT | 0.50 | 22.05% |
| BUY | retest1 | 2025-12-15 05:30:00 | 247.21 | 2025-12-30 05:30:00 | 264.09 | PARTIAL | 0.50 | 6.83% |
| BUY | retest1 | 2025-12-15 05:30:00 | 247.21 | 2026-01-09 05:30:00 | 258.40 | TARGET_HIT | 0.50 | 4.53% |
| BUY | retest1 | 2026-03-12 05:30:00 | 264.70 | 2026-03-16 05:30:00 | 248.52 | STOP_HIT | 1.00 | -6.11% |
| BUY | retest1 | 2026-03-25 05:30:00 | 277.30 | 2026-04-15 05:30:00 | 301.05 | PARTIAL | 0.50 | 8.57% |
| BUY | retest1 | 2026-03-25 05:30:00 | 277.30 | 2026-05-08 05:30:00 | 294.35 | TARGET_HIT | 0.50 | 6.15% |
