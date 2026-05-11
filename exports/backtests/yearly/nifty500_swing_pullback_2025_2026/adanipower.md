# Adani Power Ltd. (ADANIPOWER)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 225.33
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
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** 0.41% / 0.00%
- **Sum % (uncompounded):** 2.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 0 | 5 | 2 | 0.41% | 2.9% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 0 | 5 | 2 | 0.41% | 2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 0 | 5 | 2 | 0.41% | 2.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 05:30:00 | 126.77 | 114.29 | 120.46 | Stage2 pullback-breakout RSI=69 vol=5.6x ATR=3.26 |
| Stop hit — per-position SL triggered | 2025-09-11 05:30:00 | 121.88 | 114.66 | 122.08 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2025-10-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 05:30:00 | 165.98 | 122.02 | 149.84 | Stage2 pullback-breakout RSI=70 vol=1.7x ATR=6.96 |
| Stop hit — per-position SL triggered | 2025-11-03 05:30:00 | 156.73 | 125.99 | 157.95 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-01-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 05:30:00 | 148.76 | 132.79 | 144.30 | Stage2 pullback-breakout RSI=57 vol=4.9x ATR=3.75 |
| Stop hit — per-position SL triggered | 2026-01-08 05:30:00 | 143.14 | 133.45 | 144.98 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2026-02-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 05:30:00 | 143.62 | 134.20 | 138.78 | Stage2 pullback-breakout RSI=56 vol=3.2x ATR=5.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 05:30:00 | 154.11 | 134.41 | 140.35 | T1 booked 50% @ 154.11 |
| Stop hit — per-position SL triggered | 2026-02-13 05:30:00 | 143.62 | 135.43 | 144.66 | SL hit (bars_held=8) |

### Cycle 5 — BUY (started 2026-03-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 05:30:00 | 149.11 | 136.41 | 141.76 | Stage2 pullback-breakout RSI=62 vol=4.7x ATR=5.17 |
| Stop hit — per-position SL triggered | 2026-03-30 05:30:00 | 150.43 | 138.01 | 148.40 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2026-04-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 05:30:00 | 159.97 | 138.41 | 150.26 | Stage2 pullback-breakout RSI=65 vol=1.8x ATR=6.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 05:30:00 | 172.59 | 139.21 | 154.21 | T1 booked 50% @ 172.59 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-08 05:30:00 | 126.77 | 2025-09-11 05:30:00 | 121.88 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest1 | 2025-10-17 05:30:00 | 165.98 | 2025-11-03 05:30:00 | 156.73 | STOP_HIT | 1.00 | -5.57% |
| BUY | retest1 | 2026-01-01 05:30:00 | 148.76 | 2026-01-08 05:30:00 | 143.14 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest1 | 2026-02-03 05:30:00 | 143.62 | 2026-02-04 05:30:00 | 154.11 | PARTIAL | 0.50 | 7.30% |
| BUY | retest1 | 2026-02-03 05:30:00 | 143.62 | 2026-02-13 05:30:00 | 143.62 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-12 05:30:00 | 149.11 | 2026-03-30 05:30:00 | 150.43 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest1 | 2026-04-02 05:30:00 | 159.97 | 2026-04-08 05:30:00 | 172.59 | PARTIAL | 0.50 | 7.89% |
