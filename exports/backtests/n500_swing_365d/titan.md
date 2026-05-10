# Titan Company Ltd. (TITAN)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 4509.00
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
| TARGET_HIT | 3 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 7 / 1
- **Target hits / Stop hits / Partials:** 3 / 2 / 3
- **Avg / median % per leg:** 3.19% / 3.72%
- **Sum % (uncompounded):** 25.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 7 | 87.5% | 3 | 2 | 3 | 3.19% | 25.6% |
| BUY @ 2nd Alert (retest1) | 8 | 7 | 87.5% | 3 | 2 | 3 | 3.19% | 25.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 7 | 87.5% | 3 | 2 | 3 | 3.19% | 25.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 05:30:00 | 3652.20 | 3406.80 | 3513.71 | Stage2 pullback-breakout RSI=66 vol=2.9x ATR=69.86 |
| Stop hit — per-position SL triggered | 2025-07-08 05:30:00 | 3547.41 | 3428.42 | 3592.44 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2025-08-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 05:30:00 | 3554.80 | 3428.42 | 3454.66 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=61.15 |
| Stop hit — per-position SL triggered | 2025-09-02 05:30:00 | 3620.60 | 3446.23 | 3558.05 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 05:30:00 | 3565.60 | 3459.08 | 3470.92 | Stage2 pullback-breakout RSI=58 vol=3.0x ATR=68.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 05:30:00 | 3703.14 | 3466.90 | 3527.39 | T1 booked 50% @ 3703.14 |
| Target hit | 2025-12-03 05:30:00 | 3817.80 | 3562.22 | 3845.59 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2025-12-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 05:30:00 | 3929.50 | 3586.56 | 3850.64 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=64.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 05:30:00 | 4058.88 | 3621.51 | 3922.09 | T1 booked 50% @ 4058.88 |
| Target hit | 2026-01-20 05:30:00 | 4075.50 | 3687.70 | 4102.17 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2026-04-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 05:30:00 | 4246.10 | 3862.43 | 4092.38 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=124.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 05:30:00 | 4495.84 | 3872.32 | 4142.40 | T1 booked 50% @ 4495.84 |
| Target hit | 2026-05-04 05:30:00 | 4363.00 | 3958.02 | 4378.30 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2026-05-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 05:30:00 | 4509.00 | 3974.88 | 4382.86 | Stage2 pullback-breakout RSI=60 vol=3.9x ATR=124.38 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-25 05:30:00 | 3652.20 | 2025-07-08 05:30:00 | 3547.41 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest1 | 2025-08-18 05:30:00 | 3554.80 | 2025-09-02 05:30:00 | 3620.60 | STOP_HIT | 1.00 | 1.85% |
| BUY | retest1 | 2025-10-08 05:30:00 | 3565.60 | 2025-10-17 05:30:00 | 3703.14 | PARTIAL | 0.50 | 3.86% |
| BUY | retest1 | 2025-10-08 05:30:00 | 3565.60 | 2025-12-03 05:30:00 | 3817.80 | TARGET_HIT | 0.50 | 7.07% |
| BUY | retest1 | 2025-12-16 05:30:00 | 3929.50 | 2025-12-31 05:30:00 | 4058.88 | PARTIAL | 0.50 | 3.29% |
| BUY | retest1 | 2025-12-16 05:30:00 | 3929.50 | 2026-01-20 05:30:00 | 4075.50 | TARGET_HIT | 0.50 | 3.72% |
| BUY | retest1 | 2026-04-06 05:30:00 | 4246.10 | 2026-04-08 05:30:00 | 4495.84 | PARTIAL | 0.50 | 5.88% |
| BUY | retest1 | 2026-04-06 05:30:00 | 4246.10 | 2026-05-04 05:30:00 | 4363.00 | TARGET_HIT | 0.50 | 2.75% |
