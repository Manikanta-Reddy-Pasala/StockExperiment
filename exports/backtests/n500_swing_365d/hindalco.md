# Hindalco Industries Ltd. (HINDALCO)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1044.40
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
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 1
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 1.47% / 1.51%
- **Sum % (uncompounded):** 8.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 0 | 4 | 2 | 1.47% | 8.8% |
| BUY @ 2nd Alert (retest1) | 6 | 5 | 83.3% | 0 | 4 | 2 | 1.47% | 8.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 5 | 83.3% | 0 | 4 | 2 | 1.47% | 8.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 05:30:00 | 700.50 | 659.99 | 682.27 | Stage2 pullback-breakout RSI=60 vol=2.8x ATR=15.31 |
| Stop hit — per-position SL triggered | 2025-08-29 05:30:00 | 703.95 | 664.33 | 697.01 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-09-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 05:30:00 | 742.95 | 666.21 | 705.21 | Stage2 pullback-breakout RSI=69 vol=1.6x ATR=14.55 |
| Stop hit — per-position SL triggered | 2025-09-17 05:30:00 | 750.10 | 673.89 | 732.46 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-12-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 05:30:00 | 852.10 | 728.80 | 813.92 | Stage2 pullback-breakout RSI=68 vol=1.8x ATR=17.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 05:30:00 | 886.65 | 741.06 | 842.95 | T1 booked 50% @ 886.65 |
| Stop hit — per-position SL triggered | 2025-12-29 05:30:00 | 865.00 | 741.06 | 842.95 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2026-03-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 05:30:00 | 954.95 | 812.70 | 932.65 | Stage2 pullback-breakout RSI=55 vol=2.8x ATR=31.71 |
| Stop hit — per-position SL triggered | 2026-03-13 05:30:00 | 907.39 | 820.69 | 939.77 | SL hit (bars_held=6) |

### Cycle 5 — BUY (started 2026-04-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 05:30:00 | 954.50 | 830.88 | 911.24 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=32.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 05:30:00 | 1019.15 | 838.35 | 940.70 | T1 booked 50% @ 1019.15 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-13 05:30:00 | 700.50 | 2025-08-29 05:30:00 | 703.95 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest1 | 2025-09-03 05:30:00 | 742.95 | 2025-09-17 05:30:00 | 750.10 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest1 | 2025-12-12 05:30:00 | 852.10 | 2025-12-29 05:30:00 | 886.65 | PARTIAL | 0.50 | 4.05% |
| BUY | retest1 | 2025-12-12 05:30:00 | 852.10 | 2025-12-29 05:30:00 | 865.00 | STOP_HIT | 0.50 | 1.51% |
| BUY | retest1 | 2026-03-05 05:30:00 | 954.95 | 2026-03-13 05:30:00 | 907.39 | STOP_HIT | 1.00 | -4.98% |
| BUY | retest1 | 2026-04-07 05:30:00 | 954.50 | 2026-04-15 05:30:00 | 1019.15 | PARTIAL | 0.50 | 6.77% |
