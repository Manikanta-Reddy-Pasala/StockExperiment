# Chennai Petroleum Corporation Ltd. (CHENNPETRO)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 1076.90
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** -1.06% / -5.99%
- **Sum % (uncompounded):** -5.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | -1.06% | -5.3% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | -1.06% | -5.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | -1.06% | -5.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 05:30:00 | 843.20 | 706.44 | 771.21 | Stage2 pullback-breakout RSI=67 vol=5.4x ATR=37.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 05:30:00 | 918.07 | 711.92 | 803.84 | T1 booked 50% @ 918.07 |
| Target hit | 2025-11-26 05:30:00 | 896.60 | 759.74 | 977.33 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2026-01-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 05:30:00 | 875.30 | 791.45 | 842.67 | Stage2 pullback-breakout RSI=54 vol=2.9x ATR=35.44 |
| Stop hit — per-position SL triggered | 2026-01-20 05:30:00 | 822.14 | 792.80 | 840.75 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2026-03-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 05:30:00 | 999.90 | 818.62 | 911.70 | Stage2 pullback-breakout RSI=68 vol=5.6x ATR=39.90 |
| Stop hit — per-position SL triggered | 2026-03-09 05:30:00 | 940.05 | 823.34 | 928.58 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2026-03-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 05:30:00 | 1065.05 | 831.27 | 945.67 | Stage2 pullback-breakout RSI=67 vol=3.5x ATR=60.04 |
| Stop hit — per-position SL triggered | 2026-03-24 05:30:00 | 974.99 | 840.49 | 974.71 | SL hit (bars_held=5) |

### Cycle 5 — BUY (started 2026-04-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 05:30:00 | 1139.50 | 874.43 | 1024.79 | Stage2 pullback-breakout RSI=64 vol=2.4x ATR=61.74 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-28 05:30:00 | 843.20 | 2025-10-31 05:30:00 | 918.07 | PARTIAL | 0.50 | 8.88% |
| BUY | retest1 | 2025-10-28 05:30:00 | 843.20 | 2025-11-26 05:30:00 | 896.60 | TARGET_HIT | 0.50 | 6.33% |
| BUY | retest1 | 2026-01-14 05:30:00 | 875.30 | 2026-01-20 05:30:00 | 822.14 | STOP_HIT | 1.00 | -6.07% |
| BUY | retest1 | 2026-03-04 05:30:00 | 999.90 | 2026-03-09 05:30:00 | 940.05 | STOP_HIT | 1.00 | -5.99% |
| BUY | retest1 | 2026-03-17 05:30:00 | 1065.05 | 2026-03-24 05:30:00 | 974.99 | STOP_HIT | 1.00 | -8.46% |
