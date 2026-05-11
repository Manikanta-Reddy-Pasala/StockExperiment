# Craftsman Automation Ltd. (CRAFTSMAN)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 8981.50
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 1 / 6 / 1
- **Avg / median % per leg:** 2.96% / 2.09%
- **Sum % (uncompounded):** 23.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 1 | 6 | 1 | 2.96% | 23.7% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 1 | 6 | 1 | 2.96% | 23.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 1 | 6 | 1 | 2.96% | 23.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 05:30:00 | 5465.50 | 5143.79 | 5328.85 | Stage2 pullback-breakout RSI=58 vol=4.5x ATR=179.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 05:30:00 | 5824.88 | 5179.09 | 5511.43 | T1 booked 50% @ 5824.88 |
| Target hit | 2025-09-08 05:30:00 | 6837.00 | 5693.49 | 6840.10 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-10-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 05:30:00 | 6800.50 | 5931.25 | 6666.30 | Stage2 pullback-breakout RSI=56 vol=9.4x ATR=179.27 |
| Stop hit — per-position SL triggered | 2025-10-24 05:30:00 | 6531.60 | 5967.20 | 6663.71 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2025-11-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-25 05:30:00 | 7021.00 | 6118.42 | 6780.04 | Stage2 pullback-breakout RSI=60 vol=5.5x ATR=219.02 |
| Stop hit — per-position SL triggered | 2025-12-09 05:30:00 | 7076.00 | 6205.65 | 6939.57 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-12-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 05:30:00 | 7379.50 | 6303.75 | 7047.45 | Stage2 pullback-breakout RSI=67 vol=3.2x ATR=201.64 |
| Stop hit — per-position SL triggered | 2025-12-29 05:30:00 | 7077.04 | 6310.99 | 7045.93 | SL hit (bars_held=1) |

### Cycle 5 — BUY (started 2025-12-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 05:30:00 | 7689.50 | 6333.03 | 7116.71 | Stage2 pullback-breakout RSI=68 vol=7.2x ATR=270.15 |
| Stop hit — per-position SL triggered | 2026-01-14 05:30:00 | 7850.00 | 6471.31 | 7526.57 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2026-01-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 05:30:00 | 7780.00 | 6553.71 | 7535.92 | Stage2 pullback-breakout RSI=57 vol=1.9x ATR=299.54 |
| Stop hit — per-position SL triggered | 2026-01-29 05:30:00 | 7330.69 | 6561.30 | 7515.07 | SL hit (bars_held=1) |

### Cycle 7 — BUY (started 2026-04-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 05:30:00 | 7351.50 | 6856.84 | 7035.24 | Stage2 pullback-breakout RSI=56 vol=3.6x ATR=310.26 |
| Stop hit — per-position SL triggered | 2026-04-24 05:30:00 | 7570.00 | 6931.31 | 7411.29 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-25 05:30:00 | 5465.50 | 2025-07-04 05:30:00 | 5824.88 | PARTIAL | 0.50 | 6.58% |
| BUY | retest1 | 2025-06-25 05:30:00 | 5465.50 | 2025-09-08 05:30:00 | 6837.00 | TARGET_HIT | 0.50 | 25.09% |
| BUY | retest1 | 2025-10-16 05:30:00 | 6800.50 | 2025-10-24 05:30:00 | 6531.60 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest1 | 2025-11-25 05:30:00 | 7021.00 | 2025-12-09 05:30:00 | 7076.00 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest1 | 2025-12-26 05:30:00 | 7379.50 | 2025-12-29 05:30:00 | 7077.04 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest1 | 2025-12-31 05:30:00 | 7689.50 | 2026-01-14 05:30:00 | 7850.00 | STOP_HIT | 1.00 | 2.09% |
| BUY | retest1 | 2026-01-28 05:30:00 | 7780.00 | 2026-01-29 05:30:00 | 7330.69 | STOP_HIT | 1.00 | -5.78% |
| BUY | retest1 | 2026-04-08 05:30:00 | 7351.50 | 2026-04-24 05:30:00 | 7570.00 | STOP_HIT | 1.00 | 2.97% |
