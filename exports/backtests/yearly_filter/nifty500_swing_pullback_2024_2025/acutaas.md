# Acutaas Chemicals Ltd. (ACUTAAS)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 2742.70
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 2 / 1 / 2
- **Avg / median % per leg:** 9.18% / 9.79%
- **Sum % (uncompounded):** 45.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 9.18% | 45.9% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 9.18% | 45.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 9.18% | 45.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 00:00:00 | 695.20 | 604.77 | 653.58 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=30.06 |
| Stop hit — per-position SL triggered | 2024-07-18 00:00:00 | 659.03 | 612.57 | 672.48 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-08-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 00:00:00 | 684.08 | 621.79 | 651.71 | Stage2 pullback-breakout RSI=60 vol=3.5x ATR=25.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 00:00:00 | 734.56 | 625.98 | 671.54 | T1 booked 50% @ 734.56 |
| Target hit | 2024-10-17 00:00:00 | 797.75 | 671.67 | 814.80 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-10-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-29 00:00:00 | 910.18 | 680.95 | 805.76 | Stage2 pullback-breakout RSI=67 vol=9.0x ATR=44.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-01 00:00:00 | 999.27 | 689.73 | 850.67 | T1 booked 50% @ 999.27 |
| Target hit | 2024-12-12 00:00:00 | 1067.88 | 777.87 | 1069.94 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-03 00:00:00 | 695.20 | 2024-07-18 00:00:00 | 659.03 | STOP_HIT | 1.00 | -5.20% |
| BUY | retest1 | 2024-08-28 00:00:00 | 684.08 | 2024-09-05 00:00:00 | 734.56 | PARTIAL | 0.50 | 7.38% |
| BUY | retest1 | 2024-08-28 00:00:00 | 684.08 | 2024-10-17 00:00:00 | 797.75 | TARGET_HIT | 0.50 | 16.62% |
| BUY | retest1 | 2024-10-29 00:00:00 | 910.18 | 2024-11-01 00:00:00 | 999.27 | PARTIAL | 0.50 | 9.79% |
| BUY | retest1 | 2024-10-29 00:00:00 | 910.18 | 2024-12-12 00:00:00 | 1067.88 | TARGET_HIT | 0.50 | 17.33% |
