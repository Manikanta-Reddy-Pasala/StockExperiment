# Nippon Life India Asset Management Ltd. (NAM-INDIA)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 1103.40
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -1.22% / -2.01%
- **Sum % (uncompounded):** -6.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.22% | -6.1% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.22% | -6.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.22% | -6.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 00:00:00 | 686.85 | 541.92 | 645.63 | Stage2 pullback-breakout RSI=61 vol=2.8x ATR=29.38 |
| Stop hit — per-position SL triggered | 2024-08-30 00:00:00 | 673.05 | 557.11 | 678.56 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-09-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 00:00:00 | 694.40 | 574.78 | 674.95 | Stage2 pullback-breakout RSI=57 vol=2.0x ATR=22.68 |
| Stop hit — per-position SL triggered | 2024-09-27 00:00:00 | 660.38 | 578.74 | 674.83 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-10-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 00:00:00 | 700.25 | 585.41 | 664.11 | Stage2 pullback-breakout RSI=58 vol=3.6x ATR=27.28 |
| Stop hit — per-position SL triggered | 2024-10-22 00:00:00 | 659.33 | 593.10 | 680.73 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2024-12-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 00:00:00 | 734.55 | 621.16 | 698.63 | Stage2 pullback-breakout RSI=63 vol=2.4x ATR=24.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 00:00:00 | 783.38 | 623.00 | 708.85 | T1 booked 50% @ 783.38 |
| Stop hit — per-position SL triggered | 2024-12-19 00:00:00 | 734.55 | 632.94 | 738.51 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-16 00:00:00 | 686.85 | 2024-08-30 00:00:00 | 673.05 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest1 | 2024-09-23 00:00:00 | 694.40 | 2024-09-27 00:00:00 | 660.38 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest1 | 2024-10-11 00:00:00 | 700.25 | 2024-10-22 00:00:00 | 659.33 | STOP_HIT | 1.00 | -5.84% |
| BUY | retest1 | 2024-12-09 00:00:00 | 734.55 | 2024-12-10 00:00:00 | 783.38 | PARTIAL | 0.50 | 6.65% |
| BUY | retest1 | 2024-12-09 00:00:00 | 734.55 | 2024-12-19 00:00:00 | 734.55 | STOP_HIT | 0.50 | 0.00% |
