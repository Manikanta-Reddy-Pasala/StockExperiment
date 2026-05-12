# Syngene International Ltd. (SYNGENE)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 458.10
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -0.06% / 2.77%
- **Sum % (uncompounded):** -0.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.06% | -0.3% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.06% | -0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.06% | -0.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 00:00:00 | 868.75 | 748.26 | 830.16 | Stage2 pullback-breakout RSI=67 vol=2.1x ATR=22.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 00:00:00 | 913.69 | 753.45 | 847.34 | T1 booked 50% @ 913.69 |
| Stop hit — per-position SL triggered | 2024-09-18 00:00:00 | 892.80 | 767.16 | 886.30 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 00:00:00 | 903.85 | 798.72 | 871.75 | Stage2 pullback-breakout RSI=61 vol=3.0x ATR=25.75 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 865.23 | 803.19 | 878.28 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2024-11-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 00:00:00 | 916.50 | 808.69 | 879.16 | Stage2 pullback-breakout RSI=62 vol=2.4x ATR=23.96 |
| Stop hit — per-position SL triggered | 2024-12-09 00:00:00 | 880.56 | 817.46 | 901.39 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-30 00:00:00 | 868.75 | 2024-09-05 00:00:00 | 913.69 | PARTIAL | 0.50 | 5.17% |
| BUY | retest1 | 2024-08-30 00:00:00 | 868.75 | 2024-09-18 00:00:00 | 892.80 | STOP_HIT | 0.50 | 2.77% |
| BUY | retest1 | 2024-11-06 00:00:00 | 903.85 | 2024-11-13 00:00:00 | 865.23 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest1 | 2024-11-27 00:00:00 | 916.50 | 2024-12-09 00:00:00 | 880.56 | STOP_HIT | 1.00 | -3.92% |
