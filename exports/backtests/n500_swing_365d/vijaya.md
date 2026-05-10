# Vijaya Diagnostic Centre Ltd. (VIJAYA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1278.90
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -3.43% / -3.69%
- **Sum % (uncompounded):** -13.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.43% | -13.7% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.43% | -13.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.43% | -13.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 05:30:00 | 1065.20 | 1008.31 | 1038.26 | Stage2 pullback-breakout RSI=57 vol=2.7x ATR=32.91 |
| Stop hit — per-position SL triggered | 2025-09-17 05:30:00 | 1049.90 | 1013.96 | 1053.79 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-11-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 05:30:00 | 1019.70 | 1011.78 | 1001.96 | Stage2 pullback-breakout RSI=56 vol=2.4x ATR=25.09 |
| Stop hit — per-position SL triggered | 2025-11-06 05:30:00 | 982.07 | 1011.98 | 1005.50 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2025-11-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 05:30:00 | 1041.20 | 1012.62 | 1013.54 | Stage2 pullback-breakout RSI=62 vol=2.0x ATR=28.22 |
| Stop hit — per-position SL triggered | 2025-11-24 05:30:00 | 998.87 | 1013.46 | 1018.36 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2025-12-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 05:30:00 | 1080.00 | 1013.58 | 1018.16 | Stage2 pullback-breakout RSI=66 vol=13.7x ATR=32.47 |
| Stop hit — per-position SL triggered | 2025-12-05 05:30:00 | 1031.30 | 1013.61 | 1018.08 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-03 05:30:00 | 1065.20 | 2025-09-17 05:30:00 | 1049.90 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest1 | 2025-11-03 05:30:00 | 1019.70 | 2025-11-06 05:30:00 | 982.07 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest1 | 2025-11-13 05:30:00 | 1041.20 | 2025-11-24 05:30:00 | 998.87 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest1 | 2025-12-04 05:30:00 | 1080.00 | 2025-12-05 05:30:00 | 1031.30 | STOP_HIT | 1.00 | -4.51% |
