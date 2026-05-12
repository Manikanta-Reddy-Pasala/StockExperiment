# J.B. Chemicals & Pharmaceuticals Ltd. (JBCHEPHARM)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 2160.00
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -5.04% / -4.89%
- **Sum % (uncompounded):** -15.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.04% | -15.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.04% | -15.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.04% | -15.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 00:00:00 | 1355.57 | 1071.93 | 1261.06 | Stage2 pullback-breakout RSI=67 vol=4.1x ATR=48.83 |
| Stop hit — per-position SL triggered | 2023-08-18 00:00:00 | 1282.33 | 1086.92 | 1291.09 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2023-09-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-12 00:00:00 | 1462.05 | 1135.94 | 1386.33 | Stage2 pullback-breakout RSI=65 vol=4.6x ATR=47.69 |
| Stop hit — per-position SL triggered | 2023-09-22 00:00:00 | 1390.52 | 1157.82 | 1420.59 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2023-11-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 00:00:00 | 1503.30 | 1229.42 | 1415.70 | Stage2 pullback-breakout RSI=66 vol=3.1x ATR=48.36 |
| Stop hit — per-position SL triggered | 2023-11-23 00:00:00 | 1430.76 | 1254.99 | 1465.91 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-09 00:00:00 | 1355.57 | 2023-08-18 00:00:00 | 1282.33 | STOP_HIT | 1.00 | -5.40% |
| BUY | retest1 | 2023-09-12 00:00:00 | 1462.05 | 2023-09-22 00:00:00 | 1390.52 | STOP_HIT | 1.00 | -4.89% |
| BUY | retest1 | 2023-11-09 00:00:00 | 1503.30 | 2023-11-23 00:00:00 | 1430.76 | STOP_HIT | 1.00 | -4.83% |
