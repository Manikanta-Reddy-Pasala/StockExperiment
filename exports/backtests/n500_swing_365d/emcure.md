# Emcure Pharmaceuticals Ltd. (EMCURE)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1643.00
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
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 6
- **Target hits / Stop hits / Partials:** 0 / 6 / 1
- **Avg / median % per leg:** -2.97% / -4.84%
- **Sum % (uncompounded):** -20.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -2.97% | -20.8% |
| BUY @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -2.97% | -20.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 1 | 14.3% | 0 | 6 | 1 | -2.97% | -20.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 05:30:00 | 1435.30 | 1301.59 | 1344.83 | Stage2 pullback-breakout RSI=64 vol=7.0x ATR=43.74 |
| Stop hit — per-position SL triggered | 2025-10-13 05:30:00 | 1369.69 | 1306.23 | 1364.99 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2025-11-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 05:30:00 | 1359.00 | 1312.58 | 1349.00 | Stage2 pullback-breakout RSI=51 vol=4.8x ATR=40.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 05:30:00 | 1439.85 | 1314.20 | 1357.54 | T1 booked 50% @ 1439.85 |
| Stop hit — per-position SL triggered | 2025-11-12 05:30:00 | 1359.00 | 1314.98 | 1356.67 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2026-01-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 05:30:00 | 1489.50 | 1341.19 | 1408.69 | Stage2 pullback-breakout RSI=64 vol=3.7x ATR=48.09 |
| Stop hit — per-position SL triggered | 2026-01-23 05:30:00 | 1417.37 | 1365.55 | 1506.72 | SL hit (bars_held=13) |

### Cycle 4 — BUY (started 2026-02-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 05:30:00 | 1534.90 | 1374.16 | 1493.39 | Stage2 pullback-breakout RSI=58 vol=1.8x ATR=57.85 |
| Stop hit — per-position SL triggered | 2026-02-13 05:30:00 | 1448.13 | 1382.65 | 1495.57 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2026-03-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 05:30:00 | 1521.10 | 1392.41 | 1464.00 | Stage2 pullback-breakout RSI=58 vol=9.1x ATR=59.96 |
| Stop hit — per-position SL triggered | 2026-03-16 05:30:00 | 1431.16 | 1398.00 | 1478.54 | SL hit (bars_held=5) |

### Cycle 6 — BUY (started 2026-03-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 05:30:00 | 1652.70 | 1408.26 | 1512.87 | Stage2 pullback-breakout RSI=66 vol=2.4x ATR=63.76 |
| Stop hit — per-position SL triggered | 2026-04-02 05:30:00 | 1557.06 | 1413.26 | 1529.28 | SL hit (bars_held=3) |

### Cycle 7 — BUY (started 2026-04-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 05:30:00 | 1729.90 | 1445.45 | 1611.51 | Stage2 pullback-breakout RSI=66 vol=4.3x ATR=72.94 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-06 05:30:00 | 1435.30 | 2025-10-13 05:30:00 | 1369.69 | STOP_HIT | 1.00 | -4.57% |
| BUY | retest1 | 2025-11-06 05:30:00 | 1359.00 | 2025-11-10 05:30:00 | 1439.85 | PARTIAL | 0.50 | 5.95% |
| BUY | retest1 | 2025-11-06 05:30:00 | 1359.00 | 2025-11-12 05:30:00 | 1359.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-05 05:30:00 | 1489.50 | 2026-01-23 05:30:00 | 1417.37 | STOP_HIT | 1.00 | -4.84% |
| BUY | retest1 | 2026-02-04 05:30:00 | 1534.90 | 2026-02-13 05:30:00 | 1448.13 | STOP_HIT | 1.00 | -5.65% |
| BUY | retest1 | 2026-03-09 05:30:00 | 1521.10 | 2026-03-16 05:30:00 | 1431.16 | STOP_HIT | 1.00 | -5.91% |
| BUY | retest1 | 2026-03-27 05:30:00 | 1652.70 | 2026-04-02 05:30:00 | 1557.06 | STOP_HIT | 1.00 | -5.79% |
