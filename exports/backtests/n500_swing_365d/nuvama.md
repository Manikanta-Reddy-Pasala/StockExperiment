# Nuvama Wealth Management Ltd. (NUVAMA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1631.50
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
- **Avg / median % per leg:** -1.52% / -4.22%
- **Sum % (uncompounded):** -7.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.52% | -7.6% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.52% | -7.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.52% | -7.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 05:30:00 | 1580.20 | 1284.17 | 1440.89 | Stage2 pullback-breakout RSI=68 vol=2.5x ATR=56.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 05:30:00 | 1692.72 | 1287.89 | 1461.52 | T1 booked 50% @ 1692.72 |
| Stop hit — per-position SL triggered | 2025-07-04 05:30:00 | 1580.20 | 1303.07 | 1512.71 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2025-07-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 05:30:00 | 1563.70 | 1319.55 | 1504.53 | Stage2 pullback-breakout RSI=58 vol=1.9x ATR=64.48 |
| Stop hit — per-position SL triggered | 2025-07-25 05:30:00 | 1466.98 | 1331.96 | 1516.05 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2025-11-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 05:30:00 | 1492.50 | 1365.48 | 1452.67 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=43.09 |
| Stop hit — per-position SL triggered | 2025-12-03 05:30:00 | 1427.87 | 1368.07 | 1452.18 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2025-12-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 05:30:00 | 1523.00 | 1378.55 | 1450.54 | Stage2 pullback-breakout RSI=63 vol=3.9x ATR=42.83 |
| Stop hit — per-position SL triggered | 2025-12-29 05:30:00 | 1458.76 | 1380.50 | 1455.24 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-26 05:30:00 | 1580.20 | 2025-06-27 05:30:00 | 1692.72 | PARTIAL | 0.50 | 7.12% |
| BUY | retest1 | 2025-06-26 05:30:00 | 1580.20 | 2025-07-04 05:30:00 | 1580.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-17 05:30:00 | 1563.70 | 2025-07-25 05:30:00 | 1466.98 | STOP_HIT | 1.00 | -6.19% |
| BUY | retest1 | 2025-11-28 05:30:00 | 1492.50 | 2025-12-03 05:30:00 | 1427.87 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest1 | 2025-12-24 05:30:00 | 1523.00 | 2025-12-29 05:30:00 | 1458.76 | STOP_HIT | 1.00 | -4.22% |
