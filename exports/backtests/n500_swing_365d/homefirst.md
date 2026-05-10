# Home First Finance Company India Ltd. (HOMEFIRST)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1200.30
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
- **Avg / median % per leg:** -1.38% / -4.19%
- **Sum % (uncompounded):** -6.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.38% | -6.9% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.38% | -6.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.38% | -6.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 05:30:00 | 1340.70 | 1127.36 | 1268.32 | Stage2 pullback-breakout RSI=68 vol=2.3x ATR=47.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 05:30:00 | 1435.19 | 1135.83 | 1306.77 | T1 booked 50% @ 1435.19 |
| Stop hit — per-position SL triggered | 2025-07-01 05:30:00 | 1340.70 | 1140.05 | 1314.22 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2025-07-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-25 05:30:00 | 1479.00 | 1181.29 | 1386.12 | Stage2 pullback-breakout RSI=68 vol=3.3x ATR=54.14 |
| Stop hit — per-position SL triggered | 2025-07-28 05:30:00 | 1397.79 | 1183.18 | 1384.75 | SL hit (bars_held=1) |

### Cycle 3 — BUY (started 2025-10-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 05:30:00 | 1263.10 | 1211.27 | 1236.63 | Stage2 pullback-breakout RSI=54 vol=1.7x ATR=35.71 |
| Stop hit — per-position SL triggered | 2025-10-17 05:30:00 | 1209.54 | 1211.61 | 1235.02 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2025-11-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 05:30:00 | 1254.70 | 1212.38 | 1225.02 | Stage2 pullback-breakout RSI=55 vol=1.9x ATR=35.08 |
| Stop hit — per-position SL triggered | 2025-11-04 05:30:00 | 1202.08 | 1212.21 | 1222.25 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-24 05:30:00 | 1340.70 | 2025-06-27 05:30:00 | 1435.19 | PARTIAL | 0.50 | 7.05% |
| BUY | retest1 | 2025-06-24 05:30:00 | 1340.70 | 2025-07-01 05:30:00 | 1340.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-25 05:30:00 | 1479.00 | 2025-07-28 05:30:00 | 1397.79 | STOP_HIT | 1.00 | -5.49% |
| BUY | retest1 | 2025-10-15 05:30:00 | 1263.10 | 2025-10-17 05:30:00 | 1209.54 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest1 | 2025-11-03 05:30:00 | 1254.70 | 2025-11-04 05:30:00 | 1202.08 | STOP_HIT | 1.00 | -4.19% |
