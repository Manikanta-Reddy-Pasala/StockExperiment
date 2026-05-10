# Adani Ports and Special Economic Zone Ltd. (ADANIPORTS)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1760.40
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 2.14% / 2.32%
- **Sum % (uncompounded):** 12.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 0 | 4 | 2 | 2.14% | 12.8% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 0 | 4 | 2 | 2.14% | 12.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 4 | 2 | 2.14% | 12.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 05:30:00 | 1380.90 | 1333.46 | 1344.97 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=28.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 05:30:00 | 1437.12 | 1338.84 | 1378.51 | T1 booked 50% @ 1437.12 |
| Stop hit — per-position SL triggered | 2025-09-25 05:30:00 | 1407.20 | 1342.45 | 1395.37 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-10-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 05:30:00 | 1479.50 | 1352.09 | 1416.73 | Stage2 pullback-breakout RSI=69 vol=2.5x ATR=26.95 |
| Stop hit — per-position SL triggered | 2025-10-24 05:30:00 | 1439.07 | 1357.37 | 1433.00 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2025-11-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 05:30:00 | 1506.90 | 1367.93 | 1448.99 | Stage2 pullback-breakout RSI=68 vol=1.8x ATR=29.41 |
| Stop hit — per-position SL triggered | 2025-11-26 05:30:00 | 1506.20 | 1379.97 | 1477.05 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2026-02-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 05:30:00 | 1530.80 | 1408.59 | 1421.21 | Stage2 pullback-breakout RSI=61 vol=4.3x ATR=51.13 |
| Stop hit — per-position SL triggered | 2026-02-17 05:30:00 | 1566.30 | 1422.29 | 1503.19 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2026-04-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 05:30:00 | 1453.30 | 1422.37 | 1394.40 | Stage2 pullback-breakout RSI=56 vol=2.5x ATR=53.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 05:30:00 | 1559.67 | 1427.16 | 1447.22 | T1 booked 50% @ 1559.67 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-09 05:30:00 | 1380.90 | 2025-09-19 05:30:00 | 1437.12 | PARTIAL | 0.50 | 4.07% |
| BUY | retest1 | 2025-09-09 05:30:00 | 1380.90 | 2025-09-25 05:30:00 | 1407.20 | STOP_HIT | 0.50 | 1.90% |
| BUY | retest1 | 2025-10-16 05:30:00 | 1479.50 | 2025-10-24 05:30:00 | 1439.07 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest1 | 2025-11-12 05:30:00 | 1506.90 | 2025-11-26 05:30:00 | 1506.20 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest1 | 2026-02-03 05:30:00 | 1530.80 | 2026-02-17 05:30:00 | 1566.30 | STOP_HIT | 1.00 | 2.32% |
| BUY | retest1 | 2026-04-08 05:30:00 | 1453.30 | 2026-04-17 05:30:00 | 1559.67 | PARTIAL | 0.50 | 7.32% |
