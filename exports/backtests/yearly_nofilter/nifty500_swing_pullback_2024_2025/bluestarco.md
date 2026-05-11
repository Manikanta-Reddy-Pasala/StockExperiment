# Blue Star Ltd. (BLUESTARCO)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 1692.80
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 4.99% / 6.72%
- **Sum % (uncompounded):** 24.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 1 | 2 | 2 | 4.99% | 24.9% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 2 | 2 | 4.99% | 24.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 1 | 2 | 2 | 4.99% | 24.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 00:00:00 | 1705.25 | 1263.74 | 1634.36 | Stage2 pullback-breakout RSI=63 vol=7.9x ATR=69.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 00:00:00 | 1843.65 | 1273.39 | 1656.27 | T1 booked 50% @ 1843.65 |
| Stop hit — per-position SL triggered | 2024-07-10 00:00:00 | 1705.25 | 1277.74 | 1661.49 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2024-08-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 00:00:00 | 1713.70 | 1360.06 | 1665.12 | Stage2 pullback-breakout RSI=55 vol=3.5x ATR=76.30 |
| Stop hit — per-position SL triggered | 2024-08-28 00:00:00 | 1746.65 | 1394.52 | 1701.68 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-09-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 00:00:00 | 1789.30 | 1418.55 | 1707.61 | Stage2 pullback-breakout RSI=61 vol=2.1x ATR=60.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 00:00:00 | 1909.56 | 1439.41 | 1764.88 | T1 booked 50% @ 1909.56 |
| Target hit | 2024-10-07 00:00:00 | 1935.60 | 1514.92 | 1968.13 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-05 00:00:00 | 1705.25 | 2024-07-09 00:00:00 | 1843.65 | PARTIAL | 0.50 | 8.12% |
| BUY | retest1 | 2024-07-05 00:00:00 | 1705.25 | 2024-07-10 00:00:00 | 1705.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-13 00:00:00 | 1713.70 | 2024-08-28 00:00:00 | 1746.65 | STOP_HIT | 1.00 | 1.92% |
| BUY | retest1 | 2024-09-09 00:00:00 | 1789.30 | 2024-09-16 00:00:00 | 1909.56 | PARTIAL | 0.50 | 6.72% |
| BUY | retest1 | 2024-09-09 00:00:00 | 1789.30 | 2024-10-07 00:00:00 | 1935.60 | TARGET_HIT | 0.50 | 8.18% |
