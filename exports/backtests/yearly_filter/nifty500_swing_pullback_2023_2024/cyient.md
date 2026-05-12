# Cyient Ltd. (CYIENT)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 885.70
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 4.52% / 5.83%
- **Sum % (uncompounded):** 27.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 4.52% | 27.1% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 4.52% | 27.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 4.52% | 27.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-02 00:00:00 | 1518.10 | 1146.40 | 1465.01 | Stage2 pullback-breakout RSI=61 vol=2.6x ATR=48.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-07 00:00:00 | 1615.57 | 1158.89 | 1493.08 | T1 booked 50% @ 1615.57 |
| Stop hit — per-position SL triggered | 2023-08-11 00:00:00 | 1518.10 | 1174.55 | 1514.01 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2023-10-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 00:00:00 | 1774.50 | 1343.42 | 1691.48 | Stage2 pullback-breakout RSI=63 vol=1.5x ATR=55.09 |
| Stop hit — per-position SL triggered | 2023-10-18 00:00:00 | 1691.86 | 1358.41 | 1702.49 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-11-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 00:00:00 | 1733.30 | 1415.82 | 1676.44 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=50.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 00:00:00 | 1834.42 | 1419.99 | 1691.49 | T1 booked 50% @ 1834.42 |
| Target hit | 2024-01-11 00:00:00 | 2151.40 | 1637.72 | 2209.97 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 2067.10 | 1815.49 | 1998.39 | Stage2 pullback-breakout RSI=58 vol=2.1x ATR=71.26 |
| Stop hit — per-position SL triggered | 2024-04-19 00:00:00 | 1971.55 | 1840.48 | 2039.89 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-02 00:00:00 | 1518.10 | 2023-08-07 00:00:00 | 1615.57 | PARTIAL | 0.50 | 6.42% |
| BUY | retest1 | 2023-08-02 00:00:00 | 1518.10 | 2023-08-11 00:00:00 | 1518.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-12 00:00:00 | 1774.50 | 2023-10-18 00:00:00 | 1691.86 | STOP_HIT | 1.00 | -4.66% |
| BUY | retest1 | 2023-11-17 00:00:00 | 1733.30 | 2023-11-20 00:00:00 | 1834.42 | PARTIAL | 0.50 | 5.83% |
| BUY | retest1 | 2023-11-17 00:00:00 | 1733.30 | 2024-01-11 00:00:00 | 2151.40 | TARGET_HIT | 0.50 | 24.12% |
| BUY | retest1 | 2024-04-03 00:00:00 | 2067.10 | 2024-04-19 00:00:00 | 1971.55 | STOP_HIT | 1.00 | -4.62% |
