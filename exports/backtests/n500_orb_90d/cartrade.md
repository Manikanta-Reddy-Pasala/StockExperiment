# Cartrade Tech Ltd. (CARTRADE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1949.90
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
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** -0.00% / 0.00%
- **Sum % (uncompounded):** -0.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | -0.00% | -0.0% |
| SELL @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | -0.00% | -0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | -0.00% | -0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:00:00 | 2134.40 | 2141.61 | 0.00 | ORB-short ORB[2135.00,2160.00] vol=1.8x ATR=8.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 12:55:00 | 2121.94 | 2138.19 | 0.00 | T1 1.5R @ 2121.94 |
| Stop hit — per-position SL triggered | 2026-02-12 15:05:00 | 2134.40 | 2124.47 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 2021.30 | 2034.89 | 0.00 | ORB-short ORB[2022.30,2046.10] vol=1.7x ATR=9.31 |
| Stop hit — per-position SL triggered | 2026-02-18 09:40:00 | 2030.61 | 2028.38 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 2019.00 | 2034.82 | 0.00 | ORB-short ORB[2022.10,2049.70] vol=1.6x ATR=8.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:45:00 | 2006.74 | 2028.98 | 0.00 | T1 1.5R @ 2006.74 |
| Target hit | 2026-02-19 11:35:00 | 2015.30 | 2011.84 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2026-02-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 11:05:00 | 1938.60 | 1971.58 | 0.00 | ORB-short ORB[1972.10,2000.10] vol=4.7x ATR=8.42 |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 1947.02 | 1966.14 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 1698.00 | 1705.80 | 0.00 | ORB-short ORB[1703.10,1724.00] vol=2.2x ATR=8.59 |
| Stop hit — per-position SL triggered | 2026-04-27 09:50:00 | 1706.59 | 1705.44 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 11:00:00 | 2134.40 | 2026-02-12 12:55:00 | 2121.94 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-02-12 11:00:00 | 2134.40 | 2026-02-12 15:05:00 | 2134.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 09:30:00 | 2021.30 | 2026-02-18 09:40:00 | 2030.61 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-02-19 09:30:00 | 2019.00 | 2026-02-19 09:45:00 | 2006.74 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-02-19 09:30:00 | 2019.00 | 2026-02-19 11:35:00 | 2015.30 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2026-02-20 11:05:00 | 1938.60 | 2026-02-20 11:15:00 | 1947.02 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-27 09:45:00 | 1698.00 | 2026-04-27 09:50:00 | 1706.59 | STOP_HIT | 1.00 | -0.51% |
