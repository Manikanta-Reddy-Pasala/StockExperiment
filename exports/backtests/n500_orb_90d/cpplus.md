# Aditya Infotech Ltd. (CPPLUS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2511.00
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
- **Avg / median % per leg:** 0.61% / 0.00%
- **Sum % (uncompounded):** 4.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.48% | -1.0% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.48% | -1.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 1.04% | 5.2% |
| SELL @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 1.04% | 5.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.61% | 4.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:10:00 | 1580.20 | 1586.05 | 0.00 | ORB-short ORB[1587.80,1609.10] vol=1.6x ATR=6.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:35:00 | 1570.42 | 1585.42 | 0.00 | T1 1.5R @ 1570.42 |
| Stop hit — per-position SL triggered | 2026-02-18 11:30:00 | 1580.20 | 1582.02 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-03-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:20:00 | 1670.20 | 1663.65 | 0.00 | ORB-long ORB[1652.30,1670.10] vol=1.5x ATR=6.82 |
| Stop hit — per-position SL triggered | 2026-03-10 11:05:00 | 1663.38 | 1664.35 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:35:00 | 1632.80 | 1639.46 | 0.00 | ORB-short ORB[1644.50,1667.30] vol=1.5x ATR=6.90 |
| Stop hit — per-position SL triggered | 2026-03-11 09:45:00 | 1639.70 | 1639.07 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 2262.00 | 2254.00 | 0.00 | ORB-long ORB[2231.10,2261.90] vol=6.4x ATR=12.26 |
| Stop hit — per-position SL triggered | 2026-04-22 09:55:00 | 2249.74 | 2255.33 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:35:00 | 2446.90 | 2474.16 | 0.00 | ORB-short ORB[2470.00,2490.00] vol=2.4x ATR=12.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:40:00 | 2428.64 | 2461.03 | 0.00 | T1 1.5R @ 2428.64 |
| Target hit | 2026-05-06 14:45:00 | 2342.90 | 2312.42 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 10:10:00 | 1580.20 | 2026-02-18 10:35:00 | 1570.42 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-02-18 10:10:00 | 1580.20 | 2026-02-18 11:30:00 | 1580.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 10:20:00 | 1670.20 | 2026-03-10 11:05:00 | 1663.38 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-11 09:35:00 | 1632.80 | 2026-03-11 09:45:00 | 1639.70 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-22 09:30:00 | 2262.00 | 2026-04-22 09:55:00 | 2249.74 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2026-05-06 09:35:00 | 2446.90 | 2026-05-06 09:40:00 | 2428.64 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2026-05-06 09:35:00 | 2446.90 | 2026-05-06 14:45:00 | 2342.90 | TARGET_HIT | 0.50 | 4.25% |
