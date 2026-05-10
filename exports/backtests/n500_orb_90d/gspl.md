# Gujarat State Petronet Ltd. (GSPL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 289.00
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 3
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 0.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.10% | -0.5% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.10% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.18% | 0.9% |
| SELL @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.18% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.04% | 0.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:40:00 | 309.15 | 308.21 | 0.00 | ORB-long ORB[306.70,309.00] vol=2.0x ATR=1.23 |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 307.92 | 308.33 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:10:00 | 308.50 | 310.52 | 0.00 | ORB-short ORB[309.80,313.50] vol=3.0x ATR=0.98 |
| Stop hit — per-position SL triggered | 2026-02-16 12:35:00 | 309.48 | 309.93 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 301.55 | 302.63 | 0.00 | ORB-short ORB[302.55,304.05] vol=2.2x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:50:00 | 300.31 | 301.99 | 0.00 | T1 1.5R @ 300.31 |
| Target hit | 2026-02-18 10:55:00 | 300.15 | 299.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2026-02-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:05:00 | 302.65 | 303.80 | 0.00 | ORB-short ORB[303.95,306.00] vol=3.1x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:35:00 | 301.61 | 303.57 | 0.00 | T1 1.5R @ 301.61 |
| Stop hit — per-position SL triggered | 2026-02-19 12:25:00 | 302.65 | 302.78 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 306.00 | 304.99 | 0.00 | ORB-long ORB[302.65,305.30] vol=3.0x ATR=0.82 |
| Stop hit — per-position SL triggered | 2026-02-27 12:00:00 | 305.18 | 305.33 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 11:05:00 | 293.75 | 290.61 | 0.00 | ORB-long ORB[288.35,292.50] vol=1.8x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:40:00 | 295.27 | 291.53 | 0.00 | T1 1.5R @ 295.27 |
| Stop hit — per-position SL triggered | 2026-03-06 12:25:00 | 293.75 | 292.39 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:35:00 | 291.50 | 290.61 | 0.00 | ORB-long ORB[289.30,291.00] vol=1.7x ATR=0.97 |
| Stop hit — per-position SL triggered | 2026-05-06 09:45:00 | 290.53 | 290.70 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-13 09:40:00 | 309.15 | 2026-02-13 10:15:00 | 307.92 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-02-16 11:10:00 | 308.50 | 2026-02-16 12:35:00 | 309.48 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-18 09:35:00 | 301.55 | 2026-02-18 09:50:00 | 300.31 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-18 09:35:00 | 301.55 | 2026-02-18 10:55:00 | 300.15 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-19 11:05:00 | 302.65 | 2026-02-19 11:35:00 | 301.61 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-19 11:05:00 | 302.65 | 2026-02-19 12:25:00 | 302.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-27 11:10:00 | 306.00 | 2026-02-27 12:00:00 | 305.18 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-06 11:05:00 | 293.75 | 2026-03-06 11:40:00 | 295.27 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-03-06 11:05:00 | 293.75 | 2026-03-06 12:25:00 | 293.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 09:35:00 | 291.50 | 2026-05-06 09:45:00 | 290.53 | STOP_HIT | 1.00 | -0.33% |
