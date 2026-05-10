# Vardhman Textiles Ltd. (VTL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 583.10
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
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 0.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.46% | -0.9% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.46% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.36% | 1.8% |
| SELL @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.36% | 1.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.12% | 0.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:10:00 | 500.85 | 504.63 | 0.00 | ORB-short ORB[505.10,509.35] vol=2.2x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:25:00 | 498.69 | 504.32 | 0.00 | T1 1.5R @ 498.69 |
| Stop hit — per-position SL triggered | 2026-02-18 11:55:00 | 500.85 | 503.69 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:15:00 | 524.75 | 528.21 | 0.00 | ORB-short ORB[526.75,531.30] vol=4.8x ATR=2.00 |
| Stop hit — per-position SL triggered | 2026-03-16 10:25:00 | 526.75 | 527.80 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:35:00 | 557.55 | 553.63 | 0.00 | ORB-long ORB[549.15,555.90] vol=2.4x ATR=2.52 |
| Stop hit — per-position SL triggered | 2026-04-10 10:00:00 | 555.03 | 554.18 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-05-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:50:00 | 620.25 | 615.96 | 0.00 | ORB-long ORB[610.35,618.00] vol=3.9x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-05-07 12:50:00 | 617.32 | 617.03 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-05-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:10:00 | 585.20 | 593.72 | 0.00 | ORB-short ORB[592.30,600.00] vol=1.7x ATR=3.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:50:00 | 580.30 | 590.02 | 0.00 | T1 1.5R @ 580.30 |
| Target hit | 2026-05-08 15:05:00 | 580.00 | 578.59 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 11:10:00 | 500.85 | 2026-02-18 11:25:00 | 498.69 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-18 11:10:00 | 500.85 | 2026-02-18 11:55:00 | 500.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:15:00 | 524.75 | 2026-03-16 10:25:00 | 526.75 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-10 09:35:00 | 557.55 | 2026-04-10 10:00:00 | 555.03 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-07 10:50:00 | 620.25 | 2026-05-07 12:50:00 | 617.32 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-05-08 10:10:00 | 585.20 | 2026-05-08 10:50:00 | 580.30 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2026-05-08 10:10:00 | 585.20 | 2026-05-08 15:05:00 | 580.00 | TARGET_HIT | 0.50 | 0.89% |
