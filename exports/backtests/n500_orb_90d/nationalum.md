# National Aluminium Co. Ltd. (NATIONALUM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 401.75
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
- **Avg / median % per leg:** 0.03% / 0.09%
- **Sum % (uncompounded):** 0.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.06% | -0.2% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.06% | -0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.22% | 0.4% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.22% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.03% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:35:00 | 370.35 | 368.77 | 0.00 | ORB-long ORB[366.50,369.65] vol=1.6x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 09:50:00 | 372.00 | 369.68 | 0.00 | T1 1.5R @ 372.00 |
| Target hit | 2026-02-12 10:15:00 | 370.70 | 371.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-02-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:55:00 | 342.45 | 345.96 | 0.00 | ORB-short ORB[346.40,351.45] vol=1.6x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:10:00 | 340.93 | 345.00 | 0.00 | T1 1.5R @ 340.93 |
| Stop hit — per-position SL triggered | 2026-02-19 12:20:00 | 342.45 | 344.85 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 429.70 | 427.23 | 0.00 | ORB-long ORB[424.10,428.75] vol=1.5x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-04-21 09:35:00 | 428.52 | 427.35 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-05-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:55:00 | 416.75 | 410.88 | 0.00 | ORB-long ORB[407.65,412.70] vol=2.9x ATR=2.09 |
| Stop hit — per-position SL triggered | 2026-05-05 10:00:00 | 414.66 | 411.52 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 09:35:00 | 370.35 | 2026-02-12 09:50:00 | 372.00 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-12 09:35:00 | 370.35 | 2026-02-12 10:15:00 | 370.70 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2026-02-19 10:55:00 | 342.45 | 2026-02-19 12:10:00 | 340.93 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-19 10:55:00 | 342.45 | 2026-02-19 12:20:00 | 342.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:30:00 | 429.70 | 2026-04-21 09:35:00 | 428.52 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-05 09:55:00 | 416.75 | 2026-05-05 10:00:00 | 414.66 | STOP_HIT | 1.00 | -0.50% |
