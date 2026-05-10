# Pine Labs Ltd. (PINELABS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 196.60
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
- **Avg / median % per leg:** 0.20% / 0.66%
- **Sum % (uncompounded):** 1.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.11% | 0.5% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.11% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.37% | 0.7% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.37% | 0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.20% | 1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:10:00 | 210.53 | 213.02 | 0.00 | ORB-short ORB[212.55,215.06] vol=2.6x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:15:00 | 208.96 | 212.73 | 0.00 | T1 1.5R @ 208.96 |
| Stop hit — per-position SL triggered | 2026-02-10 10:25:00 | 210.53 | 212.57 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 205.54 | 203.70 | 0.00 | ORB-long ORB[200.92,203.99] vol=2.5x ATR=0.99 |
| Stop hit — per-position SL triggered | 2026-02-17 10:25:00 | 204.55 | 203.85 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:30:00 | 161.14 | 159.78 | 0.00 | ORB-long ORB[158.55,160.58] vol=2.3x ATR=0.96 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 160.18 | 160.45 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:35:00 | 203.80 | 201.34 | 0.00 | ORB-long ORB[198.38,199.98] vol=2.3x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:40:00 | 205.57 | 202.74 | 0.00 | T1 1.5R @ 205.57 |
| Target hit | 2026-04-28 11:25:00 | 205.15 | 205.67 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:10:00 | 210.53 | 2026-02-10 10:15:00 | 208.96 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2026-02-10 10:10:00 | 210.53 | 2026-02-10 10:25:00 | 210.53 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:20:00 | 205.54 | 2026-02-17 10:25:00 | 204.55 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-03-16 09:30:00 | 161.14 | 2026-03-16 10:15:00 | 160.18 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2026-04-28 09:35:00 | 203.80 | 2026-04-28 09:40:00 | 205.57 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2026-04-28 09:35:00 | 203.80 | 2026-04-28 11:25:00 | 205.15 | TARGET_HIT | 0.50 | 0.66% |
