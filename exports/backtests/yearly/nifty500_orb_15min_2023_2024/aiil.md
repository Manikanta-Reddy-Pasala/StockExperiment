# Authum Investment & Infrastructure Ltd. (AIIL)

## Backtest Summary

- **Window:** 2024-04-23 09:15:00 → 2024-11-01 18:55:00 (9933 bars)
- **Last close:** 335.00
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 0.67% / 0.93%
- **Sum % (uncompounded):** 2.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.47% | 0.9% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.47% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 2 | 100.0% | 1 | 0 | 1 | 0.86% | 1.7% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 0.86% | 1.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.67% | 2.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 09:35:00 | 166.08 | 167.34 | 0.00 | ORB-short ORB[166.20,168.51] vol=1.8x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:50:00 | 164.77 | 166.85 | 0.00 | T1 1.5R @ 164.77 |
| Target hit | 2024-05-06 11:00:00 | 164.53 | 163.52 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2024-05-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-09 10:00:00 | 158.11 | 157.11 | 0.00 | ORB-long ORB[155.07,157.03] vol=1.6x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:05:00 | 159.59 | 157.47 | 0.00 | T1 1.5R @ 159.59 |
| Stop hit — per-position SL triggered | 2024-05-09 10:40:00 | 158.11 | 157.92 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-06 09:35:00 | 166.08 | 2024-05-06 09:50:00 | 164.77 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2024-05-06 09:35:00 | 166.08 | 2024-05-06 11:00:00 | 164.53 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2024-05-09 10:00:00 | 158.11 | 2024-05-09 10:05:00 | 159.59 | PARTIAL | 0.50 | 0.94% |
| BUY | retest1 | 2024-05-09 10:00:00 | 158.11 | 2024-05-09 10:40:00 | 158.11 | STOP_HIT | 0.50 | 0.00% |
