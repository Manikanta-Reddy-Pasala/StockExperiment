# Jaiprakash Power Ventures Ltd. (JPPOWER)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 19.02
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
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** -0.03% / 0.00%
- **Sum % (uncompounded):** -0.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.09% | -0.4% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.09% | -0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.06% | 0.2% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.06% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.03% | -0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:15:00 | 15.02 | 14.96 | 0.00 | ORB-long ORB[14.78,14.99] vol=1.7x ATR=0.05 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 14.97 | 14.96 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 14.60 | 14.50 | 0.00 | ORB-long ORB[14.43,14.57] vol=3.0x ATR=0.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:20:00 | 14.67 | 14.53 | 0.00 | T1 1.5R @ 14.67 |
| Stop hit — per-position SL triggered | 2026-02-25 11:30:00 | 14.60 | 14.53 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-04-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:25:00 | 19.28 | 19.13 | 0.00 | ORB-long ORB[18.96,19.25] vol=2.1x ATR=0.10 |
| Stop hit — per-position SL triggered | 2026-04-21 10:35:00 | 19.18 | 19.14 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 19.69 | 19.80 | 0.00 | ORB-short ORB[19.73,20.00] vol=2.3x ATR=0.10 |
| Stop hit — per-position SL triggered | 2026-05-04 10:35:00 | 19.79 | 19.77 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 19.20 | 19.27 | 0.00 | ORB-short ORB[19.22,19.42] vol=1.6x ATR=0.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:40:00 | 19.07 | 19.24 | 0.00 | T1 1.5R @ 19.07 |
| Stop hit — per-position SL triggered | 2026-05-08 09:50:00 | 19.20 | 19.22 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 10:15:00 | 15.02 | 2026-02-17 10:40:00 | 14.97 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-25 11:05:00 | 14.60 | 2026-02-25 11:20:00 | 14.67 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-25 11:05:00 | 14.60 | 2026-02-25 11:30:00 | 14.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 10:25:00 | 19.28 | 2026-04-21 10:35:00 | 19.18 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2026-05-04 09:35:00 | 19.69 | 2026-05-04 10:35:00 | 19.79 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-05-08 09:35:00 | 19.20 | 2026-05-08 09:40:00 | 19.07 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2026-05-08 09:35:00 | 19.20 | 2026-05-08 09:50:00 | 19.20 | STOP_HIT | 0.50 | 0.00% |
