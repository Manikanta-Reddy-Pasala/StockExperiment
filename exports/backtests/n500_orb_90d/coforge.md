# Coforge Ltd. (COFORGE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1365.20
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
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 0.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.13% | 0.6% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.13% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.31% | -0.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.31% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.06% | 0.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 1563.60 | 1557.29 | 0.00 | ORB-long ORB[1542.40,1561.00] vol=1.9x ATR=4.52 |
| Stop hit — per-position SL triggered | 2026-02-10 10:10:00 | 1559.08 | 1559.72 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:05:00 | 1531.30 | 1544.47 | 0.00 | ORB-short ORB[1550.30,1560.00] vol=2.0x ATR=4.68 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 1535.98 | 1543.22 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:55:00 | 1293.50 | 1289.35 | 0.00 | ORB-long ORB[1280.00,1292.40] vol=6.9x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:05:00 | 1298.23 | 1289.59 | 0.00 | T1 1.5R @ 1298.23 |
| Stop hit — per-position SL triggered | 2026-04-21 14:10:00 | 1293.50 | 1293.00 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:45:00 | 1217.70 | 1209.01 | 0.00 | ORB-long ORB[1201.50,1214.00] vol=1.7x ATR=4.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 09:50:00 | 1224.61 | 1214.07 | 0.00 | T1 1.5R @ 1224.61 |
| Stop hit — per-position SL triggered | 2026-04-29 09:55:00 | 1217.70 | 1214.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:40:00 | 1563.60 | 2026-02-10 10:10:00 | 1559.08 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-11 10:05:00 | 1531.30 | 2026-02-11 10:15:00 | 1535.98 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-21 10:55:00 | 1293.50 | 2026-04-21 11:05:00 | 1298.23 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-04-21 10:55:00 | 1293.50 | 2026-04-21 14:10:00 | 1293.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 09:45:00 | 1217.70 | 2026-04-29 09:50:00 | 1224.61 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-29 09:45:00 | 1217.70 | 2026-04-29 09:55:00 | 1217.70 | STOP_HIT | 0.50 | 0.00% |
