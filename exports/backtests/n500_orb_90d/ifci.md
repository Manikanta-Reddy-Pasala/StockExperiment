# IFCI Ltd. (IFCI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 64.27
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / Stop hits / Partials:** 0 / 6 / 1
- **Avg / median % per leg:** -0.27% / -0.44%
- **Sum % (uncompounded):** -1.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.48% | -1.9% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.48% | -1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.27% | -1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:50:00 | 62.24 | 61.94 | 0.00 | ORB-long ORB[61.43,62.19] vol=2.9x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-02-12 10:35:00 | 61.88 | 62.09 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 59.90 | 60.53 | 0.00 | ORB-short ORB[60.35,61.24] vol=2.7x ATR=0.28 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 60.18 | 60.42 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 58.60 | 58.84 | 0.00 | ORB-short ORB[58.67,59.30] vol=1.8x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 09:45:00 | 58.32 | 58.75 | 0.00 | T1 1.5R @ 58.32 |
| Stop hit — per-position SL triggered | 2026-02-25 09:55:00 | 58.60 | 58.72 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:15:00 | 59.53 | 58.98 | 0.00 | ORB-long ORB[58.45,59.20] vol=3.0x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-02-26 10:20:00 | 59.28 | 59.01 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 61.75 | 61.50 | 0.00 | ORB-long ORB[61.01,61.69] vol=2.3x ATR=0.27 |
| Stop hit — per-position SL triggered | 2026-04-28 10:15:00 | 61.48 | 61.54 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:30:00 | 58.66 | 58.06 | 0.00 | ORB-long ORB[57.64,58.39] vol=1.9x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-04-30 09:35:00 | 58.37 | 58.08 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 09:50:00 | 62.24 | 2026-02-12 10:35:00 | 61.88 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2026-02-13 09:30:00 | 59.90 | 2026-02-13 09:40:00 | 60.18 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-02-25 09:35:00 | 58.60 | 2026-02-25 09:45:00 | 58.32 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-25 09:35:00 | 58.60 | 2026-02-25 09:55:00 | 58.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 10:15:00 | 59.53 | 2026-02-26 10:20:00 | 59.28 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-28 09:45:00 | 61.75 | 2026-04-28 10:15:00 | 61.48 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-30 09:30:00 | 58.66 | 2026-04-30 09:35:00 | 58.37 | STOP_HIT | 1.00 | -0.49% |
