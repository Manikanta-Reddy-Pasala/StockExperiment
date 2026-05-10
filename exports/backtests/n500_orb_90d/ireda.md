# Indian Renewable Energy Development Agency Ltd. (IREDA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 134.70
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 11
- **Target hits / Stop hits / Partials:** 0 / 11 / 5
- **Avg / median % per leg:** 0.01% / 0.00%
- **Sum % (uncompounded):** 0.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.09% | -0.7% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.09% | -0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 3 | 37.5% | 0 | 5 | 3 | 0.12% | 1.0% |
| SELL @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 0 | 5 | 3 | 0.12% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 5 | 31.2% | 0 | 11 | 5 | 0.01% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 125.26 | 123.88 | 0.00 | ORB-long ORB[122.26,124.06] vol=2.9x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-02-16 09:40:00 | 124.82 | 124.19 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:50:00 | 125.65 | 126.49 | 0.00 | ORB-short ORB[126.16,127.44] vol=2.7x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:15:00 | 125.12 | 126.36 | 0.00 | T1 1.5R @ 125.12 |
| Stop hit — per-position SL triggered | 2026-02-23 11:35:00 | 125.65 | 126.31 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 126.96 | 126.03 | 0.00 | ORB-long ORB[125.40,126.77] vol=1.9x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:25:00 | 127.43 | 126.20 | 0.00 | T1 1.5R @ 127.43 |
| Stop hit — per-position SL triggered | 2026-02-24 11:45:00 | 126.96 | 126.30 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:30:00 | 123.08 | 123.58 | 0.00 | ORB-short ORB[123.11,124.50] vol=2.3x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:55:00 | 122.41 | 123.21 | 0.00 | T1 1.5R @ 122.41 |
| Stop hit — per-position SL triggered | 2026-02-27 10:35:00 | 123.08 | 123.09 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 126.11 | 125.34 | 0.00 | ORB-long ORB[123.74,125.60] vol=4.3x ATR=0.56 |
| Stop hit — per-position SL triggered | 2026-04-10 09:35:00 | 125.55 | 125.39 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 128.02 | 129.16 | 0.00 | ORB-short ORB[128.39,130.20] vol=1.9x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:55:00 | 127.13 | 128.78 | 0.00 | T1 1.5R @ 127.13 |
| Stop hit — per-position SL triggered | 2026-04-16 10:25:00 | 128.02 | 128.64 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 130.61 | 129.73 | 0.00 | ORB-long ORB[128.68,129.95] vol=2.1x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:40:00 | 131.15 | 129.99 | 0.00 | T1 1.5R @ 131.15 |
| Stop hit — per-position SL triggered | 2026-04-22 09:45:00 | 130.61 | 130.04 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 136.39 | 137.46 | 0.00 | ORB-short ORB[136.71,138.64] vol=2.0x ATR=0.58 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 136.97 | 137.04 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:55:00 | 138.80 | 137.48 | 0.00 | ORB-long ORB[136.22,138.24] vol=2.3x ATR=0.52 |
| Stop hit — per-position SL triggered | 2026-04-27 11:55:00 | 138.28 | 137.99 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 139.46 | 138.94 | 0.00 | ORB-long ORB[138.06,139.30] vol=2.1x ATR=0.48 |
| Stop hit — per-position SL triggered | 2026-04-28 09:55:00 | 138.98 | 139.19 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:45:00 | 133.95 | 134.77 | 0.00 | ORB-short ORB[134.52,135.99] vol=2.1x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-05-08 09:50:00 | 134.33 | 134.75 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 09:30:00 | 125.26 | 2026-02-16 09:40:00 | 124.82 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-23 10:50:00 | 125.65 | 2026-02-23 11:15:00 | 125.12 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-23 10:50:00 | 125.65 | 2026-02-23 11:35:00 | 125.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 11:10:00 | 126.96 | 2026-02-24 11:25:00 | 127.43 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-24 11:10:00 | 126.96 | 2026-02-24 11:45:00 | 126.96 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 09:30:00 | 123.08 | 2026-02-27 09:55:00 | 122.41 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-02-27 09:30:00 | 123.08 | 2026-02-27 10:35:00 | 123.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:30:00 | 126.11 | 2026-04-10 09:35:00 | 125.55 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-16 09:30:00 | 128.02 | 2026-04-16 09:55:00 | 127.13 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-04-16 09:30:00 | 128.02 | 2026-04-16 10:25:00 | 128.02 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:35:00 | 130.61 | 2026-04-22 09:40:00 | 131.15 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-04-22 09:35:00 | 130.61 | 2026-04-22 09:45:00 | 130.61 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 09:30:00 | 136.39 | 2026-04-24 10:00:00 | 136.97 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-27 10:55:00 | 138.80 | 2026-04-27 11:55:00 | 138.28 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-28 09:30:00 | 139.46 | 2026-04-28 09:55:00 | 138.98 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-05-08 09:45:00 | 133.95 | 2026-05-08 09:50:00 | 134.33 | STOP_HIT | 1.00 | -0.28% |
