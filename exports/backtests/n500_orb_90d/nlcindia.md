# NLC India Ltd. (NLCINDIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 328.00
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 10
- **Target hits / Stop hits / Partials:** 0 / 10 / 7
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 3.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 0 | 8 | 5 | 0.22% | 2.9% |
| BUY @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 0 | 8 | 5 | 0.22% | 2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 0 | 2 | 2 | 0.23% | 0.9% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 2 | 2 | 0.23% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 7 | 41.2% | 0 | 10 | 7 | 0.22% | 3.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:10:00 | 252.65 | 254.46 | 0.00 | ORB-short ORB[254.70,257.90] vol=2.7x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:30:00 | 251.42 | 253.97 | 0.00 | T1 1.5R @ 251.42 |
| Stop hit — per-position SL triggered | 2026-02-13 12:25:00 | 252.65 | 253.57 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:45:00 | 263.80 | 260.35 | 0.00 | ORB-long ORB[257.20,260.70] vol=2.7x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:00:00 | 266.12 | 261.97 | 0.00 | T1 1.5R @ 266.12 |
| Stop hit — per-position SL triggered | 2026-02-20 11:45:00 | 263.80 | 263.48 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:45:00 | 264.25 | 265.74 | 0.00 | ORB-short ORB[264.40,267.50] vol=3.7x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:35:00 | 263.12 | 265.36 | 0.00 | T1 1.5R @ 263.12 |
| Stop hit — per-position SL triggered | 2026-02-25 14:25:00 | 264.25 | 264.69 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:00:00 | 257.75 | 254.04 | 0.00 | ORB-long ORB[249.25,252.85] vol=5.6x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:05:00 | 259.76 | 255.72 | 0.00 | T1 1.5R @ 259.76 |
| Stop hit — per-position SL triggered | 2026-03-05 10:25:00 | 257.75 | 257.87 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:40:00 | 258.00 | 255.83 | 0.00 | ORB-long ORB[251.80,254.80] vol=3.0x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 09:45:00 | 260.03 | 256.70 | 0.00 | T1 1.5R @ 260.03 |
| Stop hit — per-position SL triggered | 2026-03-06 10:05:00 | 258.00 | 257.09 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:10:00 | 245.20 | 243.21 | 0.00 | ORB-long ORB[242.20,244.55] vol=2.5x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:20:00 | 246.41 | 243.62 | 0.00 | T1 1.5R @ 246.41 |
| Stop hit — per-position SL triggered | 2026-03-10 12:00:00 | 245.20 | 244.09 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 272.65 | 271.82 | 0.00 | ORB-long ORB[269.40,272.55] vol=1.7x ATR=0.85 |
| Stop hit — per-position SL triggered | 2026-04-10 09:35:00 | 271.80 | 271.85 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:30:00 | 271.15 | 266.65 | 0.00 | ORB-long ORB[261.10,265.15] vol=1.9x ATR=1.22 |
| Stop hit — per-position SL triggered | 2026-04-13 10:45:00 | 269.93 | 267.27 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:55:00 | 278.35 | 276.43 | 0.00 | ORB-long ORB[274.00,277.00] vol=2.9x ATR=1.17 |
| Stop hit — per-position SL triggered | 2026-04-15 10:40:00 | 277.18 | 277.16 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:05:00 | 314.30 | 309.96 | 0.00 | ORB-long ORB[302.25,306.95] vol=1.6x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 12:25:00 | 317.75 | 312.66 | 0.00 | T1 1.5R @ 317.75 |
| Stop hit — per-position SL triggered | 2026-04-27 13:55:00 | 314.30 | 314.58 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 11:10:00 | 252.65 | 2026-02-13 11:30:00 | 251.42 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-02-13 11:10:00 | 252.65 | 2026-02-13 12:25:00 | 252.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 09:45:00 | 263.80 | 2026-02-20 10:00:00 | 266.12 | PARTIAL | 0.50 | 0.88% |
| BUY | retest1 | 2026-02-20 09:45:00 | 263.80 | 2026-02-20 11:45:00 | 263.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 10:45:00 | 264.25 | 2026-02-25 12:35:00 | 263.12 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-25 10:45:00 | 264.25 | 2026-02-25 14:25:00 | 264.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-05 10:00:00 | 257.75 | 2026-03-05 10:05:00 | 259.76 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2026-03-05 10:00:00 | 257.75 | 2026-03-05 10:25:00 | 257.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 09:40:00 | 258.00 | 2026-03-06 09:45:00 | 260.03 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2026-03-06 09:40:00 | 258.00 | 2026-03-06 10:05:00 | 258.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 11:10:00 | 245.20 | 2026-03-10 11:20:00 | 246.41 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-03-10 11:10:00 | 245.20 | 2026-03-10 12:00:00 | 245.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:30:00 | 272.65 | 2026-04-10 09:35:00 | 271.80 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-13 10:30:00 | 271.15 | 2026-04-13 10:45:00 | 269.93 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-15 09:55:00 | 278.35 | 2026-04-15 10:40:00 | 277.18 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-27 10:05:00 | 314.30 | 2026-04-27 12:25:00 | 317.75 | PARTIAL | 0.50 | 1.10% |
| BUY | retest1 | 2026-04-27 10:05:00 | 314.30 | 2026-04-27 13:55:00 | 314.30 | STOP_HIT | 0.50 | 0.00% |
