# Coal India Ltd. (COALINDIA)

## Backtest Summary

- **Window:** 2025-11-07 09:15:00 → 2026-05-05 15:25:00 (5850 bars)
- **Last close:** 472.85
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
| ENTRY1 | 27 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 5 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 22
- **Target hits / Stop hits / Partials:** 5 / 22 / 11
- **Avg / median % per leg:** 0.30% / 0.00%
- **Sum % (uncompounded):** 11.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 14 | 56.0% | 5 | 11 | 9 | 0.50% | 12.6% |
| BUY @ 2nd Alert (retest1) | 25 | 14 | 56.0% | 5 | 11 | 9 | 0.50% | 12.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 2 | 15.4% | 0 | 11 | 2 | -0.08% | -1.1% |
| SELL @ 2nd Alert (retest1) | 13 | 2 | 15.4% | 0 | 11 | 2 | -0.08% | -1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 38 | 16 | 42.1% | 5 | 22 | 11 | 0.30% | 11.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 10:45:00 | 374.40 | 372.89 | 0.00 | ORB-long ORB[371.55,373.25] vol=3.3x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 11:35:00 | 375.75 | 373.63 | 0.00 | T1 1.5R @ 375.75 |
| Stop hit — per-position SL triggered | 2025-11-07 12:00:00 | 374.40 | 373.86 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-11-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:45:00 | 379.65 | 378.49 | 0.00 | ORB-long ORB[376.10,378.85] vol=4.1x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 10:55:00 | 380.65 | 378.72 | 0.00 | T1 1.5R @ 380.65 |
| Target hit | 2025-11-10 15:20:00 | 381.35 | 380.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2025-11-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:45:00 | 385.25 | 384.04 | 0.00 | ORB-long ORB[382.30,384.60] vol=1.6x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 10:50:00 | 386.48 | 384.34 | 0.00 | T1 1.5R @ 386.48 |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 385.25 | 384.64 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-11-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:30:00 | 383.95 | 385.68 | 0.00 | ORB-short ORB[386.15,387.85] vol=2.5x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-11-18 10:35:00 | 384.60 | 385.59 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-11-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 10:50:00 | 380.90 | 380.94 | 0.00 | ORB-short ORB[381.35,383.85] vol=2.0x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 11:00:00 | 379.92 | 380.85 | 0.00 | T1 1.5R @ 379.92 |
| Stop hit — per-position SL triggered | 2025-11-19 11:40:00 | 380.90 | 380.75 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-11-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 10:35:00 | 375.70 | 376.18 | 0.00 | ORB-short ORB[375.75,378.20] vol=3.2x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-11-24 10:45:00 | 376.26 | 376.19 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-11-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 10:55:00 | 371.65 | 372.66 | 0.00 | ORB-short ORB[371.95,373.35] vol=1.7x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-11-25 11:15:00 | 372.17 | 372.40 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-11-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 11:00:00 | 375.00 | 374.49 | 0.00 | ORB-long ORB[371.30,374.90] vol=9.0x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 11:10:00 | 375.94 | 374.66 | 0.00 | T1 1.5R @ 375.94 |
| Stop hit — per-position SL triggered | 2025-11-26 11:55:00 | 375.00 | 375.37 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-11-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 11:05:00 | 376.35 | 377.02 | 0.00 | ORB-short ORB[377.05,379.20] vol=1.6x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-11-28 11:10:00 | 376.87 | 377.02 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-12-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 11:05:00 | 375.85 | 376.89 | 0.00 | ORB-short ORB[376.70,378.00] vol=2.3x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 376.33 | 376.86 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-12-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:55:00 | 373.60 | 374.61 | 0.00 | ORB-short ORB[375.80,379.30] vol=1.7x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-12-03 11:05:00 | 374.24 | 374.36 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-12-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 11:10:00 | 376.10 | 375.22 | 0.00 | ORB-long ORB[373.80,375.75] vol=2.4x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 13:05:00 | 376.88 | 375.79 | 0.00 | T1 1.5R @ 376.88 |
| Target hit | 2025-12-04 15:20:00 | 379.05 | 377.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2025-12-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 10:45:00 | 377.15 | 375.19 | 0.00 | ORB-long ORB[373.35,376.50] vol=1.5x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 11:10:00 | 378.44 | 375.43 | 0.00 | T1 1.5R @ 378.44 |
| Target hit | 2025-12-09 15:20:00 | 378.75 | 378.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2025-12-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:45:00 | 382.05 | 380.65 | 0.00 | ORB-long ORB[378.25,380.80] vol=2.8x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-12-10 10:55:00 | 381.28 | 380.89 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-12-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:10:00 | 384.65 | 383.27 | 0.00 | ORB-long ORB[381.50,383.85] vol=2.2x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-12-11 11:35:00 | 384.06 | 383.47 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-12-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 10:30:00 | 381.10 | 382.36 | 0.00 | ORB-short ORB[382.55,384.20] vol=1.7x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:15:00 | 380.24 | 381.78 | 0.00 | T1 1.5R @ 380.24 |
| Stop hit — per-position SL triggered | 2025-12-16 12:25:00 | 381.10 | 380.20 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-12-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:55:00 | 384.05 | 382.79 | 0.00 | ORB-long ORB[380.80,382.75] vol=2.0x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-12-17 10:25:00 | 383.20 | 383.37 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-12-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:30:00 | 386.45 | 385.01 | 0.00 | ORB-long ORB[383.65,385.60] vol=5.2x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-12-18 10:35:00 | 385.76 | 385.07 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-12-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 11:10:00 | 384.55 | 385.12 | 0.00 | ORB-short ORB[385.00,387.35] vol=2.2x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-12-19 11:35:00 | 385.13 | 384.98 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-12-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 09:55:00 | 402.80 | 401.30 | 0.00 | ORB-long ORB[398.15,401.50] vol=1.6x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-12-31 10:05:00 | 401.72 | 401.40 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-01-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:10:00 | 400.90 | 399.75 | 0.00 | ORB-long ORB[398.55,400.75] vol=2.5x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 11:20:00 | 401.99 | 399.96 | 0.00 | T1 1.5R @ 401.99 |
| Stop hit — per-position SL triggered | 2026-01-01 11:30:00 | 400.90 | 400.08 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2026-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:30:00 | 404.45 | 403.10 | 0.00 | ORB-long ORB[401.00,403.85] vol=1.9x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 09:35:00 | 405.62 | 403.96 | 0.00 | T1 1.5R @ 405.62 |
| Target hit | 2026-01-02 15:20:00 | 428.10 | 420.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2026-03-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:25:00 | 453.50 | 448.98 | 0.00 | ORB-long ORB[445.00,451.00] vol=1.9x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:45:00 | 455.91 | 450.68 | 0.00 | T1 1.5R @ 455.91 |
| Target hit | 2026-03-12 15:20:00 | 470.00 | 464.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — SELL (started 2026-03-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:30:00 | 442.40 | 444.30 | 0.00 | ORB-short ORB[443.50,447.50] vol=4.1x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-03-27 09:50:00 | 443.61 | 443.88 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2026-04-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:35:00 | 436.00 | 434.47 | 0.00 | ORB-long ORB[430.50,435.45] vol=2.7x ATR=0.93 |
| Stop hit — per-position SL triggered | 2026-04-17 10:45:00 | 435.07 | 434.57 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2026-04-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:50:00 | 446.95 | 444.71 | 0.00 | ORB-long ORB[442.55,445.35] vol=1.7x ATR=0.79 |
| Stop hit — per-position SL triggered | 2026-04-22 09:55:00 | 446.16 | 444.86 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2026-05-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:45:00 | 472.50 | 475.08 | 0.00 | ORB-short ORB[476.10,480.00] vol=3.6x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-05-05 11:10:00 | 473.62 | 474.39 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-11-07 10:45:00 | 374.40 | 2025-11-07 11:35:00 | 375.75 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-11-07 10:45:00 | 374.40 | 2025-11-07 12:00:00 | 374.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-10 10:45:00 | 379.65 | 2025-11-10 10:55:00 | 380.65 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-11-10 10:45:00 | 379.65 | 2025-11-10 15:20:00 | 381.35 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2025-11-14 10:45:00 | 385.25 | 2025-11-14 10:50:00 | 386.48 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-11-14 10:45:00 | 385.25 | 2025-11-14 11:15:00 | 385.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-18 10:30:00 | 383.95 | 2025-11-18 10:35:00 | 384.60 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-11-19 10:50:00 | 380.90 | 2025-11-19 11:00:00 | 379.92 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-11-19 10:50:00 | 380.90 | 2025-11-19 11:40:00 | 380.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-24 10:35:00 | 375.70 | 2025-11-24 10:45:00 | 376.26 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-11-25 10:55:00 | 371.65 | 2025-11-25 11:15:00 | 372.17 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-11-26 11:00:00 | 375.00 | 2025-11-26 11:10:00 | 375.94 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-11-26 11:00:00 | 375.00 | 2025-11-26 11:55:00 | 375.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-28 11:05:00 | 376.35 | 2025-11-28 11:10:00 | 376.87 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-12-01 11:05:00 | 375.85 | 2025-12-01 11:15:00 | 376.33 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-12-03 10:55:00 | 373.60 | 2025-12-03 11:05:00 | 374.24 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-12-04 11:10:00 | 376.10 | 2025-12-04 13:05:00 | 376.88 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-12-04 11:10:00 | 376.10 | 2025-12-04 15:20:00 | 379.05 | TARGET_HIT | 0.50 | 0.78% |
| BUY | retest1 | 2025-12-09 10:45:00 | 377.15 | 2025-12-09 11:10:00 | 378.44 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-12-09 10:45:00 | 377.15 | 2025-12-09 15:20:00 | 378.75 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2025-12-10 10:45:00 | 382.05 | 2025-12-10 10:55:00 | 381.28 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-11 11:10:00 | 384.65 | 2025-12-11 11:35:00 | 384.06 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-12-16 10:30:00 | 381.10 | 2025-12-16 11:15:00 | 380.24 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-12-16 10:30:00 | 381.10 | 2025-12-16 12:25:00 | 381.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-17 09:55:00 | 384.05 | 2025-12-17 10:25:00 | 383.20 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-18 10:30:00 | 386.45 | 2025-12-18 10:35:00 | 385.76 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-19 11:10:00 | 384.55 | 2025-12-19 11:35:00 | 385.13 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-12-31 09:55:00 | 402.80 | 2025-12-31 10:05:00 | 401.72 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-01-01 11:10:00 | 400.90 | 2026-01-01 11:20:00 | 401.99 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-01-01 11:10:00 | 400.90 | 2026-01-01 11:30:00 | 400.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 09:30:00 | 404.45 | 2026-01-02 09:35:00 | 405.62 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2026-01-02 09:30:00 | 404.45 | 2026-01-02 15:20:00 | 428.10 | TARGET_HIT | 0.50 | 5.85% |
| BUY | retest1 | 2026-03-12 10:25:00 | 453.50 | 2026-03-12 10:45:00 | 455.91 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-03-12 10:25:00 | 453.50 | 2026-03-12 15:20:00 | 470.00 | TARGET_HIT | 0.50 | 3.64% |
| SELL | retest1 | 2026-03-27 09:30:00 | 442.40 | 2026-03-27 09:50:00 | 443.61 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-17 10:35:00 | 436.00 | 2026-04-17 10:45:00 | 435.07 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-04-22 09:50:00 | 446.95 | 2026-04-22 09:55:00 | 446.16 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-05-05 10:45:00 | 472.50 | 2026-05-05 11:10:00 | 473.62 | STOP_HIT | 1.00 | -0.24% |
