# Central Bank of India (CENTRALBK)

## Backtest Summary

- **Window:** 2024-11-07 09:15:00 → 2026-05-08 15:25:00 (27688 bars)
- **Last close:** 36.55
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 16
- **Target hits / Stop hits / Partials:** 2 / 16 / 7
- **Avg / median % per leg:** 0.27% / 0.00%
- **Sum % (uncompounded):** 6.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 4 | 28.6% | 1 | 10 | 3 | 0.37% | 5.1% |
| BUY @ 2nd Alert (retest1) | 14 | 4 | 28.6% | 1 | 10 | 3 | 0.37% | 5.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 5 | 45.5% | 1 | 6 | 4 | 0.14% | 1.5% |
| SELL @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 1 | 6 | 4 | 0.14% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 9 | 36.0% | 2 | 16 | 7 | 0.27% | 6.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-11-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:55:00 | 56.48 | 56.16 | 0.00 | ORB-long ORB[55.80,56.45] vol=1.5x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 10:15:00 | 56.81 | 56.28 | 0.00 | T1 1.5R @ 56.81 |
| Stop hit — per-position SL triggered | 2024-11-27 10:25:00 | 56.48 | 56.31 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-12-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:35:00 | 57.49 | 57.27 | 0.00 | ORB-long ORB[56.90,57.48] vol=3.0x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 09:40:00 | 57.81 | 57.69 | 0.00 | T1 1.5R @ 57.81 |
| Target hit | 2024-12-04 15:20:00 | 61.22 | 60.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-12-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:05:00 | 59.54 | 60.05 | 0.00 | ORB-short ORB[59.88,60.49] vol=3.9x ATR=0.29 |
| Stop hit — per-position SL triggered | 2024-12-06 10:10:00 | 59.83 | 60.00 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-12-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:10:00 | 58.32 | 58.55 | 0.00 | ORB-short ORB[58.55,59.09] vol=2.0x ATR=0.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 12:10:00 | 58.14 | 58.50 | 0.00 | T1 1.5R @ 58.14 |
| Stop hit — per-position SL triggered | 2024-12-12 12:45:00 | 58.32 | 58.47 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-12-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 09:45:00 | 58.66 | 58.21 | 0.00 | ORB-long ORB[57.85,58.38] vol=1.5x ATR=0.22 |
| Stop hit — per-position SL triggered | 2024-12-16 09:55:00 | 58.44 | 58.37 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 09:30:00 | 54.59 | 54.45 | 0.00 | ORB-long ORB[54.19,54.50] vol=3.6x ATR=0.23 |
| Stop hit — per-position SL triggered | 2024-12-26 09:45:00 | 54.36 | 54.46 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-12-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:50:00 | 53.66 | 53.80 | 0.00 | ORB-short ORB[53.69,54.15] vol=3.2x ATR=0.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 12:15:00 | 53.49 | 53.76 | 0.00 | T1 1.5R @ 53.49 |
| Target hit | 2024-12-27 15:20:00 | 53.08 | 53.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2024-12-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 09:55:00 | 52.93 | 52.63 | 0.00 | ORB-long ORB[52.20,52.78] vol=2.0x ATR=0.23 |
| Stop hit — per-position SL triggered | 2024-12-31 10:05:00 | 52.70 | 52.65 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 53.61 | 53.38 | 0.00 | ORB-long ORB[53.05,53.48] vol=1.5x ATR=0.15 |
| Stop hit — per-position SL triggered | 2025-01-01 11:25:00 | 53.46 | 53.40 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-01-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:05:00 | 53.47 | 53.80 | 0.00 | ORB-short ORB[53.73,54.10] vol=1.6x ATR=0.15 |
| Stop hit — per-position SL triggered | 2025-01-02 11:15:00 | 53.62 | 53.62 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 52.03 | 52.32 | 0.00 | ORB-short ORB[52.12,52.70] vol=1.9x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:05:00 | 51.80 | 52.25 | 0.00 | T1 1.5R @ 51.80 |
| Stop hit — per-position SL triggered | 2025-01-09 12:15:00 | 52.03 | 52.18 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-01-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:40:00 | 53.37 | 53.11 | 0.00 | ORB-long ORB[52.61,53.30] vol=1.6x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-01-20 09:55:00 | 53.16 | 53.17 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-01-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 09:30:00 | 51.81 | 51.97 | 0.00 | ORB-short ORB[51.85,52.41] vol=4.5x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-01-23 09:35:00 | 51.98 | 51.97 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-01-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 10:30:00 | 51.63 | 51.37 | 0.00 | ORB-long ORB[50.86,51.60] vol=2.1x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 10:40:00 | 51.90 | 51.42 | 0.00 | T1 1.5R @ 51.90 |
| Stop hit — per-position SL triggered | 2025-01-30 10:50:00 | 51.63 | 51.45 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-02-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 09:30:00 | 52.54 | 52.31 | 0.00 | ORB-long ORB[52.05,52.43] vol=1.9x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-02-01 09:45:00 | 52.35 | 52.38 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-02-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:45:00 | 51.09 | 50.82 | 0.00 | ORB-long ORB[50.52,50.99] vol=1.6x ATR=0.20 |
| Stop hit — per-position SL triggered | 2025-02-04 10:05:00 | 50.89 | 50.90 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-02-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:00:00 | 51.34 | 51.54 | 0.00 | ORB-short ORB[51.36,51.99] vol=5.3x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 12:30:00 | 51.10 | 51.48 | 0.00 | T1 1.5R @ 51.10 |
| Stop hit — per-position SL triggered | 2025-02-06 14:20:00 | 51.34 | 51.44 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-02-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 09:35:00 | 51.34 | 51.13 | 0.00 | ORB-long ORB[50.89,51.21] vol=1.5x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-02-07 09:50:00 | 51.16 | 51.23 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-11-27 09:55:00 | 56.48 | 2024-11-27 10:15:00 | 56.81 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-11-27 09:55:00 | 56.48 | 2024-11-27 10:25:00 | 56.48 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-04 09:35:00 | 57.49 | 2024-12-04 09:40:00 | 57.81 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-12-04 09:35:00 | 57.49 | 2024-12-04 15:20:00 | 61.22 | TARGET_HIT | 0.50 | 6.49% |
| SELL | retest1 | 2024-12-06 10:05:00 | 59.54 | 2024-12-06 10:10:00 | 59.83 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-12-12 11:10:00 | 58.32 | 2024-12-12 12:10:00 | 58.14 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-12-12 11:10:00 | 58.32 | 2024-12-12 12:45:00 | 58.32 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-16 09:45:00 | 58.66 | 2024-12-16 09:55:00 | 58.44 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-26 09:30:00 | 54.59 | 2024-12-26 09:45:00 | 54.36 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-27 10:50:00 | 53.66 | 2024-12-27 12:15:00 | 53.49 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-12-27 10:50:00 | 53.66 | 2024-12-27 15:20:00 | 53.08 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2024-12-31 09:55:00 | 52.93 | 2024-12-31 10:05:00 | 52.70 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-01-01 10:50:00 | 53.61 | 2025-01-01 11:25:00 | 53.46 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-02 10:05:00 | 53.47 | 2025-01-02 11:15:00 | 53.62 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-09 10:45:00 | 52.03 | 2025-01-09 11:05:00 | 51.80 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-01-09 10:45:00 | 52.03 | 2025-01-09 12:15:00 | 52.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-20 09:40:00 | 53.37 | 2025-01-20 09:55:00 | 53.16 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-01-23 09:30:00 | 51.81 | 2025-01-23 09:35:00 | 51.98 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-30 10:30:00 | 51.63 | 2025-01-30 10:40:00 | 51.90 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-01-30 10:30:00 | 51.63 | 2025-01-30 10:50:00 | 51.63 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-01 09:30:00 | 52.54 | 2025-02-01 09:45:00 | 52.35 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-02-04 09:45:00 | 51.09 | 2025-02-04 10:05:00 | 50.89 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-02-06 11:00:00 | 51.34 | 2025-02-06 12:30:00 | 51.10 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-02-06 11:00:00 | 51.34 | 2025-02-06 14:20:00 | 51.34 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-07 09:35:00 | 51.34 | 2025-02-07 09:50:00 | 51.16 | STOP_HIT | 1.00 | -0.35% |
