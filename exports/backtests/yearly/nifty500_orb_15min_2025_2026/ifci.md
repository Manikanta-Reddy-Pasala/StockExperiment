# IFCI Ltd. (IFCI)

## Backtest Summary

- **Window:** 2025-10-08 09:15:00 → 2026-05-08 15:25:00 (9088 bars)
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
| ENTRY1 | 26 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 22
- **Target hits / Stop hits / Partials:** 4 / 22 / 8
- **Avg / median % per leg:** 0.13% / -0.28%
- **Sum % (uncompounded):** 4.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 2 | 14.3% | 0 | 12 | 2 | -0.23% | -3.2% |
| BUY @ 2nd Alert (retest1) | 14 | 2 | 14.3% | 0 | 12 | 2 | -0.23% | -3.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 20 | 10 | 50.0% | 4 | 10 | 6 | 0.39% | 7.8% |
| SELL @ 2nd Alert (retest1) | 20 | 10 | 50.0% | 4 | 10 | 6 | 0.39% | 7.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 34 | 12 | 35.3% | 4 | 22 | 8 | 0.13% | 4.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:00:00 | 58.64 | 58.26 | 0.00 | ORB-long ORB[57.65,58.33] vol=4.0x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-10-10 12:15:00 | 58.36 | 58.38 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-10-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:55:00 | 56.20 | 56.83 | 0.00 | ORB-short ORB[56.79,57.28] vol=2.0x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:30:00 | 55.89 | 56.49 | 0.00 | T1 1.5R @ 55.89 |
| Target hit | 2025-10-14 15:20:00 | 54.90 | 55.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2025-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 09:30:00 | 55.00 | 55.12 | 0.00 | ORB-short ORB[55.05,55.47] vol=3.0x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-10-15 09:45:00 | 55.21 | 55.10 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-10-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 10:20:00 | 54.95 | 55.21 | 0.00 | ORB-short ORB[55.12,55.60] vol=1.9x ATR=0.15 |
| Stop hit — per-position SL triggered | 2025-10-17 11:10:00 | 55.10 | 55.17 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-10-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 10:20:00 | 57.08 | 56.44 | 0.00 | ORB-long ORB[56.09,56.65] vol=3.1x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-10-23 10:25:00 | 56.86 | 56.58 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-10-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:40:00 | 56.48 | 56.26 | 0.00 | ORB-long ORB[55.86,56.39] vol=1.9x ATR=0.16 |
| Stop hit — per-position SL triggered | 2025-10-24 10:10:00 | 56.32 | 56.34 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 10:15:00 | 53.10 | 53.57 | 0.00 | ORB-short ORB[53.53,54.08] vol=3.4x ATR=0.24 |
| Stop hit — per-position SL triggered | 2025-11-07 10:20:00 | 53.34 | 53.56 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-11-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:55:00 | 54.96 | 54.74 | 0.00 | ORB-long ORB[54.50,54.89] vol=3.3x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 11:15:00 | 55.25 | 54.77 | 0.00 | T1 1.5R @ 55.25 |
| Stop hit — per-position SL triggered | 2025-11-10 11:30:00 | 54.96 | 54.79 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-11-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:30:00 | 54.06 | 54.33 | 0.00 | ORB-short ORB[54.10,54.88] vol=2.3x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 09:55:00 | 53.81 | 54.18 | 0.00 | T1 1.5R @ 53.81 |
| Stop hit — per-position SL triggered | 2025-11-11 10:50:00 | 54.06 | 54.10 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-11-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:45:00 | 54.72 | 54.44 | 0.00 | ORB-long ORB[53.86,54.48] vol=1.6x ATR=0.16 |
| Stop hit — per-position SL triggered | 2025-11-14 09:50:00 | 54.56 | 54.45 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-11-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:40:00 | 52.45 | 52.66 | 0.00 | ORB-short ORB[52.63,53.03] vol=1.9x ATR=0.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:50:00 | 52.27 | 52.62 | 0.00 | T1 1.5R @ 52.27 |
| Target hit | 2025-11-21 11:40:00 | 52.35 | 52.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — SELL (started 2025-11-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 10:05:00 | 51.76 | 52.10 | 0.00 | ORB-short ORB[51.87,52.51] vol=1.8x ATR=0.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:40:00 | 51.55 | 51.92 | 0.00 | T1 1.5R @ 51.55 |
| Target hit | 2025-11-24 15:20:00 | 50.33 | 51.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2025-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:30:00 | 51.82 | 52.09 | 0.00 | ORB-short ORB[51.85,52.42] vol=1.8x ATR=0.16 |
| Stop hit — per-position SL triggered | 2025-11-27 10:00:00 | 51.98 | 52.02 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-11-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 09:35:00 | 50.75 | 50.98 | 0.00 | ORB-short ORB[50.76,51.34] vol=1.7x ATR=0.16 |
| Stop hit — per-position SL triggered | 2025-11-28 09:55:00 | 50.91 | 50.93 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-12-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 10:35:00 | 52.33 | 51.52 | 0.00 | ORB-long ORB[50.94,51.55] vol=5.2x ATR=0.23 |
| Stop hit — per-position SL triggered | 2025-12-02 10:40:00 | 52.10 | 51.67 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-12-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:55:00 | 50.09 | 50.60 | 0.00 | ORB-short ORB[50.76,51.20] vol=2.0x ATR=0.15 |
| Stop hit — per-position SL triggered | 2025-12-03 11:35:00 | 50.24 | 50.52 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:15:00 | 50.31 | 49.88 | 0.00 | ORB-long ORB[49.17,49.68] vol=3.5x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 10:30:00 | 50.58 | 49.99 | 0.00 | T1 1.5R @ 50.58 |
| Stop hit — per-position SL triggered | 2025-12-23 10:35:00 | 50.31 | 50.01 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:35:00 | 53.46 | 53.08 | 0.00 | ORB-long ORB[52.46,53.08] vol=2.1x ATR=0.24 |
| Stop hit — per-position SL triggered | 2025-12-26 09:40:00 | 53.22 | 53.11 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-12-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 09:40:00 | 52.74 | 53.08 | 0.00 | ORB-short ORB[53.00,53.46] vol=1.6x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:20:00 | 52.44 | 52.93 | 0.00 | T1 1.5R @ 52.44 |
| Target hit | 2025-12-29 15:20:00 | 51.38 | 52.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2025-12-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 09:35:00 | 50.66 | 51.00 | 0.00 | ORB-short ORB[50.90,51.41] vol=1.8x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-12-30 09:40:00 | 50.85 | 50.97 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-02-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:50:00 | 62.24 | 61.94 | 0.00 | ORB-long ORB[61.43,62.19] vol=2.9x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-02-12 10:35:00 | 61.88 | 62.09 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 59.90 | 60.53 | 0.00 | ORB-short ORB[60.35,61.24] vol=2.7x ATR=0.28 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 60.18 | 60.42 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 58.60 | 58.84 | 0.00 | ORB-short ORB[58.67,59.30] vol=1.8x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 09:45:00 | 58.32 | 58.75 | 0.00 | T1 1.5R @ 58.32 |
| Stop hit — per-position SL triggered | 2026-02-25 09:55:00 | 58.60 | 58.72 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:15:00 | 59.53 | 58.98 | 0.00 | ORB-long ORB[58.45,59.20] vol=3.0x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-02-26 10:20:00 | 59.28 | 59.01 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 61.75 | 61.50 | 0.00 | ORB-long ORB[61.01,61.69] vol=2.3x ATR=0.27 |
| Stop hit — per-position SL triggered | 2026-04-28 10:15:00 | 61.48 | 61.54 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2026-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:30:00 | 58.66 | 58.06 | 0.00 | ORB-long ORB[57.64,58.39] vol=1.9x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-04-30 09:35:00 | 58.37 | 58.08 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-10 10:00:00 | 58.64 | 2025-10-10 12:15:00 | 58.36 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-10-14 09:55:00 | 56.20 | 2025-10-14 10:30:00 | 55.89 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-10-14 09:55:00 | 56.20 | 2025-10-14 15:20:00 | 54.90 | TARGET_HIT | 0.50 | 2.31% |
| SELL | retest1 | 2025-10-15 09:30:00 | 55.00 | 2025-10-15 09:45:00 | 55.21 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-10-17 10:20:00 | 54.95 | 2025-10-17 11:10:00 | 55.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-23 10:20:00 | 57.08 | 2025-10-23 10:25:00 | 56.86 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-10-24 09:40:00 | 56.48 | 2025-10-24 10:10:00 | 56.32 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-11-07 10:15:00 | 53.10 | 2025-11-07 10:20:00 | 53.34 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-11-10 10:55:00 | 54.96 | 2025-11-10 11:15:00 | 55.25 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-11-10 10:55:00 | 54.96 | 2025-11-10 11:30:00 | 54.96 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 09:30:00 | 54.06 | 2025-11-11 09:55:00 | 53.81 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-11-11 09:30:00 | 54.06 | 2025-11-11 10:50:00 | 54.06 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-14 09:45:00 | 54.72 | 2025-11-14 09:50:00 | 54.56 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-21 10:40:00 | 52.45 | 2025-11-21 10:50:00 | 52.27 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-11-21 10:40:00 | 52.45 | 2025-11-21 11:40:00 | 52.35 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2025-11-24 10:05:00 | 51.76 | 2025-11-24 11:40:00 | 51.55 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-11-24 10:05:00 | 51.76 | 2025-11-24 15:20:00 | 50.33 | TARGET_HIT | 0.50 | 2.76% |
| SELL | retest1 | 2025-11-27 09:30:00 | 51.82 | 2025-11-27 10:00:00 | 51.98 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-11-28 09:35:00 | 50.75 | 2025-11-28 09:55:00 | 50.91 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-12-02 10:35:00 | 52.33 | 2025-12-02 10:40:00 | 52.10 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-12-03 10:55:00 | 50.09 | 2025-12-03 11:35:00 | 50.24 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-12-23 10:15:00 | 50.31 | 2025-12-23 10:30:00 | 50.58 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-12-23 10:15:00 | 50.31 | 2025-12-23 10:35:00 | 50.31 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-26 09:35:00 | 53.46 | 2025-12-26 09:40:00 | 53.22 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-12-29 09:40:00 | 52.74 | 2025-12-29 10:20:00 | 52.44 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-12-29 09:40:00 | 52.74 | 2025-12-29 15:20:00 | 51.38 | TARGET_HIT | 0.50 | 2.58% |
| SELL | retest1 | 2025-12-30 09:35:00 | 50.66 | 2025-12-30 09:40:00 | 50.85 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-12 09:50:00 | 62.24 | 2026-02-12 10:35:00 | 61.88 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2026-02-13 09:30:00 | 59.90 | 2026-02-13 09:40:00 | 60.18 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-02-25 09:35:00 | 58.60 | 2026-02-25 09:45:00 | 58.32 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-25 09:35:00 | 58.60 | 2026-02-25 09:55:00 | 58.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 10:15:00 | 59.53 | 2026-02-26 10:20:00 | 59.28 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-28 09:45:00 | 61.75 | 2026-04-28 10:15:00 | 61.48 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-30 09:30:00 | 58.66 | 2026-04-30 09:35:00 | 58.37 | STOP_HIT | 1.00 | -0.49% |
