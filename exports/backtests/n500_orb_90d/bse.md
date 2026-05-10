# BSE Ltd. (BSE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3905.00
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
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 9
- **Target hits / Stop hits / Partials:** 2 / 9 / 5
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 3.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.14% | 1.7% |
| BUY @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.14% | 1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.51% | 2.0% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.51% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.23% | 3.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 2805.00 | 2780.61 | 0.00 | ORB-long ORB[2752.00,2784.40] vol=2.8x ATR=9.60 |
| Stop hit — per-position SL triggered | 2026-02-26 10:00:00 | 2795.40 | 2785.31 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:00:00 | 2734.00 | 2756.77 | 0.00 | ORB-short ORB[2742.20,2781.00] vol=1.7x ATR=11.49 |
| Stop hit — per-position SL triggered | 2026-02-27 10:35:00 | 2745.49 | 2751.85 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 09:40:00 | 2842.30 | 2823.28 | 0.00 | ORB-long ORB[2805.30,2832.00] vol=1.9x ATR=14.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 12:05:00 | 2864.12 | 2840.19 | 0.00 | T1 1.5R @ 2864.12 |
| Target hit | 2026-03-10 15:20:00 | 2862.90 | 2850.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-03-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:00:00 | 2839.90 | 2857.65 | 0.00 | ORB-short ORB[2840.00,2868.10] vol=2.0x ATR=9.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:15:00 | 2826.12 | 2854.87 | 0.00 | T1 1.5R @ 2826.12 |
| Target hit | 2026-03-27 15:20:00 | 2775.30 | 2826.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:15:00 | 3394.20 | 3375.48 | 0.00 | ORB-long ORB[3350.50,3390.00] vol=1.5x ATR=12.93 |
| Stop hit — per-position SL triggered | 2026-04-15 10:50:00 | 3381.27 | 3378.48 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:45:00 | 3487.00 | 3450.46 | 0.00 | ORB-long ORB[3425.00,3468.80] vol=2.4x ATR=13.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:15:00 | 3506.97 | 3471.19 | 0.00 | T1 1.5R @ 3506.97 |
| Stop hit — per-position SL triggered | 2026-04-17 11:00:00 | 3487.00 | 3475.90 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:35:00 | 3506.10 | 3480.13 | 0.00 | ORB-long ORB[3451.00,3491.00] vol=2.2x ATR=12.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:00:00 | 3524.23 | 3492.41 | 0.00 | T1 1.5R @ 3524.23 |
| Stop hit — per-position SL triggered | 2026-04-27 10:20:00 | 3506.10 | 3495.09 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:55:00 | 3583.80 | 3561.41 | 0.00 | ORB-long ORB[3543.60,3568.00] vol=2.2x ATR=8.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:05:00 | 3597.17 | 3569.49 | 0.00 | T1 1.5R @ 3597.17 |
| Stop hit — per-position SL triggered | 2026-04-28 10:35:00 | 3583.80 | 3575.26 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:40:00 | 3665.00 | 3647.66 | 0.00 | ORB-long ORB[3620.30,3662.00] vol=2.7x ATR=9.05 |
| Stop hit — per-position SL triggered | 2026-04-29 12:40:00 | 3655.95 | 3653.20 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 10:15:00 | 3719.60 | 3706.08 | 0.00 | ORB-long ORB[3691.60,3719.00] vol=1.6x ATR=9.59 |
| Stop hit — per-position SL triggered | 2026-05-05 10:25:00 | 3710.01 | 3706.49 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 3764.00 | 3781.82 | 0.00 | ORB-short ORB[3771.00,3803.00] vol=1.6x ATR=11.84 |
| Stop hit — per-position SL triggered | 2026-05-06 09:55:00 | 3775.84 | 3778.50 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-26 09:45:00 | 2805.00 | 2026-02-26 10:00:00 | 2795.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-27 10:00:00 | 2734.00 | 2026-02-27 10:35:00 | 2745.49 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-10 09:40:00 | 2842.30 | 2026-03-10 12:05:00 | 2864.12 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2026-03-10 09:40:00 | 2842.30 | 2026-03-10 15:20:00 | 2862.90 | TARGET_HIT | 0.50 | 0.72% |
| SELL | retest1 | 2026-03-27 11:00:00 | 2839.90 | 2026-03-27 11:15:00 | 2826.12 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-27 11:00:00 | 2839.90 | 2026-03-27 15:20:00 | 2775.30 | TARGET_HIT | 0.50 | 2.27% |
| BUY | retest1 | 2026-04-15 10:15:00 | 3394.20 | 2026-04-15 10:50:00 | 3381.27 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-17 09:45:00 | 3487.00 | 2026-04-17 10:15:00 | 3506.97 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-17 09:45:00 | 3487.00 | 2026-04-17 11:00:00 | 3487.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:35:00 | 3506.10 | 2026-04-27 10:00:00 | 3524.23 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-27 09:35:00 | 3506.10 | 2026-04-27 10:20:00 | 3506.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 09:55:00 | 3583.80 | 2026-04-28 10:05:00 | 3597.17 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-04-28 09:55:00 | 3583.80 | 2026-04-28 10:35:00 | 3583.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:40:00 | 3665.00 | 2026-04-29 12:40:00 | 3655.95 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-05-05 10:15:00 | 3719.60 | 2026-05-05 10:25:00 | 3710.01 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-05-06 09:30:00 | 3764.00 | 2026-05-06 09:55:00 | 3775.84 | STOP_HIT | 1.00 | -0.31% |
