# Emami Ltd. (EMAMILTD)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 456.50
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 14
- **Target hits / Stop hits / Partials:** 3 / 14 / 6
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 1.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.24% | 1.7% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.24% | 1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 6 | 37.5% | 2 | 10 | 4 | 0.00% | 0.1% |
| SELL @ 2nd Alert (retest1) | 16 | 6 | 37.5% | 2 | 10 | 4 | 0.00% | 0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 9 | 39.1% | 3 | 14 | 6 | 0.08% | 1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 500.80 | 504.01 | 0.00 | ORB-short ORB[503.00,508.20] vol=3.7x ATR=2.04 |
| Stop hit — per-position SL triggered | 2026-02-10 09:45:00 | 502.84 | 503.13 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 482.55 | 484.53 | 0.00 | ORB-short ORB[484.60,488.20] vol=2.0x ATR=1.31 |
| Stop hit — per-position SL triggered | 2026-02-19 09:45:00 | 483.86 | 483.59 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:45:00 | 477.20 | 478.96 | 0.00 | ORB-short ORB[477.25,483.80] vol=1.6x ATR=1.44 |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 478.64 | 477.82 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:50:00 | 473.50 | 475.43 | 0.00 | ORB-short ORB[474.05,479.75] vol=1.8x ATR=0.88 |
| Stop hit — per-position SL triggered | 2026-02-24 11:15:00 | 474.38 | 475.22 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:50:00 | 469.50 | 472.31 | 0.00 | ORB-short ORB[472.10,476.05] vol=4.1x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 467.45 | 471.13 | 0.00 | T1 1.5R @ 467.45 |
| Target hit | 2026-02-27 14:05:00 | 468.80 | 467.68 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2026-03-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 10:30:00 | 447.85 | 449.69 | 0.00 | ORB-short ORB[448.00,453.95] vol=1.7x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-03-04 11:45:00 | 449.52 | 449.11 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 11:10:00 | 441.80 | 442.41 | 0.00 | ORB-short ORB[442.25,447.70] vol=12.6x ATR=1.00 |
| Stop hit — per-position SL triggered | 2026-03-10 11:40:00 | 442.80 | 442.39 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:50:00 | 441.95 | 443.92 | 0.00 | ORB-short ORB[442.45,447.05] vol=1.9x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-03-11 10:00:00 | 443.48 | 443.14 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 423.90 | 425.67 | 0.00 | ORB-short ORB[425.15,429.75] vol=2.9x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:20:00 | 421.82 | 424.99 | 0.00 | T1 1.5R @ 421.82 |
| Stop hit — per-position SL triggered | 2026-03-13 10:45:00 | 423.90 | 424.52 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:15:00 | 416.15 | 416.44 | 0.00 | ORB-short ORB[418.95,422.05] vol=15.6x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 12:00:00 | 413.83 | 416.32 | 0.00 | T1 1.5R @ 413.83 |
| Stop hit — per-position SL triggered | 2026-03-18 12:20:00 | 416.15 | 416.26 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 411.90 | 409.58 | 0.00 | ORB-long ORB[406.30,411.00] vol=1.7x ATR=1.82 |
| Stop hit — per-position SL triggered | 2026-03-20 09:35:00 | 410.08 | 409.60 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 09:35:00 | 418.55 | 421.38 | 0.00 | ORB-short ORB[419.55,424.15] vol=1.6x ATR=1.72 |
| Stop hit — per-position SL triggered | 2026-04-10 10:10:00 | 420.27 | 420.72 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:50:00 | 419.95 | 416.31 | 0.00 | ORB-long ORB[412.75,418.00] vol=1.6x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:05:00 | 423.11 | 417.47 | 0.00 | T1 1.5R @ 423.11 |
| Target hit | 2026-04-13 14:00:00 | 425.50 | 425.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2026-04-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 11:10:00 | 431.55 | 429.89 | 0.00 | ORB-long ORB[425.45,429.90] vol=2.2x ATR=1.15 |
| Stop hit — per-position SL triggered | 2026-04-15 12:00:00 | 430.40 | 430.08 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:25:00 | 462.40 | 459.02 | 0.00 | ORB-long ORB[455.50,461.95] vol=1.7x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 11:15:00 | 465.18 | 460.58 | 0.00 | T1 1.5R @ 465.18 |
| Stop hit — per-position SL triggered | 2026-04-27 11:40:00 | 462.40 | 460.75 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:35:00 | 461.25 | 459.36 | 0.00 | ORB-long ORB[456.60,460.80] vol=1.8x ATR=1.26 |
| Stop hit — per-position SL triggered | 2026-04-28 10:55:00 | 459.99 | 459.60 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 451.95 | 453.67 | 0.00 | ORB-short ORB[452.70,457.90] vol=3.4x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:00:00 | 448.96 | 452.55 | 0.00 | T1 1.5R @ 448.96 |
| Target hit | 2026-05-05 13:15:00 | 450.60 | 449.46 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 09:30:00 | 500.80 | 2026-02-10 09:45:00 | 502.84 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-02-19 09:35:00 | 482.55 | 2026-02-19 09:45:00 | 483.86 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-23 09:45:00 | 477.20 | 2026-02-23 12:15:00 | 478.64 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-24 10:50:00 | 473.50 | 2026-02-24 11:15:00 | 474.38 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-27 09:50:00 | 469.50 | 2026-02-27 10:15:00 | 467.45 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-27 09:50:00 | 469.50 | 2026-02-27 14:05:00 | 468.80 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2026-03-04 10:30:00 | 447.85 | 2026-03-04 11:45:00 | 449.52 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-10 11:10:00 | 441.80 | 2026-03-10 11:40:00 | 442.80 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-03-11 09:50:00 | 441.95 | 2026-03-11 10:00:00 | 443.48 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-13 09:50:00 | 423.90 | 2026-03-13 10:20:00 | 421.82 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-13 09:50:00 | 423.90 | 2026-03-13 10:45:00 | 423.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-18 10:15:00 | 416.15 | 2026-03-18 12:00:00 | 413.83 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-18 10:15:00 | 416.15 | 2026-03-18 12:20:00 | 416.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-20 09:30:00 | 411.90 | 2026-03-20 09:35:00 | 410.08 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-04-10 09:35:00 | 418.55 | 2026-04-10 10:10:00 | 420.27 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-13 09:50:00 | 419.95 | 2026-04-13 10:05:00 | 423.11 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2026-04-13 09:50:00 | 419.95 | 2026-04-13 14:00:00 | 425.50 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2026-04-15 11:10:00 | 431.55 | 2026-04-15 12:00:00 | 430.40 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-27 10:25:00 | 462.40 | 2026-04-27 11:15:00 | 465.18 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-27 10:25:00 | 462.40 | 2026-04-27 11:40:00 | 462.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 10:35:00 | 461.25 | 2026-04-28 10:55:00 | 459.99 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-05-05 09:30:00 | 451.95 | 2026-05-05 10:00:00 | 448.96 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-05-05 09:30:00 | 451.95 | 2026-05-05 13:15:00 | 450.60 | TARGET_HIT | 0.50 | 0.30% |
