# Dabur India Ltd. (DABUR)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 487.00
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
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 14
- **Target hits / Stop hits / Partials:** 4 / 14 / 8
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 3.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 8 | 53.3% | 3 | 7 | 5 | 0.26% | 3.9% |
| BUY @ 2nd Alert (retest1) | 15 | 8 | 53.3% | 3 | 7 | 5 | 0.26% | 3.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 12 | 46.2% | 4 | 14 | 8 | 0.15% | 3.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 511.70 | 509.70 | 0.00 | ORB-long ORB[508.50,510.85] vol=4.0x ATR=1.40 |
| Stop hit — per-position SL triggered | 2026-02-09 11:20:00 | 510.30 | 509.93 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:15:00 | 519.30 | 520.63 | 0.00 | ORB-short ORB[520.00,522.85] vol=3.6x ATR=1.02 |
| Stop hit — per-position SL triggered | 2026-02-12 10:20:00 | 520.32 | 520.60 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:45:00 | 514.80 | 516.69 | 0.00 | ORB-short ORB[515.90,518.95] vol=2.0x ATR=1.14 |
| Stop hit — per-position SL triggered | 2026-02-13 09:50:00 | 515.94 | 516.53 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:00:00 | 513.80 | 510.74 | 0.00 | ORB-long ORB[508.50,512.95] vol=3.6x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:10:00 | 515.35 | 511.23 | 0.00 | T1 1.5R @ 515.35 |
| Stop hit — per-position SL triggered | 2026-02-16 15:15:00 | 513.80 | 513.69 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 518.25 | 516.74 | 0.00 | ORB-long ORB[511.10,514.55] vol=1.6x ATR=0.99 |
| Stop hit — per-position SL triggered | 2026-02-17 10:45:00 | 517.26 | 516.96 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:05:00 | 513.25 | 512.34 | 0.00 | ORB-long ORB[509.40,512.85] vol=1.8x ATR=1.25 |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 512.00 | 512.33 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:30:00 | 508.40 | 504.82 | 0.00 | ORB-long ORB[500.40,502.50] vol=1.6x ATR=1.24 |
| Stop hit — per-position SL triggered | 2026-02-20 11:05:00 | 507.16 | 505.54 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:25:00 | 519.00 | 520.89 | 0.00 | ORB-short ORB[521.20,523.75] vol=1.8x ATR=1.15 |
| Stop hit — per-position SL triggered | 2026-02-25 10:40:00 | 520.15 | 520.22 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-02-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:05:00 | 516.00 | 516.54 | 0.00 | ORB-short ORB[519.15,525.50] vol=1.9x ATR=1.29 |
| Stop hit — per-position SL triggered | 2026-02-27 11:55:00 | 517.29 | 516.44 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 486.40 | 488.18 | 0.00 | ORB-short ORB[488.05,494.00] vol=1.5x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:25:00 | 484.70 | 487.62 | 0.00 | T1 1.5R @ 484.70 |
| Stop hit — per-position SL triggered | 2026-03-05 12:55:00 | 486.40 | 486.16 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:00:00 | 477.70 | 475.12 | 0.00 | ORB-long ORB[471.20,476.55] vol=3.0x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 12:55:00 | 479.61 | 476.31 | 0.00 | T1 1.5R @ 479.61 |
| Target hit | 2026-03-10 15:20:00 | 482.50 | 478.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2026-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:50:00 | 460.65 | 457.65 | 0.00 | ORB-long ORB[453.85,458.50] vol=1.7x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-03-13 11:05:00 | 459.20 | 457.74 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 455.10 | 456.01 | 0.00 | ORB-short ORB[456.55,460.70] vol=1.6x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:35:00 | 453.27 | 455.77 | 0.00 | T1 1.5R @ 453.27 |
| Stop hit — per-position SL triggered | 2026-03-17 11:55:00 | 455.10 | 454.75 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 11:15:00 | 433.35 | 431.03 | 0.00 | ORB-long ORB[428.45,431.95] vol=8.1x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:45:00 | 435.02 | 431.25 | 0.00 | T1 1.5R @ 435.02 |
| Stop hit — per-position SL triggered | 2026-04-15 11:50:00 | 433.35 | 431.78 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:55:00 | 433.00 | 436.21 | 0.00 | ORB-short ORB[435.15,441.40] vol=1.6x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-04-16 10:25:00 | 434.28 | 433.86 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 436.20 | 430.43 | 0.00 | ORB-long ORB[424.20,429.80] vol=1.7x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:15:00 | 438.59 | 432.03 | 0.00 | T1 1.5R @ 438.59 |
| Target hit | 2026-04-17 15:20:00 | 443.50 | 440.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 447.00 | 443.74 | 0.00 | ORB-long ORB[439.05,444.80] vol=1.8x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:05:00 | 449.11 | 446.57 | 0.00 | T1 1.5R @ 449.11 |
| Target hit | 2026-04-21 12:15:00 | 448.55 | 449.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — SELL (started 2026-04-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:00:00 | 452.50 | 457.23 | 0.00 | ORB-short ORB[457.45,462.50] vol=1.9x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:15:00 | 450.80 | 456.13 | 0.00 | T1 1.5R @ 450.80 |
| Target hit | 2026-04-24 15:10:00 | 452.15 | 452.09 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 511.70 | 2026-02-09 11:20:00 | 510.30 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-12 10:15:00 | 519.30 | 2026-02-12 10:20:00 | 520.32 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-13 09:45:00 | 514.80 | 2026-02-13 09:50:00 | 515.94 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-16 11:00:00 | 513.80 | 2026-02-16 11:10:00 | 515.35 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-16 11:00:00 | 513.80 | 2026-02-16 15:15:00 | 513.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:20:00 | 518.25 | 2026-02-17 10:45:00 | 517.26 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-19 10:05:00 | 513.25 | 2026-02-19 10:15:00 | 512.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-20 10:30:00 | 508.40 | 2026-02-20 11:05:00 | 507.16 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-25 10:25:00 | 519.00 | 2026-02-25 10:40:00 | 520.15 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-27 11:05:00 | 516.00 | 2026-02-27 11:55:00 | 517.29 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-05 10:55:00 | 486.40 | 2026-03-05 11:25:00 | 484.70 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-03-05 10:55:00 | 486.40 | 2026-03-05 12:55:00 | 486.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 11:00:00 | 477.70 | 2026-03-10 12:55:00 | 479.61 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-03-10 11:00:00 | 477.70 | 2026-03-10 15:20:00 | 482.50 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2026-03-13 10:50:00 | 460.65 | 2026-03-13 11:05:00 | 459.20 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-17 10:25:00 | 455.10 | 2026-03-17 10:35:00 | 453.27 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-03-17 10:25:00 | 455.10 | 2026-03-17 11:55:00 | 455.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 11:15:00 | 433.35 | 2026-04-15 11:45:00 | 435.02 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-15 11:15:00 | 433.35 | 2026-04-15 11:50:00 | 433.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 09:55:00 | 433.00 | 2026-04-16 10:25:00 | 434.28 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-17 10:05:00 | 436.20 | 2026-04-17 10:15:00 | 438.59 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-17 10:05:00 | 436.20 | 2026-04-17 15:20:00 | 443.50 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2026-04-21 09:45:00 | 447.00 | 2026-04-21 10:05:00 | 449.11 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-21 09:45:00 | 447.00 | 2026-04-21 12:15:00 | 448.55 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-24 11:00:00 | 452.50 | 2026-04-24 11:15:00 | 450.80 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-24 11:00:00 | 452.50 | 2026-04-24 15:10:00 | 452.15 | TARGET_HIT | 0.50 | 0.08% |
