# Balrampur Chini Mills Ltd. (BALRAMCHIN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 522.00
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
| PARTIAL | 9 |
| TARGET_HIT | 6 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 12
- **Target hits / Stop hits / Partials:** 6 / 12 / 9
- **Avg / median % per leg:** 0.20% / 0.13%
- **Sum % (uncompounded):** 5.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | -0.01% | -0.1% |
| BUY @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | -0.01% | -0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 10 | 62.5% | 4 | 6 | 6 | 0.35% | 5.6% |
| SELL @ 2nd Alert (retest1) | 16 | 10 | 62.5% | 4 | 6 | 6 | 0.35% | 5.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 27 | 15 | 55.6% | 6 | 12 | 9 | 0.20% | 5.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:35:00 | 453.95 | 454.88 | 0.00 | ORB-short ORB[454.00,457.50] vol=2.2x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:20:00 | 451.83 | 454.18 | 0.00 | T1 1.5R @ 451.83 |
| Target hit | 2026-02-10 11:35:00 | 453.35 | 453.04 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:15:00 | 469.70 | 466.12 | 0.00 | ORB-long ORB[461.00,467.00] vol=2.3x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-02-17 10:30:00 | 467.91 | 466.93 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:15:00 | 463.40 | 463.88 | 0.00 | ORB-short ORB[464.55,469.95] vol=3.8x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 13:55:00 | 461.71 | 463.32 | 0.00 | T1 1.5R @ 461.71 |
| Target hit | 2026-02-19 15:20:00 | 455.70 | 461.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:30:00 | 462.55 | 463.57 | 0.00 | ORB-short ORB[463.00,467.85] vol=1.5x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:20:00 | 460.98 | 462.96 | 0.00 | T1 1.5R @ 460.98 |
| Stop hit — per-position SL triggered | 2026-02-26 11:35:00 | 462.55 | 462.57 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:15:00 | 466.25 | 464.06 | 0.00 | ORB-long ORB[461.00,465.00] vol=1.6x ATR=1.89 |
| Stop hit — per-position SL triggered | 2026-02-27 10:30:00 | 464.36 | 464.23 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:30:00 | 446.65 | 448.85 | 0.00 | ORB-short ORB[448.05,453.20] vol=4.7x ATR=2.02 |
| Stop hit — per-position SL triggered | 2026-03-04 09:40:00 | 448.67 | 448.60 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:45:00 | 484.50 | 478.21 | 0.00 | ORB-long ORB[468.65,475.85] vol=2.3x ATR=2.75 |
| Stop hit — per-position SL triggered | 2026-03-18 11:20:00 | 481.75 | 481.22 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:00:00 | 477.05 | 479.24 | 0.00 | ORB-short ORB[478.05,482.40] vol=2.8x ATR=1.55 |
| Stop hit — per-position SL triggered | 2026-04-10 10:10:00 | 478.60 | 479.05 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 483.00 | 481.58 | 0.00 | ORB-long ORB[477.35,482.90] vol=4.5x ATR=2.18 |
| Stop hit — per-position SL triggered | 2026-04-15 09:35:00 | 480.82 | 481.56 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:50:00 | 482.50 | 484.91 | 0.00 | ORB-short ORB[484.50,489.50] vol=1.6x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-04-16 10:20:00 | 484.06 | 484.23 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 10:35:00 | 491.70 | 487.56 | 0.00 | ORB-long ORB[483.00,488.85] vol=3.4x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-04-20 10:40:00 | 490.33 | 488.21 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:45:00 | 514.50 | 520.27 | 0.00 | ORB-short ORB[516.10,522.20] vol=2.4x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 12:35:00 | 511.12 | 518.29 | 0.00 | T1 1.5R @ 511.12 |
| Target hit | 2026-04-21 15:20:00 | 507.65 | 516.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 515.80 | 512.77 | 0.00 | ORB-long ORB[509.60,514.00] vol=1.8x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:50:00 | 518.67 | 515.21 | 0.00 | T1 1.5R @ 518.67 |
| Target hit | 2026-04-22 10:30:00 | 516.50 | 517.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2026-04-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:10:00 | 514.05 | 517.58 | 0.00 | ORB-short ORB[516.00,521.45] vol=1.7x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:15:00 | 511.67 | 516.52 | 0.00 | T1 1.5R @ 511.67 |
| Target hit | 2026-04-28 15:05:00 | 509.90 | 509.46 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — SELL (started 2026-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:35:00 | 507.55 | 513.11 | 0.00 | ORB-short ORB[511.35,518.45] vol=2.6x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 09:50:00 | 503.63 | 510.35 | 0.00 | T1 1.5R @ 503.63 |
| Stop hit — per-position SL triggered | 2026-04-29 11:35:00 | 507.55 | 506.53 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 528.40 | 526.12 | 0.00 | ORB-long ORB[522.00,527.80] vol=2.1x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:40:00 | 531.60 | 528.82 | 0.00 | T1 1.5R @ 531.60 |
| Stop hit — per-position SL triggered | 2026-05-04 09:50:00 | 528.40 | 528.90 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:05:00 | 527.85 | 523.90 | 0.00 | ORB-long ORB[522.05,527.70] vol=2.1x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:10:00 | 530.77 | 526.89 | 0.00 | T1 1.5R @ 530.77 |
| Target hit | 2026-05-05 13:30:00 | 528.50 | 528.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — SELL (started 2026-05-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:45:00 | 521.00 | 522.82 | 0.00 | ORB-short ORB[522.20,525.40] vol=1.9x ATR=1.57 |
| Stop hit — per-position SL triggered | 2026-05-08 10:05:00 | 522.57 | 522.35 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 09:35:00 | 453.95 | 2026-02-10 10:20:00 | 451.83 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-10 09:35:00 | 453.95 | 2026-02-10 11:35:00 | 453.35 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2026-02-17 10:15:00 | 469.70 | 2026-02-17 10:30:00 | 467.91 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-19 10:15:00 | 463.40 | 2026-02-19 13:55:00 | 461.71 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-19 10:15:00 | 463.40 | 2026-02-19 15:20:00 | 455.70 | TARGET_HIT | 0.50 | 1.66% |
| SELL | retest1 | 2026-02-26 10:30:00 | 462.55 | 2026-02-26 11:20:00 | 460.98 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-26 10:30:00 | 462.55 | 2026-02-26 11:35:00 | 462.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-27 10:15:00 | 466.25 | 2026-02-27 10:30:00 | 464.36 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-04 09:30:00 | 446.65 | 2026-03-04 09:40:00 | 448.67 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-03-18 09:45:00 | 484.50 | 2026-03-18 11:20:00 | 481.75 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2026-04-10 10:00:00 | 477.05 | 2026-04-10 10:10:00 | 478.60 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-15 09:30:00 | 483.00 | 2026-04-15 09:35:00 | 480.82 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-16 09:50:00 | 482.50 | 2026-04-16 10:20:00 | 484.06 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-20 10:35:00 | 491.70 | 2026-04-20 10:40:00 | 490.33 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-21 10:45:00 | 514.50 | 2026-04-21 12:35:00 | 511.12 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-04-21 10:45:00 | 514.50 | 2026-04-21 15:20:00 | 507.65 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2026-04-22 09:35:00 | 515.80 | 2026-04-22 09:50:00 | 518.67 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-22 09:35:00 | 515.80 | 2026-04-22 10:30:00 | 516.50 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2026-04-28 10:10:00 | 514.05 | 2026-04-28 10:15:00 | 511.67 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-28 10:10:00 | 514.05 | 2026-04-28 15:05:00 | 509.90 | TARGET_HIT | 0.50 | 0.81% |
| SELL | retest1 | 2026-04-29 09:35:00 | 507.55 | 2026-04-29 09:50:00 | 503.63 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2026-04-29 09:35:00 | 507.55 | 2026-04-29 11:35:00 | 507.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 09:35:00 | 528.40 | 2026-05-04 09:40:00 | 531.60 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-05-04 09:35:00 | 528.40 | 2026-05-04 09:50:00 | 528.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 11:05:00 | 527.85 | 2026-05-05 11:10:00 | 530.77 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-05-05 11:05:00 | 527.85 | 2026-05-05 13:30:00 | 528.50 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2026-05-08 09:45:00 | 521.00 | 2026-05-08 10:05:00 | 522.57 | STOP_HIT | 1.00 | -0.30% |
