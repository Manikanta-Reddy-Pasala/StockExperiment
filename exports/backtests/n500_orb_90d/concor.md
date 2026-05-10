# Container Corporation of India Ltd. (CONCOR)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 533.75
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
| ENTRY1 | 22 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 21
- **Target hits / Stop hits / Partials:** 1 / 21 / 6
- **Avg / median % per leg:** -0.07% / -0.20%
- **Sum % (uncompounded):** -1.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 4 | 26.7% | 1 | 11 | 3 | -0.07% | -1.0% |
| BUY @ 2nd Alert (retest1) | 15 | 4 | 26.7% | 1 | 11 | 3 | -0.07% | -1.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 3 | 23.1% | 0 | 10 | 3 | -0.07% | -0.9% |
| SELL @ 2nd Alert (retest1) | 13 | 3 | 23.1% | 0 | 10 | 3 | -0.07% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 7 | 25.0% | 1 | 21 | 6 | -0.07% | -1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 521.00 | 518.22 | 0.00 | ORB-long ORB[512.10,518.55] vol=2.3x ATR=1.49 |
| Stop hit — per-position SL triggered | 2026-02-10 09:50:00 | 519.51 | 518.60 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 500.00 | 498.26 | 0.00 | ORB-long ORB[494.00,499.10] vol=2.6x ATR=1.01 |
| Stop hit — per-position SL triggered | 2026-02-16 11:45:00 | 498.99 | 498.35 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 505.50 | 504.41 | 0.00 | ORB-long ORB[501.60,504.85] vol=2.5x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:40:00 | 506.99 | 504.93 | 0.00 | T1 1.5R @ 506.99 |
| Target hit | 2026-02-17 10:50:00 | 507.60 | 508.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2026-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:45:00 | 503.00 | 505.60 | 0.00 | ORB-short ORB[503.75,507.05] vol=1.8x ATR=1.26 |
| Stop hit — per-position SL triggered | 2026-02-18 10:55:00 | 504.26 | 504.51 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:10:00 | 504.65 | 505.04 | 0.00 | ORB-short ORB[504.75,508.35] vol=4.1x ATR=0.93 |
| Stop hit — per-position SL triggered | 2026-02-19 11:20:00 | 505.58 | 505.01 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:00:00 | 511.30 | 508.75 | 0.00 | ORB-long ORB[504.00,509.55] vol=2.4x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-02-23 10:10:00 | 509.96 | 508.93 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:05:00 | 492.40 | 495.29 | 0.00 | ORB-short ORB[496.55,500.35] vol=1.9x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:20:00 | 490.68 | 494.29 | 0.00 | T1 1.5R @ 490.68 |
| Stop hit — per-position SL triggered | 2026-02-27 10:25:00 | 492.40 | 494.30 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-02 11:10:00 | 479.60 | 484.47 | 0.00 | ORB-short ORB[482.50,489.10] vol=2.5x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 11:25:00 | 477.22 | 483.67 | 0.00 | T1 1.5R @ 477.22 |
| Stop hit — per-position SL triggered | 2026-03-02 11:35:00 | 479.60 | 483.09 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 462.50 | 462.76 | 0.00 | ORB-short ORB[463.25,467.30] vol=1.6x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-03-05 11:00:00 | 464.06 | 462.88 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:35:00 | 487.50 | 485.09 | 0.00 | ORB-long ORB[476.20,483.10] vol=1.8x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-03-06 10:45:00 | 485.94 | 485.29 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:55:00 | 459.25 | 460.60 | 0.00 | ORB-short ORB[462.85,466.10] vol=1.9x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-03-13 11:30:00 | 460.83 | 460.37 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:45:00 | 459.40 | 457.95 | 0.00 | ORB-long ORB[452.55,458.00] vol=1.6x ATR=1.47 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 457.93 | 458.09 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-03-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:55:00 | 450.50 | 447.12 | 0.00 | ORB-long ORB[443.55,448.70] vol=1.6x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:20:00 | 452.51 | 448.04 | 0.00 | T1 1.5R @ 452.51 |
| Stop hit — per-position SL triggered | 2026-03-20 12:20:00 | 450.50 | 450.29 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:10:00 | 435.00 | 436.89 | 0.00 | ORB-short ORB[437.30,442.75] vol=1.7x ATR=1.38 |
| Stop hit — per-position SL triggered | 2026-03-27 11:35:00 | 436.38 | 436.67 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-03-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 11:10:00 | 427.40 | 430.92 | 0.00 | ORB-short ORB[428.90,434.20] vol=1.7x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-03-30 11:15:00 | 428.85 | 430.60 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 10:55:00 | 475.00 | 470.43 | 0.00 | ORB-long ORB[468.60,473.95] vol=3.3x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 11:05:00 | 477.45 | 471.36 | 0.00 | T1 1.5R @ 477.45 |
| Stop hit — per-position SL triggered | 2026-04-09 13:40:00 | 475.00 | 474.75 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:10:00 | 510.55 | 505.25 | 0.00 | ORB-long ORB[501.35,506.80] vol=2.3x ATR=2.07 |
| Stop hit — per-position SL triggered | 2026-04-17 10:15:00 | 508.48 | 505.78 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:40:00 | 507.90 | 506.17 | 0.00 | ORB-long ORB[501.75,505.30] vol=5.6x ATR=1.46 |
| Stop hit — per-position SL triggered | 2026-04-21 11:05:00 | 506.44 | 506.33 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:15:00 | 514.35 | 516.37 | 0.00 | ORB-short ORB[514.80,519.15] vol=2.4x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:30:00 | 513.02 | 516.23 | 0.00 | T1 1.5R @ 513.02 |
| Stop hit — per-position SL triggered | 2026-04-28 11:40:00 | 514.35 | 516.17 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:15:00 | 520.80 | 512.74 | 0.00 | ORB-long ORB[508.50,514.45] vol=2.0x ATR=1.54 |
| Stop hit — per-position SL triggered | 2026-04-29 11:50:00 | 519.26 | 513.38 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 519.00 | 516.10 | 0.00 | ORB-long ORB[512.90,517.00] vol=2.2x ATR=1.65 |
| Stop hit — per-position SL triggered | 2026-05-05 10:00:00 | 517.35 | 517.62 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2026-05-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:45:00 | 519.35 | 520.81 | 0.00 | ORB-short ORB[519.45,523.95] vol=1.9x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-05-06 11:30:00 | 520.53 | 520.48 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:40:00 | 521.00 | 2026-02-10 09:50:00 | 519.51 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-16 11:15:00 | 500.00 | 2026-02-16 11:45:00 | 498.99 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-17 09:30:00 | 505.50 | 2026-02-17 09:40:00 | 506.99 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2026-02-17 09:30:00 | 505.50 | 2026-02-17 10:50:00 | 507.60 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-18 09:45:00 | 503.00 | 2026-02-18 10:55:00 | 504.26 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-19 11:10:00 | 504.65 | 2026-02-19 11:20:00 | 505.58 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-02-23 10:00:00 | 511.30 | 2026-02-23 10:10:00 | 509.96 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-27 10:05:00 | 492.40 | 2026-02-27 10:20:00 | 490.68 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-27 10:05:00 | 492.40 | 2026-02-27 10:25:00 | 492.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-02 11:10:00 | 479.60 | 2026-03-02 11:25:00 | 477.22 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-03-02 11:10:00 | 479.60 | 2026-03-02 11:35:00 | 479.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:45:00 | 462.50 | 2026-03-05 11:00:00 | 464.06 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-06 10:35:00 | 487.50 | 2026-03-06 10:45:00 | 485.94 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-13 10:55:00 | 459.25 | 2026-03-13 11:30:00 | 460.83 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-18 09:45:00 | 459.40 | 2026-03-18 09:55:00 | 457.93 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-20 10:55:00 | 450.50 | 2026-03-20 11:20:00 | 452.51 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-03-20 10:55:00 | 450.50 | 2026-03-20 12:20:00 | 450.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-27 11:10:00 | 435.00 | 2026-03-27 11:35:00 | 436.38 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-30 11:10:00 | 427.40 | 2026-03-30 11:15:00 | 428.85 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-09 10:55:00 | 475.00 | 2026-04-09 11:05:00 | 477.45 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-09 10:55:00 | 475.00 | 2026-04-09 13:40:00 | 475.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:10:00 | 510.55 | 2026-04-17 10:15:00 | 508.48 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-21 10:40:00 | 507.90 | 2026-04-21 11:05:00 | 506.44 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-28 11:15:00 | 514.35 | 2026-04-28 11:30:00 | 513.02 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-04-28 11:15:00 | 514.35 | 2026-04-28 11:40:00 | 514.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 11:15:00 | 520.80 | 2026-04-29 11:50:00 | 519.26 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-05 09:30:00 | 519.00 | 2026-05-05 10:00:00 | 517.35 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-05-06 10:45:00 | 519.35 | 2026-05-06 11:30:00 | 520.53 | STOP_HIT | 1.00 | -0.23% |
