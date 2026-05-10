# PG Electroplast Ltd. (PGEL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 530.45
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 9
- **Target hits / Stop hits / Partials:** 4 / 9 / 4
- **Avg / median % per leg:** 0.21% / -0.28%
- **Sum % (uncompounded):** 3.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 2 | 5 | 2 | 0.03% | 0.3% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 2 | 5 | 2 | 0.03% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 0.40% | 3.2% |
| SELL @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 2 | 4 | 2 | 0.40% | 3.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 8 | 47.1% | 4 | 9 | 4 | 0.21% | 3.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:10:00 | 606.60 | 602.66 | 0.00 | ORB-long ORB[598.30,605.60] vol=1.7x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:20:00 | 610.39 | 604.20 | 0.00 | T1 1.5R @ 610.39 |
| Target hit | 2026-02-11 11:50:00 | 614.30 | 614.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 604.65 | 610.54 | 0.00 | ORB-short ORB[607.80,616.75] vol=1.7x ATR=2.85 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 607.50 | 609.06 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:35:00 | 624.10 | 619.31 | 0.00 | ORB-long ORB[612.05,617.85] vol=1.8x ATR=2.33 |
| Stop hit — per-position SL triggered | 2026-02-17 10:45:00 | 621.77 | 619.91 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:55:00 | 618.95 | 622.54 | 0.00 | ORB-short ORB[623.25,630.60] vol=2.0x ATR=1.74 |
| Stop hit — per-position SL triggered | 2026-02-19 11:00:00 | 620.69 | 622.31 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:50:00 | 612.55 | 608.55 | 0.00 | ORB-long ORB[603.00,611.00] vol=1.5x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:10:00 | 615.84 | 609.46 | 0.00 | T1 1.5R @ 615.84 |
| Target hit | 2026-02-20 15:20:00 | 614.00 | 612.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:40:00 | 610.00 | 616.83 | 0.00 | ORB-short ORB[614.20,621.50] vol=1.8x ATR=2.02 |
| Stop hit — per-position SL triggered | 2026-02-23 11:00:00 | 612.02 | 615.33 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 623.20 | 616.05 | 0.00 | ORB-long ORB[611.40,619.55] vol=1.8x ATR=2.53 |
| Stop hit — per-position SL triggered | 2026-02-25 12:25:00 | 620.67 | 619.54 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 628.80 | 625.12 | 0.00 | ORB-long ORB[619.00,627.40] vol=2.2x ATR=2.74 |
| Stop hit — per-position SL triggered | 2026-02-26 10:45:00 | 626.06 | 627.17 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:40:00 | 528.60 | 520.11 | 0.00 | ORB-long ORB[511.60,519.40] vol=2.0x ATR=4.04 |
| Stop hit — per-position SL triggered | 2026-03-17 09:45:00 | 524.56 | 520.67 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:15:00 | 502.75 | 506.14 | 0.00 | ORB-short ORB[507.55,514.95] vol=1.6x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:45:00 | 499.23 | 505.47 | 0.00 | T1 1.5R @ 499.23 |
| Target hit | 2026-03-27 15:20:00 | 487.50 | 498.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 568.00 | 563.75 | 0.00 | ORB-long ORB[559.35,565.60] vol=3.1x ATR=2.33 |
| Stop hit — per-position SL triggered | 2026-04-21 10:40:00 | 565.67 | 564.47 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:15:00 | 550.20 | 551.31 | 0.00 | ORB-short ORB[550.60,558.50] vol=1.5x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:55:00 | 546.35 | 550.36 | 0.00 | T1 1.5R @ 546.35 |
| Target hit | 2026-04-24 14:25:00 | 548.95 | 548.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:15:00 | 556.30 | 560.44 | 0.00 | ORB-short ORB[562.05,567.90] vol=1.6x ATR=1.99 |
| Stop hit — per-position SL triggered | 2026-04-29 10:20:00 | 558.29 | 560.36 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 10:10:00 | 606.60 | 2026-02-11 10:20:00 | 610.39 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-02-11 10:10:00 | 606.60 | 2026-02-11 11:50:00 | 614.30 | TARGET_HIT | 0.50 | 1.27% |
| SELL | retest1 | 2026-02-13 09:30:00 | 604.65 | 2026-02-13 09:40:00 | 607.50 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-02-17 10:35:00 | 624.10 | 2026-02-17 10:45:00 | 621.77 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-19 10:55:00 | 618.95 | 2026-02-19 11:00:00 | 620.69 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-20 10:50:00 | 612.55 | 2026-02-20 11:10:00 | 615.84 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-02-20 10:50:00 | 612.55 | 2026-02-20 15:20:00 | 614.00 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2026-02-23 10:40:00 | 610.00 | 2026-02-23 11:00:00 | 612.02 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-25 11:00:00 | 623.20 | 2026-02-25 12:25:00 | 620.67 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-02-26 09:35:00 | 628.80 | 2026-02-26 10:45:00 | 626.06 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-17 09:40:00 | 528.60 | 2026-03-17 09:45:00 | 524.56 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest1 | 2026-03-27 11:15:00 | 502.75 | 2026-03-27 11:45:00 | 499.23 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-03-27 11:15:00 | 502.75 | 2026-03-27 15:20:00 | 487.50 | TARGET_HIT | 0.50 | 3.03% |
| BUY | retest1 | 2026-04-21 10:00:00 | 568.00 | 2026-04-21 10:40:00 | 565.67 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-24 10:15:00 | 550.20 | 2026-04-24 10:55:00 | 546.35 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-04-24 10:15:00 | 550.20 | 2026-04-24 14:25:00 | 548.95 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2026-04-29 10:15:00 | 556.30 | 2026-04-29 10:20:00 | 558.29 | STOP_HIT | 1.00 | -0.36% |
