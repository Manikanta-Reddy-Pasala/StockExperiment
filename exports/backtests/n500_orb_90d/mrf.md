# MRF Ltd. (MRF)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 130490.00
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 8
- **Target hits / Stop hits / Partials:** 4 / 8 / 4
- **Avg / median % per leg:** 0.28% / 0.26%
- **Sum % (uncompounded):** 4.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.12% | 0.6% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.12% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 6 | 54.5% | 3 | 5 | 3 | 0.36% | 3.9% |
| SELL @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 3 | 5 | 3 | 0.36% | 3.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 8 | 50.0% | 4 | 8 | 4 | 0.28% | 4.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:50:00 | 152400.00 | 150951.05 | 0.00 | ORB-long ORB[149800.00,152000.00] vol=2.6x ATR=504.09 |
| Stop hit — per-position SL triggered | 2026-02-12 10:00:00 | 151895.91 | 151129.01 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 148535.00 | 149752.31 | 0.00 | ORB-short ORB[149735.00,150990.00] vol=1.9x ATR=310.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:55:00 | 148069.83 | 149539.87 | 0.00 | T1 1.5R @ 148069.83 |
| Target hit | 2026-02-16 15:20:00 | 146800.00 | 148490.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:45:00 | 148585.00 | 148111.13 | 0.00 | ORB-long ORB[147750.00,148500.00] vol=2.5x ATR=343.67 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 148241.33 | 148275.26 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 143215.00 | 144258.00 | 0.00 | ORB-short ORB[144205.00,146200.00] vol=1.5x ATR=299.40 |
| Stop hit — per-position SL triggered | 2026-02-24 11:45:00 | 143514.40 | 144065.57 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:15:00 | 146345.00 | 145614.36 | 0.00 | ORB-long ORB[144245.00,145975.00] vol=3.0x ATR=320.26 |
| Stop hit — per-position SL triggered | 2026-02-25 12:30:00 | 146024.74 | 145729.94 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:00:00 | 144800.00 | 146199.60 | 0.00 | ORB-short ORB[146605.00,147815.00] vol=2.3x ATR=332.39 |
| Stop hit — per-position SL triggered | 2026-02-26 11:35:00 | 145132.39 | 145906.91 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 143080.00 | 143718.04 | 0.00 | ORB-short ORB[143520.00,144700.00] vol=2.3x ATR=329.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 12:00:00 | 142586.31 | 143620.20 | 0.00 | T1 1.5R @ 142586.31 |
| Target hit | 2026-02-27 15:20:00 | 140770.00 | 141669.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-03-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 10:30:00 | 134855.00 | 136584.12 | 0.00 | ORB-short ORB[136500.00,138490.00] vol=2.4x ATR=515.45 |
| Stop hit — per-position SL triggered | 2026-03-04 11:15:00 | 135370.45 | 136251.58 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:15:00 | 136600.00 | 135684.24 | 0.00 | ORB-long ORB[134435.00,135900.00] vol=2.0x ATR=353.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:50:00 | 137130.20 | 135821.58 | 0.00 | T1 1.5R @ 137130.20 |
| Target hit | 2026-03-12 15:20:00 | 137955.00 | 136799.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-04-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:50:00 | 138480.00 | 139148.60 | 0.00 | ORB-short ORB[138965.00,140095.00] vol=1.8x ATR=243.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:15:00 | 138115.06 | 138842.08 | 0.00 | T1 1.5R @ 138115.06 |
| Target hit | 2026-04-22 15:20:00 | 136500.00 | 137801.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-04-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:35:00 | 131440.00 | 132006.76 | 0.00 | ORB-short ORB[131900.00,132695.00] vol=1.8x ATR=314.24 |
| Stop hit — per-position SL triggered | 2026-04-28 09:40:00 | 131754.24 | 131975.61 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:55:00 | 128480.00 | 128960.56 | 0.00 | ORB-short ORB[128600.00,130095.00] vol=2.3x ATR=200.32 |
| Stop hit — per-position SL triggered | 2026-05-05 11:05:00 | 128680.32 | 128944.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 09:50:00 | 152400.00 | 2026-02-12 10:00:00 | 151895.91 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-16 11:15:00 | 148535.00 | 2026-02-16 11:55:00 | 148069.83 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-16 11:15:00 | 148535.00 | 2026-02-16 15:20:00 | 146800.00 | TARGET_HIT | 0.50 | 1.17% |
| BUY | retest1 | 2026-02-18 10:45:00 | 148585.00 | 2026-02-18 12:15:00 | 148241.33 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-24 11:10:00 | 143215.00 | 2026-02-24 11:45:00 | 143514.40 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-25 11:15:00 | 146345.00 | 2026-02-25 12:30:00 | 146024.74 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-26 11:00:00 | 144800.00 | 2026-02-26 11:35:00 | 145132.39 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-27 10:55:00 | 143080.00 | 2026-02-27 12:00:00 | 142586.31 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-27 10:55:00 | 143080.00 | 2026-02-27 15:20:00 | 140770.00 | TARGET_HIT | 0.50 | 1.61% |
| SELL | retest1 | 2026-03-04 10:30:00 | 134855.00 | 2026-03-04 11:15:00 | 135370.45 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-12 11:15:00 | 136600.00 | 2026-03-12 11:50:00 | 137130.20 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-03-12 11:15:00 | 136600.00 | 2026-03-12 15:20:00 | 137955.00 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2026-04-22 09:50:00 | 138480.00 | 2026-04-22 10:15:00 | 138115.06 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-04-22 09:50:00 | 138480.00 | 2026-04-22 15:20:00 | 136500.00 | TARGET_HIT | 0.50 | 1.43% |
| SELL | retest1 | 2026-04-28 09:35:00 | 131440.00 | 2026-04-28 09:40:00 | 131754.24 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-05 10:55:00 | 128480.00 | 2026-05-05 11:05:00 | 128680.32 | STOP_HIT | 1.00 | -0.16% |
