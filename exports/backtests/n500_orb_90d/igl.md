# Indraprastha Gas Ltd. (IGL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 165.97
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
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 11
- **Target hits / Stop hits / Partials:** 1 / 11 / 4
- **Avg / median % per leg:** -0.02% / 0.00%
- **Sum % (uncompounded):** -0.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.19% | -1.3% |
| BUY @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.19% | -1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.12% | 1.0% |
| SELL @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.12% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 5 | 31.2% | 1 | 11 | 4 | -0.02% | -0.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 170.02 | 170.46 | 0.00 | ORB-short ORB[170.10,171.56] vol=1.5x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:20:00 | 169.38 | 170.26 | 0.00 | T1 1.5R @ 169.38 |
| Target hit | 2026-02-18 15:20:00 | 168.81 | 169.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:35:00 | 171.25 | 170.13 | 0.00 | ORB-long ORB[168.42,169.74] vol=1.7x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:45:00 | 171.84 | 170.39 | 0.00 | T1 1.5R @ 171.84 |
| Stop hit — per-position SL triggered | 2026-02-26 11:00:00 | 171.25 | 170.55 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 156.91 | 157.92 | 0.00 | ORB-short ORB[157.64,159.20] vol=2.0x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:50:00 | 156.16 | 157.76 | 0.00 | T1 1.5R @ 156.16 |
| Stop hit — per-position SL triggered | 2026-03-06 10:55:00 | 156.91 | 157.73 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:30:00 | 158.42 | 157.07 | 0.00 | ORB-long ORB[156.04,158.00] vol=3.0x ATR=0.59 |
| Stop hit — per-position SL triggered | 2026-03-10 10:35:00 | 157.83 | 157.14 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:15:00 | 159.03 | 157.94 | 0.00 | ORB-long ORB[156.82,158.10] vol=3.8x ATR=0.41 |
| Stop hit — per-position SL triggered | 2026-03-18 12:30:00 | 158.62 | 158.24 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 157.69 | 156.29 | 0.00 | ORB-long ORB[154.65,156.61] vol=1.6x ATR=0.66 |
| Stop hit — per-position SL triggered | 2026-03-20 09:35:00 | 157.03 | 156.43 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:15:00 | 147.51 | 148.37 | 0.00 | ORB-short ORB[148.15,150.30] vol=2.8x ATR=0.66 |
| Stop hit — per-position SL triggered | 2026-03-27 10:30:00 | 148.17 | 148.22 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 162.45 | 163.58 | 0.00 | ORB-short ORB[163.00,165.00] vol=1.6x ATR=0.65 |
| Stop hit — per-position SL triggered | 2026-04-15 09:40:00 | 163.10 | 163.55 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 09:30:00 | 167.41 | 168.53 | 0.00 | ORB-short ORB[167.60,170.06] vol=1.5x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 09:45:00 | 166.44 | 168.12 | 0.00 | T1 1.5R @ 166.44 |
| Stop hit — per-position SL triggered | 2026-04-20 09:50:00 | 167.41 | 168.09 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:10:00 | 169.00 | 167.66 | 0.00 | ORB-long ORB[166.25,168.48] vol=1.8x ATR=0.61 |
| Stop hit — per-position SL triggered | 2026-04-22 10:35:00 | 168.39 | 168.11 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:50:00 | 166.03 | 165.18 | 0.00 | ORB-long ORB[163.81,165.93] vol=1.5x ATR=0.42 |
| Stop hit — per-position SL triggered | 2026-04-27 11:25:00 | 165.61 | 165.23 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:35:00 | 166.45 | 167.05 | 0.00 | ORB-short ORB[166.51,168.00] vol=2.3x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-05-06 09:40:00 | 166.89 | 167.03 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 09:55:00 | 170.02 | 2026-02-18 10:20:00 | 169.38 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-18 09:55:00 | 170.02 | 2026-02-18 15:20:00 | 168.81 | TARGET_HIT | 0.50 | 0.71% |
| BUY | retest1 | 2026-02-26 10:35:00 | 171.25 | 2026-02-26 10:45:00 | 171.84 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-02-26 10:35:00 | 171.25 | 2026-02-26 11:00:00 | 171.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 156.91 | 2026-03-06 10:50:00 | 156.16 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-03-06 10:45:00 | 156.91 | 2026-03-06 10:55:00 | 156.91 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 10:30:00 | 158.42 | 2026-03-10 10:35:00 | 157.83 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-18 11:15:00 | 159.03 | 2026-03-18 12:30:00 | 158.62 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-20 09:30:00 | 157.69 | 2026-03-20 09:35:00 | 157.03 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-27 10:15:00 | 147.51 | 2026-03-27 10:30:00 | 148.17 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-15 09:35:00 | 162.45 | 2026-04-15 09:40:00 | 163.10 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-20 09:30:00 | 167.41 | 2026-04-20 09:45:00 | 166.44 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-04-20 09:30:00 | 167.41 | 2026-04-20 09:50:00 | 167.41 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:10:00 | 169.00 | 2026-04-22 10:35:00 | 168.39 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-27 10:50:00 | 166.03 | 2026-04-27 11:25:00 | 165.61 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-05-06 09:35:00 | 166.45 | 2026-05-06 09:40:00 | 166.89 | STOP_HIT | 1.00 | -0.26% |
