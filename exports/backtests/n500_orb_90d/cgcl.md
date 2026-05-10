# Capri Global Capital Ltd. (CGCL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 197.75
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
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 10
- **Target hits / Stop hits / Partials:** 2 / 10 / 6
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 1.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.00% | -0.0% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.00% | -0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 6 | 60.0% | 2 | 4 | 4 | 0.20% | 2.0% |
| SELL @ 2nd Alert (retest1) | 10 | 6 | 60.0% | 2 | 4 | 4 | 0.20% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 8 | 44.4% | 2 | 10 | 6 | 0.11% | 2.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:35:00 | 176.00 | 177.03 | 0.00 | ORB-short ORB[176.45,178.40] vol=1.9x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 12:10:00 | 175.17 | 176.25 | 0.00 | T1 1.5R @ 175.17 |
| Target hit | 2026-02-10 15:20:00 | 175.14 | 175.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:50:00 | 178.00 | 176.75 | 0.00 | ORB-long ORB[174.97,176.14] vol=2.8x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-02-17 10:00:00 | 177.38 | 176.94 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:40:00 | 171.71 | 172.59 | 0.00 | ORB-short ORB[172.10,174.12] vol=4.5x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-02-23 11:20:00 | 172.22 | 172.53 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 172.60 | 171.38 | 0.00 | ORB-long ORB[169.79,171.38] vol=1.5x ATR=0.64 |
| Stop hit — per-position SL triggered | 2026-03-18 09:35:00 | 171.96 | 171.46 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 09:35:00 | 169.96 | 168.61 | 0.00 | ORB-long ORB[166.20,168.63] vol=4.3x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 09:40:00 | 171.30 | 169.36 | 0.00 | T1 1.5R @ 171.30 |
| Stop hit — per-position SL triggered | 2026-03-19 09:50:00 | 169.96 | 170.55 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-25 11:15:00 | 169.51 | 169.97 | 0.00 | ORB-short ORB[170.00,172.00] vol=6.2x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 11:40:00 | 168.69 | 169.89 | 0.00 | T1 1.5R @ 168.69 |
| Stop hit — per-position SL triggered | 2026-03-25 15:05:00 | 169.51 | 169.51 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-01 11:05:00 | 172.92 | 170.16 | 0.00 | ORB-long ORB[168.00,170.40] vol=7.4x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 12:10:00 | 174.43 | 171.60 | 0.00 | T1 1.5R @ 174.43 |
| Stop hit — per-position SL triggered | 2026-04-01 13:50:00 | 172.92 | 173.01 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:30:00 | 183.00 | 185.32 | 0.00 | ORB-short ORB[186.35,189.02] vol=2.0x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 12:00:00 | 181.96 | 184.67 | 0.00 | T1 1.5R @ 181.96 |
| Target hit | 2026-04-16 14:45:00 | 182.86 | 181.97 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 189.35 | 186.69 | 0.00 | ORB-long ORB[184.21,186.89] vol=4.5x ATR=1.05 |
| Stop hit — per-position SL triggered | 2026-04-24 09:35:00 | 188.30 | 186.82 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:35:00 | 184.05 | 184.72 | 0.00 | ORB-short ORB[184.25,186.10] vol=2.2x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:45:00 | 182.99 | 184.68 | 0.00 | T1 1.5R @ 182.99 |
| Stop hit — per-position SL triggered | 2026-04-29 11:00:00 | 184.05 | 184.66 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:20:00 | 182.10 | 182.78 | 0.00 | ORB-short ORB[182.14,184.57] vol=1.9x ATR=0.66 |
| Stop hit — per-position SL triggered | 2026-04-30 10:35:00 | 182.76 | 182.76 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 202.00 | 200.47 | 0.00 | ORB-long ORB[198.97,200.80] vol=1.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2026-05-08 09:35:00 | 201.20 | 200.60 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 09:35:00 | 176.00 | 2026-02-10 12:10:00 | 175.17 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-10 09:35:00 | 176.00 | 2026-02-10 15:20:00 | 175.14 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-17 09:50:00 | 178.00 | 2026-02-17 10:00:00 | 177.38 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-23 10:40:00 | 171.71 | 2026-02-23 11:20:00 | 172.22 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-18 09:30:00 | 172.60 | 2026-03-18 09:35:00 | 171.96 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-19 09:35:00 | 169.96 | 2026-03-19 09:40:00 | 171.30 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2026-03-19 09:35:00 | 169.96 | 2026-03-19 09:50:00 | 169.96 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-25 11:15:00 | 169.51 | 2026-03-25 11:40:00 | 168.69 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-25 11:15:00 | 169.51 | 2026-03-25 15:05:00 | 169.51 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-01 11:05:00 | 172.92 | 2026-04-01 12:10:00 | 174.43 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2026-04-01 11:05:00 | 172.92 | 2026-04-01 13:50:00 | 172.92 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 10:30:00 | 183.00 | 2026-04-16 12:00:00 | 181.96 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-04-16 10:30:00 | 183.00 | 2026-04-16 14:45:00 | 182.86 | TARGET_HIT | 0.50 | 0.08% |
| BUY | retest1 | 2026-04-24 09:30:00 | 189.35 | 2026-04-24 09:35:00 | 188.30 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2026-04-29 10:35:00 | 184.05 | 2026-04-29 10:45:00 | 182.99 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-04-29 10:35:00 | 184.05 | 2026-04-29 11:00:00 | 184.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-30 10:20:00 | 182.10 | 2026-04-30 10:35:00 | 182.76 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-05-08 09:30:00 | 202.00 | 2026-05-08 09:35:00 | 201.20 | STOP_HIT | 1.00 | -0.39% |
