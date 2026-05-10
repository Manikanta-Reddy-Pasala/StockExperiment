# Union Bank of India (UNIONBANK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 166.50
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
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 6
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 2.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.29% | 2.6% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.29% | 2.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 3 | 30.0% | 0 | 7 | 3 | -0.02% | -0.2% |
| SELL @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 0 | 7 | 3 | -0.02% | -0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 8 | 42.1% | 2 | 11 | 6 | 0.13% | 2.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:25:00 | 181.99 | 181.06 | 0.00 | ORB-long ORB[178.92,181.00] vol=1.6x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-02-09 11:00:00 | 180.88 | 181.19 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 178.52 | 179.12 | 0.00 | ORB-short ORB[178.55,180.28] vol=2.0x ATR=0.65 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 179.17 | 179.12 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 185.25 | 183.55 | 0.00 | ORB-long ORB[181.86,183.92] vol=2.7x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:30:00 | 186.31 | 184.21 | 0.00 | T1 1.5R @ 186.31 |
| Target hit | 2026-02-17 15:20:00 | 189.10 | 187.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:45:00 | 191.79 | 190.44 | 0.00 | ORB-long ORB[189.00,191.25] vol=1.6x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:00:00 | 192.81 | 191.19 | 0.00 | T1 1.5R @ 192.81 |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 191.79 | 191.60 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 199.20 | 197.67 | 0.00 | ORB-long ORB[196.21,197.70] vol=1.7x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:15:00 | 200.20 | 198.51 | 0.00 | T1 1.5R @ 200.20 |
| Target hit | 2026-02-24 12:30:00 | 199.80 | 199.89 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2026-02-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:45:00 | 197.78 | 201.06 | 0.00 | ORB-short ORB[200.40,202.54] vol=2.9x ATR=0.80 |
| Stop hit — per-position SL triggered | 2026-02-25 10:50:00 | 198.58 | 200.77 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 192.13 | 193.86 | 0.00 | ORB-short ORB[192.83,195.40] vol=1.6x ATR=0.64 |
| Stop hit — per-position SL triggered | 2026-03-05 11:05:00 | 192.77 | 193.77 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 191.83 | 190.76 | 0.00 | ORB-long ORB[189.75,191.22] vol=1.8x ATR=0.68 |
| Stop hit — per-position SL triggered | 2026-04-21 09:35:00 | 191.15 | 190.88 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 11:10:00 | 176.25 | 177.15 | 0.00 | ORB-short ORB[176.87,178.80] vol=1.7x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 11:35:00 | 175.45 | 177.03 | 0.00 | T1 1.5R @ 175.45 |
| Stop hit — per-position SL triggered | 2026-04-27 11:40:00 | 176.25 | 176.99 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:35:00 | 171.76 | 172.73 | 0.00 | ORB-short ORB[172.50,174.30] vol=1.5x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-04-28 10:55:00 | 172.27 | 172.63 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:40:00 | 164.10 | 163.18 | 0.00 | ORB-long ORB[162.20,163.77] vol=2.3x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-05-05 09:50:00 | 163.48 | 163.27 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:05:00 | 166.84 | 168.28 | 0.00 | ORB-short ORB[168.21,169.90] vol=2.7x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:40:00 | 166.22 | 167.87 | 0.00 | T1 1.5R @ 166.22 |
| Stop hit — per-position SL triggered | 2026-05-07 12:20:00 | 166.84 | 167.62 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 165.48 | 165.87 | 0.00 | ORB-short ORB[165.56,167.28] vol=1.8x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:45:00 | 164.82 | 165.69 | 0.00 | T1 1.5R @ 164.82 |
| Stop hit — per-position SL triggered | 2026-05-08 10:00:00 | 165.48 | 165.59 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:25:00 | 181.99 | 2026-02-09 11:00:00 | 180.88 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2026-02-13 09:30:00 | 178.52 | 2026-02-13 09:40:00 | 179.17 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-17 10:20:00 | 185.25 | 2026-02-17 10:30:00 | 186.31 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-02-17 10:20:00 | 185.25 | 2026-02-17 15:20:00 | 189.10 | TARGET_HIT | 0.50 | 2.08% |
| BUY | retest1 | 2026-02-18 09:45:00 | 191.79 | 2026-02-18 10:00:00 | 192.81 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-18 09:45:00 | 191.79 | 2026-02-18 10:15:00 | 191.79 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 09:45:00 | 199.20 | 2026-02-24 10:15:00 | 200.20 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-24 09:45:00 | 199.20 | 2026-02-24 12:30:00 | 199.80 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-25 10:45:00 | 197.78 | 2026-02-25 10:50:00 | 198.58 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-05 10:55:00 | 192.13 | 2026-03-05 11:05:00 | 192.77 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-21 09:30:00 | 191.83 | 2026-04-21 09:35:00 | 191.15 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-27 11:10:00 | 176.25 | 2026-04-27 11:35:00 | 175.45 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-27 11:10:00 | 176.25 | 2026-04-27 11:40:00 | 176.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 10:35:00 | 171.76 | 2026-04-28 10:55:00 | 172.27 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-05 09:40:00 | 164.10 | 2026-05-05 09:50:00 | 163.48 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-05-07 11:05:00 | 166.84 | 2026-05-07 11:40:00 | 166.22 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-05-07 11:05:00 | 166.84 | 2026-05-07 12:20:00 | 166.84 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 09:35:00 | 165.48 | 2026-05-08 09:45:00 | 164.82 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-05-08 09:35:00 | 165.48 | 2026-05-08 10:00:00 | 165.48 | STOP_HIT | 0.50 | 0.00% |
