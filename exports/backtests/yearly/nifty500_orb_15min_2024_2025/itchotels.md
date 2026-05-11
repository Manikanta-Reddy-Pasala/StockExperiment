# ITC Hotels Ltd. (ITCHOTELS)

## Backtest Summary

- **Window:** 2025-01-29 09:40:00 → 2026-05-08 15:25:00 (21983 bars)
- **Last close:** 164.58
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
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 7
- **Target hits / Stop hits / Partials:** 4 / 7 / 7
- **Avg / median % per leg:** 0.39% / 0.44%
- **Sum % (uncompounded):** 7.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.19% | 2.3% |
| BUY @ 2nd Alert (retest1) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.19% | 2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 5 | 83.3% | 2 | 1 | 3 | 0.78% | 4.7% |
| SELL @ 2nd Alert (retest1) | 6 | 5 | 83.3% | 2 | 1 | 3 | 0.78% | 4.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 11 | 61.1% | 4 | 7 | 7 | 0.39% | 7.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-02-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 10:50:00 | 171.00 | 170.52 | 0.00 | ORB-long ORB[168.52,170.00] vol=2.1x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 11:45:00 | 172.79 | 170.92 | 0.00 | T1 1.5R @ 172.79 |
| Target hit | 2025-02-06 15:20:00 | 172.00 | 171.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2025-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:35:00 | 164.86 | 163.76 | 0.00 | ORB-long ORB[162.70,164.61] vol=2.1x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-02-20 09:45:00 | 164.26 | 163.94 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-02-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:40:00 | 164.97 | 165.85 | 0.00 | ORB-short ORB[165.24,166.90] vol=2.0x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 10:10:00 | 164.04 | 165.42 | 0.00 | T1 1.5R @ 164.04 |
| Target hit | 2025-02-21 15:20:00 | 162.99 | 163.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2025-02-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-28 11:10:00 | 162.23 | 160.61 | 0.00 | ORB-long ORB[159.08,161.47] vol=1.5x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-02-28 11:40:00 | 161.65 | 160.97 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-03-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 10:00:00 | 170.38 | 169.51 | 0.00 | ORB-long ORB[168.60,170.27] vol=1.7x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 10:10:00 | 171.12 | 170.04 | 0.00 | T1 1.5R @ 171.12 |
| Stop hit — per-position SL triggered | 2025-03-13 10:20:00 | 170.38 | 170.13 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:35:00 | 169.72 | 168.86 | 0.00 | ORB-long ORB[167.40,169.25] vol=3.0x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-03-18 09:50:00 | 169.12 | 169.25 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-04-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:20:00 | 201.67 | 200.82 | 0.00 | ORB-long ORB[199.11,201.19] vol=3.0x ATR=0.73 |
| Stop hit — per-position SL triggered | 2025-04-24 10:30:00 | 200.94 | 200.84 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:35:00 | 198.11 | 198.83 | 0.00 | ORB-short ORB[198.20,199.81] vol=2.1x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 09:40:00 | 197.24 | 198.65 | 0.00 | T1 1.5R @ 197.24 |
| Stop hit — per-position SL triggered | 2025-04-29 09:45:00 | 198.11 | 198.50 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:30:00 | 196.40 | 195.85 | 0.00 | ORB-long ORB[193.78,195.90] vol=4.5x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:45:00 | 197.40 | 196.41 | 0.00 | T1 1.5R @ 197.40 |
| Target hit | 2025-05-05 11:00:00 | 197.60 | 197.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2025-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 09:30:00 | 199.39 | 197.51 | 0.00 | ORB-long ORB[196.12,198.70] vol=2.7x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 09:35:00 | 200.57 | 198.67 | 0.00 | T1 1.5R @ 200.57 |
| Stop hit — per-position SL triggered | 2025-05-06 09:50:00 | 199.39 | 199.01 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-05-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 10:45:00 | 189.99 | 191.49 | 0.00 | ORB-short ORB[191.30,193.64] vol=1.8x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 11:25:00 | 189.25 | 191.16 | 0.00 | T1 1.5R @ 189.25 |
| Target hit | 2025-05-08 15:20:00 | 186.01 | 189.16 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-02-06 10:50:00 | 171.00 | 2025-02-06 11:45:00 | 172.79 | PARTIAL | 0.50 | 1.05% |
| BUY | retest1 | 2025-02-06 10:50:00 | 171.00 | 2025-02-06 15:20:00 | 172.00 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2025-02-20 09:35:00 | 164.86 | 2025-02-20 09:45:00 | 164.26 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-02-21 09:40:00 | 164.97 | 2025-02-21 10:10:00 | 164.04 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-02-21 09:40:00 | 164.97 | 2025-02-21 15:20:00 | 162.99 | TARGET_HIT | 0.50 | 1.20% |
| BUY | retest1 | 2025-02-28 11:10:00 | 162.23 | 2025-02-28 11:40:00 | 161.65 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-13 10:00:00 | 170.38 | 2025-03-13 10:10:00 | 171.12 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-03-13 10:00:00 | 170.38 | 2025-03-13 10:20:00 | 170.38 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 09:35:00 | 169.72 | 2025-03-18 09:50:00 | 169.12 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-24 10:20:00 | 201.67 | 2025-04-24 10:30:00 | 200.94 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-04-29 09:35:00 | 198.11 | 2025-04-29 09:40:00 | 197.24 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-04-29 09:35:00 | 198.11 | 2025-04-29 09:45:00 | 198.11 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 09:30:00 | 196.40 | 2025-05-05 09:45:00 | 197.40 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-05-05 09:30:00 | 196.40 | 2025-05-05 11:00:00 | 197.60 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2025-05-06 09:30:00 | 199.39 | 2025-05-06 09:35:00 | 200.57 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-05-06 09:30:00 | 199.39 | 2025-05-06 09:50:00 | 199.39 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-08 10:45:00 | 189.99 | 2025-05-08 11:25:00 | 189.25 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-05-08 10:45:00 | 189.99 | 2025-05-08 15:20:00 | 186.01 | TARGET_HIT | 0.50 | 2.09% |
