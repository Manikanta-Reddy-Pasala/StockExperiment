# Housing & Urban Development Corporation Ltd. (HUDCO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 232.00
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
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 11
- **Target hits / Stop hits / Partials:** 0 / 11 / 4
- **Avg / median % per leg:** -0.03% / 0.00%
- **Sum % (uncompounded):** -0.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 0 | 7 | 3 | 0.02% | 0.2% |
| BUY @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 0 | 7 | 3 | 0.02% | 0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.13% | -0.6% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.13% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 4 | 26.7% | 0 | 11 | 4 | -0.03% | -0.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 191.22 | 192.37 | 0.00 | ORB-short ORB[192.00,194.41] vol=2.1x ATR=0.54 |
| Stop hit — per-position SL triggered | 2026-02-11 09:55:00 | 191.76 | 191.99 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 198.54 | 197.78 | 0.00 | ORB-long ORB[196.00,197.80] vol=5.2x ATR=0.66 |
| Stop hit — per-position SL triggered | 2026-02-19 09:55:00 | 197.88 | 198.10 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:30:00 | 175.41 | 176.71 | 0.00 | ORB-short ORB[176.18,178.78] vol=1.5x ATR=0.76 |
| Stop hit — per-position SL triggered | 2026-03-05 10:40:00 | 176.17 | 176.00 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:00:00 | 177.66 | 175.09 | 0.00 | ORB-long ORB[174.02,176.10] vol=2.1x ATR=0.76 |
| Stop hit — per-position SL triggered | 2026-03-12 11:20:00 | 176.90 | 175.41 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:45:00 | 174.26 | 175.25 | 0.00 | ORB-short ORB[175.01,176.31] vol=2.6x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:10:00 | 173.29 | 174.74 | 0.00 | T1 1.5R @ 173.29 |
| Stop hit — per-position SL triggered | 2026-03-13 10:30:00 | 174.26 | 174.62 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:00:00 | 173.51 | 174.64 | 0.00 | ORB-short ORB[174.19,176.50] vol=1.8x ATR=0.82 |
| Stop hit — per-position SL triggered | 2026-03-19 10:10:00 | 174.33 | 174.50 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:30:00 | 194.24 | 191.70 | 0.00 | ORB-long ORB[190.00,192.39] vol=3.8x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:45:00 | 195.40 | 193.16 | 0.00 | T1 1.5R @ 195.40 |
| Stop hit — per-position SL triggered | 2026-04-16 11:40:00 | 194.24 | 193.54 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:10:00 | 195.62 | 194.49 | 0.00 | ORB-long ORB[193.13,195.17] vol=1.9x ATR=0.71 |
| Stop hit — per-position SL triggered | 2026-04-17 10:30:00 | 194.91 | 194.62 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:00:00 | 216.84 | 214.87 | 0.00 | ORB-long ORB[213.16,216.00] vol=1.8x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 12:25:00 | 218.01 | 216.32 | 0.00 | T1 1.5R @ 218.01 |
| Stop hit — per-position SL triggered | 2026-04-29 13:45:00 | 216.84 | 216.55 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:30:00 | 224.60 | 223.08 | 0.00 | ORB-long ORB[221.10,224.00] vol=1.8x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-05-04 12:05:00 | 223.49 | 223.35 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:10:00 | 231.77 | 228.75 | 0.00 | ORB-long ORB[225.36,228.50] vol=1.6x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:15:00 | 233.37 | 230.49 | 0.00 | T1 1.5R @ 233.37 |
| Stop hit — per-position SL triggered | 2026-05-08 10:55:00 | 231.77 | 231.19 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:30:00 | 191.22 | 2026-02-11 09:55:00 | 191.76 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-19 09:30:00 | 198.54 | 2026-02-19 09:55:00 | 197.88 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-03-05 09:30:00 | 175.41 | 2026-03-05 10:40:00 | 176.17 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-12 11:00:00 | 177.66 | 2026-03-12 11:20:00 | 176.90 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-03-13 09:45:00 | 174.26 | 2026-03-13 10:10:00 | 173.29 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-03-13 09:45:00 | 174.26 | 2026-03-13 10:30:00 | 174.26 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 10:00:00 | 173.51 | 2026-03-19 10:10:00 | 174.33 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-16 10:30:00 | 194.24 | 2026-04-16 10:45:00 | 195.40 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-16 10:30:00 | 194.24 | 2026-04-16 11:40:00 | 194.24 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:10:00 | 195.62 | 2026-04-17 10:30:00 | 194.91 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-29 11:00:00 | 216.84 | 2026-04-29 12:25:00 | 218.01 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-04-29 11:00:00 | 216.84 | 2026-04-29 13:45:00 | 216.84 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 10:30:00 | 224.60 | 2026-05-04 12:05:00 | 223.49 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-05-08 10:10:00 | 231.77 | 2026-05-08 10:15:00 | 233.37 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-05-08 10:10:00 | 231.77 | 2026-05-08 10:55:00 | 231.77 | STOP_HIT | 0.50 | 0.00% |
