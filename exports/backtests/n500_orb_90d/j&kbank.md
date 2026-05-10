# Jammu & Kashmir Bank Ltd. (J&KBANK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 141.24
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 8
- **Target hits / Stop hits / Partials:** 2 / 8 / 4
- **Avg / median % per leg:** 0.30% / 0.00%
- **Sum % (uncompounded):** 4.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.48% | 5.3% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.48% | 5.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.36% | -1.1% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.36% | -1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.30% | 4.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:40:00 | 103.98 | 104.52 | 0.00 | ORB-short ORB[104.45,105.90] vol=1.9x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-02-11 10:30:00 | 104.28 | 104.33 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 101.74 | 102.28 | 0.00 | ORB-short ORB[102.26,103.03] vol=1.9x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 102.10 | 102.15 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:10:00 | 103.50 | 103.11 | 0.00 | ORB-long ORB[102.50,103.45] vol=2.2x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:20:00 | 103.91 | 103.25 | 0.00 | T1 1.5R @ 103.91 |
| Target hit | 2026-02-17 14:45:00 | 104.47 | 104.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:35:00 | 103.80 | 103.45 | 0.00 | ORB-long ORB[102.50,103.77] vol=2.1x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-02-20 11:00:00 | 103.35 | 103.66 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 107.28 | 106.06 | 0.00 | ORB-long ORB[105.31,106.14] vol=3.7x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:55:00 | 107.97 | 106.56 | 0.00 | T1 1.5R @ 107.97 |
| Target hit | 2026-02-24 10:30:00 | 111.27 | 111.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-04-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:00:00 | 124.46 | 123.56 | 0.00 | ORB-long ORB[122.27,123.95] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 123.96 | 123.59 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 127.55 | 126.53 | 0.00 | ORB-long ORB[125.30,126.95] vol=2.5x ATR=0.68 |
| Stop hit — per-position SL triggered | 2026-04-15 09:50:00 | 126.87 | 126.59 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:15:00 | 127.92 | 128.07 | 0.00 | ORB-short ORB[128.24,129.60] vol=1.6x ATR=0.55 |
| Stop hit — per-position SL triggered | 2026-04-16 10:25:00 | 128.47 | 128.10 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:10:00 | 134.19 | 133.41 | 0.00 | ORB-long ORB[132.72,134.00] vol=1.7x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:25:00 | 134.83 | 133.65 | 0.00 | T1 1.5R @ 134.83 |
| Stop hit — per-position SL triggered | 2026-04-22 11:35:00 | 134.19 | 133.68 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:10:00 | 131.85 | 131.32 | 0.00 | ORB-long ORB[130.01,131.70] vol=1.7x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:40:00 | 132.47 | 131.43 | 0.00 | T1 1.5R @ 132.47 |
| Stop hit — per-position SL triggered | 2026-05-04 12:15:00 | 131.85 | 131.58 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:40:00 | 103.98 | 2026-02-11 10:30:00 | 104.28 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-13 09:30:00 | 101.74 | 2026-02-13 09:40:00 | 102.10 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-17 10:10:00 | 103.50 | 2026-02-17 10:20:00 | 103.91 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-17 10:10:00 | 103.50 | 2026-02-17 14:45:00 | 104.47 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2026-02-20 09:35:00 | 103.80 | 2026-02-20 11:00:00 | 103.35 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-02-24 09:45:00 | 107.28 | 2026-02-24 09:55:00 | 107.97 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-02-24 09:45:00 | 107.28 | 2026-02-24 10:30:00 | 111.27 | TARGET_HIT | 0.50 | 3.72% |
| BUY | retest1 | 2026-04-10 10:00:00 | 124.46 | 2026-04-10 10:05:00 | 123.96 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-15 09:40:00 | 127.55 | 2026-04-15 09:50:00 | 126.87 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2026-04-16 10:15:00 | 127.92 | 2026-04-16 10:25:00 | 128.47 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-22 11:10:00 | 134.19 | 2026-04-22 11:25:00 | 134.83 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-22 11:10:00 | 134.19 | 2026-04-22 11:35:00 | 134.19 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 11:10:00 | 131.85 | 2026-05-04 11:40:00 | 132.47 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-05-04 11:10:00 | 131.85 | 2026-05-04 12:15:00 | 131.85 | STOP_HIT | 0.50 | 0.00% |
