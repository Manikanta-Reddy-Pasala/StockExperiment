# ZF Commercial Vehicle Control Systems India Ltd. (ZFCVINDIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 14532.00
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 9
- **Target hits / Stop hits / Partials:** 1 / 9 / 2
- **Avg / median % per leg:** 0.02% / -0.26%
- **Sum % (uncompounded):** 0.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.06% | -0.2% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.06% | -0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 2 | 25.0% | 1 | 6 | 1 | 0.05% | 0.4% |
| SELL @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 1 | 6 | 1 | 0.05% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 3 | 25.0% | 1 | 9 | 2 | 0.02% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:00:00 | 15844.00 | 15916.20 | 0.00 | ORB-short ORB[15903.00,16140.00] vol=2.7x ATR=41.95 |
| Stop hit — per-position SL triggered | 2026-02-17 11:05:00 | 15885.95 | 15915.61 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 15890.00 | 15978.11 | 0.00 | ORB-short ORB[15930.00,16118.00] vol=3.5x ATR=36.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:05:00 | 15835.32 | 15925.66 | 0.00 | T1 1.5R @ 15835.32 |
| Target hit | 2026-02-18 15:20:00 | 15585.00 | 15771.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 15645.00 | 15789.80 | 0.00 | ORB-short ORB[15798.00,15976.00] vol=1.8x ATR=40.37 |
| Stop hit — per-position SL triggered | 2026-02-25 10:50:00 | 15685.37 | 15778.56 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:10:00 | 14512.00 | 14388.00 | 0.00 | ORB-long ORB[14255.00,14411.00] vol=2.2x ATR=60.48 |
| Stop hit — per-position SL triggered | 2026-03-06 10:15:00 | 14451.52 | 14391.59 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:35:00 | 13470.00 | 13536.23 | 0.00 | ORB-short ORB[13527.00,13691.00] vol=1.8x ATR=46.83 |
| Stop hit — per-position SL triggered | 2026-03-19 11:15:00 | 13516.83 | 13516.16 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 09:45:00 | 13553.00 | 13613.51 | 0.00 | ORB-short ORB[13559.00,13700.00] vol=2.0x ATR=43.75 |
| Stop hit — per-position SL triggered | 2026-04-07 09:55:00 | 13596.75 | 13612.99 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:15:00 | 15038.00 | 14904.14 | 0.00 | ORB-long ORB[14762.00,14934.00] vol=8.5x ATR=41.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:20:00 | 15099.61 | 14929.28 | 0.00 | T1 1.5R @ 15099.61 |
| Stop hit — per-position SL triggered | 2026-04-28 11:45:00 | 15038.00 | 14982.04 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:35:00 | 14860.00 | 14941.40 | 0.00 | ORB-short ORB[14898.00,14996.00] vol=1.8x ATR=53.47 |
| Stop hit — per-position SL triggered | 2026-04-29 09:45:00 | 14913.47 | 14937.88 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:10:00 | 14659.00 | 14724.41 | 0.00 | ORB-short ORB[14744.00,14893.00] vol=2.0x ATR=40.00 |
| Stop hit — per-position SL triggered | 2026-05-05 10:35:00 | 14699.00 | 14704.97 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:20:00 | 14741.00 | 14658.24 | 0.00 | ORB-long ORB[14546.00,14693.00] vol=2.2x ATR=33.45 |
| Stop hit — per-position SL triggered | 2026-05-06 10:35:00 | 14707.55 | 14664.08 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-17 11:00:00 | 15844.00 | 2026-02-17 11:05:00 | 15885.95 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-18 11:00:00 | 15890.00 | 2026-02-18 11:05:00 | 15835.32 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-18 11:00:00 | 15890.00 | 2026-02-18 15:20:00 | 15585.00 | TARGET_HIT | 0.50 | 1.92% |
| SELL | retest1 | 2026-02-25 10:40:00 | 15645.00 | 2026-02-25 10:50:00 | 15685.37 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-06 10:10:00 | 14512.00 | 2026-03-06 10:15:00 | 14451.52 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-19 10:35:00 | 13470.00 | 2026-03-19 11:15:00 | 13516.83 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-07 09:45:00 | 13553.00 | 2026-04-07 09:55:00 | 13596.75 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-28 11:15:00 | 15038.00 | 2026-04-28 11:20:00 | 15099.61 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-04-28 11:15:00 | 15038.00 | 2026-04-28 11:45:00 | 15038.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 09:35:00 | 14860.00 | 2026-04-29 09:45:00 | 14913.47 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-05-05 10:10:00 | 14659.00 | 2026-05-05 10:35:00 | 14699.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-06 10:20:00 | 14741.00 | 2026-05-06 10:35:00 | 14707.55 | STOP_HIT | 1.00 | -0.23% |
