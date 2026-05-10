# Neuland Laboratories Ltd. (NEULANDLAB)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 17713.00
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
- **Avg / median % per leg:** -0.16% / -0.32%
- **Sum % (uncompounded):** -1.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 1 | 7 | 2 | -0.11% | -1.1% |
| BUY @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 1 | 7 | 2 | -0.11% | -1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.38% | -0.8% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.38% | -0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 3 | 25.0% | 1 | 9 | 2 | -0.16% | -1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 13545.00 | 13298.42 | 0.00 | ORB-long ORB[13056.00,13229.00] vol=2.3x ATR=108.24 |
| Stop hit — per-position SL triggered | 2026-02-09 11:50:00 | 13436.76 | 13432.28 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:30:00 | 13029.00 | 12925.76 | 0.00 | ORB-long ORB[12818.00,12971.00] vol=1.8x ATR=62.33 |
| Stop hit — per-position SL triggered | 2026-03-05 10:00:00 | 12966.67 | 12948.35 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 11:00:00 | 13118.00 | 13016.36 | 0.00 | ORB-long ORB[12924.00,13045.00] vol=2.6x ATR=41.39 |
| Stop hit — per-position SL triggered | 2026-03-06 11:10:00 | 13076.61 | 13020.24 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:30:00 | 12860.00 | 12777.68 | 0.00 | ORB-long ORB[12670.00,12799.00] vol=2.2x ATR=45.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 09:45:00 | 12928.84 | 12823.76 | 0.00 | T1 1.5R @ 12928.84 |
| Target hit | 2026-03-11 13:10:00 | 12928.00 | 12988.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 12200.00 | 12325.26 | 0.00 | ORB-short ORB[12264.00,12399.00] vol=1.9x ATR=52.08 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 12252.08 | 12327.18 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:45:00 | 12498.00 | 12409.21 | 0.00 | ORB-long ORB[12294.00,12452.00] vol=2.5x ATR=49.18 |
| Stop hit — per-position SL triggered | 2026-03-18 11:05:00 | 12448.82 | 12419.41 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:55:00 | 12087.00 | 12172.63 | 0.00 | ORB-short ORB[12158.00,12305.00] vol=1.7x ATR=39.20 |
| Stop hit — per-position SL triggered | 2026-03-19 11:05:00 | 12126.20 | 12167.07 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 15165.00 | 15075.66 | 0.00 | ORB-long ORB[14960.00,15129.00] vol=3.1x ATR=55.04 |
| Stop hit — per-position SL triggered | 2026-04-17 10:10:00 | 15109.96 | 15090.05 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:50:00 | 14756.00 | 14609.11 | 0.00 | ORB-long ORB[14503.00,14640.00] vol=2.9x ATR=53.98 |
| Stop hit — per-position SL triggered | 2026-04-29 12:35:00 | 14702.02 | 14657.41 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:40:00 | 14743.00 | 14638.98 | 0.00 | ORB-long ORB[14464.00,14650.00] vol=2.0x ATR=53.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:50:00 | 14822.73 | 14682.73 | 0.00 | T1 1.5R @ 14822.73 |
| Stop hit — per-position SL triggered | 2026-04-30 10:25:00 | 14743.00 | 14714.38 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 13545.00 | 2026-02-09 11:50:00 | 13436.76 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest1 | 2026-03-05 09:30:00 | 13029.00 | 2026-03-05 10:00:00 | 12966.67 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-03-06 11:00:00 | 13118.00 | 2026-03-06 11:10:00 | 13076.61 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-11 09:30:00 | 12860.00 | 2026-03-11 09:45:00 | 12928.84 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-03-11 09:30:00 | 12860.00 | 2026-03-11 13:10:00 | 12928.00 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2026-03-17 10:25:00 | 12200.00 | 2026-03-17 10:30:00 | 12252.08 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-18 10:45:00 | 12498.00 | 2026-03-18 11:05:00 | 12448.82 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-03-19 10:55:00 | 12087.00 | 2026-03-19 11:05:00 | 12126.20 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-17 10:05:00 | 15165.00 | 2026-04-17 10:10:00 | 15109.96 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-29 10:50:00 | 14756.00 | 2026-04-29 12:35:00 | 14702.02 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-30 09:40:00 | 14743.00 | 2026-04-30 09:50:00 | 14822.73 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-04-30 09:40:00 | 14743.00 | 2026-04-30 10:25:00 | 14743.00 | STOP_HIT | 0.50 | 0.00% |
