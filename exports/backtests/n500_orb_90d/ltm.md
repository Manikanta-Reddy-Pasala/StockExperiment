# LTM Ltd. (LTM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4360.00
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / Stop hits / Partials:** 0 / 7 / 2
- **Avg / median % per leg:** -0.09% / -0.21%
- **Sum % (uncompounded):** -0.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.12% | -0.5% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.12% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.06% | -0.3% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.06% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.09% | -0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 5657.00 | 5623.16 | 0.00 | ORB-long ORB[5586.00,5641.50] vol=1.6x ATR=14.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 12:55:00 | 5678.52 | 5651.51 | 0.00 | T1 1.5R @ 5678.52 |
| Stop hit — per-position SL triggered | 2026-02-10 13:20:00 | 5657.00 | 5653.81 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:10:00 | 5635.00 | 5640.57 | 0.00 | ORB-short ORB[5649.50,5692.00] vol=1.9x ATR=11.64 |
| Stop hit — per-position SL triggered | 2026-02-11 11:30:00 | 5646.64 | 5639.84 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:40:00 | 4805.50 | 4841.71 | 0.00 | ORB-short ORB[4814.50,4876.00] vol=1.6x ATR=18.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 09:55:00 | 4778.43 | 4828.39 | 0.00 | T1 1.5R @ 4778.43 |
| Stop hit — per-position SL triggered | 2026-02-23 11:00:00 | 4805.50 | 4812.50 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 10:20:00 | 4280.70 | 4244.69 | 0.00 | ORB-long ORB[4195.50,4250.00] vol=1.8x ATR=16.01 |
| Stop hit — per-position SL triggered | 2026-03-09 10:30:00 | 4264.69 | 4247.01 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:55:00 | 4189.60 | 4207.65 | 0.00 | ORB-short ORB[4195.60,4254.50] vol=2.0x ATR=15.65 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 4205.25 | 4201.15 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 11:10:00 | 4089.70 | 4133.35 | 0.00 | ORB-short ORB[4125.40,4178.90] vol=1.5x ATR=12.02 |
| Stop hit — per-position SL triggered | 2026-03-30 11:15:00 | 4101.72 | 4132.22 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 4689.70 | 4671.07 | 0.00 | ORB-long ORB[4632.20,4688.00] vol=2.3x ATR=23.32 |
| Stop hit — per-position SL triggered | 2026-04-22 09:40:00 | 4666.38 | 4671.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:45:00 | 5657.00 | 2026-02-10 12:55:00 | 5678.52 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-10 09:45:00 | 5657.00 | 2026-02-10 13:20:00 | 5657.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 11:10:00 | 5635.00 | 2026-02-11 11:30:00 | 5646.64 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-23 09:40:00 | 4805.50 | 2026-02-23 09:55:00 | 4778.43 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-02-23 09:40:00 | 4805.50 | 2026-02-23 11:00:00 | 4805.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-09 10:20:00 | 4280.70 | 2026-03-09 10:30:00 | 4264.69 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-16 10:55:00 | 4189.60 | 2026-03-16 11:15:00 | 4205.25 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-30 11:10:00 | 4089.70 | 2026-03-30 11:15:00 | 4101.72 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-22 09:35:00 | 4689.70 | 2026-04-22 09:40:00 | 4666.38 | STOP_HIT | 1.00 | -0.50% |
