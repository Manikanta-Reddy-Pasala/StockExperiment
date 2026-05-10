# Redington Ltd. (REDINGTON)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 223.29
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 2
- **Target hits / Stop hits / Partials:** 3 / 2 / 4
- **Avg / median % per leg:** 0.57% / 0.57%
- **Sum % (uncompounded):** 5.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 1 | 0 | 1 | 0.60% | 1.2% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 0.60% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 5 | 71.4% | 2 | 2 | 3 | 0.55% | 3.9% |
| SELL @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 2 | 2 | 3 | 0.55% | 3.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 7 | 77.8% | 3 | 2 | 4 | 0.57% | 5.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 259.05 | 257.01 | 0.00 | ORB-long ORB[254.50,256.90] vol=2.0x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:35:00 | 260.26 | 257.56 | 0.00 | T1 1.5R @ 260.26 |
| Target hit | 2026-02-17 15:20:00 | 260.95 | 259.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:05:00 | 257.45 | 258.36 | 0.00 | ORB-short ORB[257.95,259.85] vol=3.5x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:50:00 | 256.53 | 258.16 | 0.00 | T1 1.5R @ 256.53 |
| Target hit | 2026-02-19 15:20:00 | 254.85 | 256.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:40:00 | 250.80 | 251.79 | 0.00 | ORB-short ORB[250.85,254.00] vol=1.5x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 09:50:00 | 249.31 | 251.37 | 0.00 | T1 1.5R @ 249.31 |
| Target hit | 2026-02-23 15:20:00 | 246.35 | 247.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 241.80 | 243.41 | 0.00 | ORB-short ORB[243.00,246.50] vol=2.4x ATR=1.03 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 242.83 | 243.07 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:20:00 | 212.76 | 214.50 | 0.00 | ORB-short ORB[215.40,217.50] vol=2.0x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:25:00 | 211.54 | 214.12 | 0.00 | T1 1.5R @ 211.54 |
| Stop hit — per-position SL triggered | 2026-04-29 10:50:00 | 212.76 | 213.72 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 10:25:00 | 259.05 | 2026-02-17 10:35:00 | 260.26 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-17 10:25:00 | 259.05 | 2026-02-17 15:20:00 | 260.95 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2026-02-19 11:05:00 | 257.45 | 2026-02-19 11:50:00 | 256.53 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-19 11:05:00 | 257.45 | 2026-02-19 15:20:00 | 254.85 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2026-02-23 09:40:00 | 250.80 | 2026-02-23 09:50:00 | 249.31 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-02-23 09:40:00 | 250.80 | 2026-02-23 15:20:00 | 246.35 | TARGET_HIT | 0.50 | 1.77% |
| SELL | retest1 | 2026-02-24 09:30:00 | 241.80 | 2026-02-24 09:45:00 | 242.83 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-29 10:20:00 | 212.76 | 2026-04-29 10:25:00 | 211.54 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-04-29 10:20:00 | 212.76 | 2026-04-29 10:50:00 | 212.76 | STOP_HIT | 0.50 | 0.00% |
