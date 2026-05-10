# Sumitomo Chemical India Ltd. (SUMICHEM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 485.90
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
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 6
- **Target hits / Stop hits / Partials:** 5 / 6 / 6
- **Avg / median % per leg:** 0.43% / 0.39%
- **Sum % (uncompounded):** 7.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.19% | 1.7% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.19% | 1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.70% | 5.6% |
| SELL @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.70% | 5.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 11 | 64.7% | 5 | 6 | 6 | 0.43% | 7.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 422.00 | 423.62 | 0.00 | ORB-short ORB[423.00,427.90] vol=2.7x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:55:00 | 419.54 | 423.40 | 0.00 | T1 1.5R @ 419.54 |
| Target hit | 2026-02-10 12:45:00 | 421.25 | 421.14 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2026-02-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:00:00 | 434.45 | 430.91 | 0.00 | ORB-long ORB[420.55,426.95] vol=4.2x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:05:00 | 437.38 | 433.18 | 0.00 | T1 1.5R @ 437.38 |
| Target hit | 2026-02-11 10:50:00 | 436.40 | 436.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-02-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:10:00 | 408.00 | 406.04 | 0.00 | ORB-long ORB[401.45,407.30] vol=3.1x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 12:40:00 | 410.06 | 406.91 | 0.00 | T1 1.5R @ 410.06 |
| Target hit | 2026-02-17 15:20:00 | 411.20 | 407.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:10:00 | 411.25 | 412.25 | 0.00 | ORB-short ORB[411.60,415.05] vol=3.1x ATR=0.91 |
| Stop hit — per-position SL triggered | 2026-02-19 10:25:00 | 412.16 | 412.23 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 11:15:00 | 411.25 | 409.44 | 0.00 | ORB-long ORB[406.15,410.70] vol=2.8x ATR=1.07 |
| Stop hit — per-position SL triggered | 2026-02-23 11:35:00 | 410.18 | 409.51 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:25:00 | 406.35 | 408.07 | 0.00 | ORB-short ORB[407.00,411.75] vol=3.0x ATR=1.30 |
| Stop hit — per-position SL triggered | 2026-02-24 10:30:00 | 407.65 | 408.05 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 406.85 | 405.16 | 0.00 | ORB-long ORB[401.60,404.95] vol=2.6x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:05:00 | 408.45 | 406.78 | 0.00 | T1 1.5R @ 408.45 |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 406.85 | 406.81 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:15:00 | 431.30 | 429.54 | 0.00 | ORB-long ORB[426.80,430.70] vol=2.1x ATR=1.44 |
| Stop hit — per-position SL triggered | 2026-04-17 10:20:00 | 429.86 | 429.58 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:50:00 | 451.05 | 448.29 | 0.00 | ORB-long ORB[442.15,447.00] vol=5.0x ATR=2.12 |
| Stop hit — per-position SL triggered | 2026-04-23 10:55:00 | 448.93 | 450.29 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:45:00 | 442.85 | 444.12 | 0.00 | ORB-short ORB[443.30,446.70] vol=2.4x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:55:00 | 441.21 | 443.35 | 0.00 | T1 1.5R @ 441.21 |
| Target hit | 2026-04-28 15:20:00 | 432.05 | 437.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-04-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:05:00 | 428.50 | 431.99 | 0.00 | ORB-short ORB[432.15,436.00] vol=3.5x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:05:00 | 426.16 | 430.05 | 0.00 | T1 1.5R @ 426.16 |
| Target hit | 2026-04-30 15:20:00 | 419.70 | 423.67 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:35:00 | 422.00 | 2026-02-10 10:55:00 | 419.54 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-02-10 10:35:00 | 422.00 | 2026-02-10 12:45:00 | 421.25 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2026-02-11 10:00:00 | 434.45 | 2026-02-11 10:05:00 | 437.38 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-02-11 10:00:00 | 434.45 | 2026-02-11 10:50:00 | 436.40 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-17 11:10:00 | 408.00 | 2026-02-17 12:40:00 | 410.06 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-17 11:10:00 | 408.00 | 2026-02-17 15:20:00 | 411.20 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2026-02-19 10:10:00 | 411.25 | 2026-02-19 10:25:00 | 412.16 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-23 11:15:00 | 411.25 | 2026-02-23 11:35:00 | 410.18 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-24 10:25:00 | 406.35 | 2026-02-24 10:30:00 | 407.65 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 10:50:00 | 406.85 | 2026-02-26 11:05:00 | 408.45 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-26 10:50:00 | 406.85 | 2026-02-26 11:15:00 | 406.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:15:00 | 431.30 | 2026-04-17 10:20:00 | 429.86 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-23 09:50:00 | 451.05 | 2026-04-23 10:55:00 | 448.93 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-04-28 10:45:00 | 442.85 | 2026-04-28 10:55:00 | 441.21 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-04-28 10:45:00 | 442.85 | 2026-04-28 15:20:00 | 432.05 | TARGET_HIT | 0.50 | 2.44% |
| SELL | retest1 | 2026-04-30 10:05:00 | 428.50 | 2026-04-30 11:05:00 | 426.16 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-04-30 10:05:00 | 428.50 | 2026-04-30 15:20:00 | 419.70 | TARGET_HIT | 0.50 | 2.05% |
