# Latent View Analytics Ltd. (LATENTVIEW)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-06-09 15:25:00 (1575 bars)
- **Last close:** 414.75
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
- **Avg / median % per leg:** -0.15% / -0.24%
- **Sum % (uncompounded):** -1.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.17% | -0.9% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.17% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.13% | -0.9% |
| SELL @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.13% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 3 | 25.0% | 1 | 9 | 2 | -0.15% | -1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 09:30:00 | 419.75 | 421.36 | 0.00 | ORB-short ORB[421.00,424.80] vol=4.3x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-19 09:35:00 | 417.87 | 420.63 | 0.00 | T1 1.5R @ 417.87 |
| Target hit | 2025-05-19 10:15:00 | 419.50 | 419.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2025-05-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 09:30:00 | 407.60 | 408.84 | 0.00 | ORB-short ORB[407.95,412.60] vol=2.2x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-05-22 10:05:00 | 409.06 | 408.62 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:40:00 | 414.95 | 411.94 | 0.00 | ORB-long ORB[407.70,411.90] vol=1.9x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-05-23 09:45:00 | 413.17 | 412.09 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:30:00 | 422.65 | 421.49 | 0.00 | ORB-long ORB[417.65,422.00] vol=3.9x ATR=1.50 |
| Stop hit — per-position SL triggered | 2025-05-26 09:50:00 | 421.15 | 421.73 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 10:40:00 | 417.30 | 419.13 | 0.00 | ORB-short ORB[418.50,422.00] vol=1.6x ATR=0.99 |
| Stop hit — per-position SL triggered | 2025-05-28 10:55:00 | 418.29 | 418.98 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 10:35:00 | 418.20 | 420.80 | 0.00 | ORB-short ORB[420.45,423.50] vol=1.7x ATR=1.02 |
| Stop hit — per-position SL triggered | 2025-05-29 10:45:00 | 419.22 | 420.78 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 10:00:00 | 417.20 | 414.98 | 0.00 | ORB-long ORB[412.20,416.70] vol=5.1x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-06-02 10:35:00 | 415.76 | 415.88 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 10:20:00 | 415.00 | 416.43 | 0.00 | ORB-short ORB[415.05,419.70] vol=4.6x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-06-03 10:35:00 | 416.03 | 416.16 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:30:00 | 413.30 | 414.72 | 0.00 | ORB-short ORB[413.60,417.20] vol=2.0x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-06-04 09:40:00 | 414.66 | 414.64 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 11:10:00 | 411.00 | 410.12 | 0.00 | ORB-long ORB[408.10,410.95] vol=3.1x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 11:30:00 | 412.08 | 410.33 | 0.00 | T1 1.5R @ 412.08 |
| Stop hit — per-position SL triggered | 2025-06-06 11:40:00 | 411.00 | 410.35 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-19 09:30:00 | 419.75 | 2025-05-19 09:35:00 | 417.87 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-05-19 09:30:00 | 419.75 | 2025-05-19 10:15:00 | 419.50 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2025-05-22 09:30:00 | 407.60 | 2025-05-22 10:05:00 | 409.06 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-05-23 09:40:00 | 414.95 | 2025-05-23 09:45:00 | 413.17 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-05-26 09:30:00 | 422.65 | 2025-05-26 09:50:00 | 421.15 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-05-28 10:40:00 | 417.30 | 2025-05-28 10:55:00 | 418.29 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-05-29 10:35:00 | 418.20 | 2025-05-29 10:45:00 | 419.22 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-02 10:00:00 | 417.20 | 2025-06-02 10:35:00 | 415.76 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-06-03 10:20:00 | 415.00 | 2025-06-03 10:35:00 | 416.03 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-06-04 09:30:00 | 413.30 | 2025-06-04 09:40:00 | 414.66 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-06 11:10:00 | 411.00 | 2025-06-06 11:30:00 | 412.08 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-06-06 11:10:00 | 411.00 | 2025-06-06 11:40:00 | 411.00 | STOP_HIT | 0.50 | 0.00% |
