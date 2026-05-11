# Sundaram Finance Ltd. (SUNDARMFIN)

## Backtest Summary

- **Window:** 2025-03-07 09:15:00 → 2026-05-08 15:25:00 (21463 bars)
- **Last close:** 4700.10
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 0.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.32% | 0.6% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.32% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.06% | -0.3% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.06% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.06% | 0.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 11:00:00 | 4601.05 | 4558.22 | 0.00 | ORB-long ORB[4484.20,4544.10] vol=3.8x ATR=19.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:05:00 | 4630.35 | 4575.88 | 0.00 | T1 1.5R @ 4630.35 |
| Stop hit — per-position SL triggered | 2025-03-12 11:15:00 | 4601.05 | 4586.72 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-03-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-18 09:40:00 | 4753.00 | 4773.89 | 0.00 | ORB-short ORB[4760.00,4816.90] vol=2.0x ATR=18.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 09:55:00 | 4725.49 | 4767.22 | 0.00 | T1 1.5R @ 4725.49 |
| Stop hit — per-position SL triggered | 2025-03-18 12:00:00 | 4753.00 | 4755.42 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 11:15:00 | 5260.40 | 5286.70 | 0.00 | ORB-short ORB[5280.20,5330.10] vol=4.2x ATR=17.18 |
| Stop hit — per-position SL triggered | 2025-04-23 14:15:00 | 5277.58 | 5278.99 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-02 10:35:00 | 5204.00 | 5215.98 | 0.00 | ORB-short ORB[5205.00,5260.00] vol=2.8x ATR=26.64 |
| Stop hit — per-position SL triggered | 2025-05-02 10:50:00 | 5230.64 | 5216.02 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-03-12 11:00:00 | 4601.05 | 2025-03-12 11:05:00 | 4630.35 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-03-12 11:00:00 | 4601.05 | 2025-03-12 11:15:00 | 4601.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-18 09:40:00 | 4753.00 | 2025-03-18 09:55:00 | 4725.49 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-03-18 09:40:00 | 4753.00 | 2025-03-18 12:00:00 | 4753.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-23 11:15:00 | 5260.40 | 2025-04-23 14:15:00 | 5277.58 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-05-02 10:35:00 | 5204.00 | 2025-05-02 10:50:00 | 5230.64 | STOP_HIT | 1.00 | -0.51% |
