# Reliance Power Ltd. (RPOWER)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 28.74
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** -0.17% / -0.34%
- **Sum % (uncompounded):** -1.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.41% | -1.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.41% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.08% | 0.2% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.08% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.17% | -1.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:30:00 | 64.44 | 64.15 | 0.00 | ORB-long ORB[63.90,64.30] vol=1.7x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-06-25 09:40:00 | 64.22 | 64.21 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-07-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:35:00 | 68.45 | 68.97 | 0.00 | ORB-short ORB[68.75,69.63] vol=2.0x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 11:10:00 | 67.99 | 68.42 | 0.00 | T1 1.5R @ 67.99 |
| Stop hit — per-position SL triggered | 2025-07-02 11:40:00 | 68.45 | 68.25 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-07-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 10:20:00 | 66.10 | 65.61 | 0.00 | ORB-long ORB[65.03,66.00] vol=2.3x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-07-08 10:25:00 | 65.77 | 65.65 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:30:00 | 65.29 | 64.99 | 0.00 | ORB-long ORB[64.31,65.20] vol=2.6x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-07-16 09:40:00 | 65.03 | 64.99 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 09:35:00 | 63.88 | 64.35 | 0.00 | ORB-short ORB[64.20,65.00] vol=3.8x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-07-18 09:50:00 | 64.16 | 64.24 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-25 09:30:00 | 64.44 | 2025-06-25 09:40:00 | 64.22 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-07-02 09:35:00 | 68.45 | 2025-07-02 11:10:00 | 67.99 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2025-07-02 09:35:00 | 68.45 | 2025-07-02 11:40:00 | 68.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-08 10:20:00 | 66.10 | 2025-07-08 10:25:00 | 65.77 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-07-16 09:30:00 | 65.29 | 2025-07-16 09:40:00 | 65.03 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-07-18 09:35:00 | 63.88 | 2025-07-18 09:50:00 | 64.16 | STOP_HIT | 1.00 | -0.43% |
