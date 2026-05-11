# Chambal Fertilizers & Chemicals Ltd. (CHAMBLFERT)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-07-09 15:25:00 (3021 bars)
- **Last close:** 511.00
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
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 9
- **Target hits / Stop hits / Partials:** 2 / 9 / 7
- **Avg / median % per leg:** 1.21% / 0.49%
- **Sum % (uncompounded):** 21.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 7 | 58.3% | 2 | 5 | 5 | 1.78% | 21.3% |
| BUY @ 2nd Alert (retest1) | 12 | 7 | 58.3% | 2 | 5 | 5 | 1.78% | 21.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.08% | 0.5% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.08% | 0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 9 | 50.0% | 2 | 9 | 7 | 1.21% | 21.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 401.35 | 402.53 | 0.00 | ORB-short ORB[401.65,404.80] vol=3.2x ATR=1.24 |
| Stop hit — per-position SL triggered | 2024-05-16 12:15:00 | 402.59 | 402.37 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 09:35:00 | 405.60 | 406.87 | 0.00 | ORB-short ORB[405.65,408.70] vol=1.9x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 09:45:00 | 403.62 | 406.48 | 0.00 | T1 1.5R @ 403.62 |
| Stop hit — per-position SL triggered | 2024-05-17 09:50:00 | 405.60 | 406.32 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 10:15:00 | 399.40 | 400.16 | 0.00 | ORB-short ORB[400.00,404.25] vol=2.1x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-05-22 10:40:00 | 400.94 | 400.05 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 09:45:00 | 399.35 | 397.05 | 0.00 | ORB-long ORB[392.25,397.90] vol=1.9x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 10:00:00 | 401.39 | 400.11 | 0.00 | T1 1.5R @ 401.39 |
| Target hit | 2024-05-24 10:55:00 | 406.20 | 406.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2024-06-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 11:00:00 | 432.50 | 426.73 | 0.00 | ORB-long ORB[422.50,428.75] vol=4.1x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 11:10:00 | 434.78 | 427.89 | 0.00 | T1 1.5R @ 434.78 |
| Stop hit — per-position SL triggered | 2024-06-18 11:30:00 | 432.50 | 428.61 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:40:00 | 474.50 | 469.36 | 0.00 | ORB-long ORB[465.25,470.25] vol=1.8x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 09:45:00 | 478.11 | 471.56 | 0.00 | T1 1.5R @ 478.11 |
| Target hit | 2024-06-20 15:20:00 | 555.90 | 523.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-06-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:35:00 | 507.00 | 503.78 | 0.00 | ORB-long ORB[499.10,505.80] vol=2.1x ATR=2.92 |
| Stop hit — per-position SL triggered | 2024-06-28 09:45:00 | 504.08 | 504.28 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:50:00 | 518.80 | 511.39 | 0.00 | ORB-long ORB[502.00,507.85] vol=1.5x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 13:55:00 | 523.39 | 518.27 | 0.00 | T1 1.5R @ 523.39 |
| Stop hit — per-position SL triggered | 2024-07-01 14:25:00 | 518.80 | 518.34 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:50:00 | 516.00 | 519.07 | 0.00 | ORB-short ORB[519.20,525.90] vol=1.8x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 10:15:00 | 512.51 | 517.66 | 0.00 | T1 1.5R @ 512.51 |
| Stop hit — per-position SL triggered | 2024-07-02 10:25:00 | 516.00 | 517.41 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:10:00 | 518.30 | 513.65 | 0.00 | ORB-long ORB[508.25,515.00] vol=2.0x ATR=2.25 |
| Stop hit — per-position SL triggered | 2024-07-03 10:30:00 | 516.05 | 514.75 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 10:00:00 | 527.60 | 523.57 | 0.00 | ORB-long ORB[519.25,526.75] vol=2.1x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 10:05:00 | 531.71 | 525.77 | 0.00 | T1 1.5R @ 531.71 |
| Stop hit — per-position SL triggered | 2024-07-08 11:00:00 | 527.60 | 529.13 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 11:15:00 | 401.35 | 2024-05-16 12:15:00 | 402.59 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-17 09:35:00 | 405.60 | 2024-05-17 09:45:00 | 403.62 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-05-17 09:35:00 | 405.60 | 2024-05-17 09:50:00 | 405.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-22 10:15:00 | 399.40 | 2024-05-22 10:40:00 | 400.94 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-05-24 09:45:00 | 399.35 | 2024-05-24 10:00:00 | 401.39 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-05-24 09:45:00 | 399.35 | 2024-05-24 10:55:00 | 406.20 | TARGET_HIT | 0.50 | 1.72% |
| BUY | retest1 | 2024-06-18 11:00:00 | 432.50 | 2024-06-18 11:10:00 | 434.78 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-06-18 11:00:00 | 432.50 | 2024-06-18 11:30:00 | 432.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-20 09:40:00 | 474.50 | 2024-06-20 09:45:00 | 478.11 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2024-06-20 09:40:00 | 474.50 | 2024-06-20 15:20:00 | 555.90 | TARGET_HIT | 0.50 | 17.15% |
| BUY | retest1 | 2024-06-28 09:35:00 | 507.00 | 2024-06-28 09:45:00 | 504.08 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2024-07-01 09:50:00 | 518.80 | 2024-07-01 13:55:00 | 523.39 | PARTIAL | 0.50 | 0.89% |
| BUY | retest1 | 2024-07-01 09:50:00 | 518.80 | 2024-07-01 14:25:00 | 518.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-02 09:50:00 | 516.00 | 2024-07-02 10:15:00 | 512.51 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-07-02 09:50:00 | 516.00 | 2024-07-02 10:25:00 | 516.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-03 10:10:00 | 518.30 | 2024-07-03 10:30:00 | 516.05 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-07-08 10:00:00 | 527.60 | 2024-07-08 10:05:00 | 531.71 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-07-08 10:00:00 | 527.60 | 2024-07-08 11:00:00 | 527.60 | STOP_HIT | 0.50 | 0.00% |
