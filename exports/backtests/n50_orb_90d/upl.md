# UPL (UPL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 644.40
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 11
- **Target hits / Stop hits / Partials:** 0 / 11 / 2
- **Avg / median % per leg:** -0.13% / -0.28%
- **Sum % (uncompounded):** -1.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 1 | 10.0% | 0 | 9 | 1 | -0.18% | -1.8% |
| BUY @ 2nd Alert (retest1) | 10 | 1 | 10.0% | 0 | 9 | 1 | -0.18% | -1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.06% | 0.2% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.06% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 2 | 15.4% | 0 | 11 | 2 | -0.13% | -1.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:55:00 | 755.40 | 747.23 | 0.00 | ORB-long ORB[741.00,749.80] vol=1.6x ATR=2.17 |
| Stop hit — per-position SL triggered | 2026-02-10 11:10:00 | 753.23 | 747.91 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 736.65 | 733.48 | 0.00 | ORB-long ORB[728.55,733.85] vol=2.0x ATR=1.83 |
| Stop hit — per-position SL triggered | 2026-02-17 10:35:00 | 734.82 | 734.04 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:50:00 | 632.30 | 628.01 | 0.00 | ORB-long ORB[622.00,628.50] vol=3.5x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:00:00 | 636.09 | 628.67 | 0.00 | T1 1.5R @ 636.09 |
| Stop hit — per-position SL triggered | 2026-02-26 10:20:00 | 632.30 | 629.39 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 616.20 | 613.38 | 0.00 | ORB-long ORB[609.30,613.80] vol=2.0x ATR=1.98 |
| Stop hit — per-position SL triggered | 2026-03-17 11:30:00 | 614.22 | 613.50 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:05:00 | 632.05 | 629.23 | 0.00 | ORB-long ORB[623.10,630.00] vol=1.6x ATR=1.75 |
| Stop hit — per-position SL triggered | 2026-03-25 11:10:00 | 630.30 | 629.31 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 11:00:00 | 648.00 | 643.84 | 0.00 | ORB-long ORB[639.55,647.65] vol=1.5x ATR=2.23 |
| Stop hit — per-position SL triggered | 2026-04-09 11:10:00 | 645.77 | 644.29 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:00:00 | 656.75 | 661.24 | 0.00 | ORB-short ORB[660.00,665.80] vol=2.2x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:45:00 | 654.04 | 658.69 | 0.00 | T1 1.5R @ 654.04 |
| Stop hit — per-position SL triggered | 2026-04-16 13:35:00 | 656.75 | 656.04 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 668.95 | 664.64 | 0.00 | ORB-long ORB[657.75,665.00] vol=2.2x ATR=2.09 |
| Stop hit — per-position SL triggered | 2026-04-17 09:40:00 | 666.86 | 664.84 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:30:00 | 641.25 | 637.42 | 0.00 | ORB-long ORB[632.50,639.15] vol=2.0x ATR=1.94 |
| Stop hit — per-position SL triggered | 2026-04-27 11:20:00 | 639.31 | 638.27 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 644.10 | 641.23 | 0.00 | ORB-long ORB[634.95,642.25] vol=2.2x ATR=2.18 |
| Stop hit — per-position SL triggered | 2026-04-28 09:55:00 | 641.92 | 641.34 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:55:00 | 641.70 | 644.56 | 0.00 | ORB-short ORB[642.10,648.90] vol=1.9x ATR=1.54 |
| Stop hit — per-position SL triggered | 2026-05-05 11:10:00 | 643.24 | 644.46 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:55:00 | 755.40 | 2026-02-10 11:10:00 | 753.23 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-17 10:20:00 | 736.65 | 2026-02-17 10:35:00 | 734.82 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-26 09:50:00 | 632.30 | 2026-02-26 10:00:00 | 636.09 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-02-26 09:50:00 | 632.30 | 2026-02-26 10:20:00 | 632.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 11:15:00 | 616.20 | 2026-03-17 11:30:00 | 614.22 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-25 11:05:00 | 632.05 | 2026-03-25 11:10:00 | 630.30 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-09 11:00:00 | 648.00 | 2026-04-09 11:10:00 | 645.77 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-16 11:00:00 | 656.75 | 2026-04-16 11:45:00 | 654.04 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-04-16 11:00:00 | 656.75 | 2026-04-16 13:35:00 | 656.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 09:35:00 | 668.95 | 2026-04-17 09:40:00 | 666.86 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-27 10:30:00 | 641.25 | 2026-04-27 11:20:00 | 639.31 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-28 09:45:00 | 644.10 | 2026-04-28 09:55:00 | 641.92 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-05-05 10:55:00 | 641.70 | 2026-05-05 11:10:00 | 643.24 | STOP_HIT | 1.00 | -0.24% |
