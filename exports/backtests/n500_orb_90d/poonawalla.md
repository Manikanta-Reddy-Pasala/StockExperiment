# Poonawalla Fincorp Ltd. (POONAWALLA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 461.50
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
| TARGET_HIT | 5 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 6
- **Target hits / Stop hits / Partials:** 5 / 6 / 7
- **Avg / median % per leg:** 0.49% / 0.67%
- **Sum % (uncompounded):** 8.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 7 | 58.3% | 3 | 5 | 4 | 0.37% | 4.5% |
| BUY @ 2nd Alert (retest1) | 12 | 7 | 58.3% | 3 | 5 | 4 | 0.37% | 4.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 5 | 83.3% | 2 | 1 | 3 | 0.73% | 4.4% |
| SELL @ 2nd Alert (retest1) | 6 | 5 | 83.3% | 2 | 1 | 3 | 0.73% | 4.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 12 | 66.7% | 5 | 6 | 7 | 0.49% | 8.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:00:00 | 466.85 | 462.94 | 0.00 | ORB-long ORB[459.10,464.40] vol=2.1x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:05:00 | 470.05 | 463.85 | 0.00 | T1 1.5R @ 470.05 |
| Target hit | 2026-02-17 15:05:00 | 475.15 | 475.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2026-02-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:30:00 | 478.80 | 475.28 | 0.00 | ORB-long ORB[471.90,476.60] vol=1.7x ATR=2.05 |
| Stop hit — per-position SL triggered | 2026-02-23 09:50:00 | 476.75 | 476.56 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:20:00 | 471.80 | 469.37 | 0.00 | ORB-long ORB[468.00,470.25] vol=2.0x ATR=1.35 |
| Stop hit — per-position SL triggered | 2026-02-25 10:40:00 | 470.45 | 469.80 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:45:00 | 428.80 | 432.09 | 0.00 | ORB-short ORB[429.90,435.70] vol=2.0x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 14:45:00 | 425.91 | 429.44 | 0.00 | T1 1.5R @ 425.91 |
| Target hit | 2026-03-06 15:20:00 | 423.70 | 427.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-03-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:30:00 | 410.35 | 406.84 | 0.00 | ORB-long ORB[402.55,408.00] vol=4.7x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:35:00 | 413.40 | 408.56 | 0.00 | T1 1.5R @ 413.40 |
| Target hit | 2026-03-16 10:05:00 | 411.95 | 412.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 408.60 | 406.51 | 0.00 | ORB-long ORB[403.35,408.00] vol=2.5x ATR=1.44 |
| Stop hit — per-position SL triggered | 2026-03-20 09:35:00 | 407.16 | 406.59 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:35:00 | 421.20 | 417.26 | 0.00 | ORB-long ORB[413.70,418.85] vol=1.7x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 10:00:00 | 425.71 | 419.23 | 0.00 | T1 1.5R @ 425.71 |
| Target hit | 2026-04-08 11:50:00 | 421.55 | 421.67 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2026-04-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:55:00 | 410.55 | 416.32 | 0.00 | ORB-short ORB[416.35,421.85] vol=4.9x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 11:40:00 | 407.74 | 414.89 | 0.00 | T1 1.5R @ 407.74 |
| Target hit | 2026-04-09 15:20:00 | 404.90 | 404.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-04-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:50:00 | 405.25 | 399.69 | 0.00 | ORB-long ORB[394.00,399.50] vol=2.9x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 09:55:00 | 409.96 | 400.53 | 0.00 | T1 1.5R @ 409.96 |
| Stop hit — per-position SL triggered | 2026-04-10 11:10:00 | 405.25 | 403.03 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 424.60 | 423.03 | 0.00 | ORB-long ORB[418.10,424.10] vol=4.3x ATR=1.59 |
| Stop hit — per-position SL triggered | 2026-04-21 10:25:00 | 423.01 | 423.10 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:05:00 | 411.40 | 413.59 | 0.00 | ORB-short ORB[416.45,421.20] vol=1.7x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:40:00 | 409.57 | 412.99 | 0.00 | T1 1.5R @ 409.57 |
| Stop hit — per-position SL triggered | 2026-04-29 13:00:00 | 411.40 | 412.08 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 10:00:00 | 466.85 | 2026-02-17 10:05:00 | 470.05 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-02-17 10:00:00 | 466.85 | 2026-02-17 15:05:00 | 475.15 | TARGET_HIT | 0.50 | 1.78% |
| BUY | retest1 | 2026-02-23 09:30:00 | 478.80 | 2026-02-23 09:50:00 | 476.75 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-02-25 10:20:00 | 471.80 | 2026-02-25 10:40:00 | 470.45 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-06 09:45:00 | 428.80 | 2026-03-06 14:45:00 | 425.91 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2026-03-06 09:45:00 | 428.80 | 2026-03-06 15:20:00 | 423.70 | TARGET_HIT | 0.50 | 1.19% |
| BUY | retest1 | 2026-03-16 09:30:00 | 410.35 | 2026-03-16 09:35:00 | 413.40 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-03-16 09:30:00 | 410.35 | 2026-03-16 10:05:00 | 411.95 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2026-03-20 09:30:00 | 408.60 | 2026-03-20 09:35:00 | 407.16 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-08 09:35:00 | 421.20 | 2026-04-08 10:00:00 | 425.71 | PARTIAL | 0.50 | 1.07% |
| BUY | retest1 | 2026-04-08 09:35:00 | 421.20 | 2026-04-08 11:50:00 | 421.55 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2026-04-09 10:55:00 | 410.55 | 2026-04-09 11:40:00 | 407.74 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-04-09 10:55:00 | 410.55 | 2026-04-09 15:20:00 | 404.90 | TARGET_HIT | 0.50 | 1.38% |
| BUY | retest1 | 2026-04-10 09:50:00 | 405.25 | 2026-04-10 09:55:00 | 409.96 | PARTIAL | 0.50 | 1.16% |
| BUY | retest1 | 2026-04-10 09:50:00 | 405.25 | 2026-04-10 11:10:00 | 405.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 10:00:00 | 424.60 | 2026-04-21 10:25:00 | 423.01 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-29 11:05:00 | 411.40 | 2026-04-29 11:40:00 | 409.57 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-04-29 11:05:00 | 411.40 | 2026-04-29 13:00:00 | 411.40 | STOP_HIT | 0.50 | 0.00% |
