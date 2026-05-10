# ACME Solar Holdings Ltd. (ACMESOLAR)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 283.00
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 7
- **Target hits / Stop hits / Partials:** 1 / 7 / 3
- **Avg / median % per leg:** 0.01% / 0.00%
- **Sum % (uncompounded):** 0.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.19% | -1.0% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.19% | -1.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.19% | 1.1% |
| SELL @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.19% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.01% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:55:00 | 224.34 | 222.76 | 0.00 | ORB-long ORB[221.00,223.00] vol=1.8x ATR=0.85 |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 223.49 | 222.98 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:55:00 | 234.24 | 233.14 | 0.00 | ORB-long ORB[232.00,234.04] vol=5.6x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:05:00 | 235.08 | 233.46 | 0.00 | T1 1.5R @ 235.08 |
| Stop hit — per-position SL triggered | 2026-02-26 12:30:00 | 234.24 | 234.17 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:35:00 | 222.81 | 225.55 | 0.00 | ORB-short ORB[225.91,229.04] vol=4.6x ATR=1.04 |
| Stop hit — per-position SL triggered | 2026-03-04 09:40:00 | 223.85 | 225.42 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 09:45:00 | 270.50 | 267.13 | 0.00 | ORB-long ORB[264.40,267.85] vol=3.3x ATR=1.69 |
| Stop hit — per-position SL triggered | 2026-04-02 09:50:00 | 268.81 | 267.88 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 272.80 | 274.01 | 0.00 | ORB-short ORB[273.30,276.20] vol=2.1x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 14:00:00 | 271.45 | 273.24 | 0.00 | T1 1.5R @ 271.45 |
| Target hit | 2026-04-10 15:20:00 | 269.75 | 270.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-04-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:40:00 | 291.80 | 290.32 | 0.00 | ORB-long ORB[288.05,291.00] vol=5.4x ATR=0.95 |
| Stop hit — per-position SL triggered | 2026-04-17 09:45:00 | 290.85 | 290.42 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:10:00 | 301.30 | 305.88 | 0.00 | ORB-short ORB[304.60,307.90] vol=5.1x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:15:00 | 299.81 | 303.99 | 0.00 | T1 1.5R @ 299.81 |
| Stop hit — per-position SL triggered | 2026-04-29 11:20:00 | 301.30 | 303.85 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 09:45:00 | 301.50 | 303.79 | 0.00 | ORB-short ORB[302.00,306.00] vol=2.0x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-05-04 11:10:00 | 303.06 | 302.83 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 09:55:00 | 224.34 | 2026-02-16 10:15:00 | 223.49 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-26 10:55:00 | 234.24 | 2026-02-26 11:05:00 | 235.08 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-26 10:55:00 | 234.24 | 2026-02-26 12:30:00 | 234.24 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-04 09:35:00 | 222.81 | 2026-03-04 09:40:00 | 223.85 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-02 09:45:00 | 270.50 | 2026-04-02 09:50:00 | 268.81 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2026-04-10 10:15:00 | 272.80 | 2026-04-10 14:00:00 | 271.45 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-10 10:15:00 | 272.80 | 2026-04-10 15:20:00 | 269.75 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2026-04-17 09:40:00 | 291.80 | 2026-04-17 09:45:00 | 290.85 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-29 11:10:00 | 301.30 | 2026-04-29 11:15:00 | 299.81 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-29 11:10:00 | 301.30 | 2026-04-29 11:20:00 | 301.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-04 09:45:00 | 301.50 | 2026-05-04 11:10:00 | 303.06 | STOP_HIT | 1.00 | -0.52% |
