# Trident Ltd. (TRIDENT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 26.60
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
- **Avg / median % per leg:** -0.17% / -0.33%
- **Sum % (uncompounded):** -2.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.20% | -1.4% |
| BUY @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.20% | -1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.13% | -0.8% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.13% | -0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 2 | 15.4% | 0 | 11 | 2 | -0.17% | -2.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 26.34 | 26.40 | 0.00 | ORB-short ORB[26.36,26.55] vol=3.0x ATR=0.07 |
| Stop hit — per-position SL triggered | 2026-02-18 09:40:00 | 26.41 | 26.40 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 25.80 | 25.64 | 0.00 | ORB-long ORB[25.46,25.75] vol=2.1x ATR=0.12 |
| Stop hit — per-position SL triggered | 2026-04-15 09:45:00 | 25.68 | 25.64 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 25.80 | 26.04 | 0.00 | ORB-short ORB[25.91,26.25] vol=1.9x ATR=0.09 |
| Stop hit — per-position SL triggered | 2026-04-16 11:00:00 | 25.89 | 25.97 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:00:00 | 26.18 | 26.04 | 0.00 | ORB-long ORB[25.90,26.10] vol=2.2x ATR=0.09 |
| Stop hit — per-position SL triggered | 2026-04-17 10:30:00 | 26.09 | 26.05 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 26.10 | 25.91 | 0.00 | ORB-long ORB[25.68,25.95] vol=2.3x ATR=0.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:05:00 | 26.23 | 26.01 | 0.00 | T1 1.5R @ 26.23 |
| Stop hit — per-position SL triggered | 2026-04-21 11:10:00 | 26.10 | 26.09 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 26.06 | 25.98 | 0.00 | ORB-long ORB[25.77,26.00] vol=3.1x ATR=0.08 |
| Stop hit — per-position SL triggered | 2026-04-22 09:50:00 | 25.98 | 26.01 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 25.77 | 25.93 | 0.00 | ORB-short ORB[25.88,26.14] vol=2.6x ATR=0.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:40:00 | 25.64 | 25.86 | 0.00 | T1 1.5R @ 25.64 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 25.77 | 25.81 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:30:00 | 26.60 | 26.44 | 0.00 | ORB-long ORB[26.07,26.29] vol=9.0x ATR=0.12 |
| Stop hit — per-position SL triggered | 2026-04-29 09:35:00 | 26.48 | 26.48 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 26.27 | 26.21 | 0.00 | ORB-long ORB[26.10,26.25] vol=2.4x ATR=0.09 |
| Stop hit — per-position SL triggered | 2026-05-05 10:05:00 | 26.18 | 26.23 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:50:00 | 26.43 | 26.55 | 0.00 | ORB-short ORB[26.44,26.70] vol=2.2x ATR=0.10 |
| Stop hit — per-position SL triggered | 2026-05-07 10:00:00 | 26.53 | 26.55 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 26.79 | 26.97 | 0.00 | ORB-short ORB[26.81,27.11] vol=2.3x ATR=0.09 |
| Stop hit — per-position SL triggered | 2026-05-08 09:50:00 | 26.88 | 26.92 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 09:30:00 | 26.34 | 2026-02-18 09:40:00 | 26.41 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-15 09:40:00 | 25.80 | 2026-04-15 09:45:00 | 25.68 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-16 09:45:00 | 25.80 | 2026-04-16 11:00:00 | 25.89 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-17 10:00:00 | 26.18 | 2026-04-17 10:30:00 | 26.09 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-21 10:00:00 | 26.10 | 2026-04-21 10:05:00 | 26.23 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-04-21 10:00:00 | 26.10 | 2026-04-21 11:10:00 | 26.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:35:00 | 26.06 | 2026-04-22 09:50:00 | 25.98 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-24 09:30:00 | 25.77 | 2026-04-24 09:40:00 | 25.64 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-04-24 09:30:00 | 25.77 | 2026-04-24 10:00:00 | 25.77 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 09:30:00 | 26.60 | 2026-04-29 09:35:00 | 26.48 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-05 09:30:00 | 26.27 | 2026-05-05 10:05:00 | 26.18 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-05-07 09:50:00 | 26.43 | 2026-05-07 10:00:00 | 26.53 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-05-08 09:35:00 | 26.79 | 2026-05-08 09:50:00 | 26.88 | STOP_HIT | 1.00 | -0.33% |
