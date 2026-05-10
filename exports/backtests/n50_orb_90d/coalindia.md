# COALINDIA (COALINDIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 456.55
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 2
- **Avg / median % per leg:** 0.15% / -0.21%
- **Sum % (uncompounded):** 2.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.71% | 3.5% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.71% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 2 | 20.0% | 1 | 8 | 1 | -0.13% | -1.3% |
| SELL @ 2nd Alert (retest1) | 10 | 2 | 20.0% | 1 | 8 | 1 | -0.13% | -1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 4 | 26.7% | 2 | 11 | 2 | 0.15% | 2.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:00:00 | 421.85 | 423.63 | 0.00 | ORB-short ORB[424.00,427.95] vol=4.4x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-02-11 11:55:00 | 422.91 | 423.27 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:00:00 | 420.35 | 422.75 | 0.00 | ORB-short ORB[422.05,425.95] vol=4.7x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:10:00 | 418.72 | 422.38 | 0.00 | T1 1.5R @ 418.72 |
| Target hit | 2026-02-12 15:20:00 | 419.50 | 419.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:10:00 | 419.60 | 421.29 | 0.00 | ORB-short ORB[420.60,423.00] vol=2.2x ATR=0.86 |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 420.46 | 420.71 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:15:00 | 419.75 | 422.39 | 0.00 | ORB-short ORB[420.95,426.00] vol=1.7x ATR=0.86 |
| Stop hit — per-position SL triggered | 2026-02-23 11:30:00 | 420.61 | 421.97 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:35:00 | 423.00 | 425.91 | 0.00 | ORB-short ORB[423.65,428.80] vol=2.3x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-02-24 11:00:00 | 424.06 | 425.25 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 435.90 | 434.43 | 0.00 | ORB-long ORB[428.55,433.00] vol=2.2x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-02-25 11:20:00 | 434.79 | 434.48 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 427.20 | 428.50 | 0.00 | ORB-short ORB[427.55,433.00] vol=1.5x ATR=0.97 |
| Stop hit — per-position SL triggered | 2026-02-27 11:00:00 | 428.17 | 428.60 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:25:00 | 453.50 | 448.98 | 0.00 | ORB-long ORB[445.00,451.00] vol=1.9x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:45:00 | 455.91 | 450.68 | 0.00 | T1 1.5R @ 455.91 |
| Target hit | 2026-03-12 15:20:00 | 470.00 | 464.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-03-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:30:00 | 442.40 | 444.30 | 0.00 | ORB-short ORB[443.50,447.50] vol=4.1x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-03-27 09:50:00 | 443.61 | 443.88 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:35:00 | 436.00 | 434.47 | 0.00 | ORB-long ORB[430.50,435.45] vol=2.7x ATR=0.93 |
| Stop hit — per-position SL triggered | 2026-04-17 10:45:00 | 435.07 | 434.57 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:50:00 | 446.95 | 444.71 | 0.00 | ORB-long ORB[442.55,445.35] vol=1.7x ATR=0.79 |
| Stop hit — per-position SL triggered | 2026-04-22 09:55:00 | 446.16 | 444.86 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:45:00 | 472.50 | 475.08 | 0.00 | ORB-short ORB[476.10,480.00] vol=3.6x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-05-05 11:10:00 | 473.62 | 474.39 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:00:00 | 468.10 | 469.11 | 0.00 | ORB-short ORB[468.60,472.40] vol=4.2x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 469.38 | 469.02 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 11:00:00 | 421.85 | 2026-02-11 11:55:00 | 422.91 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-12 11:00:00 | 420.35 | 2026-02-12 11:10:00 | 418.72 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-12 11:00:00 | 420.35 | 2026-02-12 15:20:00 | 419.50 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2026-02-17 11:10:00 | 419.60 | 2026-02-17 12:15:00 | 420.46 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-23 11:15:00 | 419.75 | 2026-02-23 11:30:00 | 420.61 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-24 10:35:00 | 423.00 | 2026-02-24 11:00:00 | 424.06 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-25 11:05:00 | 435.90 | 2026-02-25 11:20:00 | 434.79 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-27 10:55:00 | 427.20 | 2026-02-27 11:00:00 | 428.17 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-03-12 10:25:00 | 453.50 | 2026-03-12 10:45:00 | 455.91 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-03-12 10:25:00 | 453.50 | 2026-03-12 15:20:00 | 470.00 | TARGET_HIT | 0.50 | 3.64% |
| SELL | retest1 | 2026-03-27 09:30:00 | 442.40 | 2026-03-27 09:50:00 | 443.61 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-17 10:35:00 | 436.00 | 2026-04-17 10:45:00 | 435.07 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-04-22 09:50:00 | 446.95 | 2026-04-22 09:55:00 | 446.16 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-05-05 10:45:00 | 472.50 | 2026-05-05 11:10:00 | 473.62 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-07 11:00:00 | 468.10 | 2026-05-07 11:15:00 | 469.38 | STOP_HIT | 1.00 | -0.27% |
