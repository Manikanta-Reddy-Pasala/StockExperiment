# Aarti Industries Ltd. (AARTIIND)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 486.00
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 14
- **Target hits / Stop hits / Partials:** 1 / 14 / 4
- **Avg / median % per leg:** 0.01% / -0.28%
- **Sum % (uncompounded):** 0.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.17% | -1.2% |
| BUY @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.17% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.11% | 1.3% |
| SELL @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.11% | 1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 5 | 26.3% | 1 | 14 | 4 | 0.01% | 0.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:55:00 | 460.75 | 463.47 | 0.00 | ORB-short ORB[463.60,470.00] vol=1.6x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-02-11 11:30:00 | 462.12 | 463.16 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:45:00 | 460.45 | 458.21 | 0.00 | ORB-long ORB[453.60,459.00] vol=1.6x ATR=1.51 |
| Stop hit — per-position SL triggered | 2026-02-18 09:55:00 | 458.94 | 458.55 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:05:00 | 446.80 | 450.60 | 0.00 | ORB-short ORB[451.00,457.00] vol=1.7x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:35:00 | 444.90 | 449.84 | 0.00 | T1 1.5R @ 444.90 |
| Stop hit — per-position SL triggered | 2026-02-23 15:05:00 | 446.80 | 446.13 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 441.15 | 442.13 | 0.00 | ORB-short ORB[441.80,445.35] vol=2.2x ATR=1.88 |
| Stop hit — per-position SL triggered | 2026-02-24 09:35:00 | 443.03 | 442.15 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:45:00 | 428.90 | 427.15 | 0.00 | ORB-long ORB[422.15,428.10] vol=1.9x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-03-11 11:05:00 | 427.45 | 427.24 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 420.45 | 424.37 | 0.00 | ORB-short ORB[422.90,426.70] vol=5.5x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 12:10:00 | 418.12 | 423.46 | 0.00 | T1 1.5R @ 418.12 |
| Stop hit — per-position SL triggered | 2026-03-17 13:20:00 | 420.45 | 421.72 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:25:00 | 418.50 | 422.31 | 0.00 | ORB-short ORB[420.15,425.90] vol=2.7x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 13:25:00 | 415.85 | 420.32 | 0.00 | T1 1.5R @ 415.85 |
| Target hit | 2026-03-19 15:20:00 | 411.20 | 417.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-03-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:05:00 | 432.85 | 427.43 | 0.00 | ORB-long ORB[421.05,427.00] vol=2.7x ATR=2.11 |
| Stop hit — per-position SL triggered | 2026-03-25 15:15:00 | 430.74 | 431.12 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:35:00 | 428.80 | 426.78 | 0.00 | ORB-long ORB[422.95,428.00] vol=2.2x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 09:45:00 | 431.47 | 427.80 | 0.00 | T1 1.5R @ 431.47 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 428.80 | 428.38 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:55:00 | 434.05 | 438.19 | 0.00 | ORB-short ORB[437.00,442.00] vol=2.0x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-04-17 11:25:00 | 435.61 | 437.55 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:15:00 | 454.40 | 449.95 | 0.00 | ORB-long ORB[445.05,451.70] vol=5.7x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-04-21 11:55:00 | 453.12 | 450.69 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 456.50 | 452.07 | 0.00 | ORB-long ORB[448.50,452.00] vol=1.5x ATR=1.71 |
| Stop hit — per-position SL triggered | 2026-04-22 09:45:00 | 454.79 | 452.36 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 464.10 | 466.74 | 0.00 | ORB-short ORB[465.00,471.00] vol=1.9x ATR=1.87 |
| Stop hit — per-position SL triggered | 2026-04-24 09:50:00 | 465.97 | 465.67 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:15:00 | 484.30 | 485.37 | 0.00 | ORB-short ORB[486.00,492.95] vol=3.3x ATR=1.44 |
| Stop hit — per-position SL triggered | 2026-05-06 11:25:00 | 485.74 | 485.32 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:55:00 | 484.25 | 487.17 | 0.00 | ORB-short ORB[484.95,490.55] vol=1.7x ATR=1.26 |
| Stop hit — per-position SL triggered | 2026-05-08 11:00:00 | 485.51 | 487.12 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 10:55:00 | 460.75 | 2026-02-11 11:30:00 | 462.12 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-18 09:45:00 | 460.45 | 2026-02-18 09:55:00 | 458.94 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-23 11:05:00 | 446.80 | 2026-02-23 11:35:00 | 444.90 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-23 11:05:00 | 446.80 | 2026-02-23 15:05:00 | 446.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:30:00 | 441.15 | 2026-02-24 09:35:00 | 443.03 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-11 10:45:00 | 428.90 | 2026-03-11 11:05:00 | 427.45 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-17 11:15:00 | 420.45 | 2026-03-17 12:10:00 | 418.12 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-03-17 11:15:00 | 420.45 | 2026-03-17 13:20:00 | 420.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 10:25:00 | 418.50 | 2026-03-19 13:25:00 | 415.85 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-03-19 10:25:00 | 418.50 | 2026-03-19 15:20:00 | 411.20 | TARGET_HIT | 0.50 | 1.74% |
| BUY | retest1 | 2026-03-25 11:05:00 | 432.85 | 2026-03-25 15:15:00 | 430.74 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-04-10 09:35:00 | 428.80 | 2026-04-10 09:45:00 | 431.47 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-10 09:35:00 | 428.80 | 2026-04-10 10:05:00 | 428.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-17 10:55:00 | 434.05 | 2026-04-17 11:25:00 | 435.61 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-21 11:15:00 | 454.40 | 2026-04-21 11:55:00 | 453.12 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-22 09:40:00 | 456.50 | 2026-04-22 09:45:00 | 454.79 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-24 09:35:00 | 464.10 | 2026-04-24 09:50:00 | 465.97 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-05-06 11:15:00 | 484.30 | 2026-05-06 11:25:00 | 485.74 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-05-08 10:55:00 | 484.25 | 2026-05-08 11:00:00 | 485.51 | STOP_HIT | 1.00 | -0.26% |
