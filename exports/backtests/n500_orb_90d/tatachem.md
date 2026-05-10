# Tata Chemicals Ltd. (TATACHEM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 782.00
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 13
- **Target hits / Stop hits / Partials:** 1 / 13 / 3
- **Avg / median % per leg:** 0.37% / -0.26%
- **Sum % (uncompounded):** 6.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 1 | 6 | 1 | 0.88% | 7.1% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 1 | 6 | 1 | 0.88% | 7.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.08% | -0.8% |
| SELL @ 2nd Alert (retest1) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.08% | -0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 4 | 23.5% | 1 | 13 | 3 | 0.37% | 6.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:55:00 | 709.75 | 713.19 | 0.00 | ORB-short ORB[713.25,721.30] vol=1.6x ATR=2.11 |
| Stop hit — per-position SL triggered | 2026-02-11 10:35:00 | 711.86 | 712.36 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:05:00 | 699.85 | 702.35 | 0.00 | ORB-short ORB[700.00,709.00] vol=1.6x ATR=1.90 |
| Stop hit — per-position SL triggered | 2026-02-12 10:10:00 | 701.75 | 702.31 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 692.70 | 691.68 | 0.00 | ORB-long ORB[688.00,692.50] vol=2.3x ATR=1.84 |
| Stop hit — per-position SL triggered | 2026-02-16 09:45:00 | 690.86 | 691.72 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:50:00 | 699.50 | 695.02 | 0.00 | ORB-long ORB[688.80,697.35] vol=2.6x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-02-19 10:55:00 | 697.71 | 695.19 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:15:00 | 705.00 | 706.74 | 0.00 | ORB-short ORB[711.50,716.70] vol=2.1x ATR=1.57 |
| Stop hit — per-position SL triggered | 2026-02-23 11:20:00 | 706.57 | 706.73 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 724.55 | 720.71 | 0.00 | ORB-long ORB[715.50,723.50] vol=1.5x ATR=2.39 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 722.16 | 722.04 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:30:00 | 710.70 | 711.84 | 0.00 | ORB-short ORB[711.35,718.00] vol=3.7x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:55:00 | 707.48 | 711.15 | 0.00 | T1 1.5R @ 707.48 |
| Stop hit — per-position SL triggered | 2026-02-27 11:00:00 | 710.70 | 709.94 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:30:00 | 691.00 | 696.98 | 0.00 | ORB-short ORB[695.00,704.90] vol=1.9x ATR=2.25 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 693.25 | 693.23 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 10:45:00 | 604.20 | 598.52 | 0.00 | ORB-long ORB[592.50,600.85] vol=2.1x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 11:30:00 | 608.37 | 602.07 | 0.00 | T1 1.5R @ 608.37 |
| Target hit | 2026-04-02 13:50:00 | 653.50 | 655.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-04-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:25:00 | 668.15 | 663.40 | 0.00 | ORB-long ORB[656.00,662.00] vol=1.7x ATR=2.53 |
| Stop hit — per-position SL triggered | 2026-04-10 11:25:00 | 665.62 | 664.34 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:25:00 | 695.95 | 709.69 | 0.00 | ORB-short ORB[720.15,728.70] vol=9.7x ATR=3.79 |
| Stop hit — per-position SL triggered | 2026-04-16 10:30:00 | 699.74 | 708.34 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:15:00 | 716.25 | 710.79 | 0.00 | ORB-long ORB[703.50,711.80] vol=4.3x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-04-21 10:20:00 | 714.46 | 711.51 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 718.95 | 714.43 | 0.00 | ORB-long ORB[705.00,713.90] vol=5.1x ATR=2.09 |
| Stop hit — per-position SL triggered | 2026-04-23 09:35:00 | 716.86 | 715.44 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 700.00 | 705.98 | 0.00 | ORB-short ORB[704.45,713.25] vol=2.0x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:50:00 | 696.82 | 704.09 | 0.00 | T1 1.5R @ 696.82 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 700.00 | 703.45 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:55:00 | 709.75 | 2026-02-11 10:35:00 | 711.86 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-12 10:05:00 | 699.85 | 2026-02-12 10:10:00 | 701.75 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-16 09:30:00 | 692.70 | 2026-02-16 09:45:00 | 690.86 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-19 10:50:00 | 699.50 | 2026-02-19 10:55:00 | 697.71 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-23 11:15:00 | 705.00 | 2026-02-23 11:20:00 | 706.57 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-26 09:35:00 | 724.55 | 2026-02-26 09:55:00 | 722.16 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-27 09:30:00 | 710.70 | 2026-02-27 09:55:00 | 707.48 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-27 09:30:00 | 710.70 | 2026-02-27 11:00:00 | 710.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-10 09:30:00 | 691.00 | 2026-03-10 10:15:00 | 693.25 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-02 10:45:00 | 604.20 | 2026-04-02 11:30:00 | 608.37 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-04-02 10:45:00 | 604.20 | 2026-04-02 13:50:00 | 653.50 | TARGET_HIT | 0.50 | 8.16% |
| BUY | retest1 | 2026-04-10 10:25:00 | 668.15 | 2026-04-10 11:25:00 | 665.62 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-16 10:25:00 | 695.95 | 2026-04-16 10:30:00 | 699.74 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2026-04-21 10:15:00 | 716.25 | 2026-04-21 10:20:00 | 714.46 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-23 09:30:00 | 718.95 | 2026-04-23 09:35:00 | 716.86 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-24 09:35:00 | 700.00 | 2026-04-24 09:50:00 | 696.82 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-04-24 09:35:00 | 700.00 | 2026-04-24 10:00:00 | 700.00 | STOP_HIT | 0.50 | 0.00% |
