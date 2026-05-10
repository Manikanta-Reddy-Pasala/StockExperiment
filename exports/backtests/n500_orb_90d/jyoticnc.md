# Jyoti CNC Automation Ltd. (JYOTICNC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 766.00
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 11
- **Target hits / Stop hits / Partials:** 1 / 11 / 3
- **Avg / median % per leg:** -0.03% / -0.30%
- **Sum % (uncompounded):** -0.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.10% | 1.1% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.10% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.38% | -1.5% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.38% | -1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 4 | 26.7% | 1 | 11 | 3 | -0.03% | -0.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 870.60 | 866.35 | 0.00 | ORB-long ORB[857.50,870.00] vol=2.9x ATR=2.97 |
| Stop hit — per-position SL triggered | 2026-02-10 11:05:00 | 867.63 | 866.50 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:10:00 | 826.85 | 835.21 | 0.00 | ORB-short ORB[833.55,842.00] vol=1.9x ATR=3.96 |
| Stop hit — per-position SL triggered | 2026-02-13 10:40:00 | 830.81 | 833.89 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:40:00 | 834.25 | 827.90 | 0.00 | ORB-long ORB[820.05,827.55] vol=2.6x ATR=3.68 |
| Stop hit — per-position SL triggered | 2026-02-16 09:50:00 | 830.57 | 828.18 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 839.90 | 834.31 | 0.00 | ORB-long ORB[825.05,837.40] vol=1.7x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:50:00 | 845.75 | 838.31 | 0.00 | T1 1.5R @ 845.75 |
| Stop hit — per-position SL triggered | 2026-02-17 10:10:00 | 839.90 | 840.32 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 855.20 | 849.07 | 0.00 | ORB-long ORB[844.00,852.00] vol=2.4x ATR=2.61 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 852.59 | 849.28 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 11:10:00 | 831.00 | 833.22 | 0.00 | ORB-short ORB[837.05,842.95] vol=2.8x ATR=2.50 |
| Stop hit — per-position SL triggered | 2026-02-20 11:30:00 | 833.50 | 833.13 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 856.85 | 851.73 | 0.00 | ORB-long ORB[846.30,854.05] vol=3.2x ATR=2.61 |
| Stop hit — per-position SL triggered | 2026-02-25 09:35:00 | 854.24 | 852.00 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:45:00 | 769.15 | 765.35 | 0.00 | ORB-long ORB[760.00,768.00] vol=1.7x ATR=3.33 |
| Stop hit — per-position SL triggered | 2026-04-07 11:05:00 | 765.82 | 765.77 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 752.55 | 748.69 | 0.00 | ORB-long ORB[740.55,751.00] vol=2.6x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:35:00 | 756.94 | 749.60 | 0.00 | T1 1.5R @ 756.94 |
| Stop hit — per-position SL triggered | 2026-04-27 09:40:00 | 752.55 | 749.77 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:50:00 | 765.00 | 757.86 | 0.00 | ORB-long ORB[754.05,761.75] vol=2.4x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:55:00 | 771.14 | 768.00 | 0.00 | T1 1.5R @ 771.14 |
| Target hit | 2026-04-28 10:35:00 | 771.55 | 773.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 749.25 | 755.57 | 0.00 | ORB-short ORB[753.05,762.00] vol=1.7x ATR=2.62 |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 751.87 | 755.22 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 769.85 | 772.81 | 0.00 | ORB-short ORB[770.20,778.00] vol=1.7x ATR=3.10 |
| Stop hit — per-position SL triggered | 2026-05-08 09:40:00 | 772.95 | 772.60 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 11:00:00 | 870.60 | 2026-02-10 11:05:00 | 867.63 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-13 10:10:00 | 826.85 | 2026-02-13 10:40:00 | 830.81 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-02-16 09:40:00 | 834.25 | 2026-02-16 09:50:00 | 830.57 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-02-17 09:40:00 | 839.90 | 2026-02-17 09:50:00 | 845.75 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-02-17 09:40:00 | 839.90 | 2026-02-17 10:10:00 | 839.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 11:00:00 | 855.20 | 2026-02-18 11:15:00 | 852.59 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-20 11:10:00 | 831.00 | 2026-02-20 11:30:00 | 833.50 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-25 09:30:00 | 856.85 | 2026-02-25 09:35:00 | 854.24 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-07 10:45:00 | 769.15 | 2026-04-07 11:05:00 | 765.82 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-27 09:30:00 | 752.55 | 2026-04-27 09:35:00 | 756.94 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-27 09:30:00 | 752.55 | 2026-04-27 09:40:00 | 752.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 09:50:00 | 765.00 | 2026-04-28 09:55:00 | 771.14 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2026-04-28 09:50:00 | 765.00 | 2026-04-28 10:35:00 | 771.55 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2026-05-05 11:00:00 | 749.25 | 2026-05-05 11:15:00 | 751.87 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-05-08 09:35:00 | 769.85 | 2026-05-08 09:40:00 | 772.95 | STOP_HIT | 1.00 | -0.40% |
