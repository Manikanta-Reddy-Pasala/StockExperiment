# Aegis Logistics Ltd. (AEGISLOG)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 725.00
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
| PARTIAL | 9 |
| TARGET_HIT | 4 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 10
- **Target hits / Stop hits / Partials:** 4 / 10 / 9
- **Avg / median % per leg:** 0.40% / 0.33%
- **Sum % (uncompounded):** 9.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.14% | 1.3% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.14% | 1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 8 | 57.1% | 2 | 6 | 6 | 0.56% | 7.9% |
| SELL @ 2nd Alert (retest1) | 14 | 8 | 57.1% | 2 | 6 | 6 | 0.56% | 7.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 13 | 56.5% | 4 | 10 | 9 | 0.40% | 9.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:40:00 | 707.65 | 713.10 | 0.00 | ORB-short ORB[713.10,722.40] vol=2.7x ATR=2.13 |
| Stop hit — per-position SL triggered | 2026-02-11 11:35:00 | 709.78 | 712.27 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:50:00 | 703.50 | 701.60 | 0.00 | ORB-long ORB[695.65,700.80] vol=2.5x ATR=2.95 |
| Stop hit — per-position SL triggered | 2026-02-16 10:05:00 | 700.55 | 701.70 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 694.00 | 697.10 | 0.00 | ORB-short ORB[695.35,705.00] vol=1.6x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:40:00 | 691.03 | 695.82 | 0.00 | T1 1.5R @ 691.03 |
| Stop hit — per-position SL triggered | 2026-02-17 13:40:00 | 694.00 | 692.82 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:05:00 | 691.75 | 693.82 | 0.00 | ORB-short ORB[692.55,700.00] vol=8.4x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:10:00 | 689.50 | 692.51 | 0.00 | T1 1.5R @ 689.50 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 691.75 | 692.29 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:45:00 | 689.05 | 693.58 | 0.00 | ORB-short ORB[691.00,697.95] vol=4.4x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-02-19 10:50:00 | 690.68 | 692.72 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:40:00 | 686.00 | 682.42 | 0.00 | ORB-long ORB[679.50,684.75] vol=2.1x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:00:00 | 689.97 | 684.97 | 0.00 | T1 1.5R @ 689.97 |
| Target hit | 2026-02-20 13:05:00 | 690.70 | 691.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2026-03-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:55:00 | 659.80 | 663.00 | 0.00 | ORB-short ORB[661.70,670.50] vol=2.1x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 13:20:00 | 654.94 | 660.31 | 0.00 | T1 1.5R @ 654.94 |
| Target hit | 2026-03-04 15:20:00 | 647.95 | 654.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:15:00 | 639.70 | 644.50 | 0.00 | ORB-short ORB[642.65,650.20] vol=3.9x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:20:00 | 637.08 | 643.80 | 0.00 | T1 1.5R @ 637.08 |
| Target hit | 2026-03-11 15:20:00 | 616.10 | 624.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:35:00 | 611.40 | 606.29 | 0.00 | ORB-long ORB[601.15,607.80] vol=2.0x ATR=3.18 |
| Stop hit — per-position SL triggered | 2026-03-17 09:55:00 | 608.22 | 606.97 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 611.45 | 608.96 | 0.00 | ORB-long ORB[603.60,611.00] vol=2.2x ATR=2.11 |
| Stop hit — per-position SL triggered | 2026-03-27 11:20:00 | 609.34 | 609.04 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:00:00 | 714.65 | 720.76 | 0.00 | ORB-short ORB[719.10,727.90] vol=1.5x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:20:00 | 710.96 | 718.49 | 0.00 | T1 1.5R @ 710.96 |
| Stop hit — per-position SL triggered | 2026-04-22 10:45:00 | 714.65 | 717.34 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 722.10 | 715.88 | 0.00 | ORB-long ORB[709.20,715.00] vol=3.2x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:35:00 | 726.17 | 719.96 | 0.00 | T1 1.5R @ 726.17 |
| Stop hit — per-position SL triggered | 2026-04-23 09:40:00 | 722.10 | 720.31 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:55:00 | 710.40 | 713.55 | 0.00 | ORB-short ORB[711.00,721.00] vol=2.0x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:45:00 | 706.81 | 711.11 | 0.00 | T1 1.5R @ 706.81 |
| Stop hit — per-position SL triggered | 2026-05-05 12:55:00 | 710.40 | 710.87 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:50:00 | 718.00 | 716.91 | 0.00 | ORB-long ORB[713.30,717.80] vol=3.0x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:05:00 | 721.26 | 717.85 | 0.00 | T1 1.5R @ 721.26 |
| Target hit | 2026-05-06 11:00:00 | 720.00 | 720.20 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 10:40:00 | 707.65 | 2026-02-11 11:35:00 | 709.78 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-16 09:50:00 | 703.50 | 2026-02-16 10:05:00 | 700.55 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-02-17 09:55:00 | 694.00 | 2026-02-17 10:40:00 | 691.03 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-17 09:55:00 | 694.00 | 2026-02-17 13:40:00 | 694.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 11:05:00 | 691.75 | 2026-02-18 11:10:00 | 689.50 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-18 11:05:00 | 691.75 | 2026-02-18 12:15:00 | 691.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 10:45:00 | 689.05 | 2026-02-19 10:50:00 | 690.68 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-20 09:40:00 | 686.00 | 2026-02-20 10:00:00 | 689.97 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-20 09:40:00 | 686.00 | 2026-02-20 13:05:00 | 690.70 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2026-03-04 09:55:00 | 659.80 | 2026-03-04 13:20:00 | 654.94 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2026-03-04 09:55:00 | 659.80 | 2026-03-04 15:20:00 | 647.95 | TARGET_HIT | 0.50 | 1.80% |
| SELL | retest1 | 2026-03-11 11:15:00 | 639.70 | 2026-03-11 11:20:00 | 637.08 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-11 11:15:00 | 639.70 | 2026-03-11 15:20:00 | 616.10 | TARGET_HIT | 0.50 | 3.69% |
| BUY | retest1 | 2026-03-17 09:35:00 | 611.40 | 2026-03-17 09:55:00 | 608.22 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-03-27 11:05:00 | 611.45 | 2026-03-27 11:20:00 | 609.34 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-22 10:00:00 | 714.65 | 2026-04-22 10:20:00 | 710.96 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-04-22 10:00:00 | 714.65 | 2026-04-22 10:45:00 | 714.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 09:30:00 | 722.10 | 2026-04-23 09:35:00 | 726.17 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-23 09:30:00 | 722.10 | 2026-04-23 09:40:00 | 722.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 09:55:00 | 710.40 | 2026-05-05 11:45:00 | 706.81 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-05-05 09:55:00 | 710.40 | 2026-05-05 12:55:00 | 710.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 09:50:00 | 718.00 | 2026-05-06 10:05:00 | 721.26 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-05-06 09:50:00 | 718.00 | 2026-05-06 11:00:00 | 720.00 | TARGET_HIT | 0.50 | 0.28% |
