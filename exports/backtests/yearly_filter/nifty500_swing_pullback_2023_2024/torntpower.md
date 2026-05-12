# Torrent Power Ltd. (TORNTPOWER)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1699.30
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 3
- **Target hits / Stop hits / Partials:** 3 / 3 / 4
- **Avg / median % per leg:** 6.72% / 6.49%
- **Sum % (uncompounded):** 67.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 7 | 70.0% | 3 | 3 | 4 | 6.72% | 67.2% |
| BUY @ 2nd Alert (retest1) | 10 | 7 | 70.0% | 3 | 3 | 4 | 6.72% | 67.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 7 | 70.0% | 3 | 3 | 4 | 6.72% | 67.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 00:00:00 | 640.35 | 550.68 | 624.75 | Stage2 pullback-breakout RSI=57 vol=3.9x ATR=24.10 |
| Stop hit — per-position SL triggered | 2023-07-20 00:00:00 | 614.55 | 557.20 | 620.74 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 00:00:00 | 638.95 | 559.55 | 619.83 | Stage2 pullback-breakout RSI=58 vol=1.8x ATR=20.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 00:00:00 | 680.45 | 562.53 | 630.68 | T1 booked 50% @ 680.45 |
| Stop hit — per-position SL triggered | 2023-08-11 00:00:00 | 638.95 | 571.17 | 648.92 | SL hit (bars_held=12) |

### Cycle 3 — BUY (started 2023-08-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 00:00:00 | 661.95 | 575.10 | 644.75 | Stage2 pullback-breakout RSI=58 vol=2.3x ATR=20.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 00:00:00 | 702.76 | 583.10 | 659.66 | T1 booked 50% @ 702.76 |
| Target hit | 2023-10-19 00:00:00 | 726.20 | 621.46 | 727.65 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2023-11-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 00:00:00 | 745.50 | 631.35 | 726.96 | Stage2 pullback-breakout RSI=60 vol=2.0x ATR=21.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 00:00:00 | 787.70 | 636.15 | 735.94 | T1 booked 50% @ 787.70 |
| Target hit | 2023-12-15 00:00:00 | 897.50 | 691.48 | 898.82 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-01-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 00:00:00 | 1014.60 | 720.41 | 926.16 | Stage2 pullback-breakout RSI=70 vol=8.4x ATR=40.52 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 997.10 | 748.41 | 980.51 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-03-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-19 00:00:00 | 1264.00 | 877.29 | 1156.30 | Stage2 pullback-breakout RSI=67 vol=1.6x ATR=66.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-26 00:00:00 | 1397.95 | 893.43 | 1202.33 | T1 booked 50% @ 1397.95 |
| Target hit | 2024-05-06 00:00:00 | 1440.60 | 1027.24 | 1474.99 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-06 00:00:00 | 640.35 | 2023-07-20 00:00:00 | 614.55 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest1 | 2023-07-26 00:00:00 | 638.95 | 2023-07-31 00:00:00 | 680.45 | PARTIAL | 0.50 | 6.49% |
| BUY | retest1 | 2023-07-26 00:00:00 | 638.95 | 2023-08-11 00:00:00 | 638.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-22 00:00:00 | 661.95 | 2023-09-04 00:00:00 | 702.76 | PARTIAL | 0.50 | 6.17% |
| BUY | retest1 | 2023-08-22 00:00:00 | 661.95 | 2023-10-19 00:00:00 | 726.20 | TARGET_HIT | 0.50 | 9.71% |
| BUY | retest1 | 2023-11-03 00:00:00 | 745.50 | 2023-11-09 00:00:00 | 787.70 | PARTIAL | 0.50 | 5.66% |
| BUY | retest1 | 2023-11-03 00:00:00 | 745.50 | 2023-12-15 00:00:00 | 897.50 | TARGET_HIT | 0.50 | 20.39% |
| BUY | retest1 | 2024-01-04 00:00:00 | 1014.60 | 2024-01-18 00:00:00 | 997.10 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2024-03-19 00:00:00 | 1264.00 | 2024-03-26 00:00:00 | 1397.95 | PARTIAL | 0.50 | 10.60% |
| BUY | retest1 | 2024-03-19 00:00:00 | 1264.00 | 2024-05-06 00:00:00 | 1440.60 | TARGET_HIT | 0.50 | 13.97% |
