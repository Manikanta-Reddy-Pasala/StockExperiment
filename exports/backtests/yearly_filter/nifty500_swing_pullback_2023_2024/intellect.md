# Intellect Design Arena Ltd. (INTELLECT)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 773.90
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
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 2 / 4 / 2
- **Avg / median % per leg:** 4.11% / 6.25%
- **Sum % (uncompounded):** 32.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 4 | 2 | 4.11% | 32.9% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 4 | 2 | 4.11% | 32.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 2 | 4 | 2 | 4.11% | 32.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 00:00:00 | 640.85 | 519.83 | 606.14 | Stage2 pullback-breakout RSI=65 vol=2.7x ATR=20.04 |
| Stop hit — per-position SL triggered | 2023-07-26 00:00:00 | 610.79 | 528.21 | 617.39 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2023-07-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 00:00:00 | 690.90 | 530.32 | 620.93 | Stage2 pullback-breakout RSI=67 vol=8.9x ATR=27.33 |
| Stop hit — per-position SL triggered | 2023-08-11 00:00:00 | 694.10 | 544.62 | 660.07 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 735.00 | 563.53 | 690.33 | Stage2 pullback-breakout RSI=67 vol=2.0x ATR=25.12 |
| Stop hit — per-position SL triggered | 2023-09-08 00:00:00 | 697.33 | 573.24 | 708.19 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 734.45 | 621.24 | 691.86 | Stage2 pullback-breakout RSI=67 vol=6.0x ATR=22.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 00:00:00 | 780.35 | 622.47 | 696.93 | T1 booked 50% @ 780.35 |
| Target hit | 2024-02-15 00:00:00 | 905.30 | 718.65 | 920.54 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-02-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 00:00:00 | 952.80 | 730.08 | 918.29 | Stage2 pullback-breakout RSI=57 vol=2.8x ATR=39.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-27 00:00:00 | 1032.18 | 735.84 | 937.26 | T1 booked 50% @ 1032.18 |
| Target hit | 2024-03-19 00:00:00 | 1052.65 | 786.90 | 1066.48 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-04-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 00:00:00 | 1093.20 | 845.61 | 1043.80 | Stage2 pullback-breakout RSI=58 vol=3.0x ATR=44.29 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 1026.76 | 854.01 | 1048.66 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-14 00:00:00 | 640.85 | 2023-07-26 00:00:00 | 610.79 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest1 | 2023-07-28 00:00:00 | 690.90 | 2023-08-11 00:00:00 | 694.10 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest1 | 2023-08-31 00:00:00 | 735.00 | 2023-09-08 00:00:00 | 697.33 | STOP_HIT | 1.00 | -5.13% |
| BUY | retest1 | 2023-11-30 00:00:00 | 734.45 | 2023-12-01 00:00:00 | 780.35 | PARTIAL | 0.50 | 6.25% |
| BUY | retest1 | 2023-11-30 00:00:00 | 734.45 | 2024-02-15 00:00:00 | 905.30 | TARGET_HIT | 0.50 | 23.26% |
| BUY | retest1 | 2024-02-23 00:00:00 | 952.80 | 2024-02-27 00:00:00 | 1032.18 | PARTIAL | 0.50 | 8.33% |
| BUY | retest1 | 2024-02-23 00:00:00 | 952.80 | 2024-03-19 00:00:00 | 1052.65 | TARGET_HIT | 0.50 | 10.48% |
| BUY | retest1 | 2024-04-29 00:00:00 | 1093.20 | 2024-05-06 00:00:00 | 1026.76 | STOP_HIT | 1.00 | -6.08% |
