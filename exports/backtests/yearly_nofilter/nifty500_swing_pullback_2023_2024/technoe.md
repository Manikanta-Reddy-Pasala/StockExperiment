# Techno Electric & Engineering Company Ltd. (TECHNOE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (911 bars)
- **Last close:** 1234.10
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 9.98% / 7.75%
- **Sum % (uncompounded):** 49.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 1 | 2 | 2 | 9.98% | 49.9% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 2 | 2 | 9.98% | 49.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 1 | 2 | 2 | 9.98% | 49.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 00:00:00 | 371.30 | 338.90 | 365.73 | Stage2 pullback-breakout RSI=53 vol=1.6x ATR=14.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 00:00:00 | 400.06 | 341.58 | 371.23 | T1 booked 50% @ 400.06 |
| Target hit | 2023-09-13 00:00:00 | 509.25 | 395.72 | 515.66 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-12-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 00:00:00 | 783.95 | 514.07 | 736.34 | Stage2 pullback-breakout RSI=65 vol=5.0x ATR=37.02 |
| Stop hit — per-position SL triggered | 2024-01-11 00:00:00 | 795.00 | 541.50 | 776.91 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-02-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 00:00:00 | 831.20 | 596.35 | 787.84 | Stage2 pullback-breakout RSI=60 vol=3.6x ATR=39.07 |
| Stop hit — per-position SL triggered | 2024-02-21 00:00:00 | 772.59 | 602.10 | 788.59 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2024-03-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 00:00:00 | 737.35 | 622.94 | 698.59 | Stage2 pullback-breakout RSI=55 vol=3.1x ATR=39.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 00:00:00 | 815.68 | 631.03 | 735.48 | T1 booked 50% @ 815.68 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-26 00:00:00 | 371.30 | 2023-07-07 00:00:00 | 400.06 | PARTIAL | 0.50 | 7.75% |
| BUY | retest1 | 2023-06-26 00:00:00 | 371.30 | 2023-09-13 00:00:00 | 509.25 | TARGET_HIT | 0.50 | 37.15% |
| BUY | retest1 | 2023-12-28 00:00:00 | 783.95 | 2024-01-11 00:00:00 | 795.00 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest1 | 2024-02-16 00:00:00 | 831.20 | 2024-02-21 00:00:00 | 772.59 | STOP_HIT | 1.00 | -7.05% |
| BUY | retest1 | 2024-03-26 00:00:00 | 737.35 | 2024-04-03 00:00:00 | 815.68 | PARTIAL | 0.50 | 10.62% |
