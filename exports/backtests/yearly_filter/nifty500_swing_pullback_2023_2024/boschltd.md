# Bosch Ltd. (BOSCHLTD)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 37320.00
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 0.52% / 3.88%
- **Sum % (uncompounded):** 2.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.52% | 2.1% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.52% | 2.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.52% | 2.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 00:00:00 | 19514.80 | 18152.20 | 18974.42 | Stage2 pullback-breakout RSI=65 vol=2.1x ATR=342.15 |
| Stop hit — per-position SL triggered | 2023-07-14 00:00:00 | 19001.58 | 18219.07 | 19107.03 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2023-11-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 00:00:00 | 20589.20 | 18819.35 | 19711.83 | Stage2 pullback-breakout RSI=69 vol=3.6x ATR=399.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 00:00:00 | 21387.43 | 18987.12 | 20459.71 | T1 booked 50% @ 21387.43 |
| Target hit | 2023-12-20 00:00:00 | 21494.20 | 19377.70 | 21495.06 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-05-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 00:00:00 | 30670.45 | 24229.42 | 29648.46 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=728.97 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 29576.99 | 24344.95 | 29723.58 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-06 00:00:00 | 19514.80 | 2023-07-14 00:00:00 | 19001.58 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest1 | 2023-11-16 00:00:00 | 20589.20 | 2023-11-29 00:00:00 | 21387.43 | PARTIAL | 0.50 | 3.88% |
| BUY | retest1 | 2023-11-16 00:00:00 | 20589.20 | 2023-12-20 00:00:00 | 21494.20 | TARGET_HIT | 0.50 | 4.40% |
| BUY | retest1 | 2024-05-02 00:00:00 | 30670.45 | 2024-05-06 00:00:00 | 29576.99 | STOP_HIT | 1.00 | -3.57% |
