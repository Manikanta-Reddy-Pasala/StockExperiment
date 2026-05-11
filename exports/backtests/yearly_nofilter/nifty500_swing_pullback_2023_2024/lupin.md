# Lupin Ltd. (LUPIN)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 2245.60
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 2.32% / 2.31%
- **Sum % (uncompounded):** 11.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 0 | 3 | 2 | 2.32% | 11.6% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 3 | 2 | 2.32% | 11.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 3 | 2 | 2.32% | 11.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 00:00:00 | 1171.25 | 884.57 | 1124.50 | Stage2 pullback-breakout RSI=66 vol=1.9x ATR=25.80 |
| Stop hit — per-position SL triggered | 2023-10-16 00:00:00 | 1198.30 | 911.32 | 1152.51 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 1169.20 | 939.40 | 1153.73 | Stage2 pullback-breakout RSI=56 vol=2.7x ATR=26.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 00:00:00 | 1223.16 | 949.49 | 1168.16 | T1 booked 50% @ 1223.16 |
| Stop hit — per-position SL triggered | 2023-11-13 00:00:00 | 1169.20 | 958.69 | 1173.08 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2023-11-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 00:00:00 | 1243.20 | 977.66 | 1192.81 | Stage2 pullback-breakout RSI=66 vol=1.8x ATR=29.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 00:00:00 | 1301.61 | 989.06 | 1218.39 | T1 booked 50% @ 1301.61 |
| Stop hit — per-position SL triggered | 2023-12-06 00:00:00 | 1243.20 | 996.89 | 1227.50 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-29 00:00:00 | 1171.25 | 2023-10-16 00:00:00 | 1198.30 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest1 | 2023-11-02 00:00:00 | 1169.20 | 2023-11-08 00:00:00 | 1223.16 | PARTIAL | 0.50 | 4.62% |
| BUY | retest1 | 2023-11-02 00:00:00 | 1169.20 | 2023-11-13 00:00:00 | 1169.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-24 00:00:00 | 1243.20 | 2023-12-01 00:00:00 | 1301.61 | PARTIAL | 0.50 | 4.70% |
| BUY | retest1 | 2023-11-24 00:00:00 | 1243.20 | 2023-12-06 00:00:00 | 1243.20 | STOP_HIT | 0.50 | 0.00% |
