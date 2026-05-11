# 3M India Ltd. (3MINDIA)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 32045.00
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
- **Avg / median % per leg:** 1.07% / 5.65%
- **Sum % (uncompounded):** 4.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.07% | 4.3% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.07% | 4.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.07% | 4.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 05:30:00 | 28922.60 | 24658.21 | 27934.84 | Stage2 pullback-breakout RSI=64 vol=1.5x ATR=717.71 |
| Stop hit — per-position SL triggered | 2023-08-02 05:30:00 | 27846.04 | 24732.86 | 28022.98 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-08-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 05:30:00 | 29469.30 | 24953.89 | 28032.92 | Stage2 pullback-breakout RSI=62 vol=2.2x ATR=962.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 05:30:00 | 31393.75 | 25459.00 | 29372.93 | T1 booked 50% @ 31393.75 |
| Target hit | 2023-09-13 05:30:00 | 31135.30 | 26143.93 | 31156.94 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-03-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 05:30:00 | 31475.10 | 30232.51 | 30607.37 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=874.54 |
| Stop hit — per-position SL triggered | 2024-04-02 05:30:00 | 30163.29 | 30245.29 | 30615.26 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-31 05:30:00 | 28922.60 | 2023-08-02 05:30:00 | 27846.04 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest1 | 2023-08-11 05:30:00 | 29469.30 | 2023-08-29 05:30:00 | 31393.75 | PARTIAL | 0.50 | 6.53% |
| BUY | retest1 | 2023-08-11 05:30:00 | 29469.30 | 2023-09-13 05:30:00 | 31135.30 | TARGET_HIT | 0.50 | 5.65% |
| BUY | retest1 | 2024-03-27 05:30:00 | 31475.10 | 2024-04-02 05:30:00 | 30163.29 | STOP_HIT | 1.00 | -4.17% |
