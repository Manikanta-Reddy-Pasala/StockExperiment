# NCC Ltd. (NCC)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 170.04
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
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 2 / 4 / 2
- **Avg / median % per leg:** 6.76% / 6.26%
- **Sum % (uncompounded):** 54.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 6.76% | 54.1% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 2 | 4 | 2 | 6.76% | 54.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 2 | 4 | 2 | 6.76% | 54.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 05:30:00 | 129.20 | 100.57 | 121.84 | Stage2 pullback-breakout RSI=65 vol=4.8x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 05:30:00 | 137.29 | 102.34 | 126.18 | T1 booked 50% @ 137.29 |
| Target hit | 2023-09-07 05:30:00 | 157.55 | 118.33 | 157.66 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 05:30:00 | 159.45 | 122.15 | 153.63 | Stage2 pullback-breakout RSI=56 vol=2.0x ATR=7.59 |
| Stop hit — per-position SL triggered | 2023-10-11 05:30:00 | 156.10 | 125.46 | 155.66 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-10-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 05:30:00 | 165.15 | 125.85 | 156.56 | Stage2 pullback-breakout RSI=59 vol=2.1x ATR=6.70 |
| Stop hit — per-position SL triggered | 2023-10-20 05:30:00 | 155.10 | 127.83 | 157.84 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2023-11-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 05:30:00 | 165.85 | 131.08 | 153.38 | Stage2 pullback-breakout RSI=63 vol=2.8x ATR=7.27 |
| Stop hit — per-position SL triggered | 2023-11-29 05:30:00 | 165.75 | 134.33 | 160.82 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-01-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 05:30:00 | 175.75 | 142.07 | 168.28 | Stage2 pullback-breakout RSI=64 vol=2.0x ATR=5.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-10 05:30:00 | 187.26 | 143.55 | 172.28 | T1 booked 50% @ 187.26 |
| Target hit | 2024-03-12 05:30:00 | 236.10 | 172.58 | 242.27 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-04-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 05:30:00 | 261.75 | 180.05 | 238.44 | Stage2 pullback-breakout RSI=61 vol=2.3x ATR=15.48 |
| Stop hit — per-position SL triggered | 2024-04-18 05:30:00 | 244.00 | 187.48 | 249.95 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-06 05:30:00 | 129.20 | 2023-07-14 05:30:00 | 137.29 | PARTIAL | 0.50 | 6.26% |
| BUY | retest1 | 2023-07-06 05:30:00 | 129.20 | 2023-09-07 05:30:00 | 157.55 | TARGET_HIT | 0.50 | 21.94% |
| BUY | retest1 | 2023-09-26 05:30:00 | 159.45 | 2023-10-11 05:30:00 | 156.10 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest1 | 2023-10-12 05:30:00 | 165.15 | 2023-10-20 05:30:00 | 155.10 | STOP_HIT | 1.00 | -6.09% |
| BUY | retest1 | 2023-11-13 05:30:00 | 165.85 | 2023-11-29 05:30:00 | 165.75 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest1 | 2024-01-04 05:30:00 | 175.75 | 2024-01-10 05:30:00 | 187.26 | PARTIAL | 0.50 | 6.55% |
| BUY | retest1 | 2024-01-04 05:30:00 | 175.75 | 2024-03-12 05:30:00 | 236.10 | TARGET_HIT | 0.50 | 34.34% |
| BUY | retest1 | 2024-04-02 05:30:00 | 261.75 | 2024-04-18 05:30:00 | 244.00 | STOP_HIT | 1.00 | -6.78% |
