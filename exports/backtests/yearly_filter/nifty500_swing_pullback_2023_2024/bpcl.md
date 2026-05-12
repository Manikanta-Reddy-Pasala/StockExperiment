# Bharat Petroleum Corporation Ltd. (BPCL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 294.90
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 3.66% / 1.72%
- **Sum % (uncompounded):** 25.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 4 | 2 | 3.66% | 25.6% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 3.66% | 25.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 3.66% | 25.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 187.65 | 171.61 | 183.51 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=3.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 00:00:00 | 195.34 | 172.24 | 185.98 | T1 booked 50% @ 195.34 |
| Stop hit — per-position SL triggered | 2023-07-17 00:00:00 | 190.88 | 173.61 | 189.10 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-10-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 00:00:00 | 177.35 | 175.60 | 174.07 | Stage2 pullback-breakout RSI=56 vol=1.5x ATR=3.47 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 172.15 | 175.55 | 174.03 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-12-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 00:00:00 | 232.90 | 187.62 | 222.54 | Stage2 pullback-breakout RSI=68 vol=1.9x ATR=5.82 |
| Stop hit — per-position SL triggered | 2024-01-02 00:00:00 | 224.16 | 188.77 | 223.59 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2024-01-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 00:00:00 | 236.43 | 192.61 | 227.43 | Stage2 pullback-breakout RSI=65 vol=2.5x ATR=5.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-29 00:00:00 | 247.85 | 196.15 | 233.76 | T1 booked 50% @ 247.85 |
| Target hit | 2024-03-13 00:00:00 | 298.88 | 226.16 | 307.93 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-05-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 00:00:00 | 317.33 | 245.50 | 301.92 | Stage2 pullback-breakout RSI=62 vol=2.0x ATR=10.14 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 302.13 | 247.33 | 303.23 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 187.65 | 2023-07-06 00:00:00 | 195.34 | PARTIAL | 0.50 | 4.10% |
| BUY | retest1 | 2023-07-03 00:00:00 | 187.65 | 2023-07-17 00:00:00 | 190.88 | STOP_HIT | 0.50 | 1.72% |
| BUY | retest1 | 2023-10-17 00:00:00 | 177.35 | 2023-10-23 00:00:00 | 172.15 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest1 | 2023-12-28 00:00:00 | 232.90 | 2024-01-02 00:00:00 | 224.16 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest1 | 2024-01-16 00:00:00 | 236.43 | 2024-01-29 00:00:00 | 247.85 | PARTIAL | 0.50 | 4.83% |
| BUY | retest1 | 2024-01-16 00:00:00 | 236.43 | 2024-03-13 00:00:00 | 298.88 | TARGET_HIT | 0.50 | 26.41% |
| BUY | retest1 | 2024-05-02 00:00:00 | 317.33 | 2024-05-07 00:00:00 | 302.13 | STOP_HIT | 1.00 | -4.79% |
