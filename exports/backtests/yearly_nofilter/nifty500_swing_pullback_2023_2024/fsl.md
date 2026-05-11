# Firstsource Solutions Ltd. (FSL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 263.56
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 3
- **Avg / median % per leg:** 2.33% / 0.00%
- **Sum % (uncompounded):** 21.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 2.33% | 21.0% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 2.33% | 21.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 4 | 44.4% | 1 | 5 | 3 | 2.33% | 21.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 00:00:00 | 132.80 | 117.32 | 128.83 | Stage2 pullback-breakout RSI=60 vol=3.8x ATR=3.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 00:00:00 | 140.48 | 117.53 | 129.77 | T1 booked 50% @ 140.48 |
| Target hit | 2023-09-12 00:00:00 | 157.80 | 129.35 | 161.25 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 00:00:00 | 166.45 | 135.35 | 163.34 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=4.91 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 159.08 | 137.71 | 164.71 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2023-12-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 00:00:00 | 181.20 | 146.58 | 172.84 | Stage2 pullback-breakout RSI=67 vol=2.2x ATR=5.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 00:00:00 | 191.40 | 146.99 | 174.27 | T1 booked 50% @ 191.40 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 181.20 | 148.12 | 177.00 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2024-01-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 00:00:00 | 193.25 | 152.07 | 183.45 | Stage2 pullback-breakout RSI=70 vol=2.5x ATR=6.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 00:00:00 | 205.54 | 154.11 | 187.88 | T1 booked 50% @ 205.54 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 193.25 | 155.99 | 192.45 | SL hit (bars_held=9) |

### Cycle 5 — BUY (started 2024-02-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 00:00:00 | 216.65 | 162.00 | 200.89 | Stage2 pullback-breakout RSI=67 vol=1.8x ATR=8.79 |
| Stop hit — per-position SL triggered | 2024-02-12 00:00:00 | 203.47 | 162.80 | 201.16 | SL hit (bars_held=2) |

### Cycle 6 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 215.65 | 176.93 | 201.37 | Stage2 pullback-breakout RSI=68 vol=5.5x ATR=7.30 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 204.70 | 178.87 | 204.98 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-14 00:00:00 | 132.80 | 2023-07-17 00:00:00 | 140.48 | PARTIAL | 0.50 | 5.79% |
| BUY | retest1 | 2023-07-14 00:00:00 | 132.80 | 2023-09-12 00:00:00 | 157.80 | TARGET_HIT | 0.50 | 18.83% |
| BUY | retest1 | 2023-10-11 00:00:00 | 166.45 | 2023-10-23 00:00:00 | 159.08 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest1 | 2023-12-14 00:00:00 | 181.20 | 2023-12-15 00:00:00 | 191.40 | PARTIAL | 0.50 | 5.63% |
| BUY | retest1 | 2023-12-14 00:00:00 | 181.20 | 2023-12-20 00:00:00 | 181.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-05 00:00:00 | 193.25 | 2024-01-12 00:00:00 | 205.54 | PARTIAL | 0.50 | 6.36% |
| BUY | retest1 | 2024-01-05 00:00:00 | 193.25 | 2024-01-18 00:00:00 | 193.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-08 00:00:00 | 216.65 | 2024-02-12 00:00:00 | 203.47 | STOP_HIT | 1.00 | -6.08% |
| BUY | retest1 | 2024-04-25 00:00:00 | 215.65 | 2024-05-06 00:00:00 | 204.70 | STOP_HIT | 1.00 | -5.08% |
