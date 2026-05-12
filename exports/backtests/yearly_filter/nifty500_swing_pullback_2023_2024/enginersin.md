# Engineers India Ltd. (ENGINERSIN)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 253.55
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
| TARGET_HIT | 4 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 2
- **Target hits / Stop hits / Partials:** 4 / 2 / 4
- **Avg / median % per leg:** 9.07% / 8.91%
- **Sum % (uncompounded):** 90.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 8 | 80.0% | 4 | 2 | 4 | 9.07% | 90.7% |
| BUY @ 2nd Alert (retest1) | 10 | 8 | 80.0% | 4 | 2 | 4 | 9.07% | 90.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 8 | 80.0% | 4 | 2 | 4 | 9.07% | 90.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 00:00:00 | 119.70 | 88.02 | 113.76 | Stage2 pullback-breakout RSI=64 vol=3.0x ATR=4.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 00:00:00 | 128.41 | 90.40 | 118.77 | T1 booked 50% @ 128.41 |
| Target hit | 2023-09-12 00:00:00 | 146.95 | 110.84 | 155.53 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 00:00:00 | 137.70 | 119.36 | 131.79 | Stage2 pullback-breakout RSI=56 vol=6.1x ATR=5.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 00:00:00 | 149.23 | 119.89 | 134.34 | T1 booked 50% @ 149.23 |
| Target hit | 2023-12-20 00:00:00 | 152.00 | 127.41 | 156.08 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-12-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 00:00:00 | 172.50 | 129.14 | 158.98 | Stage2 pullback-breakout RSI=66 vol=3.2x ATR=7.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 00:00:00 | 187.87 | 130.75 | 165.55 | T1 booked 50% @ 187.87 |
| Target hit | 2024-02-09 00:00:00 | 226.40 | 153.32 | 229.17 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-03-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 00:00:00 | 233.15 | 163.81 | 219.92 | Stage2 pullback-breakout RSI=59 vol=3.3x ATR=11.69 |
| Stop hit — per-position SL triggered | 2024-03-11 00:00:00 | 215.61 | 165.54 | 220.47 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-03-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 00:00:00 | 206.25 | 168.06 | 202.66 | Stage2 pullback-breakout RSI=50 vol=1.9x ATR=13.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 00:00:00 | 232.65 | 171.68 | 210.43 | T1 booked 50% @ 232.65 |
| Target hit | 2024-04-15 00:00:00 | 212.35 | 173.53 | 213.10 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-04-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 00:00:00 | 236.85 | 176.66 | 214.69 | Stage2 pullback-breakout RSI=66 vol=3.5x ATR=10.33 |
| Stop hit — per-position SL triggered | 2024-05-09 00:00:00 | 221.36 | 181.01 | 223.83 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-06 00:00:00 | 119.70 | 2023-07-17 00:00:00 | 128.41 | PARTIAL | 0.50 | 7.28% |
| BUY | retest1 | 2023-07-06 00:00:00 | 119.70 | 2023-09-12 00:00:00 | 146.95 | TARGET_HIT | 0.50 | 22.77% |
| BUY | retest1 | 2023-11-10 00:00:00 | 137.70 | 2023-11-13 00:00:00 | 149.23 | PARTIAL | 0.50 | 8.37% |
| BUY | retest1 | 2023-11-10 00:00:00 | 137.70 | 2023-12-20 00:00:00 | 152.00 | TARGET_HIT | 0.50 | 10.38% |
| BUY | retest1 | 2023-12-28 00:00:00 | 172.50 | 2024-01-02 00:00:00 | 187.87 | PARTIAL | 0.50 | 8.91% |
| BUY | retest1 | 2023-12-28 00:00:00 | 172.50 | 2024-02-09 00:00:00 | 226.40 | TARGET_HIT | 0.50 | 31.25% |
| BUY | retest1 | 2024-03-05 00:00:00 | 233.15 | 2024-03-11 00:00:00 | 215.61 | STOP_HIT | 1.00 | -7.52% |
| BUY | retest1 | 2024-03-26 00:00:00 | 206.25 | 2024-04-08 00:00:00 | 232.65 | PARTIAL | 0.50 | 12.80% |
| BUY | retest1 | 2024-03-26 00:00:00 | 206.25 | 2024-04-15 00:00:00 | 212.35 | TARGET_HIT | 0.50 | 2.96% |
| BUY | retest1 | 2024-04-26 00:00:00 | 236.85 | 2024-05-09 00:00:00 | 221.36 | STOP_HIT | 1.00 | -6.54% |
