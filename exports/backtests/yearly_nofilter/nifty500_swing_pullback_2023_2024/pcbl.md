# PCBL Chemical Ltd. (PCBL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 295.55
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
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 2
- **Avg / median % per leg:** 4.00% / 6.90%
- **Sum % (uncompounded):** 24.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 4.00% | 24.0% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 4.00% | 24.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 4.00% | 24.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 00:00:00 | 157.80 | 139.34 | 156.19 | Stage2 pullback-breakout RSI=53 vol=2.6x ATR=4.67 |
| Stop hit — per-position SL triggered | 2023-08-28 00:00:00 | 156.85 | 140.88 | 156.00 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-10-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 00:00:00 | 180.05 | 146.85 | 167.49 | Stage2 pullback-breakout RSI=67 vol=4.8x ATR=6.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 00:00:00 | 192.66 | 148.01 | 172.47 | T1 booked 50% @ 192.66 |
| Target hit | 2023-10-23 00:00:00 | 186.75 | 153.48 | 190.98 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 00:00:00 | 262.70 | 188.90 | 254.56 | Stage2 pullback-breakout RSI=60 vol=2.0x ATR=9.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 00:00:00 | 280.82 | 193.42 | 260.23 | T1 booked 50% @ 280.82 |
| Target hit | 2024-02-12 00:00:00 | 295.65 | 214.43 | 308.61 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-04-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 00:00:00 | 277.00 | 238.50 | 268.95 | Stage2 pullback-breakout RSI=56 vol=2.8x ATR=10.24 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 261.64 | 239.61 | 268.14 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-11 00:00:00 | 157.80 | 2023-08-28 00:00:00 | 156.85 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2023-10-03 00:00:00 | 180.05 | 2023-10-06 00:00:00 | 192.66 | PARTIAL | 0.50 | 7.00% |
| BUY | retest1 | 2023-10-03 00:00:00 | 180.05 | 2023-10-23 00:00:00 | 186.75 | TARGET_HIT | 0.50 | 3.72% |
| BUY | retest1 | 2024-01-05 00:00:00 | 262.70 | 2024-01-15 00:00:00 | 280.82 | PARTIAL | 0.50 | 6.90% |
| BUY | retest1 | 2024-01-05 00:00:00 | 262.70 | 2024-02-12 00:00:00 | 295.65 | TARGET_HIT | 0.50 | 12.54% |
| BUY | retest1 | 2024-04-29 00:00:00 | 277.00 | 2024-05-06 00:00:00 | 261.64 | STOP_HIT | 1.00 | -5.54% |
