# Firstsource Solutions Ltd. (FSL)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
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
- **Avg / median % per leg:** 3.17% / 2.19%
- **Sum % (uncompounded):** 15.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 0 | 3 | 2 | 3.17% | 15.8% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 3 | 2 | 3.17% | 15.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 3 | 2 | 3.17% | 15.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 00:00:00 | 307.70 | 210.41 | 275.44 | Stage2 pullback-breakout RSI=70 vol=3.3x ATR=16.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 00:00:00 | 340.40 | 211.67 | 281.34 | T1 booked 50% @ 340.40 |
| Stop hit — per-position SL triggered | 2024-08-27 00:00:00 | 307.70 | 217.91 | 297.94 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2024-09-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 00:00:00 | 333.50 | 235.25 | 312.22 | Stage2 pullback-breakout RSI=64 vol=4.3x ATR=14.82 |
| Stop hit — per-position SL triggered | 2024-09-30 00:00:00 | 311.27 | 238.78 | 316.20 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-10-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 00:00:00 | 350.90 | 250.98 | 323.60 | Stage2 pullback-breakout RSI=63 vol=5.4x ATR=16.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 00:00:00 | 384.88 | 260.52 | 342.89 | T1 booked 50% @ 384.88 |
| Stop hit — per-position SL triggered | 2024-11-11 00:00:00 | 358.60 | 263.73 | 349.52 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-16 00:00:00 | 307.70 | 2024-08-19 00:00:00 | 340.40 | PARTIAL | 0.50 | 10.63% |
| BUY | retest1 | 2024-08-16 00:00:00 | 307.70 | 2024-08-27 00:00:00 | 307.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-24 00:00:00 | 333.50 | 2024-09-30 00:00:00 | 311.27 | STOP_HIT | 1.00 | -6.67% |
| BUY | retest1 | 2024-10-23 00:00:00 | 350.90 | 2024-11-06 00:00:00 | 384.88 | PARTIAL | 0.50 | 9.68% |
| BUY | retest1 | 2024-10-23 00:00:00 | 350.90 | 2024-11-11 00:00:00 | 358.60 | STOP_HIT | 0.50 | 2.19% |
