# IDFC First Bank Ltd. (IDFCFIRSTB)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 69.64
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -2.31% / -3.17%
- **Sum % (uncompounded):** -9.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 4 | 0 | -2.31% | -9.3% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -2.31% | -9.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -2.31% | -9.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 00:00:00 | 87.20 | 65.22 | 82.22 | Stage2 pullback-breakout RSI=67 vol=2.3x ATR=2.35 |
| Stop hit — per-position SL triggered | 2023-08-14 00:00:00 | 88.15 | 67.36 | 85.72 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-12-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 00:00:00 | 89.25 | 78.86 | 85.85 | Stage2 pullback-breakout RSI=62 vol=2.4x ATR=1.89 |
| Stop hit — per-position SL triggered | 2023-12-08 00:00:00 | 86.42 | 79.15 | 86.58 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2024-01-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 00:00:00 | 88.30 | 81.05 | 86.93 | Stage2 pullback-breakout RSI=55 vol=1.7x ATR=2.05 |
| Stop hit — per-position SL triggered | 2024-01-17 00:00:00 | 85.23 | 81.09 | 86.77 | SL hit (bars_held=1) |

### Cycle 4 — BUY (started 2024-02-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 00:00:00 | 83.95 | 81.40 | 82.42 | Stage2 pullback-breakout RSI=54 vol=2.0x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-02-28 00:00:00 | 80.85 | 81.41 | 82.32 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-31 00:00:00 | 87.20 | 2023-08-14 00:00:00 | 88.15 | STOP_HIT | 1.00 | 1.09% |
| BUY | retest1 | 2023-12-05 00:00:00 | 89.25 | 2023-12-08 00:00:00 | 86.42 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest1 | 2024-01-16 00:00:00 | 88.30 | 2024-01-17 00:00:00 | 85.23 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest1 | 2024-02-26 00:00:00 | 83.95 | 2024-02-28 00:00:00 | 80.85 | STOP_HIT | 1.00 | -3.69% |
