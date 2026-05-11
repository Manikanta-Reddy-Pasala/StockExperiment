# Abbott India Ltd. (ABBOTINDIA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 26610.00
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 1
- **Avg / median % per leg:** 2.29% / -1.36%
- **Sum % (uncompounded):** 13.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | 2.29% | 13.7% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 2.29% | 13.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 1 | 4 | 1 | 2.29% | 13.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 00:00:00 | 24059.90 | 21402.38 | 23300.37 | Stage2 pullback-breakout RSI=65 vol=1.7x ATR=416.35 |
| Stop hit — per-position SL triggered | 2023-08-11 00:00:00 | 23732.80 | 21645.66 | 23695.41 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 00:00:00 | 23351.20 | 21971.73 | 23130.77 | Stage2 pullback-breakout RSI=54 vol=1.7x ATR=444.00 |
| Stop hit — per-position SL triggered | 2023-09-22 00:00:00 | 22685.20 | 22010.75 | 23066.87 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-11-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 00:00:00 | 23786.40 | 22184.99 | 22751.37 | Stage2 pullback-breakout RSI=67 vol=2.2x ATR=474.20 |
| Stop hit — per-position SL triggered | 2023-11-15 00:00:00 | 23075.10 | 22268.38 | 23112.33 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2024-01-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 00:00:00 | 23327.20 | 22495.51 | 22861.47 | Stage2 pullback-breakout RSI=58 vol=3.0x ATR=405.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 00:00:00 | 24138.17 | 22558.78 | 23237.91 | T1 booked 50% @ 24138.17 |
| Target hit | 2024-03-01 00:00:00 | 28216.65 | 24086.21 | 28271.39 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-03-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-14 00:00:00 | 28308.10 | 24383.28 | 27836.30 | Stage2 pullback-breakout RSI=57 vol=2.4x ATR=661.60 |
| Stop hit — per-position SL triggered | 2024-03-22 00:00:00 | 27315.70 | 24590.78 | 27851.40 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-28 00:00:00 | 24059.90 | 2023-08-11 00:00:00 | 23732.80 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest1 | 2023-09-15 00:00:00 | 23351.20 | 2023-09-22 00:00:00 | 22685.20 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest1 | 2023-11-07 00:00:00 | 23786.40 | 2023-11-15 00:00:00 | 23075.10 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest1 | 2024-01-02 00:00:00 | 23327.20 | 2024-01-09 00:00:00 | 24138.17 | PARTIAL | 0.50 | 3.48% |
| BUY | retest1 | 2024-01-02 00:00:00 | 23327.20 | 2024-03-01 00:00:00 | 28216.65 | TARGET_HIT | 0.50 | 20.96% |
| BUY | retest1 | 2024-03-14 00:00:00 | 28308.10 | 2024-03-22 00:00:00 | 27315.70 | STOP_HIT | 1.00 | -3.51% |
