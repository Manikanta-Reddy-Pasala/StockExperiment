# L&T Finance Ltd. (LTF)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (910 bars)
- **Last close:** 303.40
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
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / Stop hits / Partials:** 0 / 6 / 2
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 0.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 0 | 6 | 2 | 0.06% | 0.5% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 0 | 6 | 2 | 0.06% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 3 | 37.5% | 0 | 6 | 2 | 0.06% | 0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 00:00:00 | 129.55 | 105.82 | 125.17 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=3.46 |
| Stop hit — per-position SL triggered | 2023-09-18 00:00:00 | 127.40 | 108.01 | 127.42 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 00:00:00 | 133.10 | 109.39 | 126.86 | Stage2 pullback-breakout RSI=63 vol=2.7x ATR=3.54 |
| Stop hit — per-position SL triggered | 2023-10-04 00:00:00 | 127.79 | 109.83 | 127.76 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-10-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 00:00:00 | 137.55 | 110.32 | 129.02 | Stage2 pullback-breakout RSI=64 vol=2.8x ATR=4.17 |
| Stop hit — per-position SL triggered | 2023-10-09 00:00:00 | 131.30 | 110.53 | 129.21 | SL hit (bars_held=1) |

### Cycle 4 — BUY (started 2023-10-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 00:00:00 | 139.10 | 111.94 | 131.80 | Stage2 pullback-breakout RSI=63 vol=2.7x ATR=4.13 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 132.90 | 112.89 | 133.25 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2023-11-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 00:00:00 | 140.85 | 114.53 | 133.97 | Stage2 pullback-breakout RSI=61 vol=2.0x ATR=4.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 00:00:00 | 150.66 | 116.72 | 139.20 | T1 booked 50% @ 150.66 |
| Stop hit — per-position SL triggered | 2023-11-17 00:00:00 | 140.85 | 117.29 | 140.29 | SL hit (bars_held=10) |

### Cycle 6 — BUY (started 2023-12-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 00:00:00 | 162.15 | 125.50 | 153.67 | Stage2 pullback-breakout RSI=66 vol=1.7x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-04 00:00:00 | 171.10 | 127.46 | 158.55 | T1 booked 50% @ 171.10 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 166.45 | 131.32 | 164.52 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-04 00:00:00 | 129.55 | 2023-09-18 00:00:00 | 127.40 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest1 | 2023-09-29 00:00:00 | 133.10 | 2023-10-04 00:00:00 | 127.79 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest1 | 2023-10-06 00:00:00 | 137.55 | 2023-10-09 00:00:00 | 131.30 | STOP_HIT | 1.00 | -4.54% |
| BUY | retest1 | 2023-10-17 00:00:00 | 139.10 | 2023-10-23 00:00:00 | 132.90 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest1 | 2023-11-03 00:00:00 | 140.85 | 2023-11-15 00:00:00 | 150.66 | PARTIAL | 0.50 | 6.97% |
| BUY | retest1 | 2023-11-03 00:00:00 | 140.85 | 2023-11-17 00:00:00 | 140.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-28 00:00:00 | 162.15 | 2024-01-04 00:00:00 | 171.10 | PARTIAL | 0.50 | 5.52% |
| BUY | retest1 | 2023-12-28 00:00:00 | 162.15 | 2024-01-18 00:00:00 | 166.45 | STOP_HIT | 0.50 | 2.65% |
