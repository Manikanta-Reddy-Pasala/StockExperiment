# Federal Bank Ltd. (FEDERALBNK)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 294.60
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
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 0 / 6 / 2
- **Avg / median % per leg:** 0.64% / 1.48%
- **Sum % (uncompounded):** 5.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 0 | 6 | 2 | 0.64% | 5.1% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 0 | 6 | 2 | 0.64% | 5.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 0 | 6 | 2 | 0.64% | 5.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 00:00:00 | 141.70 | 130.02 | 134.58 | Stage2 pullback-breakout RSI=70 vol=2.1x ATR=2.83 |
| Stop hit — per-position SL triggered | 2023-09-06 00:00:00 | 143.80 | 131.28 | 140.20 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 154.05 | 138.18 | 148.24 | Stage2 pullback-breakout RSI=64 vol=1.9x ATR=3.13 |
| Stop hit — per-position SL triggered | 2023-12-18 00:00:00 | 156.55 | 139.75 | 152.29 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-02-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 00:00:00 | 156.60 | 143.03 | 148.40 | Stage2 pullback-breakout RSI=66 vol=2.0x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-16 00:00:00 | 164.77 | 143.25 | 149.95 | T1 booked 50% @ 164.77 |
| Stop hit — per-position SL triggered | 2024-02-20 00:00:00 | 156.60 | 143.55 | 151.50 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2024-03-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-06 00:00:00 | 157.25 | 144.64 | 152.92 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=4.11 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 151.09 | 144.97 | 152.83 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2024-04-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 00:00:00 | 159.00 | 146.06 | 152.97 | Stage2 pullback-breakout RSI=65 vol=2.8x ATR=3.88 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 153.18 | 146.24 | 153.42 | SL hit (bars_held=2) |

### Cycle 6 — BUY (started 2024-04-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 00:00:00 | 160.45 | 146.96 | 154.44 | Stage2 pullback-breakout RSI=66 vol=1.7x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 00:00:00 | 167.50 | 147.32 | 156.44 | T1 booked 50% @ 167.50 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 160.45 | 147.79 | 158.12 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-23 00:00:00 | 141.70 | 2023-09-06 00:00:00 | 143.80 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest1 | 2023-12-04 00:00:00 | 154.05 | 2023-12-18 00:00:00 | 156.55 | STOP_HIT | 1.00 | 1.62% |
| BUY | retest1 | 2024-02-15 00:00:00 | 156.60 | 2024-02-16 00:00:00 | 164.77 | PARTIAL | 0.50 | 5.22% |
| BUY | retest1 | 2024-02-15 00:00:00 | 156.60 | 2024-02-20 00:00:00 | 156.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-06 00:00:00 | 157.25 | 2024-03-13 00:00:00 | 151.09 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest1 | 2024-04-10 00:00:00 | 159.00 | 2024-04-15 00:00:00 | 153.18 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest1 | 2024-04-29 00:00:00 | 160.45 | 2024-05-02 00:00:00 | 167.50 | PARTIAL | 0.50 | 4.39% |
| BUY | retest1 | 2024-04-29 00:00:00 | 160.45 | 2024-05-07 00:00:00 | 160.45 | STOP_HIT | 0.50 | 0.00% |
