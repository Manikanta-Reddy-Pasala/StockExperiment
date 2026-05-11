# Manappuram Finance Ltd. (MANAPPURAM)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 306.10
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / Stop hits / Partials:** 1 / 6 / 2
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 1.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 1 | 6 | 2 | 0.21% | 1.9% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 6 | 2 | 0.21% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 4 | 44.4% | 1 | 6 | 2 | 0.21% | 1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 00:00:00 | 135.10 | 116.86 | 127.97 | Stage2 pullback-breakout RSI=63 vol=1.9x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 00:00:00 | 143.57 | 118.60 | 133.51 | T1 booked 50% @ 143.57 |
| Target hit | 2023-08-24 00:00:00 | 142.25 | 121.56 | 142.42 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 154.60 | 122.79 | 144.36 | Stage2 pullback-breakout RSI=67 vol=2.0x ATR=5.07 |
| Stop hit — per-position SL triggered | 2023-09-06 00:00:00 | 147.00 | 123.88 | 146.34 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-09-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 00:00:00 | 147.10 | 126.26 | 143.03 | Stage2 pullback-breakout RSI=56 vol=2.2x ATR=5.15 |
| Stop hit — per-position SL triggered | 2023-10-09 00:00:00 | 139.37 | 127.52 | 143.61 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2023-11-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 00:00:00 | 150.50 | 130.33 | 140.18 | Stage2 pullback-breakout RSI=65 vol=5.1x ATR=5.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 00:00:00 | 161.77 | 130.63 | 142.11 | T1 booked 50% @ 161.77 |
| Stop hit — per-position SL triggered | 2023-11-24 00:00:00 | 150.50 | 131.90 | 146.66 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2023-11-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 00:00:00 | 161.95 | 132.39 | 148.54 | Stage2 pullback-breakout RSI=67 vol=2.4x ATR=6.33 |
| Stop hit — per-position SL triggered | 2023-12-13 00:00:00 | 164.30 | 135.48 | 158.86 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-03-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 00:00:00 | 186.30 | 154.21 | 181.04 | Stage2 pullback-breakout RSI=57 vol=4.9x ATR=6.73 |
| Stop hit — per-position SL triggered | 2024-03-06 00:00:00 | 176.20 | 154.41 | 180.36 | SL hit (bars_held=1) |

### Cycle 7 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 194.05 | 157.27 | 176.50 | Stage2 pullback-breakout RSI=69 vol=1.9x ATR=6.71 |
| Stop hit — per-position SL triggered | 2024-04-19 00:00:00 | 188.15 | 160.59 | 186.05 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-26 00:00:00 | 135.10 | 2023-08-08 00:00:00 | 143.57 | PARTIAL | 0.50 | 6.27% |
| BUY | retest1 | 2023-07-26 00:00:00 | 135.10 | 2023-08-24 00:00:00 | 142.25 | TARGET_HIT | 0.50 | 5.29% |
| BUY | retest1 | 2023-08-31 00:00:00 | 154.60 | 2023-09-06 00:00:00 | 147.00 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest1 | 2023-09-27 00:00:00 | 147.10 | 2023-10-09 00:00:00 | 139.37 | STOP_HIT | 1.00 | -5.25% |
| BUY | retest1 | 2023-11-15 00:00:00 | 150.50 | 2023-11-16 00:00:00 | 161.77 | PARTIAL | 0.50 | 7.49% |
| BUY | retest1 | 2023-11-15 00:00:00 | 150.50 | 2023-11-24 00:00:00 | 150.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-29 00:00:00 | 161.95 | 2023-12-13 00:00:00 | 164.30 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest1 | 2024-03-05 00:00:00 | 186.30 | 2024-03-06 00:00:00 | 176.20 | STOP_HIT | 1.00 | -5.42% |
| BUY | retest1 | 2024-04-03 00:00:00 | 194.05 | 2024-04-19 00:00:00 | 188.15 | STOP_HIT | 1.00 | -3.04% |
