# Railtel Corporation Of India Ltd. (RAILTEL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 342.60
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / Stop hits / Partials:** 2 / 3 / 3
- **Avg / median % per leg:** 11.08% / 9.05%
- **Sum % (uncompounded):** 88.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 2 | 3 | 3 | 11.08% | 88.6% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 2 | 3 | 3 | 11.08% | 88.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 6 | 75.0% | 2 | 3 | 3 | 11.08% | 88.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 00:00:00 | 133.10 | 116.20 | 127.28 | Stage2 pullback-breakout RSI=63 vol=2.2x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 00:00:00 | 142.61 | 117.88 | 131.09 | T1 booked 50% @ 142.61 |
| Stop hit — per-position SL triggered | 2023-07-13 00:00:00 | 137.05 | 117.88 | 131.09 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 00:00:00 | 190.85 | 130.65 | 169.05 | Stage2 pullback-breakout RSI=69 vol=6.1x ATR=8.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 00:00:00 | 208.12 | 132.73 | 177.45 | T1 booked 50% @ 208.12 |
| Target hit | 2023-09-21 00:00:00 | 211.05 | 145.10 | 213.22 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-10-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 00:00:00 | 227.10 | 155.43 | 217.34 | Stage2 pullback-breakout RSI=59 vol=6.1x ATR=10.68 |
| Stop hit — per-position SL triggered | 2023-10-25 00:00:00 | 211.08 | 160.27 | 221.59 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2023-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 00:00:00 | 240.55 | 165.37 | 225.11 | Stage2 pullback-breakout RSI=62 vol=2.1x ATR=11.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 00:00:00 | 264.24 | 172.89 | 242.21 | T1 booked 50% @ 264.24 |
| Target hit | 2024-02-09 00:00:00 | 396.85 | 249.39 | 400.65 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-02-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 00:00:00 | 469.95 | 265.65 | 400.68 | Stage2 pullback-breakout RSI=69 vol=5.2x ATR=27.87 |
| Stop hit — per-position SL triggered | 2024-03-06 00:00:00 | 428.15 | 277.67 | 421.33 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-27 00:00:00 | 133.10 | 2023-07-13 00:00:00 | 142.61 | PARTIAL | 0.50 | 7.15% |
| BUY | retest1 | 2023-06-27 00:00:00 | 133.10 | 2023-07-13 00:00:00 | 137.05 | STOP_HIT | 0.50 | 2.97% |
| BUY | retest1 | 2023-08-28 00:00:00 | 190.85 | 2023-08-31 00:00:00 | 208.12 | PARTIAL | 0.50 | 9.05% |
| BUY | retest1 | 2023-08-28 00:00:00 | 190.85 | 2023-09-21 00:00:00 | 211.05 | TARGET_HIT | 0.50 | 10.58% |
| BUY | retest1 | 2023-10-13 00:00:00 | 227.10 | 2023-10-25 00:00:00 | 211.08 | STOP_HIT | 1.00 | -7.05% |
| BUY | retest1 | 2023-11-06 00:00:00 | 240.55 | 2023-11-17 00:00:00 | 264.24 | PARTIAL | 0.50 | 9.85% |
| BUY | retest1 | 2023-11-06 00:00:00 | 240.55 | 2024-02-09 00:00:00 | 396.85 | TARGET_HIT | 0.50 | 64.98% |
| BUY | retest1 | 2024-02-27 00:00:00 | 469.95 | 2024-03-06 00:00:00 | 428.15 | STOP_HIT | 1.00 | -8.89% |
