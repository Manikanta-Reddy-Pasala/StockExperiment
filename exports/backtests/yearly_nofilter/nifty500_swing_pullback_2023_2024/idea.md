# Vodafone Idea Ltd. (IDEA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 11.24
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
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 3
- **Target hits / Stop hits / Partials:** 2 / 4 / 3
- **Avg / median % per leg:** 4.37% / 4.30%
- **Sum % (uncompounded):** 39.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 6 | 66.7% | 2 | 4 | 3 | 4.37% | 39.3% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 2 | 4 | 3 | 4.37% | 39.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 6 | 66.7% | 2 | 4 | 3 | 4.37% | 39.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-25 00:00:00 | 8.70 | 7.66 | 7.97 | Stage2 pullback-breakout RSI=67 vol=3.6x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 00:00:00 | 9.42 | 7.73 | 8.47 | T1 booked 50% @ 9.42 |
| Target hit | 2023-10-06 00:00:00 | 10.95 | 8.43 | 11.03 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 00:00:00 | 12.80 | 8.92 | 11.50 | Stage2 pullback-breakout RSI=66 vol=3.6x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-03 00:00:00 | 14.24 | 9.02 | 11.91 | T1 booked 50% @ 14.24 |
| Target hit | 2023-11-24 00:00:00 | 13.35 | 9.69 | 13.41 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-12-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 00:00:00 | 13.95 | 10.12 | 13.27 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=0.59 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 13.07 | 10.26 | 13.44 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2024-02-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 00:00:00 | 15.00 | 11.62 | 14.72 | Stage2 pullback-breakout RSI=53 vol=2.0x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-16 00:00:00 | 16.55 | 11.87 | 15.03 | T1 booked 50% @ 16.55 |
| Stop hit — per-position SL triggered | 2024-02-21 00:00:00 | 15.35 | 11.98 | 15.22 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-02-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 00:00:00 | 17.55 | 12.08 | 15.53 | Stage2 pullback-breakout RSI=67 vol=4.4x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-02-27 00:00:00 | 16.11 | 12.16 | 15.68 | SL hit (bars_held=2) |

### Cycle 6 — BUY (started 2024-04-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 00:00:00 | 14.40 | 12.53 | 13.34 | Stage2 pullback-breakout RSI=61 vol=3.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2024-04-24 00:00:00 | 13.18 | 12.54 | 13.32 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-25 00:00:00 | 8.70 | 2023-09-01 00:00:00 | 9.42 | PARTIAL | 0.50 | 8.27% |
| BUY | retest1 | 2023-08-25 00:00:00 | 8.70 | 2023-10-06 00:00:00 | 10.95 | TARGET_HIT | 0.50 | 25.86% |
| BUY | retest1 | 2023-11-01 00:00:00 | 12.80 | 2023-11-03 00:00:00 | 14.24 | PARTIAL | 0.50 | 11.25% |
| BUY | retest1 | 2023-11-01 00:00:00 | 12.80 | 2023-11-24 00:00:00 | 13.35 | TARGET_HIT | 0.50 | 4.30% |
| BUY | retest1 | 2023-12-14 00:00:00 | 13.95 | 2023-12-20 00:00:00 | 13.07 | STOP_HIT | 1.00 | -6.31% |
| BUY | retest1 | 2024-02-07 00:00:00 | 15.00 | 2024-02-16 00:00:00 | 16.55 | PARTIAL | 0.50 | 10.33% |
| BUY | retest1 | 2024-02-07 00:00:00 | 15.00 | 2024-02-21 00:00:00 | 15.35 | STOP_HIT | 0.50 | 2.33% |
| BUY | retest1 | 2024-02-23 00:00:00 | 17.55 | 2024-02-27 00:00:00 | 16.11 | STOP_HIT | 1.00 | -8.20% |
| BUY | retest1 | 2024-04-23 00:00:00 | 14.40 | 2024-04-24 00:00:00 | 13.18 | STOP_HIT | 1.00 | -8.50% |
