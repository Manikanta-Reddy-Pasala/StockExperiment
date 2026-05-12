# INDUSINDBK (INDUSINDBK)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 950.75
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 0.83% / 2.83%
- **Sum % (uncompounded):** 4.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 3 | 1 | 0.83% | 4.1% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 3 | 1 | 0.83% | 4.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 3 | 1 | 0.83% | 4.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 00:00:00 | 1334.05 | 1176.19 | 1298.73 | Stage2 pullback-breakout RSI=65 vol=1.7x ATR=26.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 00:00:00 | 1387.47 | 1182.04 | 1318.21 | T1 booked 50% @ 1387.47 |
| Target hit | 2023-08-02 00:00:00 | 1381.25 | 1222.93 | 1392.24 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 00:00:00 | 1463.70 | 1299.29 | 1428.77 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=30.35 |
| Stop hit — per-position SL triggered | 2023-10-18 00:00:00 | 1418.17 | 1303.28 | 1429.96 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2023-10-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 00:00:00 | 1469.10 | 1306.38 | 1435.45 | Stage2 pullback-breakout RSI=58 vol=1.5x ATR=30.60 |
| Stop hit — per-position SL triggered | 2023-10-25 00:00:00 | 1423.20 | 1308.70 | 1433.24 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-03-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 00:00:00 | 1512.10 | 1446.33 | 1496.25 | Stage2 pullback-breakout RSI=52 vol=1.7x ATR=37.67 |
| Stop hit — per-position SL triggered | 2024-04-09 00:00:00 | 1554.95 | 1455.94 | 1529.99 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-28 00:00:00 | 1334.05 | 2023-07-04 00:00:00 | 1387.47 | PARTIAL | 0.50 | 4.00% |
| BUY | retest1 | 2023-06-28 00:00:00 | 1334.05 | 2023-08-02 00:00:00 | 1381.25 | TARGET_HIT | 0.50 | 3.54% |
| BUY | retest1 | 2023-10-13 00:00:00 | 1463.70 | 2023-10-18 00:00:00 | 1418.17 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest1 | 2023-10-20 00:00:00 | 1469.10 | 2023-10-25 00:00:00 | 1423.20 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest1 | 2024-03-22 00:00:00 | 1512.10 | 2024-04-09 00:00:00 | 1554.95 | STOP_HIT | 1.00 | 2.83% |
