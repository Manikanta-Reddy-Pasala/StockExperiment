# Tata Communications Ltd. (TATACOMM)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 1592.30
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 1
- **Target hits / Stop hits / Partials:** 2 / 2 / 3
- **Avg / median % per leg:** 3.55% / 5.14%
- **Sum % (uncompounded):** 24.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 6 | 85.7% | 2 | 2 | 3 | 3.55% | 24.8% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 2 | 2 | 3 | 3.55% | 24.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 6 | 85.7% | 2 | 2 | 3 | 3.55% | 24.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 05:30:00 | 1638.40 | 1324.00 | 1544.62 | Stage2 pullback-breakout RSI=68 vol=3.9x ATR=49.34 |
| Stop hit — per-position SL triggered | 2023-07-13 05:30:00 | 1564.39 | 1329.40 | 1553.97 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-07-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 05:30:00 | 1667.55 | 1354.08 | 1592.42 | Stage2 pullback-breakout RSI=65 vol=2.5x ATR=50.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-28 05:30:00 | 1768.67 | 1361.92 | 1621.15 | T1 booked 50% @ 1768.67 |
| Stop hit — per-position SL triggered | 2023-08-09 05:30:00 | 1690.85 | 1388.97 | 1668.80 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-08-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 05:30:00 | 1782.15 | 1413.29 | 1691.62 | Stage2 pullback-breakout RSI=68 vol=1.7x ATR=48.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 05:30:00 | 1878.97 | 1443.95 | 1758.15 | T1 booked 50% @ 1878.97 |
| Target hit | 2023-09-25 05:30:00 | 1856.65 | 1505.97 | 1858.88 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 05:30:00 | 1759.85 | 1642.96 | 1720.68 | Stage2 pullback-breakout RSI=56 vol=1.9x ATR=45.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 05:30:00 | 1850.31 | 1659.93 | 1779.14 | T1 booked 50% @ 1850.31 |
| Target hit | 2024-03-13 05:30:00 | 1885.60 | 1692.56 | 1900.23 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-11 05:30:00 | 1638.40 | 2023-07-13 05:30:00 | 1564.39 | STOP_HIT | 1.00 | -4.52% |
| BUY | retest1 | 2023-07-26 05:30:00 | 1667.55 | 2023-07-28 05:30:00 | 1768.67 | PARTIAL | 0.50 | 6.06% |
| BUY | retest1 | 2023-07-26 05:30:00 | 1667.55 | 2023-08-09 05:30:00 | 1690.85 | STOP_HIT | 0.50 | 1.40% |
| BUY | retest1 | 2023-08-22 05:30:00 | 1782.15 | 2023-09-01 05:30:00 | 1878.97 | PARTIAL | 0.50 | 5.43% |
| BUY | retest1 | 2023-08-22 05:30:00 | 1782.15 | 2023-09-25 05:30:00 | 1856.65 | TARGET_HIT | 0.50 | 4.18% |
| BUY | retest1 | 2024-02-08 05:30:00 | 1759.85 | 2024-02-26 05:30:00 | 1850.31 | PARTIAL | 0.50 | 5.14% |
| BUY | retest1 | 2024-02-08 05:30:00 | 1759.85 | 2024-03-13 05:30:00 | 1885.60 | TARGET_HIT | 0.50 | 7.15% |
