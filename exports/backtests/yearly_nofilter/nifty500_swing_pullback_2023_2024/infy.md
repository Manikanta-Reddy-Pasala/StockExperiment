# Infosys Ltd. (INFY)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1175.60
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
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** 1.31% / 2.72%
- **Sum % (uncompounded):** 6.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 0 | 4 | 1 | 1.31% | 6.5% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 0 | 4 | 1 | 1.31% | 6.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 0 | 4 | 1 | 1.31% | 6.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 00:00:00 | 1444.90 | 1412.37 | 1401.34 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=24.78 |
| Stop hit — per-position SL triggered | 2023-12-01 00:00:00 | 1452.30 | 1415.65 | 1431.00 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-12-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 00:00:00 | 1501.45 | 1420.68 | 1457.45 | Stage2 pullback-breakout RSI=62 vol=2.3x ATR=25.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 00:00:00 | 1552.27 | 1422.25 | 1468.97 | T1 booked 50% @ 1552.27 |
| Stop hit — per-position SL triggered | 2023-12-29 00:00:00 | 1542.90 | 1433.48 | 1518.71 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-01-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 00:00:00 | 1612.75 | 1442.69 | 1527.72 | Stage2 pullback-breakout RSI=64 vol=3.5x ATR=37.55 |
| Stop hit — per-position SL triggered | 2024-01-29 00:00:00 | 1656.60 | 1462.61 | 1607.65 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-03-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-14 00:00:00 | 1653.30 | 1518.35 | 1641.33 | Stage2 pullback-breakout RSI=53 vol=1.7x ATR=31.38 |
| Stop hit — per-position SL triggered | 2024-03-18 00:00:00 | 1606.22 | 1520.33 | 1637.01 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-16 00:00:00 | 1444.90 | 2023-12-01 00:00:00 | 1452.30 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest1 | 2023-12-14 00:00:00 | 1501.45 | 2023-12-15 00:00:00 | 1552.27 | PARTIAL | 0.50 | 3.38% |
| BUY | retest1 | 2023-12-14 00:00:00 | 1501.45 | 2023-12-29 00:00:00 | 1542.90 | STOP_HIT | 0.50 | 2.76% |
| BUY | retest1 | 2024-01-12 00:00:00 | 1612.75 | 2024-01-29 00:00:00 | 1656.60 | STOP_HIT | 1.00 | 2.72% |
| BUY | retest1 | 2024-03-14 00:00:00 | 1653.30 | 2024-03-18 00:00:00 | 1606.22 | STOP_HIT | 1.00 | -2.85% |
