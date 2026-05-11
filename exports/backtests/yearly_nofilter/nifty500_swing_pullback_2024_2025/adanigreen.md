# Adani Green Energy Ltd. (ADANIGREEN)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 1327.60
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -0.10% / 0.00%
- **Sum % (uncompounded):** -0.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.10% | -0.4% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.10% | -0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.10% | -0.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 00:00:00 | 1820.50 | 1623.34 | 1759.82 | Stage2 pullback-breakout RSI=59 vol=4.2x ATR=59.55 |
| Stop hit — per-position SL triggered | 2024-08-08 00:00:00 | 1778.15 | 1641.72 | 1791.42 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-09-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 00:00:00 | 1945.70 | 1674.77 | 1856.94 | Stage2 pullback-breakout RSI=63 vol=2.5x ATR=62.87 |
| Stop hit — per-position SL triggered | 2024-09-09 00:00:00 | 1851.39 | 1685.23 | 1868.76 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2024-09-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 00:00:00 | 1930.80 | 1692.75 | 1858.25 | Stage2 pullback-breakout RSI=59 vol=5.1x ATR=65.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 00:00:00 | 2061.87 | 1710.43 | 1923.57 | T1 booked 50% @ 2061.87 |
| Stop hit — per-position SL triggered | 2024-09-30 00:00:00 | 1930.80 | 1721.85 | 1947.17 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-25 00:00:00 | 1820.50 | 2024-08-08 00:00:00 | 1778.15 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest1 | 2024-09-02 00:00:00 | 1945.70 | 2024-09-09 00:00:00 | 1851.39 | STOP_HIT | 1.00 | -4.85% |
| BUY | retest1 | 2024-09-16 00:00:00 | 1930.80 | 2024-09-24 00:00:00 | 2061.87 | PARTIAL | 0.50 | 6.79% |
| BUY | retest1 | 2024-09-16 00:00:00 | 1930.80 | 2024-09-30 00:00:00 | 1930.80 | STOP_HIT | 0.50 | 0.00% |
