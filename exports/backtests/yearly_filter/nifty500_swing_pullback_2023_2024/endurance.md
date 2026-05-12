# Endurance Technologies Ltd. (ENDURANCE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 2513.40
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
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 2.12% / 6.19%
- **Sum % (uncompounded):** 8.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.12% | 8.5% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.12% | 8.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.12% | 8.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 00:00:00 | 1669.45 | 1536.99 | 1604.53 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=47.49 |
| Stop hit — per-position SL triggered | 2023-11-10 00:00:00 | 1598.22 | 1544.15 | 1633.87 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-12-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 00:00:00 | 1774.20 | 1570.43 | 1684.57 | Stage2 pullback-breakout RSI=66 vol=3.7x ATR=54.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 00:00:00 | 1884.08 | 1589.66 | 1764.66 | T1 booked 50% @ 1884.08 |
| Target hit | 2024-02-07 00:00:00 | 1983.25 | 1694.20 | 2036.32 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 00:00:00 | 1898.35 | 1731.19 | 1815.71 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=66.27 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 1798.95 | 1743.60 | 1856.44 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2024-05-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 00:00:00 | 2093.45 | 1773.79 | 1936.84 | Stage2 pullback-breakout RSI=69 vol=5.4x ATR=64.67 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-03 00:00:00 | 1669.45 | 2023-11-10 00:00:00 | 1598.22 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest1 | 2023-12-18 00:00:00 | 1774.20 | 2023-12-29 00:00:00 | 1884.08 | PARTIAL | 0.50 | 6.19% |
| BUY | retest1 | 2023-12-18 00:00:00 | 1774.20 | 2024-02-07 00:00:00 | 1983.25 | TARGET_HIT | 0.50 | 11.78% |
| BUY | retest1 | 2024-04-02 00:00:00 | 1898.35 | 2024-04-15 00:00:00 | 1798.95 | STOP_HIT | 1.00 | -5.24% |
