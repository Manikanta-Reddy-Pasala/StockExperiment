# BEML Ltd. (BEML)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 1894.10
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -4.37% / -5.80%
- **Sum % (uncompounded):** -17.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 4 | 0 | -4.37% | -17.5% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -4.37% | -17.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -4.37% | -17.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 00:00:00 | 2529.98 | 1686.06 | 2236.23 | Stage2 pullback-breakout RSI=69 vol=5.7x ATR=135.42 |
| Stop hit — per-position SL triggered | 2024-07-19 00:00:00 | 2326.85 | 1754.28 | 2372.03 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2024-10-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 00:00:00 | 2019.53 | 1849.24 | 1893.15 | Stage2 pullback-breakout RSI=63 vol=4.4x ATR=78.09 |
| Stop hit — per-position SL triggered | 2024-10-22 00:00:00 | 1902.40 | 1851.59 | 1906.17 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 00:00:00 | 2073.82 | 1864.24 | 1960.88 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=81.06 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 1952.23 | 1872.35 | 1984.13 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2024-11-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 00:00:00 | 2134.48 | 1879.57 | 1980.35 | Stage2 pullback-breakout RSI=65 vol=2.0x ATR=88.94 |
| Stop hit — per-position SL triggered | 2024-12-18 00:00:00 | 2181.35 | 1918.85 | 2139.21 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-05 00:00:00 | 2529.98 | 2024-07-19 00:00:00 | 2326.85 | STOP_HIT | 1.00 | -8.03% |
| BUY | retest1 | 2024-10-18 00:00:00 | 2019.53 | 2024-10-22 00:00:00 | 1902.40 | STOP_HIT | 1.00 | -5.80% |
| BUY | retest1 | 2024-11-06 00:00:00 | 2073.82 | 2024-11-13 00:00:00 | 1952.23 | STOP_HIT | 1.00 | -5.86% |
| BUY | retest1 | 2024-11-28 00:00:00 | 2134.48 | 2024-12-18 00:00:00 | 2181.35 | STOP_HIT | 1.00 | 2.20% |
