# Can Fin Homes Ltd. (CANFINHOME)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 879.55
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -4.33% / -4.31%
- **Sum % (uncompounded):** -12.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.33% | -13.0% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.33% | -13.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.33% | -13.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 00:00:00 | 882.90 | 803.16 | 845.06 | Stage2 pullback-breakout RSI=66 vol=3.6x ATR=23.06 |
| Stop hit — per-position SL triggered | 2024-09-10 00:00:00 | 848.32 | 808.51 | 859.58 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2024-09-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 00:00:00 | 908.35 | 810.06 | 864.74 | Stage2 pullback-breakout RSI=65 vol=2.8x ATR=26.13 |
| Stop hit — per-position SL triggered | 2024-09-18 00:00:00 | 869.16 | 813.51 | 875.09 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-10-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 00:00:00 | 901.95 | 818.96 | 877.99 | Stage2 pullback-breakout RSI=58 vol=1.8x ATR=28.62 |
| Stop hit — per-position SL triggered | 2024-10-04 00:00:00 | 859.02 | 819.88 | 875.57 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-29 00:00:00 | 882.90 | 2024-09-10 00:00:00 | 848.32 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest1 | 2024-09-12 00:00:00 | 908.35 | 2024-09-18 00:00:00 | 869.16 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest1 | 2024-10-01 00:00:00 | 901.95 | 2024-10-04 00:00:00 | 859.02 | STOP_HIT | 1.00 | -4.76% |
