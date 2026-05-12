# CIPLA (CIPLA)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 1334.90
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
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -1.92% / -3.02%
- **Sum % (uncompounded):** -5.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.92% | -5.8% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.92% | -5.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.92% | -5.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 00:00:00 | 1575.00 | 1401.18 | 1507.84 | Stage2 pullback-breakout RSI=68 vol=2.8x ATR=31.88 |
| Stop hit — per-position SL triggered | 2024-07-30 00:00:00 | 1527.19 | 1403.96 | 1513.82 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2024-08-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 00:00:00 | 1569.95 | 1413.26 | 1528.50 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=33.88 |
| Stop hit — per-position SL triggered | 2024-08-23 00:00:00 | 1574.55 | 1428.92 | 1559.85 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-09-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 00:00:00 | 1672.50 | 1475.46 | 1637.46 | Stage2 pullback-breakout RSI=61 vol=2.7x ATR=33.64 |
| Stop hit — per-position SL triggered | 2024-10-04 00:00:00 | 1622.04 | 1482.28 | 1641.07 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-26 00:00:00 | 1575.00 | 2024-07-30 00:00:00 | 1527.19 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest1 | 2024-08-08 00:00:00 | 1569.95 | 2024-08-23 00:00:00 | 1574.55 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest1 | 2024-09-27 00:00:00 | 1672.50 | 2024-10-04 00:00:00 | 1622.04 | STOP_HIT | 1.00 | -3.02% |
