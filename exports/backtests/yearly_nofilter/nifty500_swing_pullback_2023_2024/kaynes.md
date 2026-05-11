# Kaynes Technology India Ltd. (KAYNES)

## Backtest Summary

- **Window:** 2022-11-22 00:00:00 → 2026-05-11 00:00:00 (859 bars)
- **Last close:** 4465.30
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 0 / 2 / 1
- **Avg / median % per leg:** 3.02% / 2.58%
- **Sum % (uncompounded):** 9.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 0 | 2 | 1 | 3.02% | 9.1% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 3.02% | 9.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 3.02% | 9.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 00:00:00 | 2580.45 | 1842.74 | 2448.24 | Stage2 pullback-breakout RSI=64 vol=4.1x ATR=83.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 00:00:00 | 2747.57 | 1859.20 | 2489.76 | T1 booked 50% @ 2747.57 |
| Stop hit — per-position SL triggered | 2023-12-27 00:00:00 | 2580.45 | 1899.60 | 2564.56 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2024-01-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 00:00:00 | 2851.15 | 2038.69 | 2679.73 | Stage2 pullback-breakout RSI=65 vol=1.7x ATR=108.76 |
| Stop hit — per-position SL triggered | 2024-02-08 00:00:00 | 2924.65 | 2117.56 | 2800.71 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-12-15 00:00:00 | 2580.45 | 2023-12-19 00:00:00 | 2747.57 | PARTIAL | 0.50 | 6.48% |
| BUY | retest1 | 2023-12-15 00:00:00 | 2580.45 | 2023-12-27 00:00:00 | 2580.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-24 00:00:00 | 2851.15 | 2024-02-08 00:00:00 | 2924.65 | STOP_HIT | 1.00 | 2.58% |
