# Marico Ltd. (MARICO)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 835.35
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
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 0.92% / 2.54%
- **Sum % (uncompounded):** 3.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.92% | 3.7% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.92% | 3.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.92% | 3.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 00:00:00 | 538.55 | 514.36 | 532.89 | Stage2 pullback-breakout RSI=54 vol=2.0x ATR=10.58 |
| Stop hit — per-position SL triggered | 2023-07-07 00:00:00 | 522.68 | 514.52 | 530.93 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-07-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 00:00:00 | 554.05 | 517.35 | 535.94 | Stage2 pullback-breakout RSI=61 vol=2.4x ATR=11.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-28 00:00:00 | 577.43 | 517.91 | 539.59 | T1 booked 50% @ 577.43 |
| Stop hit — per-position SL triggered | 2023-08-14 00:00:00 | 568.10 | 523.54 | 562.03 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 538.65 | 535.82 | 529.47 | Stage2 pullback-breakout RSI=56 vol=2.0x ATR=8.84 |
| Stop hit — per-position SL triggered | 2023-12-14 00:00:00 | 538.05 | 535.95 | 534.52 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-05 00:00:00 | 538.55 | 2023-07-07 00:00:00 | 522.68 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest1 | 2023-07-27 00:00:00 | 554.05 | 2023-07-28 00:00:00 | 577.43 | PARTIAL | 0.50 | 4.22% |
| BUY | retest1 | 2023-07-27 00:00:00 | 554.05 | 2023-08-14 00:00:00 | 568.10 | STOP_HIT | 0.50 | 2.54% |
| BUY | retest1 | 2023-11-30 00:00:00 | 538.65 | 2023-12-14 00:00:00 | 538.05 | STOP_HIT | 1.00 | -0.11% |
