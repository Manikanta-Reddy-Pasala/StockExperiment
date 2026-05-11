# Krishna Institute of Medical Sciences Ltd. (KIMS)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 717.60
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 0
- **Target hits / Stop hits / Partials:** 0 / 2 / 0
- **Avg / median % per leg:** 1.80% / 2.74%
- **Sum % (uncompounded):** 3.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 0 | 2 | 0 | 1.80% | 3.6% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 1.80% | 3.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 2 | 0 | 1.80% | 3.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 05:30:00 | 387.60 | 350.85 | 377.77 | Stage2 pullback-breakout RSI=57 vol=2.7x ATR=10.64 |
| Stop hit — per-position SL triggered | 2023-12-05 05:30:00 | 398.21 | 354.54 | 385.83 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-02-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 05:30:00 | 447.57 | 375.60 | 424.18 | Stage2 pullback-breakout RSI=67 vol=2.2x ATR=13.81 |
| Stop hit — per-position SL triggered | 2024-02-29 05:30:00 | 451.44 | 382.60 | 440.47 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-20 05:30:00 | 387.60 | 2023-12-05 05:30:00 | 398.21 | STOP_HIT | 1.00 | 2.74% |
| BUY | retest1 | 2024-02-15 05:30:00 | 447.57 | 2024-02-29 05:30:00 | 451.44 | STOP_HIT | 1.00 | 0.86% |
