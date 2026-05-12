# Tata Teleservices (Maharashtra) Ltd. (TTML)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 42.58
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
- **Avg / median % per leg:** -4.81% / -5.21%
- **Sum % (uncompounded):** -14.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.81% | -14.4% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.81% | -14.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.81% | -14.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 00:00:00 | 93.45 | 86.14 | 89.34 | Stage2 pullback-breakout RSI=60 vol=4.1x ATR=3.33 |
| Stop hit — per-position SL triggered | 2023-12-13 00:00:00 | 90.00 | 86.74 | 91.06 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-12-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 00:00:00 | 97.55 | 86.94 | 91.74 | Stage2 pullback-breakout RSI=65 vol=4.2x ATR=3.39 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 92.46 | 87.04 | 91.76 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-01-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-08 00:00:00 | 94.80 | 87.55 | 91.76 | Stage2 pullback-breakout RSI=58 vol=4.8x ATR=3.49 |
| Stop hit — per-position SL triggered | 2024-01-16 00:00:00 | 89.56 | 87.88 | 92.32 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-29 00:00:00 | 93.45 | 2023-12-13 00:00:00 | 90.00 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest1 | 2023-12-18 00:00:00 | 97.55 | 2023-12-20 00:00:00 | 92.46 | STOP_HIT | 1.00 | -5.21% |
| BUY | retest1 | 2024-01-08 00:00:00 | 94.80 | 2024-01-16 00:00:00 | 89.56 | STOP_HIT | 1.00 | -5.52% |
