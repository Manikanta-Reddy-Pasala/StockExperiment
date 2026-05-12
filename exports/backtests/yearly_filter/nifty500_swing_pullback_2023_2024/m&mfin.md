# Mahindra & Mahindra Financial Services Ltd. (M&MFIN)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 332.75
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / Stop hits / Partials:** 0 / 6 / 0
- **Avg / median % per leg:** -3.21% / -3.81%
- **Sum % (uncompounded):** -19.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.21% | -19.3% |
| BUY @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.21% | -19.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.21% | -19.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 00:00:00 | 298.05 | 270.11 | 291.57 | Stage2 pullback-breakout RSI=53 vol=1.8x ATR=9.04 |
| Stop hit — per-position SL triggered | 2023-09-06 00:00:00 | 294.35 | 272.58 | 294.25 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-20 00:00:00 | 309.50 | 275.02 | 298.47 | Stage2 pullback-breakout RSI=61 vol=3.2x ATR=8.73 |
| Stop hit — per-position SL triggered | 2023-09-26 00:00:00 | 296.40 | 276.00 | 298.97 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-12-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 00:00:00 | 289.35 | 275.49 | 273.86 | Stage2 pullback-breakout RSI=66 vol=2.9x ATR=7.35 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 278.33 | 275.64 | 275.45 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 289.20 | 275.86 | 277.81 | Stage2 pullback-breakout RSI=62 vol=3.3x ATR=8.62 |
| Stop hit — per-position SL triggered | 2024-02-13 00:00:00 | 276.27 | 276.93 | 283.95 | SL hit (bars_held=9) |

### Cycle 5 — BUY (started 2024-03-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 00:00:00 | 290.80 | 278.63 | 287.33 | Stage2 pullback-breakout RSI=56 vol=2.3x ATR=7.48 |
| Stop hit — per-position SL triggered | 2024-03-06 00:00:00 | 279.58 | 278.68 | 287.03 | SL hit (bars_held=1) |

### Cycle 6 — BUY (started 2024-04-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 00:00:00 | 296.05 | 277.97 | 277.96 | Stage2 pullback-breakout RSI=66 vol=1.7x ATR=8.62 |
| Stop hit — per-position SL triggered | 2024-04-18 00:00:00 | 291.10 | 279.70 | 289.09 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-23 00:00:00 | 298.05 | 2023-09-06 00:00:00 | 294.35 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest1 | 2023-09-20 00:00:00 | 309.50 | 2023-09-26 00:00:00 | 296.40 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest1 | 2023-12-14 00:00:00 | 289.35 | 2023-12-20 00:00:00 | 278.33 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest1 | 2024-01-31 00:00:00 | 289.20 | 2024-02-13 00:00:00 | 276.27 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest1 | 2024-03-05 00:00:00 | 290.80 | 2024-03-06 00:00:00 | 279.58 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest1 | 2024-04-02 00:00:00 | 296.05 | 2024-04-18 00:00:00 | 291.10 | STOP_HIT | 1.00 | -1.67% |
