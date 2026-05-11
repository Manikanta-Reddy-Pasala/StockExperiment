# Power Finance Corporation Ltd. (PFC)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 461.35
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 11.95% / 6.21%
- **Sum % (uncompounded):** 59.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 11.95% | 59.8% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 11.95% | 59.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 11.95% | 59.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 257.65 | 181.24 | 243.48 | Stage2 pullback-breakout RSI=62 vol=2.5x ATR=9.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 00:00:00 | 276.86 | 185.34 | 252.34 | T1 booked 50% @ 276.86 |
| Target hit | 2024-01-17 00:00:00 | 392.55 | 256.26 | 393.40 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-01-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 00:00:00 | 446.75 | 267.02 | 405.73 | Stage2 pullback-breakout RSI=67 vol=2.6x ATR=18.66 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 418.76 | 282.91 | 433.17 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2024-04-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 00:00:00 | 441.55 | 333.84 | 407.16 | Stage2 pullback-breakout RSI=68 vol=4.9x ATR=13.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 00:00:00 | 468.98 | 335.17 | 412.94 | T1 booked 50% @ 468.98 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 441.55 | 337.63 | 421.13 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-02 00:00:00 | 257.65 | 2023-11-09 00:00:00 | 276.86 | PARTIAL | 0.50 | 7.46% |
| BUY | retest1 | 2023-11-02 00:00:00 | 257.65 | 2024-01-17 00:00:00 | 392.55 | TARGET_HIT | 0.50 | 52.36% |
| BUY | retest1 | 2024-01-29 00:00:00 | 446.75 | 2024-02-09 00:00:00 | 418.76 | STOP_HIT | 1.00 | -6.27% |
| BUY | retest1 | 2024-04-30 00:00:00 | 441.55 | 2024-05-02 00:00:00 | 468.98 | PARTIAL | 0.50 | 6.21% |
| BUY | retest1 | 2024-04-30 00:00:00 | 441.55 | 2024-05-06 00:00:00 | 441.55 | STOP_HIT | 0.50 | 0.00% |
