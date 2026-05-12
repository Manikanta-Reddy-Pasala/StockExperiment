# NTPC (NTPC)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 402.15
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 1.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.17% | 1.2% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.17% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.17% | 1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 00:00:00 | 189.10 | 172.98 | 183.86 | Stage2 pullback-breakout RSI=65 vol=2.9x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 00:00:00 | 195.57 | 173.96 | 187.56 | T1 booked 50% @ 195.57 |
| Stop hit — per-position SL triggered | 2023-07-13 00:00:00 | 189.10 | 174.79 | 188.85 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2023-09-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 00:00:00 | 245.55 | 195.06 | 236.07 | Stage2 pullback-breakout RSI=67 vol=2.0x ATR=5.75 |
| Stop hit — per-position SL triggered | 2023-10-04 00:00:00 | 236.93 | 195.91 | 236.43 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-02-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 00:00:00 | 332.50 | 245.07 | 313.74 | Stage2 pullback-breakout RSI=66 vol=1.6x ATR=9.46 |
| Stop hit — per-position SL triggered | 2024-02-12 00:00:00 | 318.30 | 249.95 | 320.06 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2024-03-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-04 00:00:00 | 353.85 | 262.98 | 335.76 | Stage2 pullback-breakout RSI=67 vol=2.6x ATR=8.88 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 340.53 | 267.82 | 339.71 | SL hit (bars_held=6) |

### Cycle 5 — BUY (started 2024-03-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 00:00:00 | 331.60 | 272.44 | 329.59 | Stage2 pullback-breakout RSI=52 vol=2.5x ATR=9.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 00:00:00 | 351.22 | 275.23 | 334.40 | T1 booked 50% @ 351.22 |
| Target hit | 2024-04-22 00:00:00 | 342.90 | 283.72 | 348.49 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-28 00:00:00 | 189.10 | 2023-07-06 00:00:00 | 195.57 | PARTIAL | 0.50 | 3.42% |
| BUY | retest1 | 2023-06-28 00:00:00 | 189.10 | 2023-07-13 00:00:00 | 189.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-29 00:00:00 | 245.55 | 2023-10-04 00:00:00 | 236.93 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest1 | 2024-02-02 00:00:00 | 332.50 | 2024-02-12 00:00:00 | 318.30 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest1 | 2024-03-04 00:00:00 | 353.85 | 2024-03-13 00:00:00 | 340.53 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest1 | 2024-03-27 00:00:00 | 331.60 | 2024-04-03 00:00:00 | 351.22 | PARTIAL | 0.50 | 5.92% |
| BUY | retest1 | 2024-03-27 00:00:00 | 331.60 | 2024-04-22 00:00:00 | 342.90 | TARGET_HIT | 0.50 | 3.41% |
