# ITI Ltd. (ITI)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 300.05
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 1
- **Avg / median % per leg:** -1.11% / -0.49%
- **Sum % (uncompounded):** -6.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | -1.11% | -6.7% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | -1.11% | -6.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 1 | 4 | 1 | -1.11% | -6.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 05:30:00 | 336.65 | 296.88 | 318.71 | Stage2 pullback-breakout RSI=64 vol=1.7x ATR=12.55 |
| Stop hit — per-position SL triggered | 2025-07-09 05:30:00 | 317.83 | 298.74 | 321.20 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2025-08-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-07 05:30:00 | 320.30 | 300.15 | 304.83 | Stage2 pullback-breakout RSI=59 vol=13.9x ATR=10.48 |
| Stop hit — per-position SL triggered | 2025-08-08 05:30:00 | 304.58 | 300.18 | 304.72 | SL hit (bars_held=1) |

### Cycle 3 — BUY (started 2025-09-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 05:30:00 | 301.90 | 299.31 | 295.31 | Stage2 pullback-breakout RSI=54 vol=4.0x ATR=10.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 05:30:00 | 322.08 | 300.38 | 305.21 | T1 booked 50% @ 322.08 |
| Target hit | 2025-09-25 05:30:00 | 310.55 | 301.81 | 312.57 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2025-11-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 05:30:00 | 325.15 | 308.48 | 317.47 | Stage2 pullback-breakout RSI=54 vol=15.5x ATR=11.37 |
| Stop hit — per-position SL triggered | 2025-11-26 05:30:00 | 308.10 | 308.51 | 316.16 | SL hit (bars_held=2) |

### Cycle 5 — BUY (started 2025-12-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 05:30:00 | 313.80 | 307.91 | 306.13 | Stage2 pullback-breakout RSI=55 vol=7.9x ATR=10.37 |
| Stop hit — per-position SL triggered | 2026-01-05 05:30:00 | 312.25 | 308.27 | 309.66 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-30 05:30:00 | 336.65 | 2025-07-09 05:30:00 | 317.83 | STOP_HIT | 1.00 | -5.59% |
| BUY | retest1 | 2025-08-07 05:30:00 | 320.30 | 2025-08-08 05:30:00 | 304.58 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest1 | 2025-09-02 05:30:00 | 301.90 | 2025-09-15 05:30:00 | 322.08 | PARTIAL | 0.50 | 6.68% |
| BUY | retest1 | 2025-09-02 05:30:00 | 301.90 | 2025-09-25 05:30:00 | 310.55 | TARGET_HIT | 0.50 | 2.87% |
| BUY | retest1 | 2025-11-24 05:30:00 | 325.15 | 2025-11-26 05:30:00 | 308.10 | STOP_HIT | 1.00 | -5.24% |
| BUY | retest1 | 2025-12-19 05:30:00 | 313.80 | 2026-01-05 05:30:00 | 312.25 | STOP_HIT | 1.00 | -0.49% |
