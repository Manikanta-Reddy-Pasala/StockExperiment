# Lloyds Metals And Energy Ltd. (LLOYDSME)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 1736.30
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
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 2
- **Avg / median % per leg:** 6.57% / 6.43%
- **Sum % (uncompounded):** 45.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 6.57% | 46.0% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 6.57% | 46.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 2 | 3 | 2 | 6.57% | 46.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 05:30:00 | 775.10 | 634.56 | 727.61 | Stage2 pullback-breakout RSI=69 vol=5.7x ATR=27.76 |
| Stop hit — per-position SL triggered | 2024-07-12 05:30:00 | 733.45 | 641.51 | 739.09 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2024-08-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 05:30:00 | 777.95 | 652.41 | 735.74 | Stage2 pullback-breakout RSI=63 vol=3.7x ATR=30.15 |
| Stop hit — per-position SL triggered | 2024-08-05 05:30:00 | 732.73 | 654.42 | 738.77 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-09-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 05:30:00 | 820.85 | 682.68 | 769.75 | Stage2 pullback-breakout RSI=67 vol=3.1x ATR=26.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 05:30:00 | 873.59 | 688.60 | 791.61 | T1 booked 50% @ 873.59 |
| Target hit | 2024-10-28 05:30:00 | 938.90 | 747.83 | 958.09 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-11-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 05:30:00 | 965.50 | 783.19 | 955.45 | Stage2 pullback-breakout RSI=53 vol=2.4x ATR=41.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 05:30:00 | 1047.56 | 790.99 | 965.24 | T1 booked 50% @ 1047.56 |
| Target hit | 2025-01-27 05:30:00 | 1294.55 | 949.11 | 1351.84 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2025-04-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 05:30:00 | 1329.80 | 1033.76 | 1230.79 | Stage2 pullback-breakout RSI=66 vol=1.6x ATR=55.19 |
| Stop hit — per-position SL triggered | 2025-04-04 05:30:00 | 1247.02 | 1038.63 | 1239.37 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-04 05:30:00 | 775.10 | 2024-07-12 05:30:00 | 733.45 | STOP_HIT | 1.00 | -5.37% |
| BUY | retest1 | 2024-08-01 05:30:00 | 777.95 | 2024-08-05 05:30:00 | 732.73 | STOP_HIT | 1.00 | -5.81% |
| BUY | retest1 | 2024-09-17 05:30:00 | 820.85 | 2024-09-23 05:30:00 | 873.59 | PARTIAL | 0.50 | 6.43% |
| BUY | retest1 | 2024-09-17 05:30:00 | 820.85 | 2024-10-28 05:30:00 | 938.90 | TARGET_HIT | 0.50 | 14.38% |
| BUY | retest1 | 2024-11-25 05:30:00 | 965.50 | 2024-11-29 05:30:00 | 1047.56 | PARTIAL | 0.50 | 8.50% |
| BUY | retest1 | 2024-11-25 05:30:00 | 965.50 | 2025-01-27 05:30:00 | 1294.55 | TARGET_HIT | 0.50 | 34.08% |
| BUY | retest1 | 2025-04-02 05:30:00 | 1329.80 | 2025-04-04 05:30:00 | 1247.02 | STOP_HIT | 1.00 | -6.23% |
