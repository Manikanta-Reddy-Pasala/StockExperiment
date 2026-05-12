# Netweb Technologies India Ltd. (NETWEB)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 4358.00
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
- **Avg / median % per leg:** -0.34% / 2.67%
- **Sum % (uncompounded):** -1.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.34% | -1.4% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.34% | -1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.34% | -1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 00:00:00 | 2542.50 | 1846.54 | 2359.08 | Stage2 pullback-breakout RSI=60 vol=2.4x ATR=127.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 00:00:00 | 2797.46 | 1912.45 | 2520.31 | T1 booked 50% @ 2797.46 |
| Stop hit — per-position SL triggered | 2024-09-06 00:00:00 | 2610.40 | 1942.17 | 2568.13 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-10-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-21 00:00:00 | 2745.70 | 2114.16 | 2594.19 | Stage2 pullback-breakout RSI=61 vol=4.4x ATR=136.95 |
| Stop hit — per-position SL triggered | 2024-10-25 00:00:00 | 2540.27 | 2132.76 | 2590.89 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 00:00:00 | 2799.45 | 2173.08 | 2634.51 | Stage2 pullback-breakout RSI=62 vol=1.9x ATR=122.64 |
| Stop hit — per-position SL triggered | 2024-11-18 00:00:00 | 2615.49 | 2212.96 | 2693.41 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-20 00:00:00 | 2542.50 | 2024-09-02 00:00:00 | 2797.46 | PARTIAL | 0.50 | 10.03% |
| BUY | retest1 | 2024-08-20 00:00:00 | 2542.50 | 2024-09-06 00:00:00 | 2610.40 | STOP_HIT | 0.50 | 2.67% |
| BUY | retest1 | 2024-10-21 00:00:00 | 2745.70 | 2024-10-25 00:00:00 | 2540.27 | STOP_HIT | 1.00 | -7.48% |
| BUY | retest1 | 2024-11-06 00:00:00 | 2799.45 | 2024-11-18 00:00:00 | 2615.49 | STOP_HIT | 1.00 | -6.57% |
