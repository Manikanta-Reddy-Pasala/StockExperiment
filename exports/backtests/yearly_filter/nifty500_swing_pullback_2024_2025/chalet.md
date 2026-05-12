# Chalet Hotels Ltd. (CHALET)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 770.55
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 2.08% / 4.15%
- **Sum % (uncompounded):** 8.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 2 | 1 | 2.08% | 8.3% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 2 | 1 | 2.08% | 8.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 2 | 1 | 2.08% | 8.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 00:00:00 | 824.35 | 761.60 | 796.57 | Stage2 pullback-breakout RSI=57 vol=1.5x ATR=29.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 00:00:00 | 882.58 | 766.85 | 820.53 | T1 booked 50% @ 882.58 |
| Target hit | 2024-09-18 00:00:00 | 858.55 | 778.10 | 860.07 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-09-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 00:00:00 | 914.85 | 782.61 | 871.34 | Stage2 pullback-breakout RSI=65 vol=2.4x ATR=33.97 |
| Stop hit — per-position SL triggered | 2024-09-26 00:00:00 | 863.90 | 784.69 | 874.25 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 897.60 | 810.17 | 859.29 | Stage2 pullback-breakout RSI=59 vol=5.5x ATR=38.38 |
| Stop hit — per-position SL triggered | 2024-12-10 00:00:00 | 921.80 | 819.16 | 886.39 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-26 00:00:00 | 824.35 | 2024-09-04 00:00:00 | 882.58 | PARTIAL | 0.50 | 7.06% |
| BUY | retest1 | 2024-08-26 00:00:00 | 824.35 | 2024-09-18 00:00:00 | 858.55 | TARGET_HIT | 0.50 | 4.15% |
| BUY | retest1 | 2024-09-24 00:00:00 | 914.85 | 2024-09-26 00:00:00 | 863.90 | STOP_HIT | 1.00 | -5.57% |
| BUY | retest1 | 2024-11-25 00:00:00 | 897.60 | 2024-12-10 00:00:00 | 921.80 | STOP_HIT | 1.00 | 2.70% |
