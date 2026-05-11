# Cohance Lifesciences Ltd. (COHANCE)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 488.20
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
- **Avg / median % per leg:** -6.54% / -6.90%
- **Sum % (uncompounded):** -19.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -6.54% | -19.6% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -6.54% | -19.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -6.54% | -19.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-02-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 05:30:00 | 1165.65 | 1033.86 | 1086.38 | Stage2 pullback-breakout RSI=64 vol=2.4x ATR=53.58 |
| Stop hit — per-position SL triggered | 2025-02-11 05:30:00 | 1085.27 | 1036.39 | 1094.39 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2025-02-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 05:30:00 | 1211.20 | 1042.15 | 1111.92 | Stage2 pullback-breakout RSI=63 vol=6.6x ATR=70.23 |
| Stop hit — per-position SL triggered | 2025-03-04 05:30:00 | 1105.85 | 1052.07 | 1148.58 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2025-04-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 05:30:00 | 1184.80 | 1071.23 | 1125.29 | Stage2 pullback-breakout RSI=57 vol=1.7x ATR=67.41 |
| Stop hit — per-position SL triggered | 2025-04-30 05:30:00 | 1137.10 | 1081.25 | 1153.80 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-02-06 05:30:00 | 1165.65 | 2025-02-11 05:30:00 | 1085.27 | STOP_HIT | 1.00 | -6.90% |
| BUY | retest1 | 2025-02-20 05:30:00 | 1211.20 | 2025-03-04 05:30:00 | 1105.85 | STOP_HIT | 1.00 | -8.70% |
| BUY | retest1 | 2025-04-15 05:30:00 | 1184.80 | 2025-04-30 05:30:00 | 1137.10 | STOP_HIT | 1.00 | -4.03% |
