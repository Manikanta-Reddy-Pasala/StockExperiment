# Max Financial Services Ltd. (MFSL)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 1680.20
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
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 1.29% / 1.77%
- **Sum % (uncompounded):** 5.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 0 | 3 | 1 | 1.29% | 5.2% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 3 | 1 | 1.29% | 5.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 3 | 1 | 1.29% | 5.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 00:00:00 | 995.70 | 961.87 | 975.13 | Stage2 pullback-breakout RSI=58 vol=1.9x ATR=25.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 00:00:00 | 1046.27 | 966.58 | 1000.74 | T1 booked 50% @ 1046.27 |
| Stop hit — per-position SL triggered | 2024-07-19 00:00:00 | 1013.30 | 968.47 | 1008.21 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-09-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 00:00:00 | 1114.25 | 994.57 | 1063.79 | Stage2 pullback-breakout RSI=60 vol=1.7x ATR=35.60 |
| Stop hit — per-position SL triggered | 2024-09-16 00:00:00 | 1133.55 | 1007.71 | 1108.33 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-10-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 00:00:00 | 1271.90 | 1047.51 | 1184.57 | Stage2 pullback-breakout RSI=68 vol=5.7x ATR=39.65 |
| Stop hit — per-position SL triggered | 2024-11-06 00:00:00 | 1228.45 | 1068.26 | 1233.74 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-01 00:00:00 | 995.70 | 2024-07-15 00:00:00 | 1046.27 | PARTIAL | 0.50 | 5.08% |
| BUY | retest1 | 2024-07-01 00:00:00 | 995.70 | 2024-07-19 00:00:00 | 1013.30 | STOP_HIT | 0.50 | 1.77% |
| BUY | retest1 | 2024-09-02 00:00:00 | 1114.25 | 2024-09-16 00:00:00 | 1133.55 | STOP_HIT | 1.00 | 1.73% |
| BUY | retest1 | 2024-10-23 00:00:00 | 1271.90 | 2024-11-06 00:00:00 | 1228.45 | STOP_HIT | 1.00 | -3.42% |
