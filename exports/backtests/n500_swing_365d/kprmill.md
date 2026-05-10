# K.P.R. Mill Ltd. (KPRMILL)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 955.75
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -2.33% / -1.39%
- **Sum % (uncompounded):** -9.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.33% | -9.3% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.33% | -9.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.33% | -9.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 05:30:00 | 1189.60 | 999.79 | 1130.36 | Stage2 pullback-breakout RSI=64 vol=1.7x ATR=42.00 |
| Stop hit — per-position SL triggered | 2025-07-18 05:30:00 | 1180.30 | 1018.16 | 1169.89 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-09-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 05:30:00 | 1090.65 | 1026.02 | 1019.71 | Stage2 pullback-breakout RSI=65 vol=4.0x ATR=37.05 |
| Stop hit — per-position SL triggered | 2025-09-30 05:30:00 | 1064.85 | 1033.39 | 1069.80 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 05:30:00 | 1082.60 | 1033.24 | 1041.23 | Stage2 pullback-breakout RSI=59 vol=5.6x ATR=34.55 |
| Stop hit — per-position SL triggered | 2025-10-27 05:30:00 | 1030.78 | 1033.62 | 1043.38 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2025-11-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 05:30:00 | 1094.20 | 1037.35 | 1061.76 | Stage2 pullback-breakout RSI=57 vol=2.4x ATR=40.15 |
| Stop hit — per-position SL triggered | 2025-11-25 05:30:00 | 1079.00 | 1042.79 | 1082.58 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-04 05:30:00 | 1189.60 | 2025-07-18 05:30:00 | 1180.30 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest1 | 2025-09-16 05:30:00 | 1090.65 | 2025-09-30 05:30:00 | 1064.85 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest1 | 2025-10-23 05:30:00 | 1082.60 | 2025-10-27 05:30:00 | 1030.78 | STOP_HIT | 1.00 | -4.79% |
| BUY | retest1 | 2025-11-11 05:30:00 | 1094.20 | 2025-11-25 05:30:00 | 1079.00 | STOP_HIT | 1.00 | -1.39% |
