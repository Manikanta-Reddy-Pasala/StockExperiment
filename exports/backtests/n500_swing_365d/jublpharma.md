# Jubilant Pharmova Ltd. (JUBLPHARMA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1008.20
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
- **Avg / median % per leg:** -4.08% / -4.31%
- **Sum % (uncompounded):** -16.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.08% | -16.3% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.08% | -16.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.08% | -16.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 05:30:00 | 1232.00 | 1036.24 | 1178.97 | Stage2 pullback-breakout RSI=66 vol=1.8x ATR=35.38 |
| Stop hit — per-position SL triggered | 2025-07-24 05:30:00 | 1178.93 | 1042.69 | 1185.61 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2025-07-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 05:30:00 | 1236.30 | 1048.30 | 1186.23 | Stage2 pullback-breakout RSI=62 vol=5.1x ATR=46.64 |
| Stop hit — per-position SL triggered | 2025-08-01 05:30:00 | 1166.34 | 1050.83 | 1184.26 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2025-09-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 05:30:00 | 1106.00 | 1055.81 | 1077.27 | Stage2 pullback-breakout RSI=55 vol=4.6x ATR=38.04 |
| Stop hit — per-position SL triggered | 2025-09-22 05:30:00 | 1091.10 | 1061.47 | 1099.72 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-11-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 05:30:00 | 1169.80 | 1071.42 | 1113.78 | Stage2 pullback-breakout RSI=63 vol=4.2x ATR=39.06 |
| Stop hit — per-position SL triggered | 2025-11-07 05:30:00 | 1111.22 | 1073.93 | 1124.37 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-18 05:30:00 | 1232.00 | 2025-07-24 05:30:00 | 1178.93 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest1 | 2025-07-30 05:30:00 | 1236.30 | 2025-08-01 05:30:00 | 1166.34 | STOP_HIT | 1.00 | -5.66% |
| BUY | retest1 | 2025-09-08 05:30:00 | 1106.00 | 2025-09-22 05:30:00 | 1091.10 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest1 | 2025-11-03 05:30:00 | 1169.80 | 2025-11-07 05:30:00 | 1111.22 | STOP_HIT | 1.00 | -5.01% |
