# SBILIFE (SBILIFE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1872.10
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
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -1.40% / -2.13%
- **Sum % (uncompounded):** -5.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.40% | -5.6% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.40% | -5.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.40% | -5.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 00:00:00 | 1349.35 | 1232.01 | 1292.74 | Stage2 pullback-breakout RSI=67 vol=2.6x ATR=29.13 |
| Stop hit — per-position SL triggered | 2023-08-11 00:00:00 | 1305.65 | 1234.95 | 1302.52 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2023-09-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 00:00:00 | 1347.50 | 1245.96 | 1307.56 | Stage2 pullback-breakout RSI=64 vol=1.6x ATR=25.91 |
| Stop hit — per-position SL triggered | 2023-09-22 00:00:00 | 1318.85 | 1255.66 | 1332.25 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-10-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 00:00:00 | 1367.85 | 1268.52 | 1324.90 | Stage2 pullback-breakout RSI=62 vol=2.2x ATR=27.96 |
| Stop hit — per-position SL triggered | 2023-11-06 00:00:00 | 1325.90 | 1271.13 | 1328.18 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2023-11-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 00:00:00 | 1413.95 | 1278.53 | 1347.42 | Stage2 pullback-breakout RSI=67 vol=3.3x ATR=29.13 |
| Stop hit — per-position SL triggered | 2023-12-04 00:00:00 | 1454.00 | 1291.96 | 1394.95 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-08 00:00:00 | 1349.35 | 2023-08-11 00:00:00 | 1305.65 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest1 | 2023-09-07 00:00:00 | 1347.50 | 2023-09-22 00:00:00 | 1318.85 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest1 | 2023-10-31 00:00:00 | 1367.85 | 2023-11-06 00:00:00 | 1325.90 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest1 | 2023-11-17 00:00:00 | 1413.95 | 2023-12-04 00:00:00 | 1454.00 | STOP_HIT | 1.00 | 2.83% |
