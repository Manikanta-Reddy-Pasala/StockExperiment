# Narayana Hrudayalaya Ltd. (NH)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 1820.30
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
- **Avg / median % per leg:** -1.70% / -2.11%
- **Sum % (uncompounded):** -6.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.70% | -6.8% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.70% | -6.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.70% | -6.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 00:00:00 | 1256.60 | 1214.35 | 1219.35 | Stage2 pullback-breakout RSI=59 vol=3.0x ATR=32.82 |
| Stop hit — per-position SL triggered | 2024-09-04 00:00:00 | 1285.10 | 1219.88 | 1253.44 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-10-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 00:00:00 | 1268.80 | 1230.23 | 1244.44 | Stage2 pullback-breakout RSI=55 vol=1.5x ATR=35.03 |
| Stop hit — per-position SL triggered | 2024-10-28 00:00:00 | 1242.05 | 1233.05 | 1251.35 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-10-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 00:00:00 | 1272.80 | 1233.32 | 1249.37 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=35.70 |
| Stop hit — per-position SL triggered | 2024-11-04 00:00:00 | 1219.25 | 1233.25 | 1245.64 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 00:00:00 | 1316.50 | 1239.22 | 1266.09 | Stage2 pullback-breakout RSI=62 vol=3.2x ATR=34.57 |
| Stop hit — per-position SL triggered | 2024-12-18 00:00:00 | 1280.50 | 1244.94 | 1285.23 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-21 00:00:00 | 1256.60 | 2024-09-04 00:00:00 | 1285.10 | STOP_HIT | 1.00 | 2.27% |
| BUY | retest1 | 2024-10-14 00:00:00 | 1268.80 | 2024-10-28 00:00:00 | 1242.05 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest1 | 2024-10-31 00:00:00 | 1272.80 | 2024-11-04 00:00:00 | 1219.25 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest1 | 2024-12-04 00:00:00 | 1316.50 | 2024-12-18 00:00:00 | 1280.50 | STOP_HIT | 1.00 | -2.73% |
