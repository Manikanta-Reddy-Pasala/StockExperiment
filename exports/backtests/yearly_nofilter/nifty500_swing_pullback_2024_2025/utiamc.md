# UTI Asset Management Company Ltd. (UTIAMC)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 966.05
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -4.52% / -5.02%
- **Sum % (uncompounded):** -22.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -4.52% | -22.6% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -4.52% | -22.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -4.52% | -22.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 00:00:00 | 1051.40 | 907.31 | 1024.86 | Stage2 pullback-breakout RSI=57 vol=2.0x ATR=34.27 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 999.99 | 914.32 | 1024.96 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2024-10-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 00:00:00 | 1329.00 | 1026.15 | 1238.08 | Stage2 pullback-breakout RSI=67 vol=6.5x ATR=47.75 |
| Stop hit — per-position SL triggered | 2024-10-22 00:00:00 | 1257.38 | 1035.73 | 1247.85 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-10-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 00:00:00 | 1289.30 | 1046.19 | 1234.40 | Stage2 pullback-breakout RSI=58 vol=2.8x ATR=60.13 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 1263.20 | 1072.96 | 1290.61 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-12-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 00:00:00 | 1366.15 | 1107.10 | 1306.98 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=45.71 |
| Stop hit — per-position SL triggered | 2024-12-19 00:00:00 | 1297.59 | 1125.92 | 1328.68 | SL hit (bars_held=8) |

### Cycle 5 — BUY (started 2024-12-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 00:00:00 | 1352.00 | 1133.41 | 1297.49 | Stage2 pullback-breakout RSI=58 vol=2.3x ATR=47.72 |
| Stop hit — per-position SL triggered | 2025-01-06 00:00:00 | 1280.42 | 1143.94 | 1317.00 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-26 00:00:00 | 1051.40 | 2024-08-05 00:00:00 | 999.99 | STOP_HIT | 1.00 | -4.89% |
| BUY | retest1 | 2024-10-16 00:00:00 | 1329.00 | 2024-10-22 00:00:00 | 1257.38 | STOP_HIT | 1.00 | -5.39% |
| BUY | retest1 | 2024-10-30 00:00:00 | 1289.30 | 2024-11-13 00:00:00 | 1263.20 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest1 | 2024-12-09 00:00:00 | 1366.15 | 2024-12-19 00:00:00 | 1297.59 | STOP_HIT | 1.00 | -5.02% |
| BUY | retest1 | 2024-12-30 00:00:00 | 1352.00 | 2025-01-06 00:00:00 | 1280.42 | STOP_HIT | 1.00 | -5.29% |
