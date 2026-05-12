# Kirloskar Oil Eng Ltd. (KIRLOSENG)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 1728.90
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
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -3.27% / -6.14%
- **Sum % (uncompounded):** -9.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 3 | 0 | -3.27% | -9.8% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -3.27% | -9.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | -3.27% | -9.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 00:00:00 | 1275.70 | 988.95 | 1222.81 | Stage2 pullback-breakout RSI=56 vol=2.3x ATR=59.31 |
| Stop hit — per-position SL triggered | 2024-08-27 00:00:00 | 1310.15 | 1018.78 | 1276.43 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 00:00:00 | 1192.95 | 1090.33 | 1143.62 | Stage2 pullback-breakout RSI=55 vol=2.2x ATR=50.74 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 1116.84 | 1093.34 | 1145.55 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2024-12-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 00:00:00 | 1176.40 | 1096.66 | 1124.41 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=48.12 |
| Stop hit — per-position SL triggered | 2024-12-13 00:00:00 | 1104.22 | 1098.76 | 1132.17 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-12 00:00:00 | 1275.70 | 2024-08-27 00:00:00 | 1310.15 | STOP_HIT | 1.00 | 2.70% |
| BUY | retest1 | 2024-11-06 00:00:00 | 1192.95 | 2024-11-13 00:00:00 | 1116.84 | STOP_HIT | 1.00 | -6.38% |
| BUY | retest1 | 2024-12-09 00:00:00 | 1176.40 | 2024-12-13 00:00:00 | 1104.22 | STOP_HIT | 1.00 | -6.14% |
