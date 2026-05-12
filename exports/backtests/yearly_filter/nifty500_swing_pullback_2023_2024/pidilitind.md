# Pidilite Industries Ltd. (PIDILITIND)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1445.80
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
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 0.27% / 0.00%
- **Sum % (uncompounded):** 1.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.27% | 1.1% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.27% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.27% | 1.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-03-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-06 00:00:00 | 1415.78 | 1296.20 | 1357.32 | Stage2 pullback-breakout RSI=65 vol=4.2x ATR=29.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 00:00:00 | 1474.16 | 1297.56 | 1364.53 | T1 booked 50% @ 1474.16 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 1415.78 | 1301.49 | 1381.39 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-03-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 00:00:00 | 1500.80 | 1313.33 | 1424.38 | Stage2 pullback-breakout RSI=70 vol=1.9x ATR=33.87 |
| Stop hit — per-position SL triggered | 2024-04-10 00:00:00 | 1500.05 | 1331.66 | 1475.98 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-04-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 00:00:00 | 1524.50 | 1346.94 | 1474.43 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=30.54 |
| Stop hit — per-position SL triggered | 2024-05-03 00:00:00 | 1478.69 | 1349.68 | 1476.30 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-03-06 00:00:00 | 1415.78 | 2024-03-07 00:00:00 | 1474.16 | PARTIAL | 0.50 | 4.12% |
| BUY | retest1 | 2024-03-06 00:00:00 | 1415.78 | 2024-03-13 00:00:00 | 1415.78 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-26 00:00:00 | 1500.80 | 2024-04-10 00:00:00 | 1500.05 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest1 | 2024-04-30 00:00:00 | 1524.50 | 2024-05-03 00:00:00 | 1478.69 | STOP_HIT | 1.00 | -3.00% |
