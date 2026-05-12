# Action Construction Equipment Ltd. (ACE)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 923.80
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -1.51% / -4.37%
- **Sum % (uncompounded):** -6.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.51% | -6.0% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.51% | -6.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.51% | -6.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 00:00:00 | 1366.05 | 1244.52 | 1290.11 | Stage2 pullback-breakout RSI=61 vol=3.8x ATR=39.79 |
| Stop hit — per-position SL triggered | 2024-09-19 00:00:00 | 1306.36 | 1246.35 | 1298.56 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2024-09-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 00:00:00 | 1398.40 | 1247.86 | 1308.07 | Stage2 pullback-breakout RSI=63 vol=4.6x ATR=47.37 |
| Stop hit — per-position SL triggered | 2024-10-04 00:00:00 | 1327.35 | 1261.08 | 1358.58 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2024-10-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 00:00:00 | 1418.40 | 1267.18 | 1352.27 | Stage2 pullback-breakout RSI=61 vol=3.4x ATR=43.49 |
| Stop hit — per-position SL triggered | 2024-10-18 00:00:00 | 1353.16 | 1269.49 | 1357.95 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-12-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 00:00:00 | 1354.00 | 1272.60 | 1285.77 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=54.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 00:00:00 | 1462.55 | 1285.59 | 1362.07 | T1 booked 50% @ 1462.55 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-17 00:00:00 | 1366.05 | 2024-09-19 00:00:00 | 1306.36 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest1 | 2024-09-20 00:00:00 | 1398.40 | 2024-10-04 00:00:00 | 1327.35 | STOP_HIT | 1.00 | -5.08% |
| BUY | retest1 | 2024-10-16 00:00:00 | 1418.40 | 2024-10-18 00:00:00 | 1353.16 | STOP_HIT | 1.00 | -4.60% |
| BUY | retest1 | 2024-12-02 00:00:00 | 1354.00 | 2024-12-17 00:00:00 | 1462.55 | PARTIAL | 0.50 | 8.02% |
