# Jyoti CNC Automation Ltd. (JYOTICNC)

## Backtest Summary

- **Window:** 2024-01-16 00:00:00 → 2026-05-11 00:00:00 (573 bars)
- **Last close:** 747.65
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 1
- **Avg / median % per leg:** 5.34% / 9.91%
- **Sum % (uncompounded):** 16.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 5.34% | 16.0% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 5.34% | 16.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 5.34% | 16.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 00:00:00 | 1115.40 | 969.05 | 1076.46 | Stage2 pullback-breakout RSI=55 vol=2.0x ATR=46.36 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 1045.86 | 975.73 | 1086.30 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2024-11-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 00:00:00 | 1167.90 | 980.20 | 1097.34 | Stage2 pullback-breakout RSI=59 vol=4.1x ATR=57.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 00:00:00 | 1283.67 | 1000.27 | 1178.39 | T1 booked 50% @ 1283.67 |
| Target hit | 2025-01-09 00:00:00 | 1312.15 | 1081.85 | 1338.84 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-11-06 00:00:00 | 1115.40 | 2024-11-13 00:00:00 | 1045.86 | STOP_HIT | 1.00 | -6.23% |
| BUY | retest1 | 2024-11-19 00:00:00 | 1167.90 | 2024-12-02 00:00:00 | 1283.67 | PARTIAL | 0.50 | 9.91% |
| BUY | retest1 | 2024-11-19 00:00:00 | 1167.90 | 2025-01-09 00:00:00 | 1312.15 | TARGET_HIT | 0.50 | 12.35% |
