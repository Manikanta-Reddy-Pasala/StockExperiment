# Aurobindo Pharma Ltd. (AUROPHARMA)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 1487.30
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 3.64% / 5.35%
- **Sum % (uncompounded):** 14.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.64% | 14.6% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.64% | 14.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.64% | 14.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 05:30:00 | 1303.55 | 1077.55 | 1231.44 | Stage2 pullback-breakout RSI=67 vol=1.7x ATR=34.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 05:30:00 | 1373.25 | 1092.41 | 1278.23 | T1 booked 50% @ 1373.25 |
| Target hit | 2024-09-09 05:30:00 | 1518.50 | 1214.10 | 1520.02 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-09-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 05:30:00 | 1561.45 | 1223.41 | 1522.75 | Stage2 pullback-breakout RSI=64 vol=3.0x ATR=36.93 |
| Stop hit — per-position SL triggered | 2024-09-20 05:30:00 | 1506.06 | 1241.90 | 1529.27 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-12-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 05:30:00 | 1334.50 | 1285.99 | 1264.80 | Stage2 pullback-breakout RSI=65 vol=2.2x ATR=33.24 |
| Stop hit — per-position SL triggered | 2025-01-08 05:30:00 | 1284.63 | 1287.88 | 1287.29 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-05 05:30:00 | 1303.55 | 2024-07-15 05:30:00 | 1373.25 | PARTIAL | 0.50 | 5.35% |
| BUY | retest1 | 2024-07-05 05:30:00 | 1303.55 | 2024-09-09 05:30:00 | 1518.50 | TARGET_HIT | 0.50 | 16.49% |
| BUY | retest1 | 2024-09-12 05:30:00 | 1561.45 | 2024-09-20 05:30:00 | 1506.06 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest1 | 2024-12-31 05:30:00 | 1334.50 | 2025-01-08 05:30:00 | 1284.63 | STOP_HIT | 1.00 | -3.74% |
