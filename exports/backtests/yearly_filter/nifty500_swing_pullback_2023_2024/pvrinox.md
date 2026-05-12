# PVR INOX Ltd. (PVRINOX)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1039.70
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
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 0.09% / 2.97%
- **Sum % (uncompounded):** 0.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.09% | 0.4% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.09% | 0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.09% | 0.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 00:00:00 | 1765.90 | 1632.81 | 1717.67 | Stage2 pullback-breakout RSI=60 vol=3.0x ATR=35.13 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 1713.21 | 1639.33 | 1728.89 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2023-11-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 00:00:00 | 1698.35 | 1640.43 | 1663.13 | Stage2 pullback-breakout RSI=58 vol=2.3x ATR=32.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 00:00:00 | 1763.34 | 1643.25 | 1682.30 | T1 booked 50% @ 1763.34 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 1748.85 | 1656.65 | 1741.80 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-01-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 00:00:00 | 1713.80 | 1658.70 | 1704.62 | Stage2 pullback-breakout RSI=52 vol=1.6x ATR=39.54 |
| Stop hit — per-position SL triggered | 2024-01-05 00:00:00 | 1654.50 | 1658.67 | 1695.94 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-12 00:00:00 | 1765.90 | 2023-10-20 00:00:00 | 1713.21 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest1 | 2023-11-29 00:00:00 | 1698.35 | 2023-12-04 00:00:00 | 1763.34 | PARTIAL | 0.50 | 3.83% |
| BUY | retest1 | 2023-11-29 00:00:00 | 1698.35 | 2023-12-20 00:00:00 | 1748.85 | STOP_HIT | 0.50 | 2.97% |
| BUY | retest1 | 2024-01-03 00:00:00 | 1713.80 | 2024-01-05 00:00:00 | 1654.50 | STOP_HIT | 1.00 | -3.46% |
