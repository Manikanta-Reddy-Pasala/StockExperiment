# Manappuram Finance Ltd. (MANAPPURAM)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 316.00
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 0
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 5.01% / 6.09%
- **Sum % (uncompounded):** 20.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 1 | 1 | 2 | 5.01% | 20.0% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 1 | 1 | 2 | 5.01% | 20.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 1 | 1 | 2 | 5.01% | 20.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 05:30:00 | 266.00 | 231.33 | 260.71 | Stage2 pullback-breakout RSI=55 vol=3.8x ATR=8.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 05:30:00 | 282.20 | 235.18 | 266.29 | T1 booked 50% @ 282.20 |
| Target hit | 2025-09-25 05:30:00 | 281.20 | 243.67 | 285.19 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-11-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 05:30:00 | 281.15 | 253.98 | 277.08 | Stage2 pullback-breakout RSI=54 vol=2.4x ATR=7.70 |
| Stop hit — per-position SL triggered | 2025-11-28 05:30:00 | 284.95 | 256.60 | 280.18 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-04-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 05:30:00 | 295.05 | 272.37 | 269.45 | Stage2 pullback-breakout RSI=68 vol=2.3x ATR=10.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 05:30:00 | 315.33 | 274.95 | 289.81 | T1 booked 50% @ 315.33 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-14 05:30:00 | 266.00 | 2025-09-02 05:30:00 | 282.20 | PARTIAL | 0.50 | 6.09% |
| BUY | retest1 | 2025-08-14 05:30:00 | 266.00 | 2025-09-25 05:30:00 | 281.20 | TARGET_HIT | 0.50 | 5.71% |
| BUY | retest1 | 2025-11-14 05:30:00 | 281.15 | 2025-11-28 05:30:00 | 284.95 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest1 | 2026-04-22 05:30:00 | 295.05 | 2026-05-07 05:30:00 | 315.33 | PARTIAL | 0.50 | 6.87% |
