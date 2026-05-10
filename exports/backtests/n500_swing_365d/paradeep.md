# Paradeep Phosphates Ltd. (PARADEEP)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 124.88
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
- **Avg / median % per leg:** 3.17% / 6.75%
- **Sum % (uncompounded):** 12.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.17% | 12.7% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.17% | 12.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.17% | 12.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 05:30:00 | 170.44 | 127.05 | 163.36 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=5.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 05:30:00 | 181.94 | 129.38 | 168.03 | T1 booked 50% @ 181.94 |
| Target hit | 2025-08-14 05:30:00 | 200.07 | 143.02 | 206.73 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-09-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 05:30:00 | 192.73 | 155.46 | 187.04 | Stage2 pullback-breakout RSI=52 vol=5.0x ATR=10.42 |
| Stop hit — per-position SL triggered | 2025-10-09 05:30:00 | 180.56 | 158.85 | 189.26 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-12-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 05:30:00 | 165.15 | 160.47 | 159.26 | Stage2 pullback-breakout RSI=58 vol=4.4x ATR=5.67 |
| Stop hit — per-position SL triggered | 2026-01-06 05:30:00 | 156.65 | 160.55 | 160.23 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-14 05:30:00 | 170.44 | 2025-07-21 05:30:00 | 181.94 | PARTIAL | 0.50 | 6.75% |
| BUY | retest1 | 2025-07-14 05:30:00 | 170.44 | 2025-08-14 05:30:00 | 200.07 | TARGET_HIT | 0.50 | 17.38% |
| BUY | retest1 | 2025-09-24 05:30:00 | 192.73 | 2025-10-09 05:30:00 | 180.56 | STOP_HIT | 1.00 | -6.31% |
| BUY | retest1 | 2025-12-30 05:30:00 | 165.15 | 2026-01-06 05:30:00 | 156.65 | STOP_HIT | 1.00 | -5.15% |
