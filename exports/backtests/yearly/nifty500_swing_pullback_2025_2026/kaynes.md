# Kaynes Technology India Ltd. (KAYNES)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 4507.50
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
- **Avg / median % per leg:** 1.90% / 6.54%
- **Sum % (uncompounded):** 7.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.90% | 7.6% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.90% | 7.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.90% | 7.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 05:30:00 | 6095.00 | 5446.90 | 5790.08 | Stage2 pullback-breakout RSI=63 vol=2.7x ATR=201.58 |
| Stop hit — per-position SL triggered | 2025-07-14 05:30:00 | 6023.00 | 5511.26 | 5991.07 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-07-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 05:30:00 | 6172.00 | 5546.21 | 5837.22 | Stage2 pullback-breakout RSI=61 vol=8.1x ATR=221.45 |
| Stop hit — per-position SL triggered | 2025-08-08 05:30:00 | 5839.83 | 5581.48 | 5966.63 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2025-09-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 05:30:00 | 6600.00 | 5658.19 | 6151.63 | Stage2 pullback-breakout RSI=66 vol=2.7x ATR=215.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 05:30:00 | 7031.32 | 5713.67 | 6412.55 | T1 booked 50% @ 7031.32 |
| Target hit | 2025-10-09 05:30:00 | 7102.50 | 6012.77 | 7159.14 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-30 05:30:00 | 6095.00 | 2025-07-14 05:30:00 | 6023.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest1 | 2025-07-31 05:30:00 | 6172.00 | 2025-08-08 05:30:00 | 5839.83 | STOP_HIT | 1.00 | -5.38% |
| BUY | retest1 | 2025-09-01 05:30:00 | 6600.00 | 2025-09-08 05:30:00 | 7031.32 | PARTIAL | 0.50 | 6.54% |
| BUY | retest1 | 2025-09-01 05:30:00 | 6600.00 | 2025-10-09 05:30:00 | 7102.50 | TARGET_HIT | 0.50 | 7.61% |
