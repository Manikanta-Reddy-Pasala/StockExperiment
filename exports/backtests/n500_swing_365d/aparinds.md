# Apar Industries Ltd. (APARINDS)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 12810.00
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** 1.08% / 1.97%
- **Sum % (uncompounded):** 7.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 0 | 5 | 2 | 1.08% | 7.6% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 0 | 5 | 2 | 1.08% | 7.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 0 | 5 | 2 | 1.08% | 7.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 05:30:00 | 9681.50 | 7987.65 | 8896.50 | Stage2 pullback-breakout RSI=67 vol=8.9x ATR=371.71 |
| Stop hit — per-position SL triggered | 2025-07-31 05:30:00 | 9123.93 | 8013.37 | 8964.16 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2025-09-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 05:30:00 | 8493.00 | 8077.31 | 8149.55 | Stage2 pullback-breakout RSI=57 vol=4.5x ATR=254.44 |
| Stop hit — per-position SL triggered | 2025-09-25 05:30:00 | 8734.50 | 8144.53 | 8541.37 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 05:30:00 | 8784.50 | 8182.99 | 8488.64 | Stage2 pullback-breakout RSI=59 vol=3.5x ATR=253.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 05:30:00 | 9290.90 | 8234.50 | 8719.99 | T1 booked 50% @ 9290.90 |
| Stop hit — per-position SL triggered | 2025-10-31 05:30:00 | 8784.50 | 8248.86 | 8761.12 | SL hit (bars_held=10) |

### Cycle 4 — BUY (started 2025-11-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 05:30:00 | 8980.00 | 8264.58 | 8631.96 | Stage2 pullback-breakout RSI=58 vol=2.2x ATR=321.17 |
| Stop hit — per-position SL triggered | 2025-11-26 05:30:00 | 9156.50 | 8345.98 | 8936.63 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2026-03-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 05:30:00 | 10608.00 | 8800.30 | 9841.07 | Stage2 pullback-breakout RSI=63 vol=2.3x ATR=501.23 |
| Stop hit — per-position SL triggered | 2026-03-30 05:30:00 | 9856.16 | 8811.05 | 9844.87 | SL hit (bars_held=1) |

### Cycle 6 — BUY (started 2026-04-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 05:30:00 | 10767.00 | 8889.85 | 10013.61 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=530.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 05:30:00 | 11827.11 | 9036.14 | 10655.98 | T1 booked 50% @ 11827.11 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-29 05:30:00 | 9681.50 | 2025-07-31 05:30:00 | 9123.93 | STOP_HIT | 1.00 | -5.76% |
| BUY | retest1 | 2025-09-10 05:30:00 | 8493.00 | 2025-09-25 05:30:00 | 8734.50 | STOP_HIT | 1.00 | 2.84% |
| BUY | retest1 | 2025-10-16 05:30:00 | 8784.50 | 2025-10-29 05:30:00 | 9290.90 | PARTIAL | 0.50 | 5.76% |
| BUY | retest1 | 2025-10-16 05:30:00 | 8784.50 | 2025-10-31 05:30:00 | 8784.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-12 05:30:00 | 8980.00 | 2025-11-26 05:30:00 | 9156.50 | STOP_HIT | 1.00 | 1.97% |
| BUY | retest1 | 2026-03-27 05:30:00 | 10608.00 | 2026-03-30 05:30:00 | 9856.16 | STOP_HIT | 1.00 | -7.09% |
| BUY | retest1 | 2026-04-09 05:30:00 | 10767.00 | 2026-04-20 05:30:00 | 11827.11 | PARTIAL | 0.50 | 9.85% |
