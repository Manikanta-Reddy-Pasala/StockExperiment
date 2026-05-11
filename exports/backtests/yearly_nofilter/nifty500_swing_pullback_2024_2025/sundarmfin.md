# Sundaram Finance Ltd. (SUNDARMFIN)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 4746.40
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
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -6.08% / -6.22%
- **Sum % (uncompounded):** -18.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -6.08% | -18.3% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -6.08% | -18.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -6.08% | -18.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 00:00:00 | 4646.75 | 4003.19 | 4472.46 | Stage2 pullback-breakout RSI=57 vol=2.1x ATR=192.64 |
| Stop hit — per-position SL triggered | 2024-07-29 00:00:00 | 4357.78 | 4019.78 | 4454.84 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-09-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 00:00:00 | 5339.00 | 4220.88 | 4909.75 | Stage2 pullback-breakout RSI=69 vol=2.7x ATR=215.28 |
| Stop hit — per-position SL triggered | 2024-10-14 00:00:00 | 5106.40 | 4317.25 | 5105.03 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-01-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 00:00:00 | 4698.90 | 4365.40 | 4336.28 | Stage2 pullback-breakout RSI=61 vol=7.9x ATR=240.51 |
| Stop hit — per-position SL triggered | 2025-01-08 00:00:00 | 4338.13 | 4370.44 | 4384.84 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-23 00:00:00 | 4646.75 | 2024-07-29 00:00:00 | 4357.78 | STOP_HIT | 1.00 | -6.22% |
| BUY | retest1 | 2024-09-27 00:00:00 | 5339.00 | 2024-10-14 00:00:00 | 5106.40 | STOP_HIT | 1.00 | -4.36% |
| BUY | retest1 | 2025-01-02 00:00:00 | 4698.90 | 2025-01-08 00:00:00 | 4338.13 | STOP_HIT | 1.00 | -7.68% |
