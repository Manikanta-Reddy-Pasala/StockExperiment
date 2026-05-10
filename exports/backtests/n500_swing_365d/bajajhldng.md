# Bajaj Holdings & Investment Ltd. (BAJAJHLDNG)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 10632.00
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
- **Avg / median % per leg:** -4.15% / -4.31%
- **Sum % (uncompounded):** -12.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.15% | -12.5% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.15% | -12.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.15% | -12.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 05:30:00 | 14523.00 | 12557.57 | 13978.83 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=417.77 |
| Stop hit — per-position SL triggered | 2025-08-19 05:30:00 | 13896.35 | 12569.78 | 13960.37 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2025-10-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 05:30:00 | 13144.00 | 12639.96 | 12563.23 | Stage2 pullback-breakout RSI=61 vol=1.9x ATR=326.56 |
| Stop hit — per-position SL triggered | 2025-10-29 05:30:00 | 12654.15 | 12649.12 | 12659.18 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2025-11-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 05:30:00 | 13139.00 | 12638.12 | 12575.70 | Stage2 pullback-breakout RSI=59 vol=9.4x ATR=386.48 |
| Stop hit — per-position SL triggered | 2025-11-07 05:30:00 | 12559.28 | 12637.54 | 12576.11 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-18 05:30:00 | 14523.00 | 2025-08-19 05:30:00 | 13896.35 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest1 | 2025-10-23 05:30:00 | 13144.00 | 2025-10-29 05:30:00 | 12654.15 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest1 | 2025-11-06 05:30:00 | 13139.00 | 2025-11-07 05:30:00 | 12559.28 | STOP_HIT | 1.00 | -4.41% |
