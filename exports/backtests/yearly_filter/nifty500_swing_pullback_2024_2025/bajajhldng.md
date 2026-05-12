# Bajaj Holdings & Investment Ltd. (BAJAJHLDNG)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 10478.00
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
- **Avg / median % per leg:** -1.71% / -0.97%
- **Sum % (uncompounded):** -5.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.71% | -5.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.71% | -5.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.71% | -5.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 00:00:00 | 10793.00 | 8813.47 | 10318.46 | Stage2 pullback-breakout RSI=66 vol=2.0x ATR=295.56 |
| Stop hit — per-position SL triggered | 2024-09-30 00:00:00 | 10349.66 | 8965.89 | 10562.35 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2024-11-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-04 00:00:00 | 10658.25 | 9273.22 | 10382.59 | Stage2 pullback-breakout RSI=59 vol=3.4x ATR=305.87 |
| Stop hit — per-position SL triggered | 2024-11-19 00:00:00 | 10555.30 | 9403.78 | 10553.48 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-12-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 00:00:00 | 11122.00 | 9544.66 | 10598.49 | Stage2 pullback-breakout RSI=64 vol=2.7x ATR=336.47 |
| Stop hit — per-position SL triggered | 2024-12-23 00:00:00 | 11115.15 | 9696.73 | 10933.61 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-18 00:00:00 | 10793.00 | 2024-09-30 00:00:00 | 10349.66 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest1 | 2024-11-04 00:00:00 | 10658.25 | 2024-11-19 00:00:00 | 10555.30 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest1 | 2024-12-09 00:00:00 | 11122.00 | 2024-12-23 00:00:00 | 11115.15 | STOP_HIT | 1.00 | -0.06% |
