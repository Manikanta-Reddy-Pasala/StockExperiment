# Redington Ltd. (REDINGTON)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 223.02
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -3.68% / -4.35%
- **Sum % (uncompounded):** -14.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 4 | 0 | -3.68% | -14.7% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -3.68% | -14.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -3.68% | -14.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 05:30:00 | 290.10 | 253.58 | 273.90 | Stage2 pullback-breakout RSI=60 vol=2.0x ATR=12.48 |
| Stop hit — per-position SL triggered | 2025-10-14 05:30:00 | 271.39 | 254.02 | 274.29 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2025-11-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 05:30:00 | 289.95 | 255.53 | 266.31 | Stage2 pullback-breakout RSI=63 vol=10.9x ATR=11.33 |
| Stop hit — per-position SL triggered | 2025-11-20 05:30:00 | 295.05 | 259.09 | 283.41 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-01-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 05:30:00 | 285.80 | 263.95 | 275.95 | Stage2 pullback-breakout RSI=61 vol=5.1x ATR=8.29 |
| Stop hit — per-position SL triggered | 2026-01-09 05:30:00 | 273.37 | 264.41 | 276.75 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2026-02-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 05:30:00 | 280.45 | 263.76 | 259.01 | Stage2 pullback-breakout RSI=62 vol=15.9x ATR=10.65 |
| Stop hit — per-position SL triggered | 2026-03-02 05:30:00 | 264.48 | 263.75 | 259.38 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-10 05:30:00 | 290.10 | 2025-10-14 05:30:00 | 271.39 | STOP_HIT | 1.00 | -6.45% |
| BUY | retest1 | 2025-11-06 05:30:00 | 289.95 | 2025-11-20 05:30:00 | 295.05 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest1 | 2026-01-06 05:30:00 | 285.80 | 2026-01-09 05:30:00 | 273.37 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest1 | 2026-02-27 05:30:00 | 280.45 | 2026-03-02 05:30:00 | 264.48 | STOP_HIT | 1.00 | -5.69% |
