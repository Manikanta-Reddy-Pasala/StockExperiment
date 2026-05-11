# ULTRACEMCO (ULTRACEMCO)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 11950.00
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -2.85% / -3.42%
- **Sum % (uncompounded):** -14.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.85% | -14.3% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.85% | -14.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.85% | -14.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 05:30:00 | 11798.30 | 10501.97 | 11555.63 | Stage2 pullback-breakout RSI=63 vol=2.2x ATR=208.21 |
| Stop hit — per-position SL triggered | 2024-10-04 05:30:00 | 11485.99 | 10615.10 | 11694.96 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2024-11-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 05:30:00 | 11457.45 | 10750.15 | 11061.67 | Stage2 pullback-breakout RSI=61 vol=1.7x ATR=261.55 |
| Stop hit — per-position SL triggered | 2024-11-27 05:30:00 | 11065.12 | 10757.68 | 11074.24 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-12-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 05:30:00 | 11852.35 | 10784.00 | 11202.41 | Stage2 pullback-breakout RSI=65 vol=1.6x ATR=292.47 |
| Stop hit — per-position SL triggered | 2024-12-17 05:30:00 | 11774.95 | 10887.10 | 11627.00 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-01-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 05:30:00 | 11420.90 | 10953.49 | 11034.46 | Stage2 pullback-breakout RSI=57 vol=4.7x ATR=303.40 |
| Stop hit — per-position SL triggered | 2025-02-01 05:30:00 | 10965.80 | 10981.73 | 11207.27 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2025-03-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 05:30:00 | 11421.20 | 10974.19 | 10857.17 | Stage2 pullback-breakout RSI=65 vol=1.6x ATR=269.64 |
| Stop hit — per-position SL triggered | 2025-04-07 05:30:00 | 11016.74 | 11009.12 | 11165.99 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-20 05:30:00 | 11798.30 | 2024-10-04 05:30:00 | 11485.99 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest1 | 2024-11-25 05:30:00 | 11457.45 | 2024-11-27 05:30:00 | 11065.12 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest1 | 2024-12-03 05:30:00 | 11852.35 | 2024-12-17 05:30:00 | 11774.95 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest1 | 2025-01-23 05:30:00 | 11420.90 | 2025-02-01 05:30:00 | 10965.80 | STOP_HIT | 1.00 | -3.98% |
| BUY | retest1 | 2025-03-25 05:30:00 | 11421.20 | 2025-04-07 05:30:00 | 11016.74 | STOP_HIT | 1.00 | -3.54% |
