# Manappuram Finance Ltd. (MANAPPURAM)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -2.38% / -3.02%
- **Sum % (uncompounded):** -7.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 3 | 0 | -2.38% | -7.1% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -2.38% | -7.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | -2.38% | -7.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 05:30:00 | 216.80 | 184.76 | 207.42 | Stage2 pullback-breakout RSI=60 vol=2.0x ATR=7.47 |
| Stop hit — per-position SL triggered | 2024-09-05 05:30:00 | 210.25 | 187.55 | 211.29 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-09-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 05:30:00 | 211.33 | 188.60 | 208.80 | Stage2 pullback-breakout RSI=53 vol=2.4x ATR=6.33 |
| Stop hit — per-position SL triggered | 2024-09-20 05:30:00 | 201.83 | 189.61 | 208.87 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2025-02-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 05:30:00 | 202.76 | 184.01 | 194.83 | Stage2 pullback-breakout RSI=56 vol=2.4x ATR=10.34 |
| Stop hit — per-position SL triggered | 2025-03-06 05:30:00 | 203.54 | 185.81 | 199.75 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-22 05:30:00 | 216.80 | 2024-09-05 05:30:00 | 210.25 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest1 | 2024-09-13 05:30:00 | 211.33 | 2024-09-20 05:30:00 | 201.83 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest1 | 2025-02-19 05:30:00 | 202.76 | 2025-03-06 05:30:00 | 203.54 | STOP_HIT | 1.00 | 0.38% |
