# HFCL Ltd. (HFCL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 146.61
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
- **Avg / median % per leg:** -2.97% / -4.02%
- **Sum % (uncompounded):** -8.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 3 | 0 | -2.97% | -8.9% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -2.97% | -8.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | -2.97% | -8.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 00:00:00 | 75.95 | 69.90 | 73.33 | Stage2 pullback-breakout RSI=58 vol=1.9x ATR=2.86 |
| Stop hit — per-position SL triggered | 2023-10-12 00:00:00 | 72.90 | 70.35 | 73.89 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-02-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 00:00:00 | 107.30 | 78.04 | 99.31 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=6.46 |
| Stop hit — per-position SL triggered | 2024-03-02 00:00:00 | 108.50 | 81.15 | 106.18 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-04-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 00:00:00 | 100.50 | 85.59 | 96.38 | Stage2 pullback-breakout RSI=58 vol=3.4x ATR=4.04 |
| Stop hit — per-position SL triggered | 2024-05-09 00:00:00 | 94.44 | 86.31 | 96.92 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-27 00:00:00 | 75.95 | 2023-10-12 00:00:00 | 72.90 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest1 | 2024-02-19 00:00:00 | 107.30 | 2024-03-02 00:00:00 | 108.50 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest1 | 2024-04-30 00:00:00 | 100.50 | 2024-05-09 00:00:00 | 94.44 | STOP_HIT | 1.00 | -6.03% |
