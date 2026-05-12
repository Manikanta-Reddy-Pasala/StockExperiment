# Alkem Laboratories Ltd. (ALKEM)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 5520.00
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
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -1.83% / -3.03%
- **Sum % (uncompounded):** -5.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.83% | -5.5% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.83% | -5.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.83% | -5.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 00:00:00 | 3446.15 | 3240.90 | 3389.07 | Stage2 pullback-breakout RSI=59 vol=3.6x ATR=67.64 |
| Stop hit — per-position SL triggered | 2023-07-13 00:00:00 | 3511.15 | 3266.23 | 3464.39 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-13 00:00:00 | 3744.75 | 3451.40 | 3705.06 | Stage2 pullback-breakout RSI=53 vol=1.9x ATR=75.62 |
| Stop hit — per-position SL triggered | 2023-09-20 00:00:00 | 3631.32 | 3460.46 | 3697.32 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-02-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 00:00:00 | 5311.15 | 4156.68 | 5020.79 | Stage2 pullback-breakout RSI=67 vol=2.6x ATR=153.86 |
| Stop hit — per-position SL triggered | 2024-02-12 00:00:00 | 5080.36 | 4178.09 | 5059.50 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-05-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 00:00:00 | 5148.95 | 4514.37 | 4863.55 | Stage2 pullback-breakout RSI=64 vol=2.7x ATR=138.11 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-28 00:00:00 | 3446.15 | 2023-07-13 00:00:00 | 3511.15 | STOP_HIT | 1.00 | 1.89% |
| BUY | retest1 | 2023-09-13 00:00:00 | 3744.75 | 2023-09-20 00:00:00 | 3631.32 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest1 | 2024-02-08 00:00:00 | 5311.15 | 2024-02-12 00:00:00 | 5080.36 | STOP_HIT | 1.00 | -4.35% |
