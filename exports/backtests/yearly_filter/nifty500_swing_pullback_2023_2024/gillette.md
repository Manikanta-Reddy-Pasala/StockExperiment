# Gillette India Ltd. (GILLETTE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 8017.00
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
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 3
- **Avg / median % per leg:** 2.52% / 4.53%
- **Sum % (uncompounded):** 15.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 0 | 3 | 3 | 2.52% | 15.1% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 0 | 3 | 3 | 2.52% | 15.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 3 | 3 | 2.52% | 15.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 00:00:00 | 5875.70 | 5064.08 | 5565.79 | Stage2 pullback-breakout RSI=68 vol=4.1x ATR=143.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 00:00:00 | 6161.97 | 5115.21 | 5738.46 | T1 booked 50% @ 6161.97 |
| Stop hit — per-position SL triggered | 2023-09-22 00:00:00 | 5995.55 | 5167.27 | 5909.09 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-12-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 00:00:00 | 6550.00 | 5688.78 | 6229.85 | Stage2 pullback-breakout RSI=64 vol=5.6x ATR=148.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 00:00:00 | 6846.78 | 5752.07 | 6444.20 | T1 booked 50% @ 6846.78 |
| Stop hit — per-position SL triggered | 2024-01-16 00:00:00 | 6550.00 | 5812.87 | 6597.67 | SL hit (bars_held=13) |

### Cycle 3 — BUY (started 2024-01-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 00:00:00 | 6689.05 | 5877.35 | 6577.13 | Stage2 pullback-breakout RSI=56 vol=1.9x ATR=221.23 |
| Stop hit — per-position SL triggered | 2024-02-13 00:00:00 | 6566.10 | 5955.36 | 6657.03 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-04-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 00:00:00 | 6667.90 | 6178.33 | 6443.98 | Stage2 pullback-breakout RSI=59 vol=11.4x ATR=184.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-08 00:00:00 | 7036.53 | 6207.31 | 6577.14 | T1 booked 50% @ 7036.53 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-06 00:00:00 | 5875.70 | 2023-09-14 00:00:00 | 6161.97 | PARTIAL | 0.50 | 4.87% |
| BUY | retest1 | 2023-09-06 00:00:00 | 5875.70 | 2023-09-22 00:00:00 | 5995.55 | STOP_HIT | 0.50 | 2.04% |
| BUY | retest1 | 2023-12-28 00:00:00 | 6550.00 | 2024-01-08 00:00:00 | 6846.78 | PARTIAL | 0.50 | 4.53% |
| BUY | retest1 | 2023-12-28 00:00:00 | 6550.00 | 2024-01-16 00:00:00 | 6550.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-30 00:00:00 | 6689.05 | 2024-02-13 00:00:00 | 6566.10 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest1 | 2024-04-30 00:00:00 | 6667.90 | 2024-05-08 00:00:00 | 7036.53 | PARTIAL | 0.50 | 5.53% |
