# Prestige Estates Projects Ltd. (PRESTIGE)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 1507.30
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
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 1
- **Target hits / Stop hits / Partials:** 0 / 2 / 0
- **Avg / median % per leg:** -1.99% / 2.51%
- **Sum % (uncompounded):** -3.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 1 | 50.0% | 0 | 2 | 0 | -1.99% | -4.0% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -1.99% | -4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | -1.99% | -4.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 00:00:00 | 1813.15 | 1433.18 | 1752.43 | Stage2 pullback-breakout RSI=55 vol=9.9x ATR=81.12 |
| Stop hit — per-position SL triggered | 2024-09-13 00:00:00 | 1858.60 | 1470.16 | 1799.50 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 1729.95 | 1567.18 | 1640.94 | Stage2 pullback-breakout RSI=57 vol=2.3x ATR=74.86 |
| Stop hit — per-position SL triggered | 2024-12-02 00:00:00 | 1617.66 | 1572.45 | 1654.87 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2024-12-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 00:00:00 | 1830.75 | 1589.75 | 1720.66 | Stage2 pullback-breakout RSI=64 vol=2.9x ATR=61.75 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-30 00:00:00 | 1813.15 | 2024-09-13 00:00:00 | 1858.60 | STOP_HIT | 1.00 | 2.51% |
| BUY | retest1 | 2024-11-25 00:00:00 | 1729.95 | 2024-12-02 00:00:00 | 1617.66 | STOP_HIT | 1.00 | -6.49% |
