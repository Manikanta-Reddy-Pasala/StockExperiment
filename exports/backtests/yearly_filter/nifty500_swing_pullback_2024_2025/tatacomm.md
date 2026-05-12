# Tata Communications Ltd. (TATACOMM)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 1592.30
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
- **Avg / median % per leg:** -3.49% / -3.67%
- **Sum % (uncompounded):** -10.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.49% | -10.5% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.49% | -10.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.49% | -10.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 00:00:00 | 1889.50 | 1823.47 | 1844.13 | Stage2 pullback-breakout RSI=60 vol=2.4x ATR=46.19 |
| Stop hit — per-position SL triggered | 2024-07-10 00:00:00 | 1820.22 | 1826.12 | 1854.22 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2024-07-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 00:00:00 | 1907.50 | 1828.87 | 1852.77 | Stage2 pullback-breakout RSI=59 vol=2.3x ATR=50.19 |
| Stop hit — per-position SL triggered | 2024-08-13 00:00:00 | 1849.45 | 1835.34 | 1875.05 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-09-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 00:00:00 | 2127.70 | 1869.35 | 1995.34 | Stage2 pullback-breakout RSI=69 vol=4.4x ATR=53.46 |
| Stop hit — per-position SL triggered | 2024-10-07 00:00:00 | 2047.51 | 1885.57 | 2048.69 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-02 00:00:00 | 1889.50 | 2024-07-10 00:00:00 | 1820.22 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest1 | 2024-07-30 00:00:00 | 1907.50 | 2024-08-13 00:00:00 | 1849.45 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest1 | 2024-09-25 00:00:00 | 2127.70 | 2024-10-07 00:00:00 | 2047.51 | STOP_HIT | 1.00 | -3.77% |
