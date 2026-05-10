# Narayana Hrudayalaya Ltd. (NH)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1820.30
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
- **Avg / median % per leg:** -3.09% / -3.66%
- **Sum % (uncompounded):** -9.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.09% | -9.3% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.09% | -9.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.09% | -9.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 05:30:00 | 1830.30 | 1711.51 | 1774.40 | Stage2 pullback-breakout RSI=60 vol=3.2x ATR=44.67 |
| Stop hit — per-position SL triggered | 2025-11-11 05:30:00 | 1763.30 | 1714.99 | 1782.20 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2026-01-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 05:30:00 | 1934.40 | 1771.96 | 1885.48 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=56.22 |
| Stop hit — per-position SL triggered | 2026-01-16 05:30:00 | 1916.80 | 1785.19 | 1901.45 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-02-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 05:30:00 | 1864.60 | 1783.43 | 1794.82 | Stage2 pullback-breakout RSI=57 vol=1.7x ATR=58.42 |
| Stop hit — per-position SL triggered | 2026-02-16 05:30:00 | 1776.96 | 1785.12 | 1806.44 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-11-04 05:30:00 | 1830.30 | 2025-11-11 05:30:00 | 1763.30 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest1 | 2026-01-01 05:30:00 | 1934.40 | 2026-01-16 05:30:00 | 1916.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest1 | 2026-02-11 05:30:00 | 1864.60 | 2026-02-16 05:30:00 | 1776.96 | STOP_HIT | 1.00 | -4.70% |
