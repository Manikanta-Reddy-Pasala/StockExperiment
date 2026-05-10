# J.B. Chemicals & Pharmaceuticals Ltd. (JBCHEPHARM)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 2149.80
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 0.58% / 0.00%
- **Sum % (uncompounded):** 3.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.58% | 3.5% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.58% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.58% | 3.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 05:30:00 | 1821.70 | 1723.31 | 1766.78 | Stage2 pullback-breakout RSI=65 vol=2.0x ATR=33.47 |
| Stop hit — per-position SL triggered | 2025-12-17 05:30:00 | 1771.50 | 1730.14 | 1786.87 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2026-01-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 05:30:00 | 1850.40 | 1738.66 | 1805.83 | Stage2 pullback-breakout RSI=66 vol=1.8x ATR=30.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 05:30:00 | 1911.85 | 1742.96 | 1826.86 | T1 booked 50% @ 1911.85 |
| Stop hit — per-position SL triggered | 2026-01-12 05:30:00 | 1850.40 | 1747.33 | 1843.11 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2026-02-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 05:30:00 | 1978.70 | 1777.16 | 1890.21 | Stage2 pullback-breakout RSI=64 vol=4.7x ATR=47.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 05:30:00 | 2073.98 | 1790.97 | 1948.35 | T1 booked 50% @ 2073.98 |
| Stop hit — per-position SL triggered | 2026-03-02 05:30:00 | 1978.70 | 1802.09 | 1989.78 | SL hit (bars_held=10) |

### Cycle 4 — BUY (started 2026-03-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 05:30:00 | 2117.40 | 1815.30 | 2023.55 | Stage2 pullback-breakout RSI=69 vol=1.9x ATR=50.47 |
| Stop hit — per-position SL triggered | 2026-03-24 05:30:00 | 2076.80 | 1843.05 | 2074.19 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-12-04 05:30:00 | 1821.70 | 2025-12-17 05:30:00 | 1771.50 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest1 | 2026-01-02 05:30:00 | 1850.40 | 2026-01-07 05:30:00 | 1911.85 | PARTIAL | 0.50 | 3.32% |
| BUY | retest1 | 2026-01-02 05:30:00 | 1850.40 | 2026-01-12 05:30:00 | 1850.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 05:30:00 | 1978.70 | 2026-02-24 05:30:00 | 2073.98 | PARTIAL | 0.50 | 4.82% |
| BUY | retest1 | 2026-02-16 05:30:00 | 1978.70 | 2026-03-02 05:30:00 | 1978.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 05:30:00 | 2117.40 | 2026-03-24 05:30:00 | 2076.80 | STOP_HIT | 1.00 | -1.92% |
