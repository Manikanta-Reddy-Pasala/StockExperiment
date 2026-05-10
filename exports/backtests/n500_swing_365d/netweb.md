# Netweb Technologies India Ltd. (NETWEB)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 4422.20
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
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 4
- **Avg / median % per leg:** 12.05% / 8.77%
- **Sum % (uncompounded):** 84.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 1 | 2 | 4 | 12.05% | 84.3% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 1 | 2 | 4 | 12.05% | 84.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 1 | 2 | 4 | 12.05% | 84.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 05:30:00 | 2526.90 | 2049.05 | 2204.85 | Stage2 pullback-breakout RSI=70 vol=6.0x ATR=117.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 05:30:00 | 2761.83 | 2065.24 | 2326.68 | T1 booked 50% @ 2761.83 |
| Target hit | 2025-10-23 05:30:00 | 3743.70 | 2495.52 | 3771.98 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2026-01-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 05:30:00 | 3274.60 | 2821.46 | 3179.50 | Stage2 pullback-breakout RSI=54 vol=9.5x ATR=134.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 05:30:00 | 3543.66 | 2865.32 | 3268.21 | T1 booked 50% @ 3543.66 |
| Stop hit — per-position SL triggered | 2026-01-20 05:30:00 | 3274.60 | 2868.99 | 3265.02 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2026-02-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 05:30:00 | 3374.40 | 2923.27 | 3178.14 | Stage2 pullback-breakout RSI=60 vol=5.7x ATR=167.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 05:30:00 | 3708.66 | 2936.32 | 3251.93 | T1 booked 50% @ 3708.66 |
| Stop hit — per-position SL triggered | 2026-03-05 05:30:00 | 3374.40 | 2986.49 | 3428.94 | SL hit (bars_held=10) |

### Cycle 4 — BUY (started 2026-04-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 05:30:00 | 3563.70 | 3045.20 | 3293.93 | Stage2 pullback-breakout RSI=64 vol=3.2x ATR=156.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 05:30:00 | 3876.11 | 3074.84 | 3463.35 | T1 booked 50% @ 3876.11 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-03 05:30:00 | 2526.90 | 2025-09-05 05:30:00 | 2761.83 | PARTIAL | 0.50 | 9.30% |
| BUY | retest1 | 2025-09-03 05:30:00 | 2526.90 | 2025-10-23 05:30:00 | 3743.70 | TARGET_HIT | 0.50 | 48.15% |
| BUY | retest1 | 2026-01-05 05:30:00 | 3274.60 | 2026-01-19 05:30:00 | 3543.66 | PARTIAL | 0.50 | 8.22% |
| BUY | retest1 | 2026-01-05 05:30:00 | 3274.60 | 2026-01-20 05:30:00 | 3274.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 05:30:00 | 3374.40 | 2026-02-20 05:30:00 | 3708.66 | PARTIAL | 0.50 | 9.91% |
| BUY | retest1 | 2026-02-18 05:30:00 | 3374.40 | 2026-03-05 05:30:00 | 3374.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 05:30:00 | 3563.70 | 2026-04-21 05:30:00 | 3876.11 | PARTIAL | 0.50 | 8.77% |
