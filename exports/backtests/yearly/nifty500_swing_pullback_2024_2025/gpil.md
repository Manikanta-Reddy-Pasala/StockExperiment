# Godawari Power & Ispat Ltd. (GPIL)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 294.35
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
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 1.69% / 6.50%
- **Sum % (uncompounded):** 10.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.69% | 10.1% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.69% | 10.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.69% | 10.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 05:30:00 | 231.12 | 165.23 | 216.03 | Stage2 pullback-breakout RSI=68 vol=3.3x ATR=9.30 |
| Stop hit — per-position SL triggered | 2024-07-19 05:30:00 | 217.18 | 169.36 | 220.92 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2024-08-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 05:30:00 | 229.04 | 175.36 | 220.34 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=9.04 |
| Stop hit — per-position SL triggered | 2024-08-12 05:30:00 | 215.49 | 176.66 | 219.93 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2024-09-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 05:30:00 | 197.72 | 179.83 | 190.02 | Stage2 pullback-breakout RSI=57 vol=3.0x ATR=6.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 05:30:00 | 210.57 | 180.11 | 191.76 | T1 booked 50% @ 210.57 |
| Stop hit — per-position SL triggered | 2024-10-07 05:30:00 | 197.72 | 182.40 | 201.12 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-12-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 05:30:00 | 197.95 | 184.75 | 190.25 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=7.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 05:30:00 | 212.50 | 185.45 | 195.03 | T1 booked 50% @ 212.50 |
| Target hit | 2024-12-23 05:30:00 | 214.23 | 189.76 | 215.32 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-09 05:30:00 | 231.12 | 2024-07-19 05:30:00 | 217.18 | STOP_HIT | 1.00 | -6.03% |
| BUY | retest1 | 2024-08-07 05:30:00 | 229.04 | 2024-08-12 05:30:00 | 215.49 | STOP_HIT | 1.00 | -5.92% |
| BUY | retest1 | 2024-09-23 05:30:00 | 197.72 | 2024-09-24 05:30:00 | 210.57 | PARTIAL | 0.50 | 6.50% |
| BUY | retest1 | 2024-09-23 05:30:00 | 197.72 | 2024-10-07 05:30:00 | 197.72 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-03 05:30:00 | 197.95 | 2024-12-06 05:30:00 | 212.50 | PARTIAL | 0.50 | 7.35% |
| BUY | retest1 | 2024-12-03 05:30:00 | 197.95 | 2024-12-23 05:30:00 | 214.23 | TARGET_HIT | 0.50 | 8.22% |
