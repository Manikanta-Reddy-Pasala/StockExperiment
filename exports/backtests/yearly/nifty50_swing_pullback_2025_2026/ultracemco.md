# ULTRACEMCO (ULTRACEMCO)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 11950.00
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 0.80% / -0.27%
- **Sum % (uncompounded):** 4.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.80% | 4.0% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.80% | 4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.80% | 4.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 05:30:00 | 11579.00 | 11379.65 | 11418.02 | Stage2 pullback-breakout RSI=56 vol=2.5x ATR=218.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 05:30:00 | 12016.04 | 11397.02 | 11564.22 | T1 booked 50% @ 12016.04 |
| Target hit | 2025-07-24 05:30:00 | 12304.00 | 11576.24 | 12312.20 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-08-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 05:30:00 | 12765.00 | 11684.71 | 12344.79 | Stage2 pullback-breakout RSI=66 vol=2.9x ATR=214.39 |
| Stop hit — per-position SL triggered | 2025-09-02 05:30:00 | 12730.00 | 11782.42 | 12570.66 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-01-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 05:30:00 | 12378.00 | 11879.86 | 11972.34 | Stage2 pullback-breakout RSI=68 vol=2.0x ATR=224.16 |
| Stop hit — per-position SL triggered | 2026-01-20 05:30:00 | 12041.76 | 11885.75 | 12008.56 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2026-01-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 05:30:00 | 12589.00 | 11905.51 | 12138.30 | Stage2 pullback-breakout RSI=67 vol=3.1x ATR=255.09 |
| Stop hit — per-position SL triggered | 2026-02-02 05:30:00 | 12206.37 | 11939.27 | 12312.90 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-24 05:30:00 | 11579.00 | 2025-06-27 05:30:00 | 12016.04 | PARTIAL | 0.50 | 3.77% |
| BUY | retest1 | 2025-06-24 05:30:00 | 11579.00 | 2025-07-24 05:30:00 | 12304.00 | TARGET_HIT | 0.50 | 6.26% |
| BUY | retest1 | 2025-08-18 05:30:00 | 12765.00 | 2025-09-02 05:30:00 | 12730.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-01-16 05:30:00 | 12378.00 | 2026-01-20 05:30:00 | 12041.76 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest1 | 2026-01-27 05:30:00 | 12589.00 | 2026-02-02 05:30:00 | 12206.37 | STOP_HIT | 1.00 | -3.04% |
