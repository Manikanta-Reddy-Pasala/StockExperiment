# Aditya Birla Sun Life AMC Ltd. (ABSLAMC)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1067.00
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 2.10% / 4.89%
- **Sum % (uncompounded):** 8.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 2.10% | 8.4% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 2.10% | 8.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 2.10% | 8.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-01-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 00:00:00 | 498.70 | 428.95 | 473.07 | Stage2 pullback-breakout RSI=70 vol=2.9x ATR=11.66 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 481.20 | 431.31 | 478.08 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-02-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 00:00:00 | 495.20 | 440.97 | 479.29 | Stage2 pullback-breakout RSI=61 vol=2.6x ATR=12.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-27 00:00:00 | 520.12 | 443.51 | 488.39 | T1 booked 50% @ 520.12 |
| Target hit | 2024-03-12 00:00:00 | 504.95 | 450.74 | 507.51 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 00:00:00 | 495.90 | 454.42 | 479.28 | Stage2 pullback-breakout RSI=59 vol=1.9x ATR=12.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 00:00:00 | 520.16 | 455.98 | 486.66 | T1 booked 50% @ 520.16 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-12 00:00:00 | 498.70 | 2024-01-18 00:00:00 | 481.20 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest1 | 2024-02-21 00:00:00 | 495.20 | 2024-02-27 00:00:00 | 520.12 | PARTIAL | 0.50 | 5.03% |
| BUY | retest1 | 2024-02-21 00:00:00 | 495.20 | 2024-03-12 00:00:00 | 504.95 | TARGET_HIT | 0.50 | 1.97% |
| BUY | retest1 | 2024-04-12 00:00:00 | 495.90 | 2024-04-18 00:00:00 | 520.16 | PARTIAL | 0.50 | 4.89% |
