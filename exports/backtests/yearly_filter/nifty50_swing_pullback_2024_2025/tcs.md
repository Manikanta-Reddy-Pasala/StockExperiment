# TCS (TCS)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 2394.40
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 2.20% / 2.72%
- **Sum % (uncompounded):** 8.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 0 | 3 | 1 | 2.20% | 8.8% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 3 | 1 | 2.20% | 8.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 3 | 1 | 2.20% | 8.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 00:00:00 | 3934.15 | 3783.03 | 3836.28 | Stage2 pullback-breakout RSI=61 vol=1.5x ATR=67.32 |
| Stop hit — per-position SL triggered | 2024-07-11 00:00:00 | 3923.70 | 3800.89 | 3919.45 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-07-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 00:00:00 | 4183.95 | 3804.70 | 3944.64 | Stage2 pullback-breakout RSI=70 vol=4.5x ATR=82.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 00:00:00 | 4349.78 | 3821.85 | 4044.74 | T1 booked 50% @ 4349.78 |
| Stop hit — per-position SL triggered | 2024-08-02 00:00:00 | 4283.05 | 3871.39 | 4235.52 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 4315.10 | 4068.93 | 4135.14 | Stage2 pullback-breakout RSI=64 vol=1.9x ATR=94.59 |
| Stop hit — per-position SL triggered | 2024-12-10 00:00:00 | 4432.55 | 4099.14 | 4295.60 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-27 00:00:00 | 3934.15 | 2024-07-11 00:00:00 | 3923.70 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-12 00:00:00 | 4183.95 | 2024-07-19 00:00:00 | 4349.78 | PARTIAL | 0.50 | 3.96% |
| BUY | retest1 | 2024-07-12 00:00:00 | 4183.95 | 2024-08-02 00:00:00 | 4283.05 | STOP_HIT | 0.50 | 2.37% |
| BUY | retest1 | 2024-11-25 00:00:00 | 4315.10 | 2024-12-10 00:00:00 | 4432.55 | STOP_HIT | 1.00 | 2.72% |
