# JK Tyre & Industries Ltd. (JKTYRE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 394.50
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
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 3.56% / 2.79%
- **Sum % (uncompounded):** 17.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 0 | 3 | 2 | 3.56% | 17.8% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 3 | 2 | 3.56% | 17.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 3 | 2 | 3.56% | 17.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-20 00:00:00 | 272.20 | 210.60 | 262.28 | Stage2 pullback-breakout RSI=60 vol=6.3x ATR=9.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-05 00:00:00 | 291.02 | 216.99 | 272.47 | T1 booked 50% @ 291.02 |
| Stop hit — per-position SL triggered | 2023-10-09 00:00:00 | 272.20 | 218.14 | 272.85 | SL hit (bars_held=12) |

### Cycle 2 — BUY (started 2023-10-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 00:00:00 | 311.30 | 219.58 | 276.28 | Stage2 pullback-breakout RSI=68 vol=8.0x ATR=12.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 00:00:00 | 336.52 | 223.21 | 288.18 | T1 booked 50% @ 336.52 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 311.30 | 226.83 | 296.92 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 337.25 | 232.11 | 302.14 | Stage2 pullback-breakout RSI=70 vol=4.2x ATR=15.86 |
| Stop hit — per-position SL triggered | 2023-11-20 00:00:00 | 346.65 | 245.41 | 335.53 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-20 00:00:00 | 272.20 | 2023-10-05 00:00:00 | 291.02 | PARTIAL | 0.50 | 6.91% |
| BUY | retest1 | 2023-09-20 00:00:00 | 272.20 | 2023-10-09 00:00:00 | 272.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-11 00:00:00 | 311.30 | 2023-10-17 00:00:00 | 336.52 | PARTIAL | 0.50 | 8.10% |
| BUY | retest1 | 2023-10-11 00:00:00 | 311.30 | 2023-10-23 00:00:00 | 311.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-02 00:00:00 | 337.25 | 2023-11-20 00:00:00 | 346.65 | STOP_HIT | 1.00 | 2.79% |
