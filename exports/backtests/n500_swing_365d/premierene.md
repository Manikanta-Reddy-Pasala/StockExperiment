# Premier Energies Ltd. (PREMIERENE)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 1011.60
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
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 0.19% / 0.28%
- **Sum % (uncompounded):** 0.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.19% | 0.8% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.19% | 0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.19% | 0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 05:30:00 | 1115.80 | 1013.36 | 1052.24 | Stage2 pullback-breakout RSI=66 vol=1.6x ATR=34.90 |
| Stop hit — per-position SL triggered | 2025-07-16 05:30:00 | 1063.45 | 1016.38 | 1064.39 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2025-09-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 05:30:00 | 1041.80 | 1018.73 | 1015.44 | Stage2 pullback-breakout RSI=56 vol=1.9x ATR=32.11 |
| Stop hit — per-position SL triggered | 2025-09-24 05:30:00 | 1044.70 | 1021.87 | 1036.19 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-14 05:30:00 | 1057.00 | 1022.78 | 1031.91 | Stage2 pullback-breakout RSI=57 vol=1.9x ATR=27.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 05:30:00 | 1111.74 | 1026.21 | 1052.54 | T1 booked 50% @ 1111.74 |
| Stop hit — per-position SL triggered | 2025-11-03 05:30:00 | 1057.00 | 1029.31 | 1066.74 | SL hit (bars_held=13) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-10 05:30:00 | 1115.80 | 2025-07-16 05:30:00 | 1063.45 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest1 | 2025-09-10 05:30:00 | 1041.80 | 2025-09-24 05:30:00 | 1044.70 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest1 | 2025-10-14 05:30:00 | 1057.00 | 2025-10-27 05:30:00 | 1111.74 | PARTIAL | 0.50 | 5.18% |
| BUY | retest1 | 2025-10-14 05:30:00 | 1057.00 | 2025-11-03 05:30:00 | 1057.00 | STOP_HIT | 0.50 | 0.00% |
