# BAJAJ-AUTO (BAJAJ-AUTO)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 10590.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 6 |
| PENDING | 17 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 1 |
| ENTRY2 | 14 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 12
- **Target hits / Stop hits / Partials:** 0 / 14 / 1
- **Avg / median % per leg:** 0.37% / -1.86%
- **Sum % (uncompounded):** 5.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 3 | 20.0% | 0 | 14 | 1 | 0.37% | 5.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.62% | -2.6% |
| BUY @ 3rd Alert (retest2) | 14 | 3 | 21.4% | 0 | 13 | 1 | 0.58% | 8.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.62% | -2.6% |
| retest2 (combined) | 14 | 3 | 21.4% | 0 | 13 | 1 | 0.58% | 8.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 8706.00 | 8165.04 | 8163.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 8744.00 | 8170.81 | 8165.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 8444.00 | 8497.01 | 8377.80 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 8444.00 | 8497.01 | 8377.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 8444.00 | 8497.01 | 8377.80 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-13 15:15:00 | 8482.50 | 8494.19 | 8379.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 8500.00 | 8494.24 | 8380.48 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2025-06-17 10:15:00 | 8471.50 | 8497.13 | 8386.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 8478.50 | 8496.94 | 8386.89 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-19 09:15:00 | 8513.00 | 8500.13 | 8394.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 8542.50 | 8500.55 | 8395.69 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 8327.00 | 8499.15 | 8398.09 | SL hit (close<static) qty=1.00 sl=8350.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 8327.00 | 8499.15 | 8398.09 | SL hit (close<static) qty=1.00 sl=8350.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 8327.00 | 8499.15 | 8398.09 | SL hit (close<static) qty=1.00 sl=8350.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-27 12:15:00 | 8476.50 | 8462.30 | 8395.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 13:15:00 | 8471.00 | 8462.39 | 8395.41 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 8450.00 | 8461.96 | 8396.20 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 8338.50 | 8456.90 | 8396.21 | SL hit (close<static) qty=1.00 sl=8350.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 14:15:00 | 8444.50 | 8364.81 | 8364.41 | EMA200 above EMA400 |

### Cycle 3 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 8397.00 | 8364.26 | 8364.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 8573.00 | 8242.35 | 8283.13 | Break + close above crossover candle high |

### Cycle 4 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 8826.50 | 8322.29 | 8321.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 8913.50 | 8469.25 | 8403.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 8872.50 | 8904.41 | 8716.21 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 14:15:00 | 8697.50 | 8891.87 | 8725.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 8697.50 | 8891.87 | 8725.91 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-06 14:15:00 | 8789.00 | 8830.92 | 8719.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 8799.00 | 8830.60 | 8719.93 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-09 11:15:00 | 8794.00 | 8829.60 | 8728.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 12:15:00 | 8791.00 | 8829.22 | 8728.80 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 8624.00 | 8932.93 | 8850.55 | SL hit (close<static) qty=1.00 sl=8668.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 8624.00 | 8932.93 | 8850.55 | SL hit (close<static) qty=1.00 sl=8668.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-11 11:15:00 | 8866.00 | 8904.17 | 8841.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:15:00 | 8900.00 | 8904.13 | 8842.13 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-18 10:15:00 | 8795.50 | 8977.97 | 8932.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:15:00 | 8800.00 | 8976.20 | 8931.97 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 8815.00 | 8974.59 | 8931.39 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-19 09:15:00 | 8884.50 | 8969.05 | 8929.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:15:00 | 8943.00 | 8968.79 | 8929.51 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-25 10:15:00 | 10120.00 | 9651.64 | 9473.60 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 12:15:00 | 9672.50 | 9722.02 | 9530.55 | SL hit (close<ema200) qty=0.50 sl=9722.02 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-16 11:15:00 | 8925.00 | 9565.82 | 9498.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 12:15:00 | 8950.00 | 9559.69 | 9495.35 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 8777.00 | 9397.91 | 9418.75 | SL hit (close<static) qty=1.00 sl=8799.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 8777.00 | 9397.91 | 9418.75 | SL hit (close<static) qty=1.00 sl=8799.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-24 09:15:00 | 8870.00 | 9386.57 | 9412.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-24 10:15:00 | 8828.50 | 9381.02 | 9409.92 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-24 11:15:00 | 8876.50 | 9376.00 | 9407.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 12:15:00 | 8980.00 | 9372.05 | 9405.13 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-30 11:15:00 | 8850.50 | 9297.08 | 9362.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-30 12:15:00 | 8826.00 | 9292.39 | 9360.12 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-03-30 14:15:00 | 8797.50 | 9282.64 | 9354.55 | SL hit (close<static) qty=1.00 sl=8799.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-01 09:15:00 | 9008.50 | 9274.80 | 9349.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 8957.00 | 9271.64 | 9347.93 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 8948.00 | 9268.42 | 9345.94 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 8698.00 | 9249.58 | 9334.50 | SL hit (close<static) qty=1.00 sl=8799.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 11:15:00 | 8648.00 | 9237.89 | 9327.78 | SL hit (close<static) qty=1.00 sl=8668.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-07 13:15:00 | 9022.50 | 9187.71 | 9294.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:15:00 | 9051.00 | 9186.35 | 9293.44 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-17 11:15:00 | 9770.00 | 9375.89 | 9374.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 9770.00 | 9375.89 | 9374.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 9799.00 | 9398.74 | 9386.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 9490.00 | 9492.89 | 9442.54 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-29 09:15:00 | 9644.50 | 9494.30 | 9443.75 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:15:00 | 9626.00 | 9495.61 | 9444.66 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 9374.00 | 9498.26 | 9447.53 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 9374.00 | 9498.26 | 9447.53 | SL hit (close<ema400) qty=1.00 sl=9447.53 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-30 10:15:00 | 9678.00 | 9500.05 | 9448.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 9755.00 | 9502.59 | 9450.21 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-16 09:15:00 | 8500.00 | 2025-06-20 09:15:00 | 8327.00 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-06-17 11:15:00 | 8478.50 | 2025-06-20 09:15:00 | 8327.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-06-19 10:15:00 | 8542.50 | 2025-06-20 09:15:00 | 8327.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-06-27 13:15:00 | 8471.00 | 2025-07-01 10:15:00 | 8338.50 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-10-06 15:15:00 | 8799.00 | 2025-11-07 09:15:00 | 8624.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-10-09 12:15:00 | 8791.00 | 2025-11-07 09:15:00 | 8624.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-11-11 12:15:00 | 8900.00 | 2026-02-25 10:15:00 | 10120.00 | PARTIAL | 0.50 | 13.71% |
| BUY | retest2 | 2025-11-11 12:15:00 | 8900.00 | 2026-03-02 12:15:00 | 9672.50 | STOP_HIT | 0.50 | 8.68% |
| BUY | retest2 | 2025-12-18 11:15:00 | 8800.00 | 2026-03-23 14:15:00 | 8777.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-12-19 10:15:00 | 8943.00 | 2026-03-23 14:15:00 | 8777.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-03-16 12:15:00 | 8950.00 | 2026-03-30 14:15:00 | 8797.50 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2026-03-24 12:15:00 | 8980.00 | 2026-04-02 09:15:00 | 8698.00 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2026-04-01 10:15:00 | 8957.00 | 2026-04-02 11:15:00 | 8648.00 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2026-04-07 14:15:00 | 9051.00 | 2026-04-17 11:15:00 | 9770.00 | STOP_HIT | 1.00 | 7.94% |
| BUY | retest1 | 2026-04-29 10:15:00 | 9626.00 | 2026-04-30 09:15:00 | 9374.00 | STOP_HIT | 1.00 | -2.62% |
