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
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 5 |
| PENDING | 17 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 1 |
| ENTRY2 | 11 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 5
- **Target hits / Stop hits / Partials:** 0 / 12 / 1
- **Avg / median % per leg:** 2.11% / 2.35%
- **Sum % (uncompounded):** 27.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 8 | 61.5% | 0 | 12 | 1 | 2.11% | 27.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.46% | -1.5% |
| SELL @ 3rd Alert (retest2) | 12 | 8 | 66.7% | 0 | 11 | 1 | 2.41% | 28.9% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.46% | -1.5% |
| retest2 (combined) | 12 | 8 | 66.7% | 0 | 11 | 1 | 2.41% | 28.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 9416.60 | 10782.97 | 10787.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 09:15:00 | 9200.00 | 10066.47 | 10348.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 8775.00 | 8707.47 | 9053.88 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 8998.60 | 8726.74 | 9037.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 8998.60 | 8726.74 | 9037.16 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-11 09:15:00 | 8768.05 | 8805.31 | 9016.75 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 11:15:00 | 8796.90 | 8804.87 | 9014.42 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-04 10:15:00 | 7477.36 | 8510.93 | 8766.20 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-21 09:15:00 | 8050.80 | 7980.46 | 8354.77 | SL hit (close>ema200) qty=0.50 sl=7980.46 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-20 09:15:00 | 8762.50 | 8060.76 | 8113.10 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-20 10:15:00 | 8827.00 | 8068.39 | 8116.66 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-20 11:15:00 | 8788.50 | 8075.55 | 8120.01 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 13:15:00 | 8606.50 | 8087.62 | 8125.63 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-23 10:15:00 | 8787.00 | 8183.26 | 8172.24 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 12:15:00 | 8767.00 | 8194.96 | 8178.23 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-28 11:15:00 | 8795.50 | 8323.32 | 8247.12 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-28 12:15:00 | 8835.00 | 8328.41 | 8250.06 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-30 09:15:00 | 8662.00 | 8382.17 | 8281.89 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 11:15:00 | 8653.50 | 8387.58 | 8285.60 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 8652.00 | 8390.21 | 8287.43 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-30 14:15:00 | 8608.00 | 8394.97 | 8290.84 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 8444.00 | 8397.48 | 8293.14 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 4020m) |
| Cross detected — sustain check pending | 2025-06-09 09:15:00 | 8623.00 | 8447.94 | 8337.01 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 11:15:00 | 8619.00 | 8451.39 | 8339.85 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-06-10 09:15:00 | 8600.00 | 8460.17 | 8347.05 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:15:00 | 8619.00 | 8463.16 | 8349.68 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-06-10 13:15:00 | 8633.00 | 8466.69 | 8352.58 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 15:15:00 | 8635.00 | 8469.96 | 8355.36 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 8743.00 | 8472.68 | 8357.29 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 8743.00 | 8472.68 | 8357.29 | SL hit (close>static) qty=1.00 sl=8671.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 8743.00 | 8472.68 | 8357.29 | SL hit (close>static) qty=1.00 sl=8671.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 8743.00 | 8472.68 | 8357.29 | SL hit (close>static) qty=1.00 sl=8671.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 8743.00 | 8472.68 | 8357.29 | SL hit (close>static) qty=1.00 sl=8671.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-12 13:15:00 | 8587.00 | 8496.37 | 8375.67 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 15:15:00 | 8550.00 | 8497.55 | 8377.47 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-06-18 10:15:00 | 8556.50 | 8499.56 | 8391.47 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 8521.00 | 8500.07 | 8392.79 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-07-17 13:15:00 | 8321.00 | 8363.82 | 8363.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-17 13:15:00 | 8321.00 | 8363.82 | 8363.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-17 13:15:00 | 8321.00 | 8363.82 | 8363.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-17 13:15:00 | 8321.00 | 8363.82 | 8363.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-17 13:15:00 | 8321.00 | 8363.82 | 8363.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 8321.00 | 8363.82 | 8363.97 | EMA200 below EMA400 |

### Cycle 3 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 8301.00 | 8363.71 | 8363.88 | EMA200 below EMA400 |

### Cycle 4 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 8285.50 | 8363.43 | 8363.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 8060.00 | 8359.67 | 8361.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 8245.00 | 8242.30 | 8291.38 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-08-08 13:15:00 | 8203.50 | 8241.51 | 8289.77 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-08 14:15:00 | 8225.50 | 8241.35 | 8289.45 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-08 15:15:00 | 8218.00 | 8241.11 | 8289.09 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 09:15:00 | 8178.50 | 8240.49 | 8288.54 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 3960m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 8274.00 | 8238.95 | 8286.56 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-11 15:15:00 | 8235.00 | 8238.91 | 8286.31 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-12 09:15:00 | 8298.00 | 8239.50 | 8286.37 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 8298.00 | 8239.50 | 8286.37 | SL hit (close>ema400) qty=1.00 sl=8286.37 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-12 14:15:00 | 8196.00 | 8239.54 | 8285.23 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-13 09:15:00 | 8242.00 | 8239.04 | 8284.52 | ENTRY2 sustain failed after 1140m |
| Cross detected — sustain check pending | 2025-08-14 09:15:00 | 8217.00 | 8240.25 | 8283.58 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:15:00 | 8220.00 | 8239.77 | 8282.91 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 8573.00 | 8242.35 | 8283.13 | SL hit (close>static) qty=1.00 sl=8292.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 9048.50 | 9437.80 | 9439.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 8836.50 | 9428.24 | 9434.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 9446.50 | 9187.50 | 9292.94 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 9446.50 | 9187.50 | 9292.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 9446.50 | 9187.50 | 9292.94 | EMA400 retest candle locked |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-02-11 11:15:00 | 8796.90 | 2025-03-04 10:15:00 | 7477.36 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-11 11:15:00 | 8796.90 | 2025-03-21 09:15:00 | 8050.80 | STOP_HIT | 0.50 | 8.48% |
| SELL | retest2 | 2025-05-20 13:15:00 | 8606.50 | 2025-06-11 09:15:00 | 8743.00 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-05-23 12:15:00 | 8767.00 | 2025-06-11 09:15:00 | 8743.00 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-05-30 11:15:00 | 8653.50 | 2025-06-11 09:15:00 | 8743.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-06-02 09:15:00 | 8444.00 | 2025-06-11 09:15:00 | 8743.00 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-06-09 11:15:00 | 8619.00 | 2025-07-17 13:15:00 | 8321.00 | STOP_HIT | 1.00 | 3.46% |
| SELL | retest2 | 2025-06-10 11:15:00 | 8619.00 | 2025-07-17 13:15:00 | 8321.00 | STOP_HIT | 1.00 | 3.46% |
| SELL | retest2 | 2025-06-10 15:15:00 | 8635.00 | 2025-07-17 13:15:00 | 8321.00 | STOP_HIT | 1.00 | 3.64% |
| SELL | retest2 | 2025-06-12 15:15:00 | 8550.00 | 2025-07-17 13:15:00 | 8321.00 | STOP_HIT | 1.00 | 2.68% |
| SELL | retest2 | 2025-06-18 12:15:00 | 8521.00 | 2025-07-17 13:15:00 | 8321.00 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest1 | 2025-08-11 09:15:00 | 8178.50 | 2025-08-12 09:15:00 | 8298.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-08-14 11:15:00 | 8220.00 | 2025-08-18 09:15:00 | 8573.00 | STOP_HIT | 1.00 | -4.29% |
