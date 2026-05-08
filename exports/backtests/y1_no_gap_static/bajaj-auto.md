# BAJAJ-AUTO (BAJAJ-AUTO)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 10696.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 6 |
| PENDING | 19 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 1 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 12
- **Target hits / Stop hits / Partials:** 3 / 12 / 0
- **Avg / median % per leg:** 0.46% / -1.64%
- **Sum % (uncompounded):** 6.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 3 | 20.0% | 3 | 12 | 0 | 0.46% | 7.0% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.62% | -2.6% |
| BUY @ 3rd Alert (retest2) | 14 | 3 | 21.4% | 3 | 11 | 0 | 0.68% | 9.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.62% | -2.6% |
| retest2 (combined) | 14 | 3 | 21.4% | 3 | 11 | 0 | 0.68% | 9.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 13:15:00 | 8765.00 | 8421.72 | 8421.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 8847.00 | 8466.02 | 8444.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 8872.50 | 8904.79 | 8740.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 13:15:00 | 8770.00 | 8894.14 | 8748.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 8770.00 | 8894.14 | 8748.12 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-07 09:15:00 | 8838.00 | 8830.89 | 8738.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-07 10:15:00 | 8817.00 | 8830.75 | 8739.16 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-07 11:15:00 | 8833.50 | 8830.78 | 8739.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:15:00 | 8873.00 | 8831.20 | 8740.29 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-08 11:15:00 | 8828.00 | 8832.65 | 8743.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:15:00 | 8848.00 | 8832.80 | 8744.24 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 8723.50 | 8830.63 | 8744.90 | SL hit (close<static) qty=1.00 sl=8731.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 8723.50 | 8830.63 | 8744.90 | SL hit (close<static) qty=1.00 sl=8731.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-10 09:15:00 | 8860.50 | 8828.88 | 8746.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 8875.00 | 8829.34 | 8747.60 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-06 14:15:00 | 8706.50 | 8938.33 | 8861.44 | SL hit (close<static) qty=1.00 sl=8731.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-11 11:15:00 | 8866.00 | 8904.22 | 8850.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:15:00 | 8900.00 | 8904.18 | 8850.37 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 8863.50 | 8904.04 | 8851.37 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-17 09:15:00 | 8980.00 | 8895.07 | 8851.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 10:15:00 | 8984.00 | 8895.95 | 8852.57 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-19 09:15:00 | 8943.00 | 8904.39 | 8859.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-19 10:15:00 | 8920.00 | 8904.55 | 8859.98 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-20 10:15:00 | 8997.00 | 8905.44 | 8861.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 11:15:00 | 8952.00 | 8905.91 | 8862.41 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-24 09:15:00 | 8992.00 | 8909.10 | 8866.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 10:15:00 | 9006.00 | 8910.07 | 8867.31 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-09 10:15:00 | 8972.00 | 8988.42 | 8928.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 11:15:00 | 8986.50 | 8988.40 | 8928.38 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 8929.00 | 8987.59 | 8929.16 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-10 14:15:00 | 8993.00 | 8985.91 | 8930.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 15:15:00 | 8980.00 | 8985.85 | 8930.27 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-12 12:15:00 | 8987.00 | 8991.22 | 8936.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 13:15:00 | 9010.00 | 8991.41 | 8936.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 8900.00 | 8990.26 | 8936.91 | SL hit (close<static) qty=1.00 sl=8921.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 8900.00 | 8990.26 | 8936.91 | SL hit (close<static) qty=1.00 sl=8921.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-16 12:15:00 | 8974.50 | 8986.36 | 8937.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:15:00 | 8990.00 | 8986.39 | 8937.52 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-17 14:15:00 | 8897.00 | 8983.32 | 8937.88 | SL hit (close<static) qty=1.00 sl=8921.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 8745.00 | 8979.81 | 8936.57 | SL hit (close<static) qty=1.00 sl=8845.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 8745.00 | 8979.81 | 8936.57 | SL hit (close<static) qty=1.00 sl=8845.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 8745.00 | 8979.81 | 8936.57 | SL hit (close<static) qty=1.00 sl=8845.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 8745.00 | 8979.81 | 8936.57 | SL hit (close<static) qty=1.00 sl=8845.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-19 13:15:00 | 8976.00 | 8968.82 | 8933.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 14:15:00 | 8999.00 | 8969.12 | 8933.51 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Target hit | 2026-01-07 14:15:00 | 9790.00 | 9207.07 | 9080.20 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 9181.50 | 9335.96 | 9187.09 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-22 09:15:00 | 9291.00 | 9322.46 | 9186.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-22 10:15:00 | 9212.50 | 9321.36 | 9186.85 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-22 12:15:00 | 9270.00 | 9319.83 | 9187.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 9286.50 | 9319.49 | 9187.91 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2026-02-03 09:15:00 | 9898.90 | 9375.64 | 9249.21 | Target hit (10%) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 9134.00 | 9664.84 | 9540.17 | SL hit (close<static) qty=1.00 sl=9176.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-18 14:15:00 | 9278.00 | 9500.11 | 9469.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-18 15:15:00 | 9229.00 | 9497.41 | 9468.37 | ENTRY2 sustain failed after 60m |

### Cycle 2 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 9048.50 | 9437.80 | 9439.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 8836.50 | 9428.24 | 9434.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 9446.50 | 9187.50 | 9293.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 9446.50 | 9187.50 | 9293.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 9446.50 | 9187.50 | 9293.19 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 9770.00 | 9375.89 | 9375.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 9799.00 | 9398.74 | 9386.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 9490.00 | 9492.89 | 9442.70 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-04-29 09:15:00 | 9644.50 | 9494.30 | 9443.90 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:15:00 | 9626.00 | 9495.61 | 9444.81 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 9374.00 | 9498.26 | 9447.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 9374.00 | 9498.26 | 9447.68 | SL hit (close<ema400) qty=1.00 sl=9447.68 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-30 10:15:00 | 9678.00 | 9500.05 | 9448.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 9755.00 | 9502.59 | 9450.35 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2026-05-07 09:15:00 | 10730.50 | 9652.30 | 9536.01 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-07 12:15:00 | 8873.00 | 2025-10-09 09:15:00 | 8723.50 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-10-08 12:15:00 | 8848.00 | 2025-10-09 09:15:00 | 8723.50 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-10-10 10:15:00 | 8875.00 | 2025-11-06 14:15:00 | 8706.50 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-11-11 12:15:00 | 8900.00 | 2025-12-15 10:15:00 | 8900.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-11-17 10:15:00 | 8984.00 | 2025-12-15 10:15:00 | 8900.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-11-20 11:15:00 | 8952.00 | 2025-12-17 14:15:00 | 8897.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-11-24 10:15:00 | 9006.00 | 2025-12-18 09:15:00 | 8745.00 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-12-09 11:15:00 | 8986.50 | 2025-12-18 09:15:00 | 8745.00 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-12-10 15:15:00 | 8980.00 | 2025-12-18 09:15:00 | 8745.00 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-12-12 13:15:00 | 9010.00 | 2025-12-18 09:15:00 | 8745.00 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-12-16 13:15:00 | 8990.00 | 2026-01-07 14:15:00 | 9790.00 | TARGET_HIT | 1.00 | 8.90% |
| BUY | retest2 | 2025-12-19 14:15:00 | 8999.00 | 2026-02-03 09:15:00 | 9898.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-22 13:15:00 | 9286.50 | 2026-03-12 09:15:00 | 9134.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest1 | 2026-04-29 10:15:00 | 9626.00 | 2026-04-30 09:15:00 | 9374.00 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2026-04-30 11:15:00 | 9755.00 | 2026-05-07 09:15:00 | 10730.50 | TARGET_HIT | 1.00 | 10.00% |
