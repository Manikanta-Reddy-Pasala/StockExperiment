# MARUTI (MARUTI)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 13726.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 5 |
| ALERT3 | 12 |
| PENDING | 47 |
| PENDING_CANCEL | 14 |
| ENTRY1 | 15 |
| ENTRY2 | 18 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 30
- **Target hits / Stop hits / Partials:** 1 / 31 / 1
- **Avg / median % per leg:** -1.30% / -2.71%
- **Sum % (uncompounded):** -43.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 3 | 15.8% | 1 | 17 | 1 | -0.07% | -1.4% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 6 | 0 | -2.73% | -16.4% |
| BUY @ 3rd Alert (retest2) | 13 | 2 | 15.4% | 1 | 11 | 1 | 1.15% | 15.0% |
| SELL (all) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.97% | -41.6% |
| SELL @ 2nd Alert (retest1) | 9 | 0 | 0.0% | 0 | 9 | 0 | -3.07% | -27.6% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.81% | -14.0% |
| retest1 (combined) | 15 | 1 | 6.7% | 0 | 15 | 0 | -2.93% | -44.0% |
| retest2 (combined) | 18 | 2 | 11.1% | 1 | 16 | 1 | 0.05% | 0.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 13:15:00 | 10380.00 | 9626.83 | 9626.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 09:15:00 | 10398.10 | 9648.40 | 9636.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 09:15:00 | 10155.90 | 10272.99 | 10053.92 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-10-05 15:15:00 | 10254.00 | 10260.15 | 10061.05 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 09:15:00 | 10316.40 | 10260.71 | 10062.32 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2023-10-09 14:15:00 | 10243.00 | 10262.51 | 10074.83 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 15:15:00 | 10244.80 | 10262.33 | 10075.67 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 10:15:00 | 10270.80 | 10469.56 | 10272.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-10-30 10:15:00 | 10270.80 | 10469.56 | 10272.64 | SL hit (close<ema400) qty=1.00 sl=10272.64 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-10-30 10:15:00 | 10270.80 | 10469.56 | 10272.64 | SL hit (close<ema400) qty=1.00 sl=10272.64 alert=retest1 |
| Cross detected — sustain check pending | 2023-10-30 13:15:00 | 10420.00 | 10467.12 | 10274.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-30 14:15:00 | 10397.05 | 10466.42 | 10274.96 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-10-31 10:15:00 | 10408.00 | 10464.53 | 10276.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-31 11:15:00 | 10404.95 | 10463.94 | 10277.49 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-10-31 13:15:00 | 10449.25 | 10463.20 | 10278.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-31 14:15:00 | 10405.25 | 10462.63 | 10279.61 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-09 09:15:00 | 10411.60 | 10400.97 | 10281.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 10:15:00 | 10434.00 | 10401.30 | 10282.13 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-11-13 09:15:00 | 10408.50 | 10400.29 | 10289.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-13 10:15:00 | 10371.50 | 10400.00 | 10289.53 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-13 13:15:00 | 10410.00 | 10399.79 | 10291.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-13 14:15:00 | 10396.90 | 10399.77 | 10291.59 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-15 09:15:00 | 10409.00 | 10399.84 | 10292.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 10:15:00 | 10412.25 | 10399.97 | 10293.30 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-12-13 15:15:00 | 10409.00 | 10513.06 | 10421.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-12-14 09:15:00 | 10377.95 | 10511.72 | 10421.41 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2023-12-15 09:15:00 | 10408.85 | 10501.92 | 10419.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-12-15 10:15:00 | 10302.05 | 10499.93 | 10418.94 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2023-12-15 13:15:00 | 10200.70 | 10492.98 | 10416.66 | SL hit (close<static) qty=1.00 sl=10245.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-12-15 13:15:00 | 10200.70 | 10492.98 | 10416.66 | SL hit (close<static) qty=1.00 sl=10245.50 alert=retest2 |

### Cycle 2 — SELL (started 2024-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 15:15:00 | 10270.00 | 10360.76 | 10361.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 09:15:00 | 10260.00 | 10359.76 | 10360.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 09:15:00 | 10219.80 | 10192.55 | 10262.92 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-01-17 09:15:00 | 10039.20 | 10191.10 | 10259.78 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 10:15:00 | 10057.20 | 10189.77 | 10258.77 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-17 14:15:00 | 10062.30 | 10185.25 | 10255.12 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 15:15:00 | 10051.70 | 10183.92 | 10254.10 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 13:15:00 | 10117.25 | 10091.80 | 10184.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-31 14:15:00 | 10199.40 | 10092.87 | 10184.65 | SL hit (close>ema400) qty=1.00 sl=10184.65 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-01-31 14:15:00 | 10199.40 | 10092.87 | 10184.65 | SL hit (close>ema400) qty=1.00 sl=10184.65 alert=retest1 |

### Cycle 3 — BUY (started 2024-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 14:15:00 | 10923.80 | 10263.78 | 10261.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 14:15:00 | 11028.95 | 10414.08 | 10343.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 13:15:00 | 12444.15 | 12454.98 | 11989.10 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-05-06 09:15:00 | 12563.70 | 12456.76 | 11996.93 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 10:15:00 | 12577.60 | 12457.96 | 11999.83 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-09 09:15:00 | 12686.90 | 12454.43 | 12041.45 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-09 10:15:00 | 12617.35 | 12456.06 | 12044.32 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-09 13:15:00 | 12579.15 | 12459.29 | 12052.08 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-09 14:15:00 | 12509.65 | 12459.79 | 12054.36 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-10 09:15:00 | 12685.85 | 12462.48 | 12059.75 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 10:15:00 | 12657.00 | 12464.42 | 12062.73 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-13 12:15:00 | 12572.20 | 12476.73 | 12086.72 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-13 13:15:00 | 12607.00 | 12478.03 | 12089.31 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 12364.75 | 12604.91 | 12324.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 12104.40 | 12599.93 | 12322.98 | SL hit (close<ema400) qty=1.00 sl=12322.98 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 12104.40 | 12599.93 | 12322.98 | SL hit (close<ema400) qty=1.00 sl=12322.98 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 12104.40 | 12599.93 | 12322.98 | SL hit (close<ema400) qty=1.00 sl=12322.98 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 12104.40 | 12599.93 | 12322.98 | SL hit (close<ema400) qty=1.00 sl=12322.98 alert=retest1 |
| Cross detected — sustain check pending | 2024-06-05 09:15:00 | 12570.10 | 12578.18 | 12320.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-05 10:15:00 | 12512.10 | 12577.52 | 12321.05 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-05 11:15:00 | 12570.30 | 12577.45 | 12322.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-05 12:15:00 | 12494.30 | 12576.62 | 12323.15 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-06 09:15:00 | 12569.05 | 12574.13 | 12326.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 10:15:00 | 12617.95 | 12574.57 | 12328.36 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-20 09:15:00 | 12209.80 | 12628.16 | 12427.16 | SL hit (close<static) qty=1.00 sl=12225.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-09 09:15:00 | 12585.00 | 12322.93 | 12315.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 10:15:00 | 12712.80 | 12326.81 | 12317.54 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-12 15:15:00 | 12567.00 | 12426.21 | 12372.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 12633.70 | 12428.27 | 12373.42 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2024-07-18 12:15:00 | 12555.90 | 12459.89 | 12394.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 13:15:00 | 12665.75 | 12461.94 | 12395.81 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 12586.00 | 12473.89 | 12405.19 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-07-22 12:15:00 | 12618.65 | 12477.32 | 12407.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 13:15:00 | 12648.90 | 12479.03 | 12409.14 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-23 13:15:00 | 12607.05 | 12489.44 | 12416.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 14:15:00 | 12611.70 | 12490.66 | 12417.82 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-26 11:15:00 | 12647.00 | 12495.82 | 12426.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 12:15:00 | 12736.65 | 12498.22 | 12428.33 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 12371.55 | 12631.79 | 12513.35 | SL hit (close<static) qty=1.00 sl=12400.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 12371.55 | 12631.79 | 12513.35 | SL hit (close<static) qty=1.00 sl=12400.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 12371.55 | 12631.79 | 12513.35 | SL hit (close<static) qty=1.00 sl=12400.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 12135.10 | 12626.85 | 12511.47 | SL hit (close<static) qty=1.00 sl=12225.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 12135.10 | 12626.85 | 12511.47 | SL hit (close<static) qty=1.00 sl=12225.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 12135.10 | 12626.85 | 12511.47 | SL hit (close<static) qty=1.00 sl=12225.65 alert=retest2 |

### Cycle 4 — SELL (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 15:15:00 | 12149.80 | 12431.14 | 12431.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 13:15:00 | 12123.45 | 12368.28 | 12392.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 13:15:00 | 12334.70 | 12330.06 | 12368.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 14:15:00 | 12393.35 | 12330.69 | 12368.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 12393.35 | 12330.69 | 12368.44 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-09-13 12:15:00 | 12308.60 | 12330.94 | 12367.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 13:15:00 | 12315.75 | 12330.79 | 12367.38 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 12556.35 | 12318.52 | 12355.33 | SL hit (close>static) qty=1.00 sl=12427.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 09:15:00 | 12670.30 | 12389.67 | 12389.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 09:15:00 | 13162.35 | 12415.62 | 12402.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 12595.65 | 12638.83 | 12529.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 14:15:00 | 12613.95 | 12637.67 | 12529.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 12613.95 | 12637.67 | 12529.51 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-10-09 09:15:00 | 12683.00 | 12622.53 | 12529.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 10:15:00 | 12712.00 | 12623.42 | 12530.83 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 12487.30 | 12653.42 | 12558.99 | SL hit (close<static) qty=1.00 sl=12514.95 alert=retest2 |

### Cycle 6 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 11983.00 | 12481.32 | 12483.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 10:15:00 | 11790.25 | 12441.11 | 12462.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 12:15:00 | 11331.15 | 11321.49 | 11641.89 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-12-10 11:15:00 | 11235.30 | 11317.24 | 11619.55 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 12:15:00 | 11216.85 | 11316.24 | 11617.54 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-12 09:15:00 | 11185.40 | 11309.69 | 11598.04 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:15:00 | 11120.00 | 11307.80 | 11595.66 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-16 10:15:00 | 11246.10 | 11294.08 | 11569.03 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-16 11:15:00 | 11267.00 | 11293.81 | 11567.52 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-16 13:15:00 | 11248.35 | 11293.18 | 11564.48 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-16 14:15:00 | 11272.50 | 11292.97 | 11563.02 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-17 09:15:00 | 11196.75 | 11291.86 | 11559.78 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:15:00 | 11178.85 | 11290.73 | 11557.88 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 11430.10 | 11101.36 | 11361.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 11430.10 | 11101.36 | 11361.41 | SL hit (close>ema400) qty=1.00 sl=11361.41 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 11430.10 | 11101.36 | 11361.41 | SL hit (close>ema400) qty=1.00 sl=11361.41 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 11430.10 | 11101.36 | 11361.41 | SL hit (close>ema400) qty=1.00 sl=11361.41 alert=retest1 |

### Cycle 7 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 11993.00 | 11526.56 | 11525.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 11:15:00 | 12108.55 | 11532.35 | 11527.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 12399.50 | 12411.45 | 12107.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 11956.60 | 12399.71 | 12150.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 11956.60 | 12399.71 | 12150.02 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 11:15:00 | 11647.20 | 11987.65 | 11988.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-19 13:15:00 | 11590.00 | 11960.42 | 11974.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 11:15:00 | 11944.85 | 11925.92 | 11954.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-24 11:15:00 | 11944.85 | 11925.92 | 11954.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 11:15:00 | 11944.85 | 11925.92 | 11954.64 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-03-26 09:15:00 | 11746.80 | 11923.11 | 11951.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 11815.00 | 11922.04 | 11950.89 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-16 09:15:00 | 11678.00 | 11729.16 | 11824.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-16 10:15:00 | 11701.00 | 11728.88 | 11824.21 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-25 10:15:00 | 11818.00 | 11741.57 | 11812.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-25 11:15:00 | 11842.00 | 11742.57 | 11812.30 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-25 12:15:00 | 11784.00 | 11742.98 | 11812.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 13:15:00 | 11786.00 | 11743.41 | 11812.03 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-28 13:15:00 | 11828.00 | 11744.23 | 11810.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-28 14:15:00 | 11843.00 | 11745.21 | 11810.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-29 09:15:00 | 11803.00 | 11746.78 | 11810.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 11815.00 | 11747.46 | 11810.40 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 11760.00 | 11747.58 | 11810.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-30 11:15:00 | 12135.00 | 11758.42 | 11813.48 | SL hit (close>static) qty=1.00 sl=11954.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-30 11:15:00 | 12135.00 | 11758.42 | 11813.48 | SL hit (close>static) qty=1.00 sl=11954.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-30 11:15:00 | 12135.00 | 11758.42 | 11813.48 | SL hit (close>static) qty=1.00 sl=11954.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-30 11:15:00 | 12135.00 | 11758.42 | 11813.48 | SL hit (close>static) qty=1.00 sl=11954.95 alert=retest2 |

### Cycle 9 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 12458.00 | 11866.54 | 11865.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 12502.00 | 11872.86 | 11868.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 12322.00 | 12347.43 | 12168.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 12165.00 | 12346.56 | 12191.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 12165.00 | 12346.56 | 12191.47 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-02 11:15:00 | 12248.00 | 12344.34 | 12191.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:15:00 | 12267.00 | 12343.57 | 12192.27 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 12130.00 | 12338.34 | 12193.37 | SL hit (close<static) qty=1.00 sl=12144.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-06 10:15:00 | 12469.00 | 12305.39 | 12190.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:15:00 | 12515.00 | 12307.47 | 12192.03 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 13:15:00 | 14392.25 | 12942.02 | 12706.14 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-23 09:15:00 | 16269.50 | 14770.93 | 14022.99 | Target hit (30%) qty=0.50 alert=retest2 |

### Cycle 10 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 14511.00 | 16031.10 | 16031.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 14425.00 | 15985.09 | 16008.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 13627.00 | 13309.02 | 14074.29 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 13180.00 | 13370.51 | 14030.42 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 10:15:00 | 13132.00 | 13368.13 | 14025.94 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-23 10:15:00 | 13183.00 | 13370.61 | 13894.02 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:15:00 | 13219.00 | 13369.10 | 13890.66 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-27 14:15:00 | 13231.00 | 13330.09 | 13827.76 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 15:15:00 | 13220.00 | 13328.99 | 13824.73 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-28 10:15:00 | 13232.00 | 13327.31 | 13818.96 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:15:00 | 13132.00 | 13325.36 | 13815.53 | SELL ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 13746.00 | 13312.44 | 13764.19 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-05-04 11:15:00 | 13525.00 | 13318.57 | 13762.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:15:00 | 13502.00 | 13320.39 | 13761.48 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 13761.00 | 13359.19 | 13741.67 | SL hit (close>ema400) qty=1.00 sl=13741.67 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 13761.00 | 13359.19 | 13741.67 | SL hit (close>ema400) qty=1.00 sl=13741.67 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 13761.00 | 13359.19 | 13741.67 | SL hit (close>ema400) qty=1.00 sl=13741.67 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 13761.00 | 13359.19 | 13741.67 | SL hit (close>ema400) qty=1.00 sl=13741.67 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-06 09:15:00 | 10316.40 | 2023-10-30 10:15:00 | 10270.80 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-10-09 15:15:00 | 10244.80 | 2023-10-30 10:15:00 | 10270.80 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2023-11-09 10:15:00 | 10434.00 | 2023-12-15 13:15:00 | 10200.70 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2023-11-15 10:15:00 | 10412.25 | 2023-12-15 13:15:00 | 10200.70 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest1 | 2024-01-17 10:15:00 | 10057.20 | 2024-01-31 14:15:00 | 10199.40 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest1 | 2024-01-17 15:15:00 | 10051.70 | 2024-01-31 14:15:00 | 10199.40 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest1 | 2024-05-06 10:15:00 | 12577.60 | 2024-06-04 10:15:00 | 12104.40 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest1 | 2024-05-09 10:15:00 | 12617.35 | 2024-06-04 10:15:00 | 12104.40 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest1 | 2024-05-10 10:15:00 | 12657.00 | 2024-06-04 10:15:00 | 12104.40 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest1 | 2024-05-13 13:15:00 | 12607.00 | 2024-06-04 10:15:00 | 12104.40 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2024-06-06 10:15:00 | 12617.95 | 2024-06-20 09:15:00 | 12209.80 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2024-07-09 10:15:00 | 12712.80 | 2024-08-05 09:15:00 | 12371.55 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2024-07-15 09:15:00 | 12633.70 | 2024-08-05 09:15:00 | 12371.55 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-07-18 13:15:00 | 12665.75 | 2024-08-05 09:15:00 | 12371.55 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-07-22 13:15:00 | 12648.90 | 2024-08-05 10:15:00 | 12135.10 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2024-07-23 14:15:00 | 12611.70 | 2024-08-05 10:15:00 | 12135.10 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2024-07-26 12:15:00 | 12736.65 | 2024-08-05 10:15:00 | 12135.10 | STOP_HIT | 1.00 | -4.72% |
| SELL | retest2 | 2024-09-13 13:15:00 | 12315.75 | 2024-09-20 09:15:00 | 12556.35 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-10-09 10:15:00 | 12712.00 | 2024-10-15 09:15:00 | 12487.30 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest1 | 2024-12-10 12:15:00 | 11216.85 | 2025-01-02 09:15:00 | 11430.10 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest1 | 2024-12-12 10:15:00 | 11120.00 | 2025-01-02 09:15:00 | 11430.10 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest1 | 2024-12-17 10:15:00 | 11178.85 | 2025-01-02 09:15:00 | 11430.10 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-03-26 10:15:00 | 11815.00 | 2025-04-30 11:15:00 | 12135.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-04-16 10:15:00 | 11701.00 | 2025-04-30 11:15:00 | 12135.00 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-04-25 13:15:00 | 11786.00 | 2025-04-30 11:15:00 | 12135.00 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-04-29 10:15:00 | 11815.00 | 2025-04-30 11:15:00 | 12135.00 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-06-02 12:15:00 | 12267.00 | 2025-06-03 10:15:00 | 12130.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-06-06 11:15:00 | 12515.00 | 2025-08-21 13:15:00 | 14392.25 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-06-06 11:15:00 | 12515.00 | 2025-09-23 09:15:00 | 16269.50 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest1 | 2026-04-13 10:15:00 | 13132.00 | 2026-05-07 10:15:00 | 13761.00 | STOP_HIT | 1.00 | -4.79% |
| SELL | retest1 | 2026-04-23 11:15:00 | 13219.00 | 2026-05-07 10:15:00 | 13761.00 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest1 | 2026-04-27 15:15:00 | 13220.00 | 2026-05-07 10:15:00 | 13761.00 | STOP_HIT | 1.00 | -4.09% |
| SELL | retest1 | 2026-04-28 11:15:00 | 13132.00 | 2026-05-07 10:15:00 | 13761.00 | STOP_HIT | 1.00 | -4.79% |
