# DIVISLAB (DIVISLAB)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 6705.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 5 |
| PENDING | 20 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 1 |
| ENTRY2 | 13 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 12
- **Target hits / Stop hits / Partials:** 0 / 14 / 2
- **Avg / median % per leg:** -0.95% / -1.64%
- **Sum % (uncompounded):** -15.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.73% | -13.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.73% | -13.8% |
| SELL (all) | 8 | 4 | 50.0% | 0 | 6 | 2 | -0.16% | -1.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.80% | -3.8% |
| SELL @ 3rd Alert (retest2) | 7 | 4 | 57.1% | 0 | 5 | 2 | 0.36% | 2.5% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.80% | -3.8% |
| retest2 (combined) | 15 | 4 | 26.7% | 0 | 13 | 2 | -0.76% | -11.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 15:15:00 | 6126.00 | 6570.93 | 6571.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 6041.00 | 6565.65 | 6568.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 6260.00 | 6145.77 | 6262.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 6260.00 | 6145.77 | 6262.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 6260.00 | 6145.77 | 6262.96 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-22 09:15:00 | 6139.00 | 6149.73 | 6260.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 6144.50 | 6149.67 | 6260.37 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 5837.27 | 6116.33 | 6228.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 11:15:00 | 6026.00 | 5993.97 | 6137.96 | SL hit (close>ema200) qty=0.50 sl=5993.97 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-08 14:15:00 | 6119.00 | 6008.80 | 6138.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 6106.00 | 6009.77 | 6138.35 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-09 11:15:00 | 6154.50 | 6014.56 | 6138.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-09 12:15:00 | 6163.00 | 6016.03 | 6138.97 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-09 13:15:00 | 6142.00 | 6017.29 | 6138.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:15:00 | 6132.50 | 6018.43 | 6138.95 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 6386.00 | 6026.04 | 6140.40 | SL hit (close>static) qty=1.00 sl=6275.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 6386.00 | 6026.04 | 6140.40 | SL hit (close>static) qty=1.00 sl=6275.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 6610.00 | 6232.76 | 6231.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 10:15:00 | 6685.00 | 6323.22 | 6281.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 6475.00 | 6497.73 | 6404.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 6416.00 | 6491.83 | 6410.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 6416.00 | 6491.83 | 6410.66 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-26 09:15:00 | 6474.00 | 6471.96 | 6408.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 6480.50 | 6472.05 | 6408.62 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-27 13:15:00 | 6491.50 | 6473.83 | 6412.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:15:00 | 6494.00 | 6474.03 | 6413.04 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-28 14:15:00 | 6480.50 | 6473.59 | 6414.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-28 15:15:00 | 6461.50 | 6473.47 | 6415.15 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-01 09:15:00 | 6464.00 | 6473.37 | 6415.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-01 10:15:00 | 6455.00 | 6473.19 | 6415.59 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 6374.50 | 6470.29 | 6415.83 | SL hit (close<static) qty=1.00 sl=6407.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 6374.50 | 6470.29 | 6415.83 | SL hit (close<static) qty=1.00 sl=6407.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-04 10:15:00 | 6483.00 | 6463.86 | 6416.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 11:15:00 | 6489.50 | 6464.11 | 6416.76 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-04 13:15:00 | 6462.50 | 6464.08 | 6417.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 14:15:00 | 6479.00 | 6464.22 | 6417.52 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 6404.00 | 6463.19 | 6417.46 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 6404.00 | 6463.19 | 6417.46 | SL hit (close<static) qty=1.00 sl=6407.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 6404.00 | 6463.19 | 6417.46 | SL hit (close<static) qty=1.00 sl=6407.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-05 14:15:00 | 6471.50 | 6462.67 | 6418.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 15:15:00 | 6486.00 | 6462.90 | 6418.67 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 6384.00 | 6461.51 | 6418.41 | SL hit (close<static) qty=1.00 sl=6394.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-11 11:15:00 | 6483.00 | 6437.97 | 6410.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-11 12:15:00 | 6452.50 | 6438.12 | 6410.62 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-19 09:15:00 | 6565.50 | 6416.35 | 6403.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:15:00 | 6515.50 | 6417.34 | 6404.26 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-19 14:15:00 | 6473.00 | 6419.04 | 6405.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-19 15:15:00 | 6460.00 | 6419.45 | 6405.66 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-22 09:15:00 | 6526.50 | 6420.52 | 6406.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 6526.00 | 6421.57 | 6406.86 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 6390.50 | 6438.59 | 6417.91 | SL hit (close<static) qty=1.00 sl=6394.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 6390.50 | 6438.59 | 6417.91 | SL hit (close<static) qty=1.00 sl=6394.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-30 15:15:00 | 6480.50 | 6431.05 | 6415.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-31 09:15:00 | 6388.00 | 6430.63 | 6415.19 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-01-06 09:15:00 | 6506.00 | 6418.23 | 6410.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 6543.50 | 6419.48 | 6411.18 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 6399.00 | 6465.48 | 6436.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 6391.50 | 6463.06 | 6436.92 | SL hit (close<static) qty=1.00 sl=6394.50 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 6083.00 | 6414.56 | 6415.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 6072.50 | 6411.16 | 6413.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 6241.50 | 6230.99 | 6307.95 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-04 10:15:00 | 6073.00 | 6228.86 | 6303.86 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 11:15:00 | 6056.50 | 6227.14 | 6302.63 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 6286.50 | 6189.93 | 6269.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 6286.50 | 6189.93 | 6269.76 | SL hit (close>ema400) qty=1.00 sl=6269.76 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-12 12:15:00 | 6196.00 | 6197.05 | 6270.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 6211.50 | 6197.19 | 6269.99 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 6313.50 | 6199.08 | 6260.31 | SL hit (close>static) qty=1.00 sl=6307.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-13 09:15:00 | 6169.00 | 6292.73 | 6293.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 10:15:00 | 6149.00 | 6291.30 | 6293.12 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 13:15:00 | 5841.55 | 6149.76 | 6209.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-10 11:15:00 | 6091.00 | 6061.97 | 6149.57 | SL hit (close>ema200) qty=0.50 sl=6061.97 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 6431.00 | 6200.63 | 6199.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 6468.00 | 6205.80 | 6202.11 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-22 10:15:00 | 6144.50 | 2025-09-26 09:15:00 | 5837.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 10:15:00 | 6144.50 | 2025-10-07 11:15:00 | 6026.00 | STOP_HIT | 0.50 | 1.93% |
| SELL | retest2 | 2025-10-08 15:15:00 | 6106.00 | 2025-10-10 11:15:00 | 6386.00 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2025-10-09 14:15:00 | 6132.50 | 2025-10-10 11:15:00 | 6386.00 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2025-11-26 10:15:00 | 6480.50 | 2025-12-02 09:15:00 | 6374.50 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-11-27 14:15:00 | 6494.00 | 2025-12-02 09:15:00 | 6374.50 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-12-04 11:15:00 | 6489.50 | 2025-12-05 09:15:00 | 6404.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-12-04 14:15:00 | 6479.00 | 2025-12-05 09:15:00 | 6404.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-12-05 15:15:00 | 6486.00 | 2025-12-08 10:15:00 | 6384.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-12-19 10:15:00 | 6515.50 | 2025-12-29 09:15:00 | 6390.50 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-12-22 10:15:00 | 6526.00 | 2025-12-29 09:15:00 | 6390.50 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2026-01-06 10:15:00 | 6543.50 | 2026-01-13 11:15:00 | 6391.50 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest1 | 2026-02-04 11:15:00 | 6056.50 | 2026-02-11 11:15:00 | 6286.50 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2026-02-12 13:15:00 | 6211.50 | 2026-02-19 09:15:00 | 6313.50 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-03-13 10:15:00 | 6149.00 | 2026-04-01 13:15:00 | 5841.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 10:15:00 | 6149.00 | 2026-04-10 11:15:00 | 6091.00 | STOP_HIT | 0.50 | 0.94% |
