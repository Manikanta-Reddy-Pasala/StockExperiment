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
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 13 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 0
- **Avg / median % per leg:** -1.73% / -1.64%
- **Sum % (uncompounded):** -13.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.73% | -13.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.73% | -13.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.73% | -13.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-21 13:15:00)

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
| CROSSOVER_SKIP | 2026-01-20 11:15:00 | 6083.00 | 6414.56 | 6415.00 | min_gap filter: gap=0.007% < 0.010% |
| TREND_RESET | 2026-01-20 11:15:00 | 6083.00 | 6414.56 | 6415.00 | EMA inversion without crossover edge (EMA200=6414.56 EMA400=6415.00) — end cycle |

### Cycle 2 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 6431.00 | 6200.63 | 6199.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 6468.00 | 6205.80 | 6202.11 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-26 10:15:00 | 6480.50 | 2025-12-02 09:15:00 | 6374.50 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-11-27 14:15:00 | 6494.00 | 2025-12-02 09:15:00 | 6374.50 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-12-04 11:15:00 | 6489.50 | 2025-12-05 09:15:00 | 6404.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-12-04 14:15:00 | 6479.00 | 2025-12-05 09:15:00 | 6404.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-12-05 15:15:00 | 6486.00 | 2025-12-08 10:15:00 | 6384.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-12-19 10:15:00 | 6515.50 | 2025-12-29 09:15:00 | 6390.50 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-12-22 10:15:00 | 6526.00 | 2025-12-29 09:15:00 | 6390.50 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2026-01-06 10:15:00 | 6543.50 | 2026-01-13 11:15:00 | 6391.50 | STOP_HIT | 1.00 | -2.32% |
