# HDFCBANK (HDFCBANK)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 781.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 4 |
| PENDING | 11 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 8
- **Target hits / Stop hits / Partials:** 1 / 9 / 1
- **Avg / median % per leg:** 0.14% / -0.79%
- **Sum % (uncompounded):** 1.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 1 | 11.1% | 0 | 9 | 0 | -1.49% | -13.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 1 | 11.1% | 0 | 9 | 0 | -1.49% | -13.4% |
| SELL (all) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 3 | 27.3% | 1 | 9 | 1 | 0.14% | 1.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 945.05 | 982.79 | 982.84 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 1004.00 | 974.95 | 974.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 1009.15 | 976.41 | 975.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 14:15:00 | 984.05 | 986.98 | 982.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 979.85 | 986.90 | 982.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 979.85 | 986.90 | 982.06 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-06 11:15:00 | 987.30 | 986.85 | 982.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 12:15:00 | 989.10 | 986.87 | 982.12 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-07 15:15:00 | 984.30 | 986.43 | 982.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 986.35 | 986.43 | 982.15 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2025-11-11 11:15:00 | 984.10 | 986.33 | 982.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:15:00 | 987.05 | 986.33 | 982.31 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-14 11:15:00 | 984.90 | 986.75 | 982.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 985.85 | 986.75 | 982.94 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 983.35 | 986.71 | 982.94 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-14 14:15:00 | 990.45 | 986.75 | 982.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:15:00 | 989.00 | 986.77 | 983.01 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-19 11:15:00 | 987.20 | 987.65 | 983.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:15:00 | 988.10 | 987.65 | 983.79 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-10 13:15:00 | 988.50 | 995.82 | 990.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 14:15:00 | 989.40 | 995.75 | 990.62 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 981.30 | 995.66 | 991.37 | SL hit (close<static) qty=1.00 sl=981.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 981.30 | 995.66 | 991.37 | SL hit (close<static) qty=1.00 sl=981.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 981.30 | 995.66 | 991.37 | SL hit (close<static) qty=1.00 sl=981.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-22 09:15:00 | 988.30 | 993.73 | 990.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 987.30 | 993.66 | 990.71 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 991.30 | 993.66 | 991.01 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-31 13:15:00 | 996.40 | 993.03 | 990.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-31 14:15:00 | 990.90 | 993.01 | 990.96 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-02 09:15:00 | 996.30 | 992.99 | 991.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:15:00 | 996.15 | 993.02 | 991.07 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 990.15 | 993.33 | 991.28 | SL hit (close<static) qty=1.00 sl=990.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 978.15 | 992.93 | 991.12 | SL hit (close<static) qty=1.00 sl=981.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.34 | 990.85 | SL hit (close<static) qty=1.00 sl=974.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.34 | 990.85 | SL hit (close<static) qty=1.00 sl=974.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.34 | 990.85 | SL hit (close<static) qty=1.00 sl=974.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.34 | 990.85 | SL hit (close<static) qty=1.00 sl=974.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 948.60 | 989.05 | 989.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 945.70 | 987.46 | 988.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.99 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-06 09:15:00 | 944.15 | 947.54 | 960.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:15:00 | 942.45 | 947.49 | 960.78 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 15:15:00 | 895.33 | 929.26 | 944.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 848.21 | 922.93 | 940.15 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-06 12:15:00 | 989.10 | 2025-12-17 12:15:00 | 981.30 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-11-10 09:15:00 | 986.35 | 2025-12-17 12:15:00 | 981.30 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-11-11 12:15:00 | 987.05 | 2025-12-17 12:15:00 | 981.30 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-11-14 12:15:00 | 985.85 | 2026-01-05 09:15:00 | 990.15 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2025-11-14 15:15:00 | 989.00 | 2026-01-05 13:15:00 | 978.15 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-11-19 12:15:00 | 988.10 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-12-10 14:15:00 | 989.40 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-12-22 10:15:00 | 987.30 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2026-01-02 10:15:00 | 996.15 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2026-02-06 10:15:00 | 942.45 | 2026-02-26 15:15:00 | 895.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 10:15:00 | 942.45 | 2026-03-04 09:15:00 | 848.21 | TARGET_HIT | 0.50 | 10.00% |
