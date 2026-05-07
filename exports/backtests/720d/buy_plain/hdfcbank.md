# HDFCBANK (HDFCBANK)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 795.55
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 4 |
| PENDING | 12 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 11
- **Target hits / Stop hits / Partials:** 0 / 12 / 1
- **Avg / median % per leg:** 0.90% / -0.58%
- **Sum % (uncompounded):** 11.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 2 | 15.4% | 0 | 12 | 1 | 0.90% | 11.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 2 | 15.4% | 0 | 12 | 1 | 0.90% | 11.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 2 | 15.4% | 0 | 12 | 1 | 0.90% | 11.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 15:15:00 | 886.00 | 856.78 | 856.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 897.25 | 857.18 | 856.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 10:15:00 | 876.75 | 878.76 | 869.60 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 12:15:00 | 872.08 | 878.64 | 869.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 12:15:00 | 872.08 | 878.64 | 869.63 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-07 14:15:00 | 879.48 | 878.60 | 869.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 15:15:00 | 880.03 | 878.62 | 869.75 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-06-26 14:15:00 | 1012.03 | 970.96 | 953.77 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 985.60 | 986.12 | 968.59 | SL hit (close<ema200) qty=0.50 sl=986.12 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 1000.55 | 974.00 | 973.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1008.40 | 974.35 | 974.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 14:15:00 | 984.05 | 986.98 | 981.78 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 979.85 | 986.89 | 981.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 979.85 | 986.89 | 981.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-06 11:15:00 | 987.30 | 986.85 | 981.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 12:15:00 | 989.10 | 986.87 | 981.86 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-07 15:15:00 | 984.30 | 986.43 | 981.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 986.35 | 986.43 | 981.90 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2025-11-11 11:15:00 | 984.10 | 986.32 | 982.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:15:00 | 987.05 | 986.33 | 982.07 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-14 11:15:00 | 984.90 | 986.75 | 982.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 985.85 | 986.74 | 982.72 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 983.35 | 986.71 | 982.73 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-14 14:15:00 | 990.45 | 986.75 | 982.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:15:00 | 989.00 | 986.77 | 982.80 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-19 11:15:00 | 987.20 | 987.65 | 983.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:15:00 | 988.10 | 987.65 | 983.60 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-10 13:15:00 | 988.50 | 995.82 | 990.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 14:15:00 | 989.40 | 995.75 | 990.50 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 981.30 | 995.66 | 991.28 | SL hit (close<static) qty=1.00 sl=981.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 981.30 | 995.66 | 991.28 | SL hit (close<static) qty=1.00 sl=981.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 981.30 | 995.66 | 991.28 | SL hit (close<static) qty=1.00 sl=981.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-22 09:15:00 | 988.30 | 993.73 | 990.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 987.30 | 993.66 | 990.62 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 986.00 | 993.59 | 990.60 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-23 09:15:00 | 994.10 | 993.33 | 990.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 10:15:00 | 993.40 | 993.33 | 990.56 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-29 12:15:00 | 989.40 | 993.43 | 990.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:15:00 | 990.50 | 993.40 | 990.92 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 985.00 | 993.21 | 990.89 | SL hit (close<static) qty=1.00 sl=985.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 985.00 | 993.21 | 990.89 | SL hit (close<static) qty=1.00 sl=985.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-30 14:15:00 | 990.90 | 993.03 | 990.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 15:15:00 | 995.00 | 993.05 | 990.85 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 983.85 | 993.19 | 991.17 | SL hit (close<static) qty=1.00 sl=985.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 978.15 | 992.93 | 991.06 | SL hit (close<static) qty=1.00 sl=981.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.34 | 990.79 | SL hit (close<static) qty=1.00 sl=974.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.34 | 990.79 | SL hit (close<static) qty=1.00 sl=974.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.34 | 990.79 | SL hit (close<static) qty=1.00 sl=974.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.34 | 990.79 | SL hit (close<static) qty=1.00 sl=974.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-04-07 15:15:00 | 880.03 | 2025-06-26 14:15:00 | 1012.03 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-07 15:15:00 | 880.03 | 2025-07-14 10:15:00 | 985.60 | STOP_HIT | 0.50 | 12.00% |
| BUY | retest2 | 2025-11-06 12:15:00 | 989.10 | 2025-12-17 12:15:00 | 981.30 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-11-10 09:15:00 | 986.35 | 2025-12-17 12:15:00 | 981.30 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-11-11 12:15:00 | 987.05 | 2025-12-17 12:15:00 | 981.30 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-11-14 12:15:00 | 985.85 | 2025-12-30 11:15:00 | 985.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-11-14 15:15:00 | 989.00 | 2025-12-30 11:15:00 | 985.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-11-19 12:15:00 | 988.10 | 2026-01-05 11:15:00 | 983.85 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-12-10 14:15:00 | 989.40 | 2026-01-05 13:15:00 | 978.15 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-12-22 10:15:00 | 987.30 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-12-23 10:15:00 | 993.40 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-12-29 13:15:00 | 990.50 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-12-30 15:15:00 | 995.00 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.19% |
