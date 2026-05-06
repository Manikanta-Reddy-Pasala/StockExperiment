# HEROMOTOCO (HEROMOTOCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 5170.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 8 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 1
- **Avg / median % per leg:** 5.66% / -0.87%
- **Sum % (uncompounded):** 39.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | 5.66% | 39.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 2 | 28.6% | 1 | 5 | 1 | 5.66% | 39.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 2 | 28.6% | 1 | 5 | 1 | 5.66% | 39.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 14:15:00 | 2997.90 | 2984.67 | 2984.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 3081.95 | 2985.77 | 2985.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 12:15:00 | 2999.00 | 3004.97 | 2995.47 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 12:15:00 | 2999.00 | 3004.97 | 2995.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 2999.00 | 3004.97 | 2995.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-09-22 09:15:00 | 3013.80 | 3004.88 | 2995.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 10:15:00 | 3032.80 | 3005.16 | 2995.80 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-09-25 09:15:00 | 2994.25 | 3006.05 | 2996.54 | SL hit qty=1.00 sl=2994.25 alert=retest2 |
| Cross detected — sustain check pending | 2023-09-26 10:15:00 | 3017.20 | 3004.85 | 2996.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-09-26 11:15:00 | 3002.80 | 3004.82 | 2996.32 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-09-26 14:15:00 | 3036.30 | 3005.14 | 2996.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-26 15:15:00 | 3037.75 | 3005.47 | 2996.81 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-09-28 10:15:00 | 2994.25 | 3005.90 | 2997.42 | SL hit qty=1.00 sl=2994.25 alert=retest2 |
| Cross detected — sustain check pending | 2023-09-29 10:15:00 | 3015.00 | 3004.81 | 2997.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 11:15:00 | 3023.85 | 3005.00 | 2997.29 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-10-03 09:15:00 | 2994.25 | 3006.48 | 2998.23 | SL hit qty=1.00 sl=2994.25 alert=retest2 |
| Cross detected — sustain check pending | 2023-10-03 10:15:00 | 3023.00 | 3006.65 | 2998.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 11:15:00 | 3019.35 | 3006.77 | 2998.46 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 3009.85 | 3007.20 | 2998.88 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2023-10-04 09:15:00 | 2994.25 | 3007.20 | 2998.88 | SL hit qty=1.00 sl=2994.25 alert=retest2 |
| Cross detected — sustain check pending | 2023-10-05 11:15:00 | 3016.80 | 3006.57 | 2998.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-05 12:15:00 | 3009.00 | 3006.59 | 2998.97 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-10-05 14:15:00 | 3017.55 | 3006.72 | 2999.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 15:15:00 | 3016.65 | 3006.82 | 2999.20 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-10-09 09:15:00 | 2990.50 | 3008.92 | 3000.57 | SL hit qty=1.00 sl=2990.50 alert=retest2 |
| Cross detected — sustain check pending | 2023-10-11 11:15:00 | 3018.00 | 3004.59 | 2998.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 12:15:00 | 3087.20 | 3005.42 | 2999.38 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2023-11-23 10:15:00 | 3550.28 | 3186.86 | 3122.89 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2023-12-26 09:15:00 | 4013.36 | 3669.33 | 3474.18 | Target hit (30%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2024-10-23 09:15:00 | 5216.70 | 5528.42 | 5529.56 | HTF filter: close above htf_sma |
| CROSSOVER_SKIP | 2025-05-16 13:15:00 | 4314.90 | 3887.26 | 3887.16 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2026-01-28 12:15:00 | 5379.00 | 5703.51 | 5705.06 | HTF filter: close above htf_sma |

### Cycle 2 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 5796.50 | 5701.05 | 5700.98 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2026-02-13 14:15:00 | 5593.50 | 5700.19 | 5700.69 | HTF filter: close above htf_sma |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-09-22 10:15:00 | 3032.80 | 2023-09-25 09:15:00 | 2994.25 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2023-09-26 15:15:00 | 3037.75 | 2023-09-28 10:15:00 | 2994.25 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2023-09-29 11:15:00 | 3023.85 | 2023-10-03 09:15:00 | 2994.25 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2023-10-03 11:15:00 | 3019.35 | 2023-10-04 09:15:00 | 2994.25 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-10-05 15:15:00 | 3016.65 | 2023-10-09 09:15:00 | 2990.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-10-11 12:15:00 | 3087.20 | 2023-11-23 10:15:00 | 3550.28 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-10-11 12:15:00 | 3087.20 | 2023-12-26 09:15:00 | 4013.36 | TARGET_HIT | 0.50 | 30.00% |
