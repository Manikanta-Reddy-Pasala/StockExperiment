# SBIN (SBIN)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2025-11-10 09:15:00 → 2026-05-07 15:15:00 (847 bars)
- **Last close:** 1092.90
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 5 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -2.08% / -1.56%
- **Sum % (uncompounded):** -8.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.08% | -8.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.08% | -8.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.08% | -8.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 15:15:00 | 1018.00 | 1074.83 | 1075.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 09:15:00 | 1014.10 | 1074.23 | 1074.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1073.00 | 1068.06 | 1071.53 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1073.00 | 1068.06 | 1071.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1073.00 | 1068.06 | 1071.53 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 11:15:00 | 1067.00 | 1068.06 | 1071.49 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 13:15:00 | 1062.40 | 1067.98 | 1071.42 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1078.95 | 1065.41 | 1069.66 | SL hit (close>static) qty=1.00 sl=1077.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-16 11:15:00 | 1067.00 | 1065.96 | 1069.75 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 13:15:00 | 1067.00 | 1065.98 | 1069.72 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-04-17 14:15:00 | 1080.10 | 1066.34 | 1069.76 | SL hit (close>static) qty=1.00 sl=1077.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-30 14:15:00 | 1067.65 | 1080.79 | 1077.39 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-30 15:15:00 | 1071.80 | 1080.70 | 1077.36 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-04 15:15:00 | 1067.50 | 1080.23 | 1077.24 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 1063.10 | 1080.06 | 1077.17 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-05-06 10:15:00 | 1065.70 | 1078.64 | 1076.55 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 1064.50 | 1078.39 | 1076.45 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1093.20 | 1078.47 | 1076.51 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 1093.20 | 1078.47 | 1076.51 | SL hit (close>static) qty=1.00 sl=1077.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 1093.20 | 1078.47 | 1076.51 | SL hit (close>static) qty=1.00 sl=1077.70 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-08 13:15:00 | 1062.40 | 2026-04-15 09:15:00 | 1078.95 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-04-16 13:15:00 | 1067.00 | 2026-04-17 14:15:00 | 1080.10 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-05-05 09:15:00 | 1063.10 | 2026-05-06 14:15:00 | 1093.20 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2026-05-06 12:15:00 | 1064.50 | 2026-05-06 14:15:00 | 1093.20 | STOP_HIT | 1.00 | -2.70% |
