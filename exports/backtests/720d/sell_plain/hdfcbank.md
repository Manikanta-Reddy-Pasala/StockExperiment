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
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 4 |
| PENDING | 12 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 1 |
| ENTRY2 | 6 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / Stop hits / Partials:** 0 / 7 / 1
- **Avg / median % per leg:** 2.53% / -1.15%
- **Sum % (uncompounded):** 20.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 2 | 25.0% | 0 | 7 | 1 | 2.53% | 20.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.59% | -1.6% |
| SELL @ 3rd Alert (retest2) | 7 | 2 | 28.6% | 0 | 6 | 1 | 3.12% | 21.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.59% | -1.6% |
| retest2 (combined) | 7 | 2 | 28.6% | 0 | 6 | 1 | 3.12% | 21.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 10:15:00 | 826.45 | 876.41 | 876.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 11:15:00 | 821.00 | 875.86 | 876.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 12:15:00 | 851.55 | 848.53 | 859.09 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-02-01 11:15:00 | 843.00 | 848.53 | 858.78 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 13:15:00 | 844.88 | 848.42 | 858.62 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 858.30 | 847.94 | 857.73 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-02-04 12:15:00 | 858.30 | 847.94 | 857.73 | SL hit (close>ema400) qty=1.00 sl=857.73 alert=retest1 |
| Cross detected — sustain check pending | 2025-02-11 09:15:00 | 849.45 | 852.29 | 858.64 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-11 10:15:00 | 851.98 | 852.29 | 858.61 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-11 12:15:00 | 848.20 | 852.25 | 858.53 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-11 14:15:00 | 851.20 | 852.19 | 858.44 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-02-12 09:15:00 | 843.88 | 852.09 | 858.32 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 11:15:00 | 848.08 | 851.95 | 858.19 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-02-13 13:15:00 | 848.53 | 852.04 | 857.96 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:15:00 | 849.48 | 851.98 | 857.88 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 859.23 | 851.50 | 857.25 | SL hit (close>static) qty=1.00 sl=858.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 859.23 | 851.50 | 857.25 | SL hit (close>static) qty=1.00 sl=858.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-20 09:15:00 | 845.03 | 852.91 | 857.54 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 11:15:00 | 844.35 | 852.75 | 857.42 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-02-27 14:15:00 | 848.73 | 850.10 | 855.31 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-28 09:15:00 | 850.95 | 850.10 | 855.25 | ENTRY2 sustain failed after 1140m |
| Stop hit — per-position SL triggered | 2025-02-28 13:15:00 | 862.48 | 850.44 | 855.32 | SL hit (close>static) qty=1.00 sl=858.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-03 13:15:00 | 848.73 | 850.86 | 855.37 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-03 14:15:00 | 850.60 | 850.86 | 855.35 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-05 12:15:00 | 849.35 | 851.06 | 855.19 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 14:15:00 | 844.70 | 850.95 | 855.09 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 852.33 | 849.39 | 853.67 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-03-12 12:15:00 | 858.75 | 849.58 | 853.70 | SL hit (close>static) qty=1.00 sl=858.30 alert=retest2 |

### Cycle 2 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 958.70 | 980.47 | 980.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 947.40 | 971.43 | 974.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 11:15:00 | 968.95 | 965.15 | 970.87 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-10-03 09:15:00 | 957.75 | 965.04 | 970.68 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-03 10:15:00 | 962.90 | 965.02 | 970.64 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 971.30 | 965.03 | 970.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 971.30 | 965.03 | 970.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-07 09:15:00 | 951.50 | 990.26 | 989.78 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 948.05 | 989.45 | 989.38 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-07 12:15:00 | 948.60 | 989.05 | 989.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 948.60 | 989.05 | 989.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 945.70 | 987.46 | 988.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.96 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.96 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-06 09:15:00 | 944.15 | 947.54 | 960.84 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 941.00 | 947.43 | 960.65 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-19 09:15:00 | 799.85 | 879.20 | 909.00 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 15:15:00 | 817.20 | 816.16 | 860.87 | SL hit (close>ema200) qty=0.50 sl=816.16 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-02-01 13:15:00 | 844.88 | 2025-02-04 12:15:00 | 858.30 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-02-12 11:15:00 | 848.08 | 2025-02-17 14:15:00 | 859.23 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-02-13 15:15:00 | 849.48 | 2025-02-17 14:15:00 | 859.23 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-02-20 11:15:00 | 844.35 | 2025-02-28 13:15:00 | 862.48 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-03-05 14:15:00 | 844.70 | 2025-03-12 12:15:00 | 858.75 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-01-07 11:15:00 | 948.05 | 2026-01-07 12:15:00 | 948.60 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2026-02-06 11:15:00 | 941.00 | 2026-03-19 09:15:00 | 799.85 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-02-06 11:15:00 | 941.00 | 2026-04-08 15:15:00 | 817.20 | STOP_HIT | 0.50 | 13.16% |
