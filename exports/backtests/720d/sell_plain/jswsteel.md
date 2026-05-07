# JSWSTEEL (JSWSTEEL)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1280.40
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
| PENDING | 5 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 1 |
| ENTRY2 | 3 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -3.65% / -3.79%
- **Sum % (uncompounded):** -14.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.65% | -14.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.79% | -3.8% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.60% | -10.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.79% | -3.8% |
| retest2 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.60% | -10.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 14:15:00 | 920.90 | 968.46 | 968.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 12:15:00 | 918.20 | 966.19 | 967.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 09:15:00 | 924.80 | 923.64 | 939.23 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-22 11:15:00 | 915.35 | 923.75 | 938.60 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 13:15:00 | 911.80 | 923.54 | 938.34 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 932.45 | 923.99 | 937.85 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-24 10:15:00 | 946.40 | 924.21 | 937.89 | SL hit (close>ema400) qty=1.00 sl=937.89 alert=retest1 |
| Cross detected — sustain check pending | 2025-01-27 09:15:00 | 918.05 | 924.79 | 937.78 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 11:15:00 | 913.10 | 924.54 | 937.53 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 954.30 | 924.57 | 936.34 | SL hit (close>static) qty=1.00 sl=945.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-01 12:15:00 | 919.00 | 928.03 | 937.19 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-01 13:15:00 | 932.55 | 928.08 | 937.17 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-03 09:15:00 | 925.85 | 928.15 | 937.08 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 11:15:00 | 926.20 | 928.11 | 936.96 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-02-05 09:15:00 | 949.00 | 929.32 | 937.06 | SL hit (close>static) qty=1.00 sl=945.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-07 10:15:00 | 929.10 | 1018.60 | 995.07 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 12:15:00 | 912.60 | 1016.53 | 994.27 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-08 11:15:00 | 947.40 | 1011.88 | 992.56 | SL hit (close>static) qty=1.00 sl=945.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1076.90 | 1134.81 | 1135.00 | EMA200 below EMA400 |

### Cycle 3 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 1120.50 | 1191.41 | 1191.63 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-01-22 13:15:00 | 911.80 | 2025-01-24 10:15:00 | 946.40 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2025-01-27 11:15:00 | 913.10 | 2025-01-30 09:15:00 | 954.30 | STOP_HIT | 1.00 | -4.51% |
| SELL | retest2 | 2025-02-03 11:15:00 | 926.20 | 2025-02-05 09:15:00 | 949.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-04-07 12:15:00 | 912.60 | 2025-04-08 11:15:00 | 947.40 | STOP_HIT | 1.00 | -3.81% |
