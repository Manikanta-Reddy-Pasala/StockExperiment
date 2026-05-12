# Zydus Lifesciences Ltd. (ZYDUSLIFE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 939.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 12 |
| TARGET_HIT | 0 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 22 / 16
- **Target hits / Stop hits / Partials:** 0 / 26 / 12
- **Avg / median % per leg:** 1.30% / 0.36%
- **Sum % (uncompounded):** 49.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.45% | -14.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.45% | -14.5% |
| SELL (all) | 28 | 22 | 78.6% | 0 | 16 | 12 | 2.28% | 64.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 22 | 78.6% | 0 | 16 | 12 | 2.28% | 64.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 38 | 22 | 57.9% | 0 | 26 | 12 | 1.30% | 49.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 931.80 | 901.81 | 901.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 938.50 | 903.68 | 902.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 965.55 | 966.52 | 946.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:00:00 | 965.55 | 966.52 | 946.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 957.10 | 969.53 | 953.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:15:00 | 960.70 | 969.53 | 953.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:15:00 | 960.90 | 968.95 | 953.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:30:00 | 960.80 | 968.77 | 953.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 951.40 | 972.52 | 958.68 | SL hit (close<static) qty=1.00 sl=951.55 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 930.00 | 991.70 | 991.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 928.20 | 973.13 | 981.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 925.00 | 924.65 | 940.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 09:45:00 | 925.65 | 924.65 | 940.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 939.35 | 924.98 | 939.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 939.35 | 924.98 | 939.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 933.05 | 925.06 | 939.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:30:00 | 928.25 | 925.09 | 939.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 881.84 | 922.53 | 937.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 901.10 | 897.38 | 915.35 | SL hit (close>ema200) qty=0.50 sl=897.38 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 931.50 | 906.06 | 906.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 957.45 | 909.17 | 907.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 913.15 | 914.88 | 910.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 913.15 | 914.88 | 910.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 913.15 | 914.88 | 910.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 10:15:00 | 925.05 | 912.61 | 910.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-22 11:15:00 | 960.70 | 2025-08-01 09:15:00 | 951.40 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-07-23 10:15:00 | 960.90 | 2025-08-01 09:15:00 | 951.40 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-23 11:30:00 | 960.80 | 2025-08-01 09:15:00 | 951.40 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-08-04 13:30:00 | 961.55 | 2025-08-05 09:15:00 | 950.40 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-08-05 09:15:00 | 960.60 | 2025-08-05 09:15:00 | 950.40 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-08-05 12:00:00 | 960.25 | 2025-08-05 12:15:00 | 954.50 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-08-05 15:00:00 | 960.80 | 2025-08-06 09:15:00 | 936.60 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-08-12 12:30:00 | 960.85 | 2025-08-12 13:15:00 | 954.55 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-08-13 11:00:00 | 966.30 | 2025-11-06 13:15:00 | 938.00 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-11-06 13:00:00 | 963.25 | 2025-11-06 13:15:00 | 938.00 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2026-01-07 11:30:00 | 928.25 | 2026-01-12 09:15:00 | 881.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 11:30:00 | 928.25 | 2026-02-03 09:15:00 | 901.10 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2026-02-26 11:30:00 | 931.50 | 2026-03-10 09:15:00 | 917.55 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2026-02-26 12:00:00 | 931.40 | 2026-03-16 10:15:00 | 884.92 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2026-02-27 09:30:00 | 928.45 | 2026-03-16 10:15:00 | 884.83 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2026-03-09 09:15:00 | 892.45 | 2026-03-16 10:15:00 | 882.03 | PARTIAL | 0.50 | 1.17% |
| SELL | retest2 | 2026-03-13 10:15:00 | 904.95 | 2026-03-23 14:15:00 | 859.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 11:00:00 | 905.20 | 2026-03-23 14:15:00 | 859.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 12:15:00 | 903.50 | 2026-03-23 14:15:00 | 858.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:00:00 | 931.40 | 2026-03-25 09:15:00 | 901.65 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2026-02-27 09:30:00 | 928.45 | 2026-03-25 09:15:00 | 901.65 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2026-03-09 09:15:00 | 892.45 | 2026-03-25 09:15:00 | 901.65 | STOP_HIT | 0.50 | -1.03% |
| SELL | retest2 | 2026-03-13 10:15:00 | 904.95 | 2026-03-25 09:15:00 | 901.65 | STOP_HIT | 0.50 | 0.36% |
| SELL | retest2 | 2026-03-13 11:00:00 | 905.20 | 2026-03-25 09:15:00 | 901.65 | STOP_HIT | 0.50 | 0.39% |
| SELL | retest2 | 2026-03-13 12:15:00 | 903.50 | 2026-03-25 09:15:00 | 901.65 | STOP_HIT | 0.50 | 0.20% |
| SELL | retest2 | 2026-03-16 10:15:00 | 887.45 | 2026-04-01 14:15:00 | 858.32 | PARTIAL | 0.50 | 3.28% |
| SELL | retest2 | 2026-03-17 11:30:00 | 892.65 | 2026-04-02 09:15:00 | 843.08 | PARTIAL | 0.50 | 5.55% |
| SELL | retest2 | 2026-03-17 15:00:00 | 891.95 | 2026-04-02 09:15:00 | 848.02 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2026-03-19 09:15:00 | 883.80 | 2026-04-02 09:15:00 | 847.35 | PARTIAL | 0.50 | 4.12% |
| SELL | retest2 | 2026-03-27 12:15:00 | 903.50 | 2026-04-02 09:15:00 | 839.61 | PARTIAL | 0.50 | 7.07% |
| SELL | retest2 | 2026-03-16 10:15:00 | 887.45 | 2026-04-08 14:15:00 | 891.80 | STOP_HIT | 0.50 | -0.49% |
| SELL | retest2 | 2026-03-17 11:30:00 | 892.65 | 2026-04-08 14:15:00 | 891.80 | STOP_HIT | 0.50 | 0.10% |
| SELL | retest2 | 2026-03-17 15:00:00 | 891.95 | 2026-04-08 14:15:00 | 891.80 | STOP_HIT | 0.50 | 0.02% |
| SELL | retest2 | 2026-03-19 09:15:00 | 883.80 | 2026-04-08 14:15:00 | 891.80 | STOP_HIT | 0.50 | -0.91% |
| SELL | retest2 | 2026-03-27 12:15:00 | 903.50 | 2026-04-08 14:15:00 | 891.80 | STOP_HIT | 0.50 | 1.29% |
| SELL | retest2 | 2026-04-09 11:45:00 | 903.75 | 2026-04-10 09:15:00 | 910.85 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-04-09 13:15:00 | 902.90 | 2026-04-10 09:15:00 | 910.85 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-04-13 09:15:00 | 905.00 | 2026-04-13 09:15:00 | 910.65 | STOP_HIT | 1.00 | -0.62% |
