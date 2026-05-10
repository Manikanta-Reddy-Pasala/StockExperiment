# SBI Cards and Payment Services Ltd. (SBICARD)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 645.00
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
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 12
- **Target hits / Stop hits / Partials:** 2 / 14 / 4
- **Avg / median % per leg:** 1.23% / -0.81%
- **Sum % (uncompounded):** 24.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.26% | -7.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.26% | -7.6% |
| SELL (all) | 14 | 8 | 57.1% | 2 | 8 | 4 | 2.29% | 32.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 8 | 57.1% | 2 | 8 | 4 | 2.29% | 32.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 8 | 40.0% | 2 | 14 | 4 | 1.23% | 24.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 11:15:00 | 844.10 | 914.46 | 914.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 840.10 | 913.72 | 914.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 14:15:00 | 819.65 | 818.61 | 844.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 15:00:00 | 819.65 | 818.61 | 844.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 844.50 | 819.24 | 843.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 844.50 | 819.24 | 843.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 843.25 | 819.48 | 843.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 840.15 | 819.48 | 843.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 845.95 | 819.74 | 843.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:45:00 | 846.45 | 819.74 | 843.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 853.90 | 820.08 | 843.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 853.90 | 820.08 | 843.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 857.00 | 856.12 | 857.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 854.00 | 856.12 | 857.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 861.40 | 856.17 | 857.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 861.40 | 856.17 | 857.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 861.30 | 856.22 | 857.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:30:00 | 863.35 | 856.22 | 857.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 890.95 | 857.95 | 857.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 899.00 | 858.36 | 858.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 15:15:00 | 898.00 | 899.93 | 883.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:15:00 | 894.00 | 899.93 | 883.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 885.35 | 900.44 | 885.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 15:00:00 | 885.35 | 900.44 | 885.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 886.00 | 900.29 | 885.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 891.20 | 900.29 | 885.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 883.50 | 899.89 | 885.70 | SL hit (close<static) qty=1.00 sl=884.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 12:45:00 | 888.45 | 898.69 | 885.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 14:30:00 | 887.60 | 898.44 | 885.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 15:00:00 | 887.60 | 898.44 | 885.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 886.00 | 898.32 | 885.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 886.20 | 898.32 | 885.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 894.75 | 898.28 | 885.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 10:15:00 | 896.85 | 898.28 | 885.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 14:15:00 | 880.40 | 897.88 | 885.80 | SL hit (close<static) qty=1.00 sl=884.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 14:15:00 | 880.40 | 897.88 | 885.80 | SL hit (close<static) qty=1.00 sl=884.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 14:15:00 | 880.40 | 897.88 | 885.80 | SL hit (close<static) qty=1.00 sl=884.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 14:15:00 | 880.40 | 897.88 | 885.80 | SL hit (close<static) qty=1.00 sl=881.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:45:00 | 898.55 | 883.52 | 881.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 12:15:00 | 877.60 | 883.58 | 881.35 | SL hit (close<static) qty=1.00 sl=881.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 870.25 | 879.77 | 879.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 860.15 | 879.48 | 879.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 882.55 | 877.41 | 878.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 882.55 | 877.41 | 878.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 882.55 | 877.41 | 878.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 882.55 | 877.41 | 878.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 875.45 | 877.39 | 878.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:00:00 | 873.05 | 877.35 | 878.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:30:00 | 872.60 | 877.30 | 878.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 15:00:00 | 872.60 | 877.30 | 878.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 873.55 | 877.22 | 878.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 12:15:00 | 829.40 | 873.38 | 876.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 12:15:00 | 829.87 | 873.38 | 876.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 870.60 | 869.69 | 874.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 870.60 | 869.69 | 874.13 | SL hit (close>ema200) qty=0.50 sl=869.69 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 870.60 | 869.69 | 874.13 | SL hit (close>ema200) qty=0.50 sl=869.69 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 10:00:00 | 868.65 | 870.03 | 874.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 14:00:00 | 868.75 | 870.08 | 874.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 879.95 | 870.18 | 874.04 | SL hit (close>static) qty=1.00 sl=878.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 879.95 | 870.18 | 874.04 | SL hit (close>static) qty=1.00 sl=878.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 13:45:00 | 869.10 | 870.29 | 874.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 10:00:00 | 866.00 | 870.18 | 873.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 868.75 | 870.17 | 873.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:30:00 | 873.05 | 870.17 | 873.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 871.20 | 870.18 | 873.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:45:00 | 872.55 | 870.18 | 873.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 874.00 | 865.94 | 871.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 875.20 | 865.94 | 871.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 878.65 | 866.07 | 871.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-02 11:15:00 | 878.65 | 866.07 | 871.04 | SL hit (close>static) qty=1.00 sl=878.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 11:15:00 | 878.65 | 866.07 | 871.04 | SL hit (close>static) qty=1.00 sl=878.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-02 11:45:00 | 880.95 | 866.07 | 871.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 875.75 | 866.25 | 871.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:00:00 | 875.75 | 866.25 | 871.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 873.45 | 866.58 | 871.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:00:00 | 873.45 | 866.58 | 871.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 870.95 | 866.62 | 871.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 886.75 | 867.33 | 871.33 | SL hit (close>static) qty=1.00 sl=884.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 886.75 | 867.33 | 871.33 | SL hit (close>static) qty=1.00 sl=884.85 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 866.50 | 869.99 | 872.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:45:00 | 868.65 | 869.96 | 872.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 825.22 | 862.73 | 868.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:15:00 | 823.17 | 861.59 | 867.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 15:15:00 | 781.78 | 855.46 | 863.96 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-23 11:15:00 | 779.85 | 848.98 | 860.22 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-31 09:15:00 | 891.20 | 2025-10-31 11:15:00 | 883.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-11-03 12:45:00 | 888.45 | 2025-11-04 14:15:00 | 880.40 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-03 14:30:00 | 887.60 | 2025-11-04 14:15:00 | 880.40 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-03 15:00:00 | 887.60 | 2025-11-04 14:15:00 | 880.40 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-04 10:15:00 | 896.85 | 2025-11-04 14:15:00 | 880.40 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-11-24 09:45:00 | 898.55 | 2025-11-24 12:15:00 | 877.60 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-12-11 14:00:00 | 873.05 | 2025-12-17 12:15:00 | 829.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-11 14:30:00 | 872.60 | 2025-12-17 12:15:00 | 829.87 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-12-11 14:00:00 | 873.05 | 2025-12-19 15:15:00 | 870.60 | STOP_HIT | 0.50 | 0.28% |
| SELL | retest2 | 2025-12-11 14:30:00 | 872.60 | 2025-12-19 15:15:00 | 870.60 | STOP_HIT | 0.50 | 0.23% |
| SELL | retest2 | 2025-12-11 15:00:00 | 872.60 | 2025-12-24 10:15:00 | 879.95 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-12 10:45:00 | 873.55 | 2025-12-24 10:15:00 | 879.95 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-12-23 10:00:00 | 868.65 | 2026-01-02 11:15:00 | 878.65 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-12-23 14:00:00 | 868.75 | 2026-01-02 11:15:00 | 878.65 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-12-24 13:45:00 | 869.10 | 2026-01-06 12:15:00 | 886.75 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-12-26 10:00:00 | 866.00 | 2026-01-06 12:15:00 | 886.75 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-01-09 09:15:00 | 866.50 | 2026-01-20 09:15:00 | 825.22 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2026-01-09 10:45:00 | 868.65 | 2026-01-20 12:15:00 | 823.17 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2026-01-09 09:15:00 | 866.50 | 2026-01-21 15:15:00 | 781.78 | TARGET_HIT | 0.50 | 9.78% |
| SELL | retest2 | 2026-01-09 10:45:00 | 868.65 | 2026-01-23 11:15:00 | 779.85 | TARGET_HIT | 0.50 | 10.22% |
