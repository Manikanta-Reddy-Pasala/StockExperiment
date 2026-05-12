# PTC Industries Ltd. (PTCIL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-11 15:15:00 (3717 bars)
- **Last close:** 17080.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 167 |
| ALERT1 | 101 |
| ALERT2 | 96 |
| ALERT2_SKIP | 56 |
| ALERT3 | 259 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 146 |
| PARTIAL | 31 |
| TARGET_HIT | 22 |
| STOP_HIT | 130 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 183 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 88 / 95
- **Target hits / Stop hits / Partials:** 22 / 130 / 31
- **Avg / median % per leg:** 1.41% / -0.23%
- **Sum % (uncompounded):** 257.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 26 | 41.9% | 15 | 46 | 1 | 2.15% | 133.4% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | 1.22% | 8.6% |
| BUY @ 3rd Alert (retest2) | 55 | 24 | 43.6% | 14 | 41 | 0 | 2.27% | 124.8% |
| SELL (all) | 121 | 62 | 51.2% | 7 | 84 | 30 | 1.02% | 123.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 121 | 62 | 51.2% | 7 | 84 | 30 | 1.02% | 123.9% |
| retest1 (combined) | 7 | 2 | 28.6% | 1 | 5 | 1 | 1.22% | 8.6% |
| retest2 (combined) | 176 | 86 | 48.9% | 21 | 125 | 30 | 1.41% | 248.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 7374.05 | 7290.36 | 7288.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 7463.95 | 7363.85 | 7327.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 7386.05 | 7397.51 | 7358.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 15:00:00 | 7386.05 | 7397.51 | 7358.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 7405.00 | 7399.01 | 7362.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 7444.10 | 7412.89 | 7389.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 11:30:00 | 7452.20 | 7429.39 | 7403.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 12:00:00 | 7441.90 | 7429.39 | 7403.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 12:45:00 | 7450.00 | 7431.30 | 7406.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 7420.75 | 7430.58 | 7410.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 14:45:00 | 7415.20 | 7430.58 | 7410.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 7433.95 | 7431.26 | 7412.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 7490.00 | 7431.26 | 7412.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 11:30:00 | 7473.80 | 7434.00 | 7416.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 12:00:00 | 7470.00 | 7434.00 | 7416.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-21 11:15:00 | 8188.51 | 7675.64 | 7539.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 12:15:00 | 7906.20 | 8055.37 | 8069.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 13:15:00 | 7847.65 | 7920.93 | 7978.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 09:15:00 | 7908.80 | 7808.81 | 7861.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 7908.80 | 7808.81 | 7861.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 7908.80 | 7808.81 | 7861.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:00:00 | 7908.80 | 7808.81 | 7861.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 7948.20 | 7836.69 | 7869.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:45:00 | 8000.00 | 7836.69 | 7869.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 7910.10 | 7855.97 | 7873.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:00:00 | 7910.10 | 7855.97 | 7873.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 7895.95 | 7863.96 | 7875.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 15:15:00 | 7814.45 | 7861.57 | 7873.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 09:15:00 | 8550.00 | 7991.72 | 7929.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 09:15:00 | 8550.00 | 7991.72 | 7929.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 8729.70 | 8394.86 | 8204.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 9365.00 | 9379.16 | 8888.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 9365.00 | 9379.16 | 8888.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 9365.00 | 9379.16 | 8888.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 9350.10 | 9379.16 | 8888.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 8260.35 | 9155.40 | 8831.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 8260.35 | 9155.40 | 8831.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 8286.95 | 8981.71 | 8781.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:15:00 | 8050.10 | 8981.71 | 8781.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 8226.75 | 8728.62 | 8706.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 8226.75 | 8728.62 | 8706.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 15:15:00 | 8250.00 | 8632.89 | 8665.02 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 9833.95 | 8890.17 | 8773.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 10918.45 | 9719.11 | 9255.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 12:15:00 | 10792.05 | 10799.32 | 10501.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 13:00:00 | 10792.05 | 10799.32 | 10501.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 13863.60 | 13934.56 | 13593.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 12:00:00 | 14291.75 | 13996.47 | 13680.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 13504.35 | 13686.86 | 13709.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 13504.35 | 13686.86 | 13709.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 13021.80 | 13489.19 | 13607.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 14:15:00 | 13495.00 | 13486.58 | 13585.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 15:00:00 | 13495.00 | 13486.58 | 13585.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 13700.50 | 13515.67 | 13580.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:45:00 | 13819.05 | 13515.67 | 13580.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 13766.30 | 13565.80 | 13597.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:45:00 | 13776.15 | 13565.80 | 13597.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 13630.00 | 13614.90 | 13616.20 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 14:15:00 | 13699.15 | 13631.75 | 13623.74 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 15:15:00 | 13474.45 | 13600.29 | 13610.17 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 13700.00 | 13620.23 | 13618.34 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 13:15:00 | 13548.00 | 13618.81 | 13620.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 14:15:00 | 13426.65 | 13580.38 | 13602.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 15:15:00 | 13270.00 | 13208.48 | 13358.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 15:15:00 | 13270.00 | 13208.48 | 13358.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 13270.00 | 13208.48 | 13358.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 11:00:00 | 13141.70 | 13214.59 | 13336.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:00:00 | 13150.10 | 13201.69 | 13319.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:45:00 | 13086.70 | 13167.40 | 13293.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 13723.55 | 13260.48 | 13312.65 | SL hit (close>static) qty=1.00 sl=13540.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 15:15:00 | 13752.00 | 13358.79 | 13352.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 14306.50 | 13729.54 | 13579.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 15:15:00 | 13935.00 | 13938.81 | 13774.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 09:15:00 | 14149.95 | 13938.81 | 13774.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 13849.95 | 13938.19 | 13803.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:30:00 | 13986.20 | 13938.19 | 13803.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 13920.05 | 13934.56 | 13814.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 12:30:00 | 13995.00 | 13938.38 | 13826.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 13:00:00 | 13953.65 | 13938.38 | 13826.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:15:00 | 13980.00 | 13938.69 | 13837.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 15:15:00 | 14646.95 | 14733.19 | 14735.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 15:15:00 | 14646.95 | 14733.19 | 14735.28 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 09:15:00 | 14854.85 | 14741.64 | 14729.24 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 14439.40 | 14681.19 | 14702.89 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 15303.75 | 14739.69 | 14703.26 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 14568.10 | 14757.98 | 14776.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 14:15:00 | 14265.00 | 14506.12 | 14610.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 14:15:00 | 14404.95 | 14211.58 | 14373.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 14:15:00 | 14404.95 | 14211.58 | 14373.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 14404.95 | 14211.58 | 14373.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:00:00 | 14404.95 | 14211.58 | 14373.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 14399.00 | 14249.06 | 14375.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 14094.90 | 14249.06 | 14375.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 13:00:00 | 14070.00 | 14104.66 | 14255.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 13:45:00 | 14000.00 | 14113.53 | 14246.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 14:15:00 | 14000.00 | 14113.53 | 14246.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 14258.75 | 14142.57 | 14247.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 14258.75 | 14142.57 | 14247.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 14299.00 | 14173.86 | 14251.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 13900.00 | 14173.86 | 14251.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 15:00:00 | 13799.60 | 13863.52 | 14035.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 13390.15 | 13741.32 | 13946.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 13366.50 | 13741.32 | 13946.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 13300.00 | 13741.32 | 13946.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 13300.00 | 13741.32 | 13946.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 13205.00 | 13741.32 | 13946.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-22 14:15:00 | 13849.70 | 13657.40 | 13815.53 | SL hit (close>ema200) qty=0.50 sl=13657.40 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 14:15:00 | 14089.00 | 13757.42 | 13733.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 10:15:00 | 14350.00 | 13974.60 | 13846.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 13:15:00 | 14080.00 | 14168.03 | 14067.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 13:15:00 | 14080.00 | 14168.03 | 14067.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 14080.00 | 14168.03 | 14067.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 13:30:00 | 14185.00 | 14168.03 | 14067.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 14120.00 | 14158.42 | 14072.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:45:00 | 14114.00 | 14158.42 | 14072.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 13930.05 | 14112.75 | 14059.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:15:00 | 13480.00 | 14112.75 | 14059.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 09:15:00 | 13475.50 | 13985.30 | 14006.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 15:15:00 | 13250.00 | 13463.71 | 13588.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 14:15:00 | 13620.00 | 13335.49 | 13442.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 14:15:00 | 13620.00 | 13335.49 | 13442.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 14:15:00 | 13620.00 | 13335.49 | 13442.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 15:00:00 | 13620.00 | 13335.49 | 13442.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 13425.00 | 13353.39 | 13440.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 09:15:00 | 13300.00 | 13353.39 | 13440.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 10:00:00 | 13370.00 | 13356.72 | 13434.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 11:00:00 | 13301.00 | 13345.57 | 13422.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 11:45:00 | 13400.00 | 13364.06 | 13423.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 13447.00 | 13380.65 | 13425.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:45:00 | 13450.00 | 13380.65 | 13425.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 13883.00 | 13481.12 | 13467.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 13883.00 | 13481.12 | 13467.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 14:15:00 | 13999.00 | 13802.68 | 13678.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 13:15:00 | 14062.00 | 14069.88 | 13891.41 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 14:45:00 | 14260.00 | 14117.90 | 13929.47 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:00:00 | 14189.00 | 14135.77 | 13985.41 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 14011.00 | 14100.29 | 13994.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 12:30:00 | 13939.50 | 14100.29 | 13994.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 14000.00 | 14080.23 | 13995.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 13:30:00 | 13902.00 | 14080.23 | 13995.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 14200.00 | 14104.19 | 14013.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:30:00 | 13975.00 | 14104.19 | 14013.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 14182.00 | 14119.75 | 14029.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-13 09:15:00 | 14000.00 | 14095.80 | 14026.47 | SL hit (close<ema400) qty=1.00 sl=14026.47 alert=retest1 |

### Cycle 20 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 13832.95 | 13965.54 | 13980.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 15:15:00 | 13832.00 | 13927.99 | 13959.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 13:15:00 | 13699.00 | 13688.72 | 13809.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 14:00:00 | 13699.00 | 13688.72 | 13809.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 14149.00 | 13780.77 | 13840.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 15:00:00 | 14149.00 | 13780.77 | 13840.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 14100.00 | 13844.62 | 13864.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:15:00 | 14096.00 | 13844.62 | 13864.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 13919.50 | 13884.46 | 13880.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 14160.00 | 13968.80 | 13923.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 09:15:00 | 13730.00 | 13950.03 | 13924.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 09:15:00 | 13730.00 | 13950.03 | 13924.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 13730.00 | 13950.03 | 13924.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:45:00 | 13734.00 | 13950.03 | 13924.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 10:15:00 | 13695.00 | 13899.03 | 13903.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-19 11:15:00 | 13637.00 | 13846.62 | 13879.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 14:15:00 | 13650.00 | 13504.95 | 13634.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 14:15:00 | 13650.00 | 13504.95 | 13634.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 13650.00 | 13504.95 | 13634.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 13650.00 | 13504.95 | 13634.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 13610.00 | 13525.96 | 13632.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-21 13:15:00 | 13250.00 | 13491.45 | 13581.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 13201.00 | 13452.89 | 13536.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 11:15:00 | 13277.00 | 13269.10 | 13356.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 09:15:00 | 14000.00 | 13465.60 | 13416.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 14000.00 | 13465.60 | 13416.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 09:15:00 | 14499.90 | 14004.42 | 13757.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 15:15:00 | 14738.00 | 14753.93 | 14467.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 09:15:00 | 14625.10 | 14753.93 | 14467.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 14700.00 | 14743.14 | 14488.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:30:00 | 14505.00 | 14743.14 | 14488.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 14250.00 | 14644.51 | 14466.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:45:00 | 14289.70 | 14644.51 | 14466.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 14079.20 | 14531.45 | 14431.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:00:00 | 14079.20 | 14531.45 | 14431.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 15:15:00 | 14189.65 | 14357.13 | 14370.64 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 14943.45 | 14428.50 | 14383.61 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 10:15:00 | 14260.00 | 14405.97 | 14411.30 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 15:15:00 | 14469.00 | 14413.42 | 14409.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 14550.00 | 14440.74 | 14422.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 10:15:00 | 14300.00 | 14412.59 | 14411.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 10:15:00 | 14300.00 | 14412.59 | 14411.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 14300.00 | 14412.59 | 14411.40 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 11:15:00 | 14259.50 | 14381.97 | 14397.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 13:15:00 | 14119.15 | 14320.87 | 14366.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 14360.00 | 14292.28 | 14339.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 14360.00 | 14292.28 | 14339.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 14360.00 | 14292.28 | 14339.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:15:00 | 14330.00 | 14292.28 | 14339.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 14263.25 | 14286.48 | 14332.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:45:00 | 14200.00 | 14279.18 | 14325.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 12:15:00 | 14209.55 | 14279.18 | 14325.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 14:00:00 | 14186.10 | 14242.46 | 14299.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 15:15:00 | 14210.00 | 14251.18 | 14298.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 14210.00 | 14242.95 | 14290.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:15:00 | 14300.00 | 14242.95 | 14290.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 14190.00 | 14232.36 | 14281.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:30:00 | 14322.00 | 14232.36 | 14281.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 14276.20 | 14237.71 | 14270.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:00:00 | 14276.20 | 14237.71 | 14270.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 14197.85 | 14229.74 | 14264.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 14:15:00 | 14095.95 | 14229.74 | 14264.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 13978.50 | 14184.23 | 14236.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 15:15:00 | 13905.05 | 14060.12 | 14129.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 14298.95 | 14083.07 | 14126.03 | SL hit (close>static) qty=1.00 sl=14276.20 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 14242.35 | 14157.68 | 14155.14 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 12:15:00 | 14104.50 | 14147.04 | 14150.53 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 14:15:00 | 14300.00 | 14169.91 | 14155.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 13:15:00 | 14300.95 | 14226.00 | 14193.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 14196.70 | 14287.53 | 14255.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 14:15:00 | 14196.70 | 14287.53 | 14255.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 14196.70 | 14287.53 | 14255.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 14:45:00 | 14250.00 | 14287.53 | 14255.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 14300.00 | 14290.02 | 14259.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 14:15:00 | 14400.00 | 14273.08 | 14262.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 10:45:00 | 14406.85 | 14387.00 | 14329.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 12:30:00 | 14399.70 | 14388.91 | 14340.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 13:30:00 | 14400.00 | 14390.70 | 14345.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 14399.95 | 14392.55 | 14350.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:30:00 | 14209.60 | 14392.55 | 14350.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 14200.00 | 14354.04 | 14336.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:30:00 | 14444.00 | 14329.91 | 14327.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 10:15:00 | 14169.00 | 14297.73 | 14313.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 14169.00 | 14297.73 | 14313.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 11:15:00 | 14150.00 | 14268.18 | 14298.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 12:15:00 | 14041.80 | 14028.95 | 14148.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 13:00:00 | 14041.80 | 14028.95 | 14148.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 14100.00 | 13953.59 | 14075.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 15:00:00 | 13500.00 | 13814.29 | 13943.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 09:15:00 | 13382.60 | 13851.43 | 13948.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 13652.35 | 13303.11 | 13285.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 13652.35 | 13303.11 | 13285.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 09:15:00 | 13750.00 | 13448.37 | 13357.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 11:15:00 | 13837.50 | 13847.58 | 13661.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 11:30:00 | 13900.00 | 13847.58 | 13661.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 14000.00 | 13883.43 | 13725.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:45:00 | 14000.00 | 13883.43 | 13725.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 13696.45 | 13827.66 | 13726.55 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 13499.70 | 13663.32 | 13669.25 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 14:15:00 | 13849.00 | 13671.12 | 13669.94 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 11:15:00 | 13420.00 | 13631.79 | 13656.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 13384.55 | 13582.34 | 13631.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 14:15:00 | 13990.00 | 13642.54 | 13649.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 14:15:00 | 13990.00 | 13642.54 | 13649.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 13990.00 | 13642.54 | 13649.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 15:00:00 | 13990.00 | 13642.54 | 13649.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 15:15:00 | 13770.00 | 13668.03 | 13660.35 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 13277.00 | 13589.82 | 13625.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-09 11:15:00 | 13266.05 | 13433.74 | 13489.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 14:15:00 | 13501.00 | 13409.47 | 13460.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 14:15:00 | 13501.00 | 13409.47 | 13460.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 13501.00 | 13409.47 | 13460.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 14:45:00 | 13494.95 | 13409.47 | 13460.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 13506.00 | 13428.77 | 13464.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:15:00 | 13608.60 | 13428.77 | 13464.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 13403.00 | 13376.00 | 13418.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 13403.00 | 13376.00 | 13418.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 13439.00 | 13388.60 | 13420.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 13500.90 | 13388.60 | 13420.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 13417.00 | 13394.28 | 13419.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 10:15:00 | 13300.00 | 13394.28 | 13419.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 11:00:00 | 13329.95 | 13381.41 | 13411.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 14:15:00 | 13331.95 | 13357.11 | 13391.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 13305.00 | 13356.15 | 13384.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 13272.00 | 13339.32 | 13374.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 10:45:00 | 13254.70 | 13328.86 | 13366.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 12:15:00 | 13420.00 | 13352.22 | 13370.93 | SL hit (close>static) qty=1.00 sl=13400.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 14:15:00 | 13534.95 | 13396.41 | 13388.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 10:15:00 | 13950.00 | 13539.49 | 13466.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 12:15:00 | 13490.00 | 13550.81 | 13485.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 12:15:00 | 13490.00 | 13550.81 | 13485.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 13490.00 | 13550.81 | 13485.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:00:00 | 13490.00 | 13550.81 | 13485.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 13301.00 | 13500.85 | 13469.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:00:00 | 13301.00 | 13500.85 | 13469.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 13399.75 | 13480.63 | 13462.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 13500.00 | 13454.50 | 13452.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 13360.00 | 13435.60 | 13444.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 13360.00 | 13435.60 | 13444.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 13305.00 | 13399.03 | 13425.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 13189.95 | 13079.42 | 13187.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 13189.95 | 13079.42 | 13187.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 13189.95 | 13079.42 | 13187.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:15:00 | 13044.45 | 13079.42 | 13187.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 15:15:00 | 12851.20 | 13019.00 | 13103.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 12392.23 | 12599.00 | 12796.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 12208.64 | 12599.00 | 12796.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 12654.35 | 12610.07 | 12783.73 | SL hit (close>ema200) qty=0.50 sl=12610.07 alert=retest2 |

### Cycle 41 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 12123.95 | 11908.64 | 11890.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 12378.20 | 12002.56 | 11934.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 12000.00 | 12039.51 | 11972.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 13:15:00 | 12000.00 | 12039.51 | 11972.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 12000.00 | 12039.51 | 11972.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 12000.00 | 12039.51 | 11972.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 11976.05 | 12026.82 | 11972.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:45:00 | 11955.30 | 12026.82 | 11972.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 11900.00 | 12001.45 | 11966.07 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 11846.25 | 11965.54 | 11969.15 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 10:15:00 | 11996.00 | 11971.64 | 11971.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-04 13:15:00 | 11999.20 | 11983.49 | 11977.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 09:15:00 | 11965.45 | 11984.63 | 11980.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 11965.45 | 11984.63 | 11980.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 11965.45 | 11984.63 | 11980.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:15:00 | 11900.05 | 11984.63 | 11980.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 11887.55 | 11965.21 | 11971.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 13:15:00 | 11797.00 | 11907.03 | 11941.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 12208.25 | 11939.67 | 11945.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 12208.25 | 11939.67 | 11945.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 12208.25 | 11939.67 | 11945.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 12208.25 | 11939.67 | 11945.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 12000.00 | 11951.74 | 11950.25 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 11735.75 | 11957.85 | 11974.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 11699.35 | 11866.18 | 11927.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 11713.05 | 11710.40 | 11823.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 11713.05 | 11710.40 | 11823.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 11873.40 | 11743.00 | 11828.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 12:00:00 | 11873.40 | 11743.00 | 11828.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 11924.00 | 11779.20 | 11836.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:15:00 | 11880.00 | 11779.20 | 11836.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 11800.00 | 11781.91 | 11828.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 14:30:00 | 11875.00 | 11781.91 | 11828.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 11625.85 | 11750.39 | 11805.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 11470.00 | 11730.57 | 11768.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 10:00:00 | 11401.10 | 11664.68 | 11735.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 10896.50 | 11263.19 | 11457.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 10831.05 | 11263.19 | 11457.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 12:15:00 | 11215.45 | 11174.66 | 11361.35 | SL hit (close>ema200) qty=0.50 sl=11174.66 alert=retest2 |

### Cycle 47 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 11749.00 | 11371.63 | 11329.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 10:15:00 | 12039.90 | 11693.27 | 11539.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 09:15:00 | 11773.00 | 11856.63 | 11709.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 11773.00 | 11856.63 | 11709.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 11667.60 | 11818.82 | 11705.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:45:00 | 11680.90 | 11818.82 | 11705.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 11701.60 | 11795.38 | 11705.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 12:30:00 | 11756.95 | 11802.27 | 11716.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 11:15:00 | 11745.00 | 11874.99 | 11843.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 12:15:00 | 11733.25 | 11842.55 | 11831.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 12:15:00 | 11717.95 | 11817.63 | 11821.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 12:15:00 | 11717.95 | 11817.63 | 11821.13 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 12268.95 | 11857.28 | 11831.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 12299.35 | 11998.99 | 11920.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 15:15:00 | 12062.60 | 12074.23 | 12002.52 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 09:15:00 | 12140.00 | 12074.23 | 12002.52 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 11974.20 | 12073.47 | 12027.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-29 12:15:00 | 11974.20 | 12073.47 | 12027.86 | SL hit (close<ema400) qty=1.00 sl=12027.86 alert=retest1 |

### Cycle 50 — SELL (started 2024-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 15:15:00 | 11905.35 | 11999.12 | 12001.68 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 12044.95 | 12004.15 | 12003.27 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 13:15:00 | 11899.00 | 11988.07 | 11996.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 15:15:00 | 11850.00 | 11946.36 | 11975.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 11955.60 | 11948.21 | 11973.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 11955.60 | 11948.21 | 11973.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 11955.60 | 11948.21 | 11973.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 15:15:00 | 11855.00 | 11938.39 | 11957.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 09:45:00 | 11880.00 | 11919.46 | 11945.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 10:15:00 | 11880.85 | 11919.46 | 11945.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 13:15:00 | 11849.95 | 11917.18 | 11937.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 11738.25 | 11842.39 | 11892.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 11:30:00 | 11660.00 | 11814.06 | 11870.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 12:15:00 | 11691.90 | 11814.06 | 11870.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 13:45:00 | 11720.00 | 11794.88 | 11851.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 11700.00 | 11775.91 | 11837.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 11694.00 | 11664.28 | 11731.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 12:45:00 | 11600.00 | 11672.14 | 11720.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 15:15:00 | 11400.00 | 11658.35 | 11704.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 09:15:00 | 11262.25 | 11525.35 | 11633.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 09:15:00 | 11286.00 | 11525.35 | 11633.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 09:15:00 | 11286.81 | 11525.35 | 11633.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 09:15:00 | 11257.45 | 11525.35 | 11633.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-10 13:15:00 | 11594.95 | 11474.00 | 11567.65 | SL hit (close>ema200) qty=0.50 sl=11474.00 alert=retest2 |

### Cycle 53 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 11732.50 | 11621.42 | 11620.41 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 11601.95 | 11668.84 | 11674.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 09:15:00 | 11512.00 | 11585.46 | 11623.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 13:15:00 | 11524.95 | 11522.02 | 11575.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 13:30:00 | 11492.30 | 11522.02 | 11575.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 11671.90 | 11552.00 | 11584.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 15:00:00 | 11671.90 | 11552.00 | 11584.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 11558.20 | 11553.24 | 11582.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 11678.70 | 11553.24 | 11582.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 11676.70 | 11577.93 | 11590.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 12:15:00 | 11521.85 | 11571.30 | 11585.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 13:15:00 | 11517.45 | 11569.18 | 11583.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:30:00 | 11524.85 | 11453.03 | 11489.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 11444.35 | 11472.42 | 11494.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 11476.35 | 11472.42 | 11490.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 10:30:00 | 11497.20 | 11472.42 | 11490.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 11475.30 | 11473.00 | 11489.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:30:00 | 11496.30 | 11473.00 | 11489.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 11480.00 | 11474.40 | 11488.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 13:45:00 | 11450.10 | 11473.46 | 11486.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 14:15:00 | 11694.45 | 11517.66 | 11505.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 14:15:00 | 11694.45 | 11517.66 | 11505.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-23 11:15:00 | 11796.85 | 11602.17 | 11572.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 15:15:00 | 12350.00 | 12352.42 | 12124.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-26 09:15:00 | 12640.05 | 12352.42 | 12124.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 12375.00 | 12342.40 | 12240.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 10:30:00 | 12679.00 | 12393.92 | 12272.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-30 09:15:00 | 13946.90 | 13010.07 | 12654.51 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 12:15:00 | 16322.35 | 16678.08 | 16696.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 16229.00 | 16588.26 | 16654.05 | Break + close below crossover candle low |

### Cycle 57 — BUY (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-13 14:15:00 | 17530.00 | 16776.61 | 16733.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 09:15:00 | 17560.00 | 17022.97 | 16858.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-14 14:15:00 | 17299.95 | 17345.05 | 17111.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-14 15:00:00 | 17299.95 | 17345.05 | 17111.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 17300.00 | 17324.39 | 17141.56 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 09:15:00 | 16830.00 | 17191.10 | 17196.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 10:15:00 | 16650.75 | 17083.03 | 17146.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 10:15:00 | 15089.00 | 15019.18 | 15369.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:45:00 | 15294.55 | 15019.18 | 15369.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 14850.00 | 15005.83 | 15210.14 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 15:15:00 | 15450.00 | 15305.82 | 15298.19 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 14868.35 | 15218.32 | 15259.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 12:15:00 | 14713.25 | 15064.18 | 15174.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 15260.30 | 15045.06 | 15142.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 14:15:00 | 15260.30 | 15045.06 | 15142.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 15260.30 | 15045.06 | 15142.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 15:00:00 | 15260.30 | 15045.06 | 15142.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 15000.00 | 15036.05 | 15129.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 09:15:00 | 14514.30 | 15036.05 | 15129.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 14:15:00 | 13788.58 | 14172.41 | 14467.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 13948.15 | 13745.96 | 13993.13 | SL hit (close>ema200) qty=0.50 sl=13745.96 alert=retest2 |

### Cycle 61 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 14488.90 | 14115.49 | 14089.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 14:15:00 | 14870.00 | 14249.01 | 14160.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 14098.00 | 14311.77 | 14210.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 14098.00 | 14311.77 | 14210.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 14098.00 | 14311.77 | 14210.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:45:00 | 14071.60 | 14311.77 | 14210.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 14090.00 | 14267.41 | 14199.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:30:00 | 14075.00 | 14267.41 | 14199.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 14094.40 | 14190.81 | 14177.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:30:00 | 14094.40 | 14190.81 | 14177.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 14212.00 | 14307.27 | 14239.92 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 12:15:00 | 14464.55 | 14553.06 | 14560.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 14254.50 | 14493.35 | 14532.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 14500.00 | 13985.56 | 14176.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 14:15:00 | 14500.00 | 13985.56 | 14176.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 14500.00 | 13985.56 | 14176.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:45:00 | 14544.60 | 13985.56 | 14176.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 14400.00 | 14068.45 | 14197.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 14027.50 | 14068.45 | 14197.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 11:15:00 | 14062.15 | 13987.07 | 13984.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 14062.15 | 13987.07 | 13984.42 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 13:15:00 | 13885.60 | 13976.84 | 13980.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 14:15:00 | 13800.00 | 13941.47 | 13964.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 11:15:00 | 10500.00 | 10398.96 | 10920.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-20 12:00:00 | 10500.00 | 10398.96 | 10920.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 10210.45 | 10204.44 | 10382.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 10210.45 | 10204.44 | 10382.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 10229.00 | 10184.64 | 10341.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:00:00 | 10229.00 | 10184.64 | 10341.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 10310.45 | 10095.66 | 10171.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 10310.45 | 10095.66 | 10171.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 10299.00 | 10136.33 | 10183.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:45:00 | 10130.00 | 10135.42 | 10178.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 15:15:00 | 10000.00 | 10126.17 | 10151.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 15:15:00 | 10485.00 | 10182.85 | 10150.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 10485.00 | 10182.85 | 10150.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 10762.95 | 10298.87 | 10206.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 14:15:00 | 12280.55 | 12468.98 | 12181.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 15:00:00 | 12280.55 | 12468.98 | 12181.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 12060.00 | 12387.18 | 12170.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 11785.65 | 12387.18 | 12170.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 11900.00 | 12289.75 | 12145.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 11848.15 | 12289.75 | 12145.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 11749.75 | 12045.86 | 12058.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 13:15:00 | 11700.00 | 11976.68 | 12025.49 | Break + close below crossover candle low |

### Cycle 67 — BUY (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 14:15:00 | 12500.00 | 12081.35 | 12068.63 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 11:15:00 | 12060.25 | 12084.09 | 12086.19 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 13:15:00 | 12183.65 | 12104.53 | 12095.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 09:15:00 | 12549.95 | 12200.60 | 12141.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 14:15:00 | 12346.95 | 12420.01 | 12295.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 14:15:00 | 12346.95 | 12420.01 | 12295.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 12346.95 | 12420.01 | 12295.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 15:00:00 | 12346.95 | 12420.01 | 12295.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 12360.00 | 12408.01 | 12301.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 12645.30 | 12408.01 | 12301.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-25 09:15:00 | 13909.83 | 13733.63 | 13408.66 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 14:15:00 | 14342.60 | 14647.47 | 14654.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 15:15:00 | 14270.00 | 14571.98 | 14619.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 14:15:00 | 13800.00 | 13793.30 | 13994.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-04 15:00:00 | 13800.00 | 13793.30 | 13994.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 13221.80 | 13015.58 | 13359.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 12769.95 | 13197.24 | 13314.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 15:15:00 | 13000.05 | 13016.30 | 13147.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 09:30:00 | 13118.75 | 13063.10 | 13145.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 14:15:00 | 13285.60 | 13202.37 | 13192.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 13285.60 | 13202.37 | 13192.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 13780.00 | 13332.72 | 13254.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 14482.00 | 14502.12 | 14300.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 14482.00 | 14502.12 | 14300.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 14409.00 | 14483.50 | 14310.34 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 14121.00 | 14313.51 | 14336.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 12:15:00 | 14043.00 | 14212.23 | 14265.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 14:15:00 | 13600.00 | 13590.73 | 13845.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-25 14:30:00 | 13626.00 | 13590.73 | 13845.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 14090.00 | 13679.27 | 13840.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 14090.00 | 13679.27 | 13840.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 13900.00 | 13723.42 | 13846.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 11:15:00 | 13898.00 | 13723.42 | 13846.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 13:45:00 | 13780.00 | 13772.75 | 13841.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 11:15:00 | 13203.10 | 13460.01 | 13613.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 12:15:00 | 13091.00 | 13386.61 | 13566.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-02 14:15:00 | 13000.00 | 12991.41 | 13193.53 | SL hit (close>ema200) qty=0.50 sl=12991.41 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 12877.00 | 12474.14 | 12435.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 13100.00 | 12735.53 | 12596.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 12789.00 | 12810.19 | 12683.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 13:15:00 | 12789.00 | 12810.19 | 12683.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 12789.00 | 12810.19 | 12683.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:30:00 | 12806.00 | 12810.19 | 12683.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 14067.00 | 14237.84 | 14056.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 14083.00 | 14237.84 | 14056.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 14025.00 | 14195.27 | 14053.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:30:00 | 13925.00 | 14195.27 | 14053.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 14113.00 | 14178.82 | 14058.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:45:00 | 14319.00 | 14082.12 | 14047.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 12:00:00 | 14200.00 | 14105.70 | 14061.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 14380.00 | 14181.25 | 14119.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:00:00 | 14166.00 | 14209.21 | 14162.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 14432.00 | 14253.77 | 14187.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 15:15:00 | 14460.00 | 14253.77 | 14187.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:45:00 | 14451.00 | 14329.41 | 14235.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-22 14:15:00 | 15620.00 | 14989.26 | 14617.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 11:15:00 | 15329.00 | 15507.45 | 15522.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 12:15:00 | 15310.00 | 15467.96 | 15502.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 15536.00 | 15420.62 | 15462.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 15536.00 | 15420.62 | 15462.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 15536.00 | 15420.62 | 15462.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 15536.00 | 15420.62 | 15462.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 15562.00 | 15448.89 | 15471.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 15562.00 | 15448.89 | 15471.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 15571.00 | 15473.31 | 15480.56 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 15547.00 | 15488.05 | 15486.60 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 15386.00 | 15482.19 | 15485.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 14997.00 | 15385.15 | 15440.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 13:15:00 | 15389.00 | 15369.14 | 15409.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 13:30:00 | 15385.00 | 15369.14 | 15409.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 15242.00 | 15343.71 | 15394.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:30:00 | 15383.00 | 15343.71 | 15394.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 15328.00 | 15319.34 | 15373.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 15:15:00 | 15276.00 | 15336.35 | 15362.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 11:00:00 | 15266.00 | 15326.50 | 15351.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 09:15:00 | 14512.20 | 14927.28 | 15108.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 09:15:00 | 14502.70 | 14927.28 | 15108.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 15:15:00 | 14858.00 | 14850.37 | 14983.20 | SL hit (close>ema200) qty=0.50 sl=14850.37 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 15670.00 | 14832.16 | 14733.37 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 10:15:00 | 15099.00 | 15219.93 | 15221.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 11:15:00 | 14907.00 | 15157.34 | 15192.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 14877.00 | 14818.54 | 14917.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 14877.00 | 14818.54 | 14917.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 14877.00 | 14818.54 | 14917.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 14882.00 | 14818.54 | 14917.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 14416.00 | 14300.53 | 14451.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:30:00 | 14480.00 | 14300.53 | 14451.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 14613.00 | 14363.02 | 14466.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 14613.00 | 14363.02 | 14466.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 14779.00 | 14446.22 | 14494.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:45:00 | 14742.00 | 14446.22 | 14494.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 14674.00 | 14530.62 | 14527.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 14749.00 | 14621.62 | 14584.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 09:15:00 | 14495.00 | 14610.60 | 14586.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 14495.00 | 14610.60 | 14586.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 14495.00 | 14610.60 | 14586.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:45:00 | 14576.00 | 14610.60 | 14586.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 10:15:00 | 14373.00 | 14563.08 | 14567.34 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 11:15:00 | 14751.00 | 14600.66 | 14584.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 14:15:00 | 14924.00 | 14697.46 | 14634.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 10:15:00 | 15267.00 | 15387.78 | 15210.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 11:00:00 | 15267.00 | 15387.78 | 15210.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 14940.00 | 15298.22 | 15185.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 14958.00 | 15298.22 | 15185.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 14982.00 | 15234.98 | 15167.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 14982.00 | 15234.98 | 15167.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 14930.00 | 15102.28 | 15117.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 14727.00 | 15027.22 | 15082.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 12:15:00 | 15050.00 | 15006.24 | 15056.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 13:00:00 | 15050.00 | 15006.24 | 15056.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 15070.00 | 15019.59 | 15053.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 15070.00 | 15019.59 | 15053.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 14973.00 | 15010.28 | 15046.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 15193.00 | 15070.02 | 15070.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 15136.00 | 15083.22 | 15076.09 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 14864.00 | 15054.70 | 15065.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 14766.00 | 14959.25 | 15017.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 13:15:00 | 14858.00 | 14851.06 | 14926.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 13:15:00 | 14858.00 | 14851.06 | 14926.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 14858.00 | 14851.06 | 14926.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:30:00 | 14931.00 | 14851.06 | 14926.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 14898.00 | 14864.28 | 14919.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 14822.00 | 14864.28 | 14919.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:00:00 | 14825.00 | 14765.73 | 14837.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 14799.00 | 14757.67 | 14821.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 14841.00 | 14753.21 | 14782.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 14769.00 | 14757.78 | 14779.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 14771.00 | 14757.78 | 14779.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 14799.00 | 14766.02 | 14781.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:15:00 | 14840.00 | 14766.02 | 14781.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 14795.00 | 14771.82 | 14782.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:30:00 | 14840.00 | 14771.82 | 14782.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 14550.00 | 14727.45 | 14761.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:15:00 | 14536.00 | 14655.39 | 14717.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 14080.90 | 14365.52 | 14519.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 14083.75 | 14365.52 | 14519.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 14098.95 | 14365.52 | 14519.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:15:00 | 14059.05 | 14312.42 | 14480.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 14:15:00 | 14423.00 | 14216.89 | 14370.01 | SL hit (close>ema200) qty=0.50 sl=14216.89 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 14494.00 | 14372.28 | 14366.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 14511.00 | 14400.02 | 14379.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 14425.00 | 14441.03 | 14412.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:45:00 | 14436.00 | 14441.03 | 14412.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 14452.00 | 14443.22 | 14415.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:00:00 | 14459.00 | 14446.38 | 14419.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:45:00 | 14466.00 | 14451.10 | 14424.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 14510.00 | 14505.81 | 14484.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 14422.00 | 14466.10 | 14469.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 14422.00 | 14466.10 | 14469.69 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 14470.00 | 14466.75 | 14466.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 14539.00 | 14481.20 | 14473.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 12:15:00 | 14485.00 | 14500.80 | 14488.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 12:15:00 | 14485.00 | 14500.80 | 14488.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 14485.00 | 14500.80 | 14488.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:45:00 | 14481.00 | 14500.80 | 14488.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 14498.00 | 14500.24 | 14489.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:15:00 | 14482.00 | 14500.24 | 14489.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 14541.00 | 14508.39 | 14493.77 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 14311.00 | 14464.90 | 14477.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 12:15:00 | 14230.00 | 14394.41 | 14441.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 14391.00 | 14366.94 | 14419.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 14391.00 | 14366.94 | 14419.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 14391.00 | 14366.94 | 14419.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 14391.00 | 14366.94 | 14419.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 14477.00 | 14386.56 | 14419.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 12:00:00 | 14308.00 | 14373.16 | 14407.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 14550.00 | 14403.65 | 14412.27 | SL hit (close>static) qty=1.00 sl=14498.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 15:15:00 | 14560.00 | 14434.92 | 14425.70 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 14398.00 | 14417.93 | 14419.60 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 12:15:00 | 14619.00 | 14458.14 | 14437.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 13:15:00 | 14790.00 | 14685.87 | 14608.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 15:15:00 | 14730.00 | 14753.58 | 14698.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 09:15:00 | 14795.00 | 14753.58 | 14698.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 14727.00 | 14748.26 | 14700.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:30:00 | 14960.00 | 14778.67 | 14732.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 13:15:00 | 14845.00 | 14835.30 | 14782.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 14:30:00 | 14848.00 | 14834.59 | 14792.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 11:15:00 | 14683.00 | 14755.09 | 14763.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 11:15:00 | 14683.00 | 14755.09 | 14763.10 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 14832.00 | 14766.64 | 14765.79 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 14727.00 | 14758.71 | 14762.27 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 14822.00 | 14771.37 | 14767.70 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 14739.00 | 14791.58 | 14793.95 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 14:15:00 | 15018.00 | 14828.09 | 14808.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 14:15:00 | 15124.00 | 14953.79 | 14899.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 15:15:00 | 14821.00 | 14927.23 | 14892.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 15:15:00 | 14821.00 | 14927.23 | 14892.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 14821.00 | 14927.23 | 14892.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 14642.00 | 14927.23 | 14892.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 14380.00 | 14817.78 | 14845.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 13778.00 | 13988.33 | 14129.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 11:15:00 | 13722.00 | 13718.49 | 13841.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 12:00:00 | 13722.00 | 13718.49 | 13841.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 13647.00 | 13573.55 | 13661.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:45:00 | 13697.00 | 13573.55 | 13661.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 13580.00 | 13574.84 | 13654.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 13593.00 | 13574.84 | 13654.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 13537.00 | 13567.27 | 13643.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 11:15:00 | 13454.00 | 13565.82 | 13635.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:30:00 | 13463.00 | 13523.18 | 13596.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 14168.00 | 13600.35 | 13608.75 | SL hit (close>static) qty=1.00 sl=13747.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 10:15:00 | 13980.00 | 13676.28 | 13642.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 14:15:00 | 14221.00 | 13894.69 | 13796.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 10:15:00 | 13964.00 | 13968.54 | 13861.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-26 10:45:00 | 13948.00 | 13968.54 | 13861.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 13893.00 | 13950.33 | 13879.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:30:00 | 13851.00 | 13950.33 | 13879.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 13969.00 | 13954.07 | 13887.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:30:00 | 13843.00 | 13954.07 | 13887.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 13761.00 | 13918.48 | 13883.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 13757.00 | 13918.48 | 13883.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 13793.00 | 13893.39 | 13875.12 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 13740.00 | 13841.37 | 13853.32 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 15:15:00 | 13965.00 | 13873.41 | 13864.58 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 13844.00 | 13856.58 | 13857.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 13755.00 | 13832.81 | 13846.56 | Break + close below crossover candle low |

### Cycle 103 — BUY (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 14:15:00 | 14000.00 | 13854.92 | 13853.52 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 10:15:00 | 13845.00 | 13870.65 | 13871.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 12:15:00 | 13819.00 | 13861.02 | 13867.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 13845.00 | 13840.78 | 13854.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 13845.00 | 13840.78 | 13854.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 13845.00 | 13840.78 | 13854.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 13812.00 | 13840.78 | 13854.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 13710.00 | 13622.86 | 13618.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 13710.00 | 13622.86 | 13618.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 14049.00 | 13708.09 | 13657.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 13799.00 | 13826.11 | 13748.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 14:00:00 | 14001.00 | 13861.09 | 13771.16 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 14520.00 | 14227.49 | 14114.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 14486.00 | 14227.49 | 14114.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 14:15:00 | 14701.05 | 14513.96 | 14342.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-09-17 12:15:00 | 15401.10 | 14897.54 | 14603.60 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 106 — SELL (started 2025-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 13:15:00 | 15125.00 | 15323.51 | 15333.39 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 15438.00 | 15291.14 | 15284.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 15699.00 | 15470.87 | 15385.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 16210.00 | 16518.73 | 16285.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 16210.00 | 16518.73 | 16285.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 16210.00 | 16518.73 | 16285.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:00:00 | 16210.00 | 16518.73 | 16285.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 16212.00 | 16457.38 | 16278.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:15:00 | 16188.00 | 16457.38 | 16278.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 16236.00 | 16329.38 | 16256.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:45:00 | 16183.00 | 16329.38 | 16256.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 16001.00 | 16263.70 | 16233.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 16001.00 | 16263.70 | 16233.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 16050.00 | 16220.96 | 16216.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 16220.00 | 16220.96 | 16216.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 16118.00 | 16200.37 | 16207.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 16118.00 | 16200.37 | 16207.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 15813.00 | 16068.83 | 16138.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 16284.00 | 16070.15 | 16118.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 16284.00 | 16070.15 | 16118.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 16284.00 | 16070.15 | 16118.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 16284.00 | 16070.15 | 16118.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 16538.00 | 16163.72 | 16156.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 14:15:00 | 16699.00 | 16376.86 | 16269.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 09:15:00 | 16415.00 | 16424.99 | 16312.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 10:00:00 | 16415.00 | 16424.99 | 16312.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 16321.00 | 16394.44 | 16317.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:00:00 | 16321.00 | 16394.44 | 16317.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 16263.00 | 16368.15 | 16312.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:45:00 | 16270.00 | 16368.15 | 16312.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 16366.00 | 16367.72 | 16317.28 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 16060.00 | 16275.06 | 16285.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 15959.00 | 16211.84 | 16255.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 16439.00 | 16156.61 | 16203.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 16439.00 | 16156.61 | 16203.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 16439.00 | 16156.61 | 16203.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 16368.00 | 16156.61 | 16203.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 16353.00 | 16195.89 | 16217.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 16465.00 | 16195.89 | 16217.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 10:15:00 | 16298.00 | 16244.01 | 16236.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 14:15:00 | 16563.00 | 16331.42 | 16280.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 11:15:00 | 16677.00 | 16703.90 | 16573.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 12:00:00 | 16677.00 | 16703.90 | 16573.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 16794.00 | 16706.31 | 16622.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 17190.00 | 16669.31 | 16634.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 11:45:00 | 16918.00 | 16773.13 | 16695.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:30:00 | 16932.00 | 16816.30 | 16722.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 17052.00 | 16887.28 | 16788.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 16774.00 | 16879.46 | 16803.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 17520.00 | 16971.51 | 16903.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 14:45:00 | 17181.00 | 17254.44 | 17189.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 17265.00 | 17225.55 | 17182.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 16868.00 | 17152.99 | 17156.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 10:15:00 | 16868.00 | 17152.99 | 17156.58 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 11:15:00 | 17154.00 | 17112.05 | 17108.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 14:15:00 | 17259.00 | 17139.28 | 17121.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 15:15:00 | 17240.00 | 17317.44 | 17247.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 15:15:00 | 17240.00 | 17317.44 | 17247.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 17240.00 | 17317.44 | 17247.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:45:00 | 17205.00 | 17287.95 | 17240.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 17125.00 | 17255.36 | 17230.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 17125.00 | 17255.36 | 17230.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 17338.00 | 17271.83 | 17241.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 17640.00 | 17241.53 | 17235.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 17172.00 | 17227.62 | 17229.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 17172.00 | 17227.62 | 17229.99 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 10:15:00 | 17392.00 | 17260.50 | 17244.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 12:15:00 | 17515.00 | 17421.62 | 17355.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 15:15:00 | 17440.00 | 17443.55 | 17383.67 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:15:00 | 17658.00 | 17443.55 | 17383.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 12:45:00 | 17528.00 | 17535.93 | 17454.87 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 17425.00 | 17513.74 | 17452.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-10 13:15:00 | 17425.00 | 17513.74 | 17452.15 | SL hit (close<ema400) qty=1.00 sl=17452.15 alert=retest1 |

### Cycle 116 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 17328.00 | 17412.60 | 17416.24 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 17519.00 | 17411.06 | 17405.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 17647.00 | 17458.25 | 17427.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 17512.00 | 17541.41 | 17493.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 15:15:00 | 17512.00 | 17541.41 | 17493.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 17512.00 | 17541.41 | 17493.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 17603.00 | 17541.41 | 17493.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 17330.00 | 17501.30 | 17483.48 | SL hit (close<static) qty=1.00 sl=17451.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 11:15:00 | 17344.00 | 17469.84 | 17470.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 12:15:00 | 17290.00 | 17433.87 | 17454.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 17227.00 | 17143.54 | 17250.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 17227.00 | 17143.54 | 17250.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 17227.00 | 17143.54 | 17250.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 17227.00 | 17143.54 | 17250.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 17178.00 | 17150.43 | 17243.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 17208.00 | 17150.43 | 17243.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 17190.00 | 17158.35 | 17238.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:45:00 | 17240.00 | 17158.35 | 17238.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 17261.00 | 17178.88 | 17240.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 17261.00 | 17178.88 | 17240.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 17197.00 | 17182.50 | 17236.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:15:00 | 17268.00 | 17182.50 | 17236.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 17315.00 | 17209.00 | 17243.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:45:00 | 17347.00 | 17209.00 | 17243.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 17345.00 | 17236.20 | 17253.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 17379.00 | 17236.20 | 17253.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 17280.00 | 17242.41 | 17251.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:45:00 | 17303.00 | 17242.41 | 17251.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 17279.00 | 17249.73 | 17254.31 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 17330.00 | 17265.78 | 17261.19 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 17227.00 | 17258.02 | 17258.08 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 15:15:00 | 17260.00 | 17258.42 | 17258.26 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 17226.00 | 17251.94 | 17255.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 17180.00 | 17237.55 | 17248.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 12:15:00 | 17256.00 | 17232.51 | 17243.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 12:15:00 | 17256.00 | 17232.51 | 17243.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 17256.00 | 17232.51 | 17243.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 17256.00 | 17232.51 | 17243.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 17280.00 | 17242.01 | 17247.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:00:00 | 17280.00 | 17242.01 | 17247.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 17235.00 | 17240.61 | 17245.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:45:00 | 17250.00 | 17240.61 | 17245.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 17249.00 | 17242.29 | 17246.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 17247.00 | 17242.29 | 17246.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 17172.00 | 17228.23 | 17239.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:15:00 | 17100.00 | 17204.69 | 17223.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:00:00 | 17088.00 | 17164.60 | 17200.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 17081.00 | 17141.42 | 17160.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:15:00 | 17141.00 | 17152.71 | 17162.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 17140.00 | 17150.17 | 17160.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 17653.00 | 17258.53 | 17207.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 17653.00 | 17258.53 | 17207.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 18200.00 | 17850.79 | 17614.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 18003.00 | 18058.33 | 17874.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:30:00 | 18027.00 | 18058.33 | 17874.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 18070.00 | 18060.31 | 17954.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 18001.00 | 18060.31 | 17954.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 18201.00 | 18275.34 | 18176.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 18438.00 | 18275.34 | 18176.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 11:30:00 | 18376.00 | 18309.02 | 18217.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 12:45:00 | 18350.00 | 18312.40 | 18268.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 09:45:00 | 18355.00 | 18414.34 | 18384.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 18480.00 | 18884.28 | 18790.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 18424.00 | 18884.28 | 18790.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 18747.00 | 18856.82 | 18786.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 18435.00 | 18856.82 | 18786.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 18906.00 | 18866.66 | 18797.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:45:00 | 18712.00 | 18866.66 | 18797.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 18802.00 | 18853.73 | 18797.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:45:00 | 18755.00 | 18853.73 | 18797.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 18787.00 | 18840.38 | 18796.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 18716.00 | 18840.38 | 18796.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 18946.00 | 18861.50 | 18810.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:30:00 | 18830.00 | 18861.50 | 18810.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 18884.00 | 18866.00 | 18817.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 18930.00 | 18866.00 | 18817.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 18805.00 | 18853.80 | 18816.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-10 13:15:00 | 18723.00 | 18798.64 | 18800.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 18723.00 | 18798.64 | 18800.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 18583.00 | 18755.51 | 18780.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 18739.00 | 18726.53 | 18761.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 18739.00 | 18726.53 | 18761.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 18739.00 | 18726.53 | 18761.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:15:00 | 18448.00 | 18687.32 | 18732.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 09:45:00 | 18475.00 | 18511.24 | 18628.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:30:00 | 18433.00 | 18498.39 | 18603.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 14:15:00 | 17525.60 | 17951.08 | 18213.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 14:15:00 | 17551.25 | 17951.08 | 18213.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 14:15:00 | 17511.35 | 17951.08 | 18213.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-18 09:15:00 | 16603.20 | 17288.35 | 17534.29 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 17490.00 | 17308.88 | 17296.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 17532.00 | 17382.48 | 17333.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 09:15:00 | 18083.00 | 18201.87 | 18051.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 10:00:00 | 18083.00 | 18201.87 | 18051.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 18082.00 | 18177.90 | 18054.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 14:45:00 | 18180.00 | 18223.82 | 18107.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 18087.00 | 18389.46 | 18415.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 14:15:00 | 18087.00 | 18389.46 | 18415.61 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 11:15:00 | 18387.00 | 18352.17 | 18349.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 18416.00 | 18364.93 | 18355.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 14:15:00 | 18329.00 | 18366.08 | 18358.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 14:15:00 | 18329.00 | 18366.08 | 18358.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 18329.00 | 18366.08 | 18358.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 18329.00 | 18366.08 | 18358.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 15:15:00 | 18250.00 | 18342.86 | 18348.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 12:15:00 | 18170.00 | 18286.71 | 18319.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 18058.00 | 18045.92 | 18134.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 18058.00 | 18045.92 | 18134.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 18058.00 | 18045.92 | 18134.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:15:00 | 17941.00 | 18047.77 | 18108.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 15:00:00 | 17916.00 | 17705.97 | 17723.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 17994.00 | 17763.57 | 17747.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 17994.00 | 17763.57 | 17747.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 18065.00 | 17823.86 | 17776.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 15:15:00 | 17991.00 | 18007.22 | 17911.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:15:00 | 18040.00 | 18007.22 | 17911.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 17866.00 | 17978.98 | 17907.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 17866.00 | 17978.98 | 17907.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 17827.00 | 17948.58 | 17899.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:45:00 | 17840.00 | 17948.58 | 17899.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 17831.00 | 17925.07 | 17893.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 17831.00 | 17925.07 | 17893.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 17863.00 | 17904.48 | 17889.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:30:00 | 17831.00 | 17904.48 | 17889.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 18040.00 | 17988.15 | 17936.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 18000.00 | 17988.15 | 17936.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 17935.00 | 17982.30 | 17942.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:00:00 | 17935.00 | 17982.30 | 17942.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 17752.00 | 17936.24 | 17925.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:30:00 | 17764.00 | 17936.24 | 17925.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 17751.00 | 17899.19 | 17909.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 17636.00 | 17789.34 | 17842.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 14:15:00 | 18081.00 | 17780.69 | 17809.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 14:15:00 | 18081.00 | 17780.69 | 17809.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 18081.00 | 17780.69 | 17809.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:00:00 | 18081.00 | 17780.69 | 17809.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 17501.00 | 17724.75 | 17781.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 17394.00 | 17724.75 | 17781.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 17729.00 | 17697.20 | 17696.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 17729.00 | 17697.20 | 17696.15 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 17542.00 | 17701.62 | 17709.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 17485.00 | 17658.30 | 17688.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 10:15:00 | 17547.00 | 17538.66 | 17604.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 11:00:00 | 17547.00 | 17538.66 | 17604.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 17595.00 | 17454.93 | 17524.49 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 17846.00 | 17599.63 | 17579.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 17968.00 | 17710.48 | 17635.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 18030.00 | 18248.57 | 18084.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 12:15:00 | 18030.00 | 18248.57 | 18084.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 18030.00 | 18248.57 | 18084.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 18030.00 | 18248.57 | 18084.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 18076.00 | 18214.05 | 18083.67 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 17887.00 | 18039.44 | 18042.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 17640.00 | 17905.52 | 17975.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 17911.00 | 17827.85 | 17907.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 17911.00 | 17827.85 | 17907.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 17911.00 | 17827.85 | 17907.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:45:00 | 17923.00 | 17827.85 | 17907.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 18150.00 | 17892.28 | 17929.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 18325.00 | 17892.28 | 17929.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 18160.00 | 17945.83 | 17950.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:30:00 | 17990.00 | 17949.06 | 17951.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 11:15:00 | 18191.00 | 17690.18 | 17624.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 18191.00 | 17690.18 | 17624.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 18377.00 | 17891.36 | 17731.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 18251.00 | 18430.90 | 18212.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 18251.00 | 18430.90 | 18212.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 18440.00 | 18395.75 | 18296.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:45:00 | 18656.00 | 18457.00 | 18333.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 18175.00 | 18372.02 | 18392.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 18175.00 | 18372.02 | 18392.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 15:15:00 | 18150.00 | 18336.41 | 18373.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 18067.00 | 18004.30 | 18137.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:15:00 | 18019.00 | 18004.30 | 18137.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 18290.00 | 18061.44 | 18151.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 18290.00 | 18061.44 | 18151.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 18320.00 | 18113.15 | 18166.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 18260.00 | 18113.15 | 18166.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 18163.00 | 18174.41 | 18187.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 18163.00 | 18174.41 | 18187.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 18270.00 | 18193.52 | 18195.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 18376.00 | 18193.52 | 18195.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 18237.00 | 18202.22 | 18199.13 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 17890.00 | 18149.36 | 18180.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 17811.00 | 18081.68 | 18147.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 17921.00 | 17754.02 | 17873.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 13:15:00 | 17921.00 | 17754.02 | 17873.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 17921.00 | 17754.02 | 17873.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 17921.00 | 17754.02 | 17873.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 17853.00 | 17773.81 | 17871.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:30:00 | 18040.00 | 17773.81 | 17871.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 17698.00 | 17758.65 | 17855.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 17932.00 | 17758.65 | 17855.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 17789.00 | 17764.72 | 17849.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 17919.00 | 17764.72 | 17849.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 17797.00 | 17771.18 | 17844.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 17797.00 | 17771.18 | 17844.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 17845.00 | 17785.94 | 17844.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 17839.00 | 17785.94 | 17844.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 17900.00 | 17808.75 | 17849.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:00:00 | 17900.00 | 17808.75 | 17849.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 17934.00 | 17833.80 | 17857.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:00:00 | 17934.00 | 17833.80 | 17857.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 17991.00 | 17865.24 | 17869.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:45:00 | 17901.00 | 17865.24 | 17869.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 17850.00 | 17862.19 | 17867.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 17711.00 | 17862.19 | 17867.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 18010.00 | 17879.81 | 17863.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 18010.00 | 17879.81 | 17863.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 18049.00 | 17935.35 | 17899.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 13:15:00 | 17977.00 | 18050.48 | 17986.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 13:15:00 | 17977.00 | 18050.48 | 17986.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 17977.00 | 18050.48 | 17986.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:00:00 | 17977.00 | 18050.48 | 17986.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 17965.00 | 18033.38 | 17984.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:15:00 | 17900.00 | 18033.38 | 17984.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 17900.00 | 18006.71 | 17976.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 17847.00 | 18006.71 | 17976.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 17961.00 | 17989.61 | 17973.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 17920.00 | 17989.61 | 17973.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 17837.00 | 17959.09 | 17961.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 17725.00 | 17880.16 | 17918.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 11:15:00 | 17834.00 | 17826.26 | 17884.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 11:15:00 | 17834.00 | 17826.26 | 17884.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 11:15:00 | 17834.00 | 17826.26 | 17884.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 11:30:00 | 17870.00 | 17826.26 | 17884.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 17762.00 | 17813.41 | 17873.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 12:30:00 | 17878.00 | 17813.41 | 17873.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 17993.00 | 17849.33 | 17884.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:30:00 | 17980.00 | 17849.33 | 17884.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 17988.00 | 17877.06 | 17893.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 17988.00 | 17877.06 | 17893.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 09:15:00 | 17930.00 | 17904.12 | 17904.07 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 17821.00 | 17887.50 | 17896.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 17696.00 | 17849.20 | 17878.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 13:15:00 | 18162.00 | 17884.85 | 17887.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 13:15:00 | 18162.00 | 17884.85 | 17887.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 18162.00 | 17884.85 | 17887.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:00:00 | 18162.00 | 17884.85 | 17887.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 14:15:00 | 17925.00 | 17892.88 | 17891.16 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 15:15:00 | 17840.00 | 17882.30 | 17886.51 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 17958.00 | 17897.44 | 17893.01 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 13:15:00 | 17852.00 | 17892.03 | 17892.56 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 18000.00 | 17913.63 | 17902.33 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 11:15:00 | 17789.00 | 17886.04 | 17894.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 17210.00 | 17701.71 | 17800.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 17459.00 | 17371.27 | 17538.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 17459.00 | 17371.27 | 17538.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 17459.00 | 17371.27 | 17538.95 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 17771.00 | 17573.46 | 17567.74 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 17451.00 | 17743.86 | 17770.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 17305.00 | 17656.09 | 17728.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 17118.00 | 16920.24 | 17200.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 17118.00 | 16920.24 | 17200.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 17118.00 | 16920.24 | 17200.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 17118.00 | 16920.24 | 17200.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 16859.00 | 16907.99 | 17169.36 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 17352.00 | 17257.25 | 17248.11 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 17087.00 | 17238.27 | 17246.98 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 15:15:00 | 17310.00 | 17225.10 | 17224.44 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 12:15:00 | 17164.00 | 17216.58 | 17221.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 17106.00 | 17194.47 | 17211.24 | Break + close below crossover candle low |

### Cycle 155 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 17480.00 | 17251.57 | 17235.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 15:15:00 | 17490.00 | 17299.26 | 17258.79 | Break + close above crossover candle high |

### Cycle 156 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 16931.00 | 17225.61 | 17228.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 16749.00 | 17130.29 | 17185.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 14:15:00 | 16985.00 | 16920.34 | 17048.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 14:15:00 | 16985.00 | 16920.34 | 17048.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 16985.00 | 16920.34 | 17048.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 15:00:00 | 16985.00 | 16920.34 | 17048.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 16708.00 | 16858.62 | 16997.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 11:15:00 | 16525.00 | 16833.29 | 16973.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:00:00 | 16604.00 | 16736.99 | 16901.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:45:00 | 16584.00 | 16694.19 | 16867.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:00:00 | 16539.00 | 16663.15 | 16837.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:15:00 | 15698.75 | 16315.74 | 16556.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:15:00 | 15773.80 | 16315.74 | 16556.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:15:00 | 15754.80 | 16315.74 | 16556.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:15:00 | 15712.05 | 16315.74 | 16556.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-30 10:15:00 | 14872.50 | 15407.22 | 15878.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 157 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 15372.00 | 15051.81 | 15050.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 15434.00 | 15128.25 | 15084.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 12:15:00 | 16043.00 | 16140.14 | 15839.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 13:00:00 | 16043.00 | 16140.14 | 15839.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 15781.00 | 16061.89 | 15855.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 15:00:00 | 15781.00 | 16061.89 | 15855.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 15:15:00 | 15750.00 | 15999.51 | 15845.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:15:00 | 15562.00 | 15999.51 | 15845.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2026-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 11:15:00 | 15460.00 | 15701.35 | 15730.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 15182.00 | 15503.29 | 15595.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 15253.00 | 15154.45 | 15336.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 10:00:00 | 15253.00 | 15154.45 | 15336.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 15655.00 | 15239.38 | 15316.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:00:00 | 15655.00 | 15239.38 | 15316.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 15699.00 | 15331.31 | 15350.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 15:15:00 | 15898.00 | 15331.31 | 15350.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2026-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 15:15:00 | 15898.00 | 15444.65 | 15400.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 15981.00 | 15551.92 | 15453.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 10:15:00 | 15685.00 | 15821.44 | 15685.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 10:15:00 | 15685.00 | 15821.44 | 15685.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 15685.00 | 15821.44 | 15685.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:45:00 | 15692.00 | 15821.44 | 15685.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 15744.00 | 15805.95 | 15691.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:15:00 | 15828.00 | 15750.71 | 15699.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:45:00 | 15805.00 | 15762.77 | 15709.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 15889.00 | 15982.37 | 15983.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 14:15:00 | 15889.00 | 15982.37 | 15983.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 15660.00 | 15880.80 | 15924.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 16121.00 | 15859.22 | 15892.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 16121.00 | 15859.22 | 15892.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 16121.00 | 15859.22 | 15892.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 16121.00 | 15859.22 | 15892.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 16050.00 | 15897.38 | 15906.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 16133.00 | 15897.38 | 15906.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 16201.00 | 15958.10 | 15933.57 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 15952.00 | 16025.29 | 16025.56 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 16047.00 | 16029.63 | 16027.51 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 10:15:00 | 15966.00 | 16016.91 | 16021.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 15882.00 | 15974.94 | 16000.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 16060.00 | 15891.44 | 15927.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 16060.00 | 15891.44 | 15927.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 16060.00 | 15891.44 | 15927.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 16067.00 | 15891.44 | 15927.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 16140.00 | 15941.16 | 15946.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 16307.00 | 15941.16 | 15946.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 16320.00 | 16016.92 | 15980.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 15:15:00 | 16490.00 | 16318.49 | 16211.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 16784.00 | 16972.30 | 16802.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 14:15:00 | 16784.00 | 16972.30 | 16802.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 16784.00 | 16972.30 | 16802.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 15:00:00 | 16784.00 | 16972.30 | 16802.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 16820.00 | 16941.84 | 16804.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 16950.00 | 16941.84 | 16804.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:00:00 | 16867.00 | 16900.00 | 16833.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-11 09:15:00 | 16598.00 | 16805.62 | 16804.14 | SL hit (close<static) qty=1.00 sl=16650.00 alert=retest2 |

### Cycle 166 — SELL (started 2026-05-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-11 10:15:00 | 16649.00 | 16774.29 | 16790.04 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2026-05-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-11 15:15:00 | 17080.00 | 16822.49 | 16796.96 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 09:15:00 | 7444.10 | 2024-05-21 11:15:00 | 8188.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-17 11:30:00 | 7452.20 | 2024-05-21 11:15:00 | 8197.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-17 12:00:00 | 7441.90 | 2024-05-21 11:15:00 | 8186.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-17 12:45:00 | 7450.00 | 2024-05-21 11:15:00 | 8195.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-18 09:15:00 | 7490.00 | 2024-05-21 11:15:00 | 8239.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-18 11:30:00 | 7473.80 | 2024-05-21 11:15:00 | 8221.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-18 12:00:00 | 7470.00 | 2024-05-21 11:15:00 | 8217.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-30 15:15:00 | 7814.45 | 2024-05-31 09:15:00 | 8550.00 | STOP_HIT | 1.00 | -9.41% |
| BUY | retest2 | 2024-06-19 12:00:00 | 14291.75 | 2024-06-21 09:15:00 | 13504.35 | STOP_HIT | 1.00 | -5.51% |
| SELL | retest2 | 2024-06-27 11:00:00 | 13141.70 | 2024-06-27 14:15:00 | 13723.55 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2024-06-27 12:00:00 | 13150.10 | 2024-06-27 14:15:00 | 13723.55 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2024-06-27 12:45:00 | 13086.70 | 2024-06-27 14:15:00 | 13723.55 | STOP_HIT | 1.00 | -4.87% |
| BUY | retest2 | 2024-07-02 12:30:00 | 13995.00 | 2024-07-08 15:15:00 | 14646.95 | STOP_HIT | 1.00 | 4.66% |
| BUY | retest2 | 2024-07-02 13:00:00 | 13953.65 | 2024-07-08 15:15:00 | 14646.95 | STOP_HIT | 1.00 | 4.97% |
| BUY | retest2 | 2024-07-02 14:15:00 | 13980.00 | 2024-07-08 15:15:00 | 14646.95 | STOP_HIT | 1.00 | 4.77% |
| SELL | retest2 | 2024-07-18 09:15:00 | 14094.90 | 2024-07-22 09:15:00 | 13390.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 13:00:00 | 14070.00 | 2024-07-22 09:15:00 | 13366.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 13:45:00 | 14000.00 | 2024-07-22 09:15:00 | 13300.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 14:15:00 | 14000.00 | 2024-07-22 09:15:00 | 13300.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 09:15:00 | 13900.00 | 2024-07-22 09:15:00 | 13205.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 09:15:00 | 14094.90 | 2024-07-22 14:15:00 | 13849.70 | STOP_HIT | 0.50 | 1.74% |
| SELL | retest2 | 2024-07-18 13:00:00 | 14070.00 | 2024-07-22 14:15:00 | 13849.70 | STOP_HIT | 0.50 | 1.57% |
| SELL | retest2 | 2024-07-18 13:45:00 | 14000.00 | 2024-07-22 14:15:00 | 13849.70 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2024-07-18 14:15:00 | 14000.00 | 2024-07-22 14:15:00 | 13849.70 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2024-07-19 09:15:00 | 13900.00 | 2024-07-22 14:15:00 | 13849.70 | STOP_HIT | 0.50 | 0.36% |
| SELL | retest2 | 2024-07-19 15:00:00 | 13799.60 | 2024-07-23 12:15:00 | 13223.67 | PARTIAL | 0.50 | 4.17% |
| SELL | retest2 | 2024-07-19 15:00:00 | 13799.60 | 2024-07-23 12:15:00 | 13799.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest2 | 2024-07-23 09:30:00 | 13919.65 | 2024-07-26 14:15:00 | 14089.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-08-07 09:15:00 | 13300.00 | 2024-08-07 13:15:00 | 13883.00 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2024-08-07 10:00:00 | 13370.00 | 2024-08-07 13:15:00 | 13883.00 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2024-08-07 11:00:00 | 13301.00 | 2024-08-07 13:15:00 | 13883.00 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2024-08-07 11:45:00 | 13400.00 | 2024-08-07 13:15:00 | 13883.00 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest1 | 2024-08-09 14:45:00 | 14260.00 | 2024-08-13 09:15:00 | 14000.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest1 | 2024-08-12 11:00:00 | 14189.00 | 2024-08-13 09:15:00 | 14000.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-08-21 13:15:00 | 13250.00 | 2024-08-26 09:15:00 | 14000.00 | STOP_HIT | 1.00 | -5.66% |
| SELL | retest2 | 2024-08-22 09:15:00 | 13201.00 | 2024-08-26 09:15:00 | 14000.00 | STOP_HIT | 1.00 | -6.05% |
| SELL | retest2 | 2024-08-23 11:15:00 | 13277.00 | 2024-08-26 09:15:00 | 14000.00 | STOP_HIT | 1.00 | -5.45% |
| SELL | retest2 | 2024-09-05 11:45:00 | 14200.00 | 2024-09-10 09:15:00 | 14298.95 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-09-05 12:15:00 | 14209.55 | 2024-09-10 09:15:00 | 14298.95 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-09-05 14:00:00 | 14186.10 | 2024-09-10 09:15:00 | 14298.95 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-09-05 15:15:00 | 14210.00 | 2024-09-10 11:15:00 | 14242.35 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-09-06 14:15:00 | 14095.95 | 2024-09-10 11:15:00 | 14242.35 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-09-09 09:15:00 | 13978.50 | 2024-09-10 11:15:00 | 14242.35 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-09-09 15:15:00 | 13905.05 | 2024-09-10 11:15:00 | 14242.35 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-09-16 14:15:00 | 14400.00 | 2024-09-18 10:15:00 | 14169.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-09-17 10:45:00 | 14406.85 | 2024-09-18 10:15:00 | 14169.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-09-17 12:30:00 | 14399.70 | 2024-09-18 10:15:00 | 14169.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-09-17 13:30:00 | 14400.00 | 2024-09-18 10:15:00 | 14169.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-09-18 09:30:00 | 14444.00 | 2024-09-18 10:15:00 | 14169.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-09-20 15:00:00 | 13500.00 | 2024-09-27 14:15:00 | 13652.35 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-09-23 09:15:00 | 13382.60 | 2024-09-27 14:15:00 | 13652.35 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-10-11 10:15:00 | 13300.00 | 2024-10-14 12:15:00 | 13420.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-10-11 11:00:00 | 13329.95 | 2024-10-14 14:15:00 | 13534.95 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-10-11 14:15:00 | 13331.95 | 2024-10-14 14:15:00 | 13534.95 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-10-14 09:15:00 | 13305.00 | 2024-10-14 14:15:00 | 13534.95 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-10-14 10:45:00 | 13254.70 | 2024-10-14 14:15:00 | 13534.95 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-10-17 09:15:00 | 13500.00 | 2024-10-17 09:15:00 | 13360.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-10-21 10:15:00 | 13044.45 | 2024-10-23 09:15:00 | 12392.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 15:15:00 | 12851.20 | 2024-10-23 09:15:00 | 12208.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 10:15:00 | 13044.45 | 2024-10-23 10:15:00 | 12654.35 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2024-10-21 15:15:00 | 12851.20 | 2024-10-23 10:15:00 | 12654.35 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2024-11-13 09:15:00 | 11470.00 | 2024-11-14 09:15:00 | 10896.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-13 10:00:00 | 11401.10 | 2024-11-14 09:15:00 | 10831.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-13 09:15:00 | 11470.00 | 2024-11-14 12:15:00 | 11215.45 | STOP_HIT | 0.50 | 2.22% |
| SELL | retest2 | 2024-11-13 10:00:00 | 11401.10 | 2024-11-14 12:15:00 | 11215.45 | STOP_HIT | 0.50 | 1.63% |
| BUY | retest2 | 2024-11-22 12:30:00 | 11756.95 | 2024-11-26 12:15:00 | 11717.95 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-11-26 11:15:00 | 11745.00 | 2024-11-26 12:15:00 | 11717.95 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-11-26 12:15:00 | 11733.25 | 2024-11-26 12:15:00 | 11717.95 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2024-11-29 09:15:00 | 12140.00 | 2024-11-29 12:15:00 | 11974.20 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-12-03 15:15:00 | 11855.00 | 2024-12-10 09:15:00 | 11262.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-04 09:45:00 | 11880.00 | 2024-12-10 09:15:00 | 11286.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-04 10:15:00 | 11880.85 | 2024-12-10 09:15:00 | 11286.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-04 13:15:00 | 11849.95 | 2024-12-10 09:15:00 | 11257.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-03 15:15:00 | 11855.00 | 2024-12-10 13:15:00 | 11594.95 | STOP_HIT | 0.50 | 2.19% |
| SELL | retest2 | 2024-12-04 09:45:00 | 11880.00 | 2024-12-10 13:15:00 | 11594.95 | STOP_HIT | 0.50 | 2.40% |
| SELL | retest2 | 2024-12-04 10:15:00 | 11880.85 | 2024-12-10 13:15:00 | 11594.95 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2024-12-04 13:15:00 | 11849.95 | 2024-12-10 13:15:00 | 11594.95 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2024-12-05 11:30:00 | 11660.00 | 2024-12-10 14:15:00 | 11818.10 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-12-05 12:15:00 | 11691.90 | 2024-12-10 14:15:00 | 11818.10 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-12-05 13:45:00 | 11720.00 | 2024-12-11 09:15:00 | 11732.50 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2024-12-05 15:00:00 | 11700.00 | 2024-12-11 09:15:00 | 11732.50 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-12-09 12:45:00 | 11600.00 | 2024-12-11 09:15:00 | 11732.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-12-09 15:15:00 | 11400.00 | 2024-12-11 09:15:00 | 11732.50 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-12-17 12:15:00 | 11521.85 | 2024-12-19 14:15:00 | 11694.45 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-12-17 13:15:00 | 11517.45 | 2024-12-19 14:15:00 | 11694.45 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-12-18 14:30:00 | 11524.85 | 2024-12-19 14:15:00 | 11694.45 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-12-19 09:15:00 | 11444.35 | 2024-12-19 14:15:00 | 11694.45 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-12-19 13:45:00 | 11450.10 | 2024-12-19 14:15:00 | 11694.45 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-12-27 10:30:00 | 12679.00 | 2024-12-30 09:15:00 | 13946.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-28 09:15:00 | 14514.30 | 2025-01-29 14:15:00 | 13788.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-28 09:15:00 | 14514.30 | 2025-01-31 10:15:00 | 13948.15 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2025-02-11 09:15:00 | 14027.50 | 2025-02-13 11:15:00 | 14062.15 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-02-28 09:45:00 | 10130.00 | 2025-03-03 15:15:00 | 10485.00 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-02-28 15:15:00 | 10000.00 | 2025-03-03 15:15:00 | 10485.00 | STOP_HIT | 1.00 | -4.85% |
| BUY | retest2 | 2025-03-18 09:15:00 | 12645.30 | 2025-03-25 09:15:00 | 13909.83 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-09 09:15:00 | 12769.95 | 2025-04-11 14:15:00 | 13285.60 | STOP_HIT | 1.00 | -4.04% |
| SELL | retest2 | 2025-04-09 15:15:00 | 13000.05 | 2025-04-11 14:15:00 | 13285.60 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-04-11 09:30:00 | 13118.75 | 2025-04-11 14:15:00 | 13285.60 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-04-28 11:15:00 | 13898.00 | 2025-04-30 11:15:00 | 13203.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-28 13:45:00 | 13780.00 | 2025-04-30 12:15:00 | 13091.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-28 11:15:00 | 13898.00 | 2025-05-02 14:15:00 | 13000.00 | STOP_HIT | 0.50 | 6.46% |
| SELL | retest2 | 2025-04-28 13:45:00 | 13780.00 | 2025-05-02 14:15:00 | 13000.00 | STOP_HIT | 0.50 | 5.66% |
| BUY | retest2 | 2025-05-20 10:45:00 | 14319.00 | 2025-05-22 14:15:00 | 15620.00 | TARGET_HIT | 1.00 | 9.09% |
| BUY | retest2 | 2025-05-20 12:00:00 | 14200.00 | 2025-05-22 14:15:00 | 15582.60 | TARGET_HIT | 1.00 | 9.74% |
| BUY | retest2 | 2025-05-21 09:15:00 | 14380.00 | 2025-05-23 09:15:00 | 15750.90 | TARGET_HIT | 1.00 | 9.53% |
| BUY | retest2 | 2025-05-21 14:00:00 | 14166.00 | 2025-05-26 13:15:00 | 15818.00 | TARGET_HIT | 1.00 | 11.66% |
| BUY | retest2 | 2025-05-21 15:15:00 | 14460.00 | 2025-05-28 09:15:00 | 15896.10 | TARGET_HIT | 1.00 | 9.93% |
| BUY | retest2 | 2025-05-22 09:45:00 | 14451.00 | 2025-05-29 11:15:00 | 15329.00 | STOP_HIT | 1.00 | 6.08% |
| SELL | retest2 | 2025-06-03 15:15:00 | 15276.00 | 2025-06-05 09:15:00 | 14512.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-04 11:00:00 | 15266.00 | 2025-06-05 09:15:00 | 14502.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-03 15:15:00 | 15276.00 | 2025-06-05 15:15:00 | 14858.00 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2025-06-04 11:00:00 | 15266.00 | 2025-06-05 15:15:00 | 14858.00 | STOP_HIT | 0.50 | 2.67% |
| SELL | retest2 | 2025-07-07 09:15:00 | 14822.00 | 2025-07-11 09:15:00 | 14080.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-07 15:00:00 | 14825.00 | 2025-07-11 09:15:00 | 14083.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 09:30:00 | 14799.00 | 2025-07-11 09:15:00 | 14098.95 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2025-07-09 09:15:00 | 14841.00 | 2025-07-11 10:15:00 | 14059.05 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2025-07-07 09:15:00 | 14822.00 | 2025-07-11 14:15:00 | 14423.00 | STOP_HIT | 0.50 | 2.69% |
| SELL | retest2 | 2025-07-07 15:00:00 | 14825.00 | 2025-07-11 14:15:00 | 14423.00 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2025-07-08 09:30:00 | 14799.00 | 2025-07-11 14:15:00 | 14423.00 | STOP_HIT | 0.50 | 2.54% |
| SELL | retest2 | 2025-07-09 09:15:00 | 14841.00 | 2025-07-11 14:15:00 | 14423.00 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2025-07-10 10:15:00 | 14536.00 | 2025-07-15 11:15:00 | 14494.00 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-07-14 09:15:00 | 14169.00 | 2025-07-15 11:15:00 | 14494.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-07-16 13:00:00 | 14459.00 | 2025-07-18 11:15:00 | 14422.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-07-16 13:45:00 | 14466.00 | 2025-07-18 11:15:00 | 14422.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-07-18 09:15:00 | 14510.00 | 2025-07-18 11:15:00 | 14422.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-07-24 12:00:00 | 14308.00 | 2025-07-24 14:15:00 | 14550.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-07-31 14:30:00 | 14960.00 | 2025-08-04 11:15:00 | 14683.00 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-08-01 13:15:00 | 14845.00 | 2025-08-04 11:15:00 | 14683.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-08-01 14:30:00 | 14848.00 | 2025-08-04 11:15:00 | 14683.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-08-21 11:15:00 | 13454.00 | 2025-08-22 09:15:00 | 14168.00 | STOP_HIT | 1.00 | -5.31% |
| SELL | retest2 | 2025-08-21 13:30:00 | 13463.00 | 2025-08-22 09:15:00 | 14168.00 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2025-09-03 10:15:00 | 13812.00 | 2025-09-10 13:15:00 | 13710.00 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest1 | 2025-09-11 14:00:00 | 14001.00 | 2025-09-16 14:15:00 | 14701.05 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-09-11 14:00:00 | 14001.00 | 2025-09-17 12:15:00 | 15401.10 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-22 09:15:00 | 15138.00 | 2025-09-26 13:15:00 | 15125.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-10-08 09:15:00 | 16220.00 | 2025-10-08 09:15:00 | 16118.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-10-20 09:15:00 | 17190.00 | 2025-10-29 10:15:00 | 16868.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-10-20 11:45:00 | 16918.00 | 2025-10-29 10:15:00 | 16868.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-10-20 12:30:00 | 16932.00 | 2025-10-29 10:15:00 | 16868.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-10-21 13:45:00 | 17052.00 | 2025-10-29 10:15:00 | 16868.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-10-27 09:15:00 | 17520.00 | 2025-10-29 10:15:00 | 16868.00 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2025-10-28 14:45:00 | 17181.00 | 2025-10-29 10:15:00 | 16868.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-10-29 09:15:00 | 17265.00 | 2025-10-29 10:15:00 | 16868.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-11-06 09:15:00 | 17640.00 | 2025-11-06 09:15:00 | 17172.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest1 | 2025-11-10 09:15:00 | 17658.00 | 2025-11-10 13:15:00 | 17425.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest1 | 2025-11-10 12:45:00 | 17528.00 | 2025-11-10 13:15:00 | 17425.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-11-13 09:15:00 | 17603.00 | 2025-11-13 10:15:00 | 17330.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-20 15:15:00 | 17100.00 | 2025-11-24 14:15:00 | 17653.00 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-11-21 10:00:00 | 17088.00 | 2025-11-24 14:15:00 | 17653.00 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2025-11-24 09:15:00 | 17081.00 | 2025-11-24 14:15:00 | 17653.00 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-11-24 11:15:00 | 17141.00 | 2025-11-24 14:15:00 | 17653.00 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-12-02 09:15:00 | 18438.00 | 2025-12-10 13:15:00 | 18723.00 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2025-12-02 11:30:00 | 18376.00 | 2025-12-10 13:15:00 | 18723.00 | STOP_HIT | 1.00 | 1.89% |
| BUY | retest2 | 2025-12-03 12:45:00 | 18350.00 | 2025-12-10 13:15:00 | 18723.00 | STOP_HIT | 1.00 | 2.03% |
| BUY | retest2 | 2025-12-05 09:45:00 | 18355.00 | 2025-12-10 13:15:00 | 18723.00 | STOP_HIT | 1.00 | 2.00% |
| SELL | retest2 | 2025-12-11 14:15:00 | 18448.00 | 2025-12-15 14:15:00 | 17525.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 09:45:00 | 18475.00 | 2025-12-15 14:15:00 | 17551.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 11:30:00 | 18433.00 | 2025-12-15 14:15:00 | 17511.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-11 14:15:00 | 18448.00 | 2025-12-18 09:15:00 | 16603.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-12 09:45:00 | 18475.00 | 2025-12-18 09:15:00 | 16627.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-12 11:30:00 | 18433.00 | 2025-12-18 10:15:00 | 16589.70 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-26 14:45:00 | 18180.00 | 2025-12-30 14:15:00 | 18087.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-01-06 14:15:00 | 17941.00 | 2026-01-12 15:15:00 | 17994.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2026-01-12 15:00:00 | 17916.00 | 2026-01-12 15:15:00 | 17994.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2026-01-21 09:15:00 | 17394.00 | 2026-01-22 10:15:00 | 17729.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-02-03 10:30:00 | 17990.00 | 2026-02-09 11:15:00 | 18191.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-02-12 10:45:00 | 18656.00 | 2026-02-13 13:15:00 | 18175.00 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-02-24 09:15:00 | 17711.00 | 2026-02-25 09:15:00 | 18010.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-03-24 11:15:00 | 16525.00 | 2026-03-27 09:15:00 | 15698.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-24 13:00:00 | 16604.00 | 2026-03-27 09:15:00 | 15773.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-24 13:45:00 | 16584.00 | 2026-03-27 09:15:00 | 15754.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-24 15:00:00 | 16539.00 | 2026-03-27 09:15:00 | 15712.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-24 11:15:00 | 16525.00 | 2026-03-30 10:15:00 | 14872.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 13:00:00 | 16604.00 | 2026-03-30 10:15:00 | 14943.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 13:45:00 | 16584.00 | 2026-03-30 10:15:00 | 14925.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 15:00:00 | 16539.00 | 2026-03-30 10:15:00 | 14885.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 14968.00 | 2026-04-06 12:15:00 | 15372.00 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-04-01 13:30:00 | 14969.00 | 2026-04-06 12:15:00 | 15372.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-04-02 09:15:00 | 14579.00 | 2026-04-06 12:15:00 | 15372.00 | STOP_HIT | 1.00 | -5.44% |
| BUY | retest2 | 2026-04-20 09:15:00 | 15828.00 | 2026-04-22 14:15:00 | 15889.00 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2026-04-20 09:45:00 | 15805.00 | 2026-04-22 14:15:00 | 15889.00 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2026-05-08 09:15:00 | 16950.00 | 2026-05-11 09:15:00 | 16598.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2026-05-08 14:00:00 | 16867.00 | 2026-05-11 09:15:00 | 16598.00 | STOP_HIT | 1.00 | -1.59% |
