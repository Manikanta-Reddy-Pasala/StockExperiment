# MRF Ltd. (MRF)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 130490.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 42 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 30 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 26
- **Target hits / Stop hits / Partials:** 4 / 30 / 7
- **Avg / median % per leg:** 1.03% / -1.02%
- **Sum % (uncompounded):** 42.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 1 | 9.1% | 1 | 10 | 0 | -0.08% | -0.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 1 | 9.1% | 1 | 10 | 0 | -0.08% | -0.9% |
| SELL (all) | 30 | 14 | 46.7% | 3 | 20 | 7 | 1.44% | 43.3% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.49% | -14.0% |
| SELL @ 3rd Alert (retest2) | 26 | 14 | 53.8% | 3 | 16 | 7 | 2.20% | 57.2% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.49% | -14.0% |
| retest2 (combined) | 37 | 15 | 40.5% | 4 | 26 | 7 | 1.52% | 56.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 15:15:00 | 129300.00 | 136221.98 | 136231.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 09:15:00 | 129098.00 | 136151.09 | 136196.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 133901.00 | 133175.51 | 134455.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-30 09:45:00 | 134000.00 | 133175.51 | 134455.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 134674.34 | 133190.42 | 134456.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 11:00:00 | 134674.34 | 133190.42 | 134456.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 134405.09 | 133202.51 | 134456.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 12:30:00 | 133795.55 | 133204.47 | 134451.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 13:45:00 | 133988.70 | 133234.85 | 134417.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 15:00:00 | 133825.20 | 133240.73 | 134414.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 09:15:00 | 130300.15 | 133248.28 | 134412.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 127105.77 | 132930.85 | 134205.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 127289.26 | 132930.85 | 134205.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 127133.94 | 132930.85 | 134205.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 10:15:00 | 123785.14 | 132382.52 | 133875.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-23 14:15:00 | 130000.00 | 129989.77 | 131867.05 | SL hit (close>ema200) qty=0.50 sl=129989.77 alert=retest2 |

### Cycle 2 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 136147.25 | 129609.99 | 129582.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 14:15:00 | 137262.91 | 129686.14 | 129621.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 09:15:00 | 132846.70 | 134069.21 | 132257.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 09:15:00 | 132846.70 | 134069.21 | 132257.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 132846.70 | 134069.21 | 132257.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:00:00 | 132846.70 | 134069.21 | 132257.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 132403.25 | 134052.63 | 132258.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:00:00 | 132403.25 | 134052.63 | 132258.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 132529.00 | 134037.47 | 132260.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:00:00 | 132529.00 | 134037.47 | 132260.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 132159.84 | 134018.79 | 132259.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:00:00 | 132159.84 | 134018.79 | 132259.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 141273.66 | 134090.97 | 132304.46 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 129700.00 | 134819.33 | 134834.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 128999.00 | 134311.32 | 134573.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 125491.75 | 124720.77 | 127907.36 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 11:30:00 | 124907.10 | 124734.47 | 127882.54 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 13:30:00 | 125068.10 | 124745.32 | 127856.68 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 11:15:00 | 125035.05 | 124747.91 | 127796.34 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:00:00 | 125215.00 | 124680.15 | 127280.84 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 126858.00 | 124839.00 | 127198.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 128086.50 | 124839.00 | 127198.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 129421.00 | 124884.59 | 127209.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-05 09:15:00 | 129421.00 | 124884.59 | 127209.62 | SL hit (close>ema400) qty=1.00 sl=127209.62 alert=retest1 |

### Cycle 4 — BUY (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 11:15:00 | 131058.60 | 128738.26 | 128736.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 15:15:00 | 131494.05 | 128826.38 | 128781.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 128314.00 | 129243.72 | 129014.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 128314.00 | 129243.72 | 129014.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 128314.00 | 129243.72 | 129014.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:45:00 | 128599.80 | 129243.72 | 129014.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 128401.05 | 129235.33 | 129011.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 14:15:00 | 128895.05 | 129213.80 | 129004.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 10:00:00 | 128905.35 | 129211.52 | 129006.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 10:15:00 | 127828.30 | 129197.76 | 129000.21 | SL hit (close<static) qty=1.00 sl=128025.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 122707.00 | 128772.63 | 128793.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 09:15:00 | 122535.80 | 128653.65 | 128733.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 108800.05 | 108676.32 | 112648.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 10:00:00 | 108800.05 | 108676.32 | 112648.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 112393.00 | 108933.39 | 112493.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 10:45:00 | 112391.00 | 108933.39 | 112493.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 112224.30 | 108966.13 | 112491.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 15:00:00 | 112000.00 | 109064.50 | 112489.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-24 09:15:00 | 113743.00 | 109140.96 | 112493.38 | SL hit (close>static) qty=1.00 sl=112500.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 14:15:00 | 126500.00 | 113958.76 | 113922.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 09:15:00 | 126740.00 | 114210.17 | 114049.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 11:15:00 | 136715.00 | 136842.67 | 130512.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 12:00:00 | 136715.00 | 136842.67 | 130512.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 135000.00 | 137172.95 | 132971.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 135000.00 | 137172.95 | 132971.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 142500.00 | 145991.33 | 142574.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 142500.00 | 145991.33 | 142574.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 142845.00 | 145960.03 | 142575.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:30:00 | 142995.00 | 145897.08 | 142577.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 11:00:00 | 143060.00 | 145897.08 | 142577.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 11:45:00 | 143005.00 | 145867.26 | 142579.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 142955.00 | 145867.26 | 142579.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 142285.00 | 145774.62 | 142581.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 142285.00 | 145774.62 | 142581.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 141500.00 | 145732.09 | 142576.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-11 15:15:00 | 141500.00 | 145732.09 | 142576.19 | SL hit (close<static) qty=1.00 sl=142000.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 148510.00 | 153067.73 | 153080.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 148100.00 | 151991.20 | 152478.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 12:15:00 | 143855.00 | 141379.96 | 145534.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 12:15:00 | 143855.00 | 141379.96 | 145534.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 143855.00 | 141379.96 | 145534.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 143855.00 | 141379.96 | 145534.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 146700.00 | 141432.90 | 145540.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 11:00:00 | 143215.00 | 144873.99 | 146277.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:45:00 | 143190.00 | 144824.09 | 146231.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:45:00 | 143080.00 | 144859.01 | 146130.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 136054.25 | 144203.07 | 145713.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 136030.50 | 144203.07 | 145713.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 135926.00 | 144203.07 | 145713.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-20 13:15:00 | 128893.50 | 138565.53 | 141846.72 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-30 12:30:00 | 133795.55 | 2024-05-06 09:15:00 | 127105.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 13:45:00 | 133988.70 | 2024-05-06 09:15:00 | 127289.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 15:00:00 | 133825.20 | 2024-05-06 09:15:00 | 127133.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 09:15:00 | 130300.15 | 2024-05-07 10:15:00 | 123785.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-30 12:30:00 | 133795.55 | 2024-05-23 14:15:00 | 130000.00 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2024-05-02 13:45:00 | 133988.70 | 2024-05-23 14:15:00 | 130000.00 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2024-05-02 15:00:00 | 133825.20 | 2024-05-23 14:15:00 | 130000.00 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2024-05-03 09:15:00 | 130300.15 | 2024-05-23 14:15:00 | 130000.00 | STOP_HIT | 0.50 | 0.23% |
| SELL | retest2 | 2024-07-02 09:30:00 | 129011.45 | 2024-07-08 10:15:00 | 130739.00 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-07-02 11:30:00 | 129107.05 | 2024-07-08 10:15:00 | 130739.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-07-05 10:15:00 | 129336.55 | 2024-07-08 10:15:00 | 130739.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-07-08 13:00:00 | 128928.00 | 2024-07-09 13:15:00 | 130534.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-07-10 10:15:00 | 129101.00 | 2024-07-10 12:15:00 | 130437.60 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-07-10 11:00:00 | 129217.40 | 2024-07-10 12:15:00 | 130437.60 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-07-19 09:45:00 | 129321.15 | 2024-07-22 10:15:00 | 130640.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest1 | 2024-11-25 11:30:00 | 124907.10 | 2024-12-05 09:15:00 | 129421.00 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest1 | 2024-11-25 13:30:00 | 125068.10 | 2024-12-05 09:15:00 | 129421.00 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest1 | 2024-11-26 11:15:00 | 125035.05 | 2024-12-05 09:15:00 | 129421.00 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest1 | 2024-12-03 10:00:00 | 125215.00 | 2024-12-05 09:15:00 | 129421.00 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2025-01-02 14:15:00 | 128895.05 | 2025-01-03 10:15:00 | 127828.30 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-01-03 10:00:00 | 128905.35 | 2025-01-03 10:15:00 | 127828.30 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-03-21 15:00:00 | 112000.00 | 2025-03-24 09:15:00 | 113743.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-04-07 09:15:00 | 110264.20 | 2025-04-08 11:15:00 | 112635.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-04-08 10:30:00 | 111740.00 | 2025-04-08 11:15:00 | 112635.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-04-09 09:45:00 | 111800.80 | 2025-04-09 12:15:00 | 113235.40 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-04-09 15:15:00 | 112853.05 | 2025-04-11 09:15:00 | 116159.65 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-08-11 10:30:00 | 142995.00 | 2025-08-11 15:15:00 | 141500.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-08-11 11:00:00 | 143060.00 | 2025-08-11 15:15:00 | 141500.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-08-11 11:45:00 | 143005.00 | 2025-08-11 15:15:00 | 141500.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-08-11 12:15:00 | 142955.00 | 2025-08-11 15:15:00 | 141500.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-08-12 09:45:00 | 142400.00 | 2025-08-12 14:15:00 | 140600.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-08-18 10:30:00 | 142400.00 | 2025-08-29 09:15:00 | 140765.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-08-28 10:30:00 | 142900.00 | 2025-08-29 09:15:00 | 140765.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-08-28 14:15:00 | 142410.00 | 2025-08-29 09:15:00 | 140765.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-09-01 14:15:00 | 144650.00 | 2025-10-08 11:15:00 | 159115.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-24 11:00:00 | 143215.00 | 2026-03-04 09:15:00 | 136054.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 13:45:00 | 143190.00 | 2026-03-04 09:15:00 | 136030.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 10:45:00 | 143080.00 | 2026-03-04 09:15:00 | 135926.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 11:00:00 | 143215.00 | 2026-03-20 13:15:00 | 128893.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 13:45:00 | 143190.00 | 2026-03-20 13:15:00 | 128871.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-27 10:45:00 | 143080.00 | 2026-03-20 13:15:00 | 128772.00 | TARGET_HIT | 0.50 | 10.00% |
