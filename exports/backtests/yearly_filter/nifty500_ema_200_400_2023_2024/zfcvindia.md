# ZF Commercial Vehicle Control Systems India Ltd. (ZFCVINDIA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 14532.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 2 |
| ALERT3 | 55 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 39 |
| PARTIAL | 12 |
| TARGET_HIT | 17 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 21
- **Target hits / Stop hits / Partials:** 17 / 23 / 12
- **Avg / median % per leg:** 3.57% / 5.00%
- **Sum % (uncompounded):** 185.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 8 | 34.8% | 7 | 15 | 1 | 1.88% | 43.2% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| BUY @ 3rd Alert (retest2) | 21 | 6 | 28.6% | 6 | 15 | 0 | 1.34% | 28.2% |
| SELL (all) | 29 | 23 | 79.3% | 10 | 8 | 11 | 4.91% | 142.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 23 | 79.3% | 10 | 8 | 11 | 4.91% | 142.5% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest2 (combined) | 50 | 29 | 58.0% | 16 | 23 | 11 | 3.41% | 170.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 14351.00 | 15831.99 | 15838.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 14:15:00 | 14149.40 | 15298.16 | 15525.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 09:15:00 | 14722.80 | 14712.23 | 15087.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-15 10:00:00 | 14722.80 | 14712.23 | 15087.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 10:15:00 | 15205.65 | 14717.14 | 15088.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 11:00:00 | 15205.65 | 14717.14 | 15088.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 11:15:00 | 15097.40 | 14720.92 | 15088.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 11:45:00 | 15099.00 | 14720.92 | 15088.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 12:15:00 | 15163.70 | 14725.33 | 15088.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 12:45:00 | 15200.30 | 14725.33 | 15088.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 13:15:00 | 15246.95 | 14730.52 | 15089.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 13:45:00 | 15189.55 | 14730.52 | 15089.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 12:15:00 | 15022.70 | 14755.98 | 15091.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 12:45:00 | 15091.05 | 14755.98 | 15091.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 13:15:00 | 15290.30 | 14761.30 | 15092.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 14:00:00 | 15290.30 | 14761.30 | 15092.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 14:15:00 | 15625.00 | 14769.89 | 15095.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 15:00:00 | 15625.00 | 14769.89 | 15095.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 15:15:00 | 15482.00 | 14905.39 | 15133.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 09:30:00 | 15100.00 | 14932.67 | 15138.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 11:30:00 | 15077.00 | 14935.92 | 15137.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 13:15:00 | 14997.15 | 14937.60 | 15137.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 15:00:00 | 15097.75 | 14941.05 | 15124.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 15:15:00 | 15080.00 | 14942.43 | 15123.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 09:15:00 | 15054.15 | 14942.43 | 15123.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 15077.05 | 14943.77 | 15123.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-03 12:45:00 | 14937.20 | 14966.71 | 15120.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-03 15:15:00 | 14910.10 | 14967.47 | 15119.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-04 09:45:00 | 14751.60 | 14965.00 | 15117.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 09:15:00 | 14807.55 | 14957.57 | 15108.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 09:15:00 | 14345.00 | 14858.88 | 15035.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 09:15:00 | 14342.86 | 14858.88 | 15035.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 11:15:00 | 14323.15 | 14849.35 | 15029.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 12:15:00 | 14247.29 | 14843.29 | 15025.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 14:15:00 | 14190.34 | 14830.66 | 15017.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 14:15:00 | 14164.59 | 14830.66 | 15017.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 14014.02 | 14816.05 | 15008.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 14067.17 | 14816.05 | 15008.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-04-16 14:15:00 | 13590.00 | 14697.81 | 14936.14 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 17300.00 | 14472.68 | 14460.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 14:15:00 | 17669.60 | 15410.58 | 14983.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 13:15:00 | 15900.20 | 15960.34 | 15371.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-20 13:45:00 | 15900.10 | 15960.34 | 15371.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 15451.00 | 15878.14 | 15519.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 13:00:00 | 15451.00 | 15878.14 | 15519.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 15628.85 | 15875.66 | 15520.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 14:45:00 | 15755.10 | 15874.29 | 15521.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 11:15:00 | 15638.10 | 15873.98 | 15593.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 14:00:00 | 15791.80 | 15866.82 | 15593.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 15158.00 | 15857.67 | 15593.06 | SL hit (close<static) qty=1.00 sl=15451.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 11:15:00 | 15351.45 | 15568.84 | 15569.26 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 15836.40 | 15569.26 | 15569.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 09:15:00 | 16245.50 | 15586.27 | 15577.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 15950.00 | 16135.81 | 15920.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 10:00:00 | 15950.00 | 16135.81 | 15920.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 16075.00 | 16135.20 | 15921.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 15824.95 | 16135.20 | 15921.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 16038.10 | 16129.98 | 15924.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 10:45:00 | 16160.00 | 16129.39 | 15925.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 12:00:00 | 16164.20 | 16129.74 | 15926.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:15:00 | 16158.40 | 16127.60 | 15929.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 12:15:00 | 15852.30 | 16121.44 | 15930.52 | SL hit (close<static) qty=1.00 sl=15902.35 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 15064.50 | 15806.12 | 15808.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 15:15:00 | 14900.00 | 15695.64 | 15749.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 14938.50 | 14738.53 | 15124.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-07 09:30:00 | 15084.40 | 14738.53 | 15124.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 15274.30 | 14764.70 | 15099.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:00:00 | 15274.30 | 14764.70 | 15099.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 15148.55 | 14768.52 | 15099.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:15:00 | 15071.80 | 14768.52 | 15099.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:00:00 | 15106.65 | 14771.89 | 15099.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 14:15:00 | 14351.32 | 14760.43 | 15010.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-27 09:15:00 | 13564.62 | 14735.25 | 14995.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 12769.90 | 11602.42 | 11598.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 13467.40 | 11640.56 | 11617.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 13:15:00 | 12109.00 | 12158.52 | 11920.25 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 09:15:00 | 12355.80 | 12156.83 | 11921.77 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 12:15:00 | 12973.59 | 12200.22 | 11956.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-04-16 12:15:00 | 13591.38 | 12278.41 | 12004.65 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 7 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 12561.00 | 13487.73 | 13492.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 10:15:00 | 12540.00 | 13461.32 | 13478.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 10:15:00 | 13501.00 | 13342.13 | 13413.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 10:15:00 | 13501.00 | 13342.13 | 13413.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 13501.00 | 13342.13 | 13413.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:00:00 | 13501.00 | 13342.13 | 13413.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 13640.00 | 13345.10 | 13414.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:45:00 | 13768.00 | 13345.10 | 13414.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 13447.00 | 13357.19 | 13418.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:15:00 | 13522.00 | 13357.19 | 13418.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 13485.00 | 13358.46 | 13418.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:30:00 | 13511.00 | 13358.46 | 13418.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 13410.00 | 13366.19 | 13417.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:45:00 | 13436.00 | 13366.19 | 13417.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 13380.00 | 13366.33 | 13417.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:45:00 | 13341.00 | 13366.07 | 13416.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 13321.00 | 13366.21 | 13416.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:15:00 | 12673.95 | 13134.31 | 13258.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:15:00 | 12654.95 | 13134.31 | 13258.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 13:15:00 | 12943.00 | 12936.07 | 13112.33 | SL hit (close>ema200) qty=0.50 sl=12936.07 alert=retest2 |

### Cycle 8 — BUY (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 12:15:00 | 14937.00 | 13153.09 | 13151.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 12:15:00 | 15258.00 | 13878.20 | 13577.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 14471.00 | 14576.60 | 14144.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 15:00:00 | 14471.00 | 14576.60 | 14144.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 14024.00 | 14546.66 | 14148.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:45:00 | 13990.00 | 14546.66 | 14148.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 13900.00 | 14540.23 | 14147.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:30:00 | 13914.00 | 14540.23 | 14147.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 14052.00 | 14524.48 | 14145.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 14052.00 | 14524.48 | 14145.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 13907.00 | 14518.34 | 14143.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 13907.00 | 14518.34 | 14143.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 13946.00 | 14502.04 | 14141.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 13946.00 | 14502.04 | 14141.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 14136.00 | 14494.46 | 14141.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 14:45:00 | 14280.00 | 14198.92 | 14069.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 14208.00 | 14199.43 | 14070.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 11:00:00 | 14196.00 | 14199.40 | 14071.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 12:30:00 | 14240.00 | 14200.14 | 14073.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-11 09:15:00 | 15628.80 | 14635.01 | 14359.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 14:15:00 | 13632.00 | 14577.79 | 14579.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 15:15:00 | 13577.00 | 14567.84 | 14574.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 14465.00 | 14120.10 | 14316.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 14465.00 | 14120.10 | 14316.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 14465.00 | 14120.10 | 14316.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 14465.00 | 14120.10 | 14316.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 14197.00 | 14120.86 | 14316.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:45:00 | 14181.00 | 14126.34 | 14316.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 15:15:00 | 14780.00 | 14134.29 | 14318.16 | SL hit (close>static) qty=1.00 sl=14480.00 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 10:15:00 | 14999.00 | 14382.19 | 14380.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 14:15:00 | 15177.00 | 14410.75 | 14394.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 14:15:00 | 14531.00 | 14561.95 | 14485.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 15:00:00 | 14531.00 | 14561.95 | 14485.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 14523.00 | 14572.12 | 14495.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 14:30:00 | 14523.00 | 14572.12 | 14495.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-03-26 09:30:00 | 15100.00 | 2024-04-12 09:15:00 | 14345.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-26 11:30:00 | 15077.00 | 2024-04-12 09:15:00 | 14342.86 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2024-03-26 13:15:00 | 14997.15 | 2024-04-12 11:15:00 | 14323.15 | PARTIAL | 0.50 | 4.49% |
| SELL | retest2 | 2024-03-28 15:00:00 | 15097.75 | 2024-04-12 12:15:00 | 14247.29 | PARTIAL | 0.50 | 5.63% |
| SELL | retest2 | 2024-04-03 12:45:00 | 14937.20 | 2024-04-12 14:15:00 | 14190.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-03 15:15:00 | 14910.10 | 2024-04-12 14:15:00 | 14164.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-04 09:45:00 | 14751.60 | 2024-04-15 09:15:00 | 14014.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-05 09:15:00 | 14807.55 | 2024-04-15 09:15:00 | 14067.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-26 09:30:00 | 15100.00 | 2024-04-16 14:15:00 | 13590.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-26 11:30:00 | 15077.00 | 2024-04-16 14:15:00 | 13569.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-26 13:15:00 | 14997.15 | 2024-04-16 14:15:00 | 13497.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-28 15:00:00 | 15097.75 | 2024-04-16 14:15:00 | 13587.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-03 12:45:00 | 14937.20 | 2024-04-16 14:15:00 | 13443.48 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-03 15:15:00 | 14910.10 | 2024-04-16 14:15:00 | 13419.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-04 09:45:00 | 14751.60 | 2024-04-16 15:15:00 | 13326.80 | TARGET_HIT | 0.50 | 9.66% |
| SELL | retest2 | 2024-04-05 09:15:00 | 14807.55 | 2024-05-09 14:15:00 | 13276.44 | TARGET_HIT | 0.50 | 10.34% |
| BUY | retest2 | 2024-07-08 14:45:00 | 15755.10 | 2024-07-19 09:15:00 | 15158.00 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2024-07-18 11:15:00 | 15638.10 | 2024-07-19 09:15:00 | 15158.00 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2024-07-18 14:00:00 | 15791.80 | 2024-07-19 09:15:00 | 15158.00 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest2 | 2024-07-24 11:15:00 | 15633.20 | 2024-07-24 14:15:00 | 15418.95 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-08-16 12:45:00 | 15632.50 | 2024-08-23 09:15:00 | 15508.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-08-16 15:15:00 | 16000.00 | 2024-08-23 09:15:00 | 15508.00 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2024-08-20 14:30:00 | 15640.60 | 2024-08-23 09:15:00 | 15508.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-08-21 12:45:00 | 15650.15 | 2024-08-23 09:15:00 | 15508.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-08-30 15:15:00 | 15880.00 | 2024-09-02 09:15:00 | 15500.05 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2024-09-26 10:45:00 | 16160.00 | 2024-09-27 12:15:00 | 15852.30 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-09-26 12:00:00 | 16164.20 | 2024-09-27 12:15:00 | 15852.30 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-09-27 09:15:00 | 16158.40 | 2024-09-27 12:15:00 | 15852.30 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-09-27 15:00:00 | 16337.10 | 2024-09-30 13:15:00 | 15894.80 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2024-10-01 15:00:00 | 16122.85 | 2024-10-03 10:15:00 | 15892.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-11-12 11:15:00 | 15071.80 | 2024-11-26 14:15:00 | 14351.32 | PARTIAL | 0.50 | 4.78% |
| SELL | retest2 | 2024-11-12 11:15:00 | 15071.80 | 2024-11-27 09:15:00 | 13564.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-12 12:00:00 | 15106.65 | 2024-11-27 09:15:00 | 13595.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-04-11 09:15:00 | 12355.80 | 2025-04-15 12:15:00 | 12973.59 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-11 09:15:00 | 12355.80 | 2025-04-16 12:15:00 | 13591.38 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-05-06 13:30:00 | 12257.00 | 2025-05-06 15:15:00 | 12051.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-05-07 11:30:00 | 12240.00 | 2025-05-15 09:15:00 | 13464.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-07 12:00:00 | 12260.00 | 2025-05-15 09:15:00 | 13486.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-09 13:45:00 | 13341.00 | 2025-10-31 10:15:00 | 12673.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 10:15:00 | 13321.00 | 2025-10-31 10:15:00 | 12654.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-09 13:45:00 | 13341.00 | 2025-11-13 13:15:00 | 12943.00 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2025-10-10 10:15:00 | 13321.00 | 2025-11-13 13:15:00 | 12943.00 | STOP_HIT | 0.50 | 2.84% |
| BUY | retest2 | 2026-01-28 14:45:00 | 14280.00 | 2026-02-11 09:15:00 | 15628.80 | TARGET_HIT | 1.00 | 9.45% |
| BUY | retest2 | 2026-01-29 10:15:00 | 14208.00 | 2026-02-11 09:15:00 | 15615.60 | TARGET_HIT | 1.00 | 9.91% |
| BUY | retest2 | 2026-01-29 11:00:00 | 14196.00 | 2026-02-11 09:15:00 | 15664.00 | TARGET_HIT | 1.00 | 10.34% |
| BUY | retest2 | 2026-01-29 12:30:00 | 14240.00 | 2026-02-11 10:15:00 | 15708.00 | TARGET_HIT | 1.00 | 10.31% |
| SELL | retest2 | 2026-04-01 13:45:00 | 14181.00 | 2026-04-01 15:15:00 | 14780.00 | STOP_HIT | 1.00 | -4.22% |
| SELL | retest2 | 2026-04-02 13:15:00 | 14160.00 | 2026-04-08 09:15:00 | 14542.00 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-04-09 09:45:00 | 14126.00 | 2026-04-17 09:15:00 | 14506.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-04-13 12:00:00 | 14167.00 | 2026-04-17 09:15:00 | 14506.00 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2026-04-15 11:15:00 | 14050.00 | 2026-04-17 09:15:00 | 14506.00 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2026-04-15 12:30:00 | 14101.00 | 2026-04-17 09:15:00 | 14506.00 | STOP_HIT | 1.00 | -2.87% |
