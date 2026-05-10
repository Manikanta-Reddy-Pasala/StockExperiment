# Dixon Technologies (India) Ltd. (DIXON)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 10825.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 58 |
| ALERT1 | 42 |
| ALERT2 | 42 |
| ALERT2_SKIP | 17 |
| ALERT3 | 97 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 63 |
| PARTIAL | 11 |
| TARGET_HIT | 1 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 26 / 50
- **Target hits / Stop hits / Partials:** 1 / 64 / 11
- **Avg / median % per leg:** 0.17% / -0.45%
- **Sum % (uncompounded):** 13.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 6 | 15.8% | 1 | 37 | 0 | -0.71% | -26.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 38 | 6 | 15.8% | 1 | 37 | 0 | -0.71% | -26.9% |
| SELL (all) | 38 | 20 | 52.6% | 0 | 27 | 11 | 1.06% | 40.2% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.46% | -4.4% |
| SELL @ 3rd Alert (retest2) | 35 | 20 | 57.1% | 0 | 24 | 11 | 1.27% | 44.6% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.46% | -4.4% |
| retest2 (combined) | 73 | 26 | 35.6% | 1 | 61 | 11 | 0.24% | 17.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 15993.00 | 15808.05 | 15793.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 16028.00 | 15852.04 | 15814.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 11:15:00 | 16141.00 | 16210.19 | 16103.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 11:15:00 | 16141.00 | 16210.19 | 16103.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 16141.00 | 16210.19 | 16103.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 16115.00 | 16210.19 | 16103.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 16120.00 | 16192.15 | 16104.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 16063.00 | 16192.15 | 16104.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 16087.00 | 16171.12 | 16103.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:00:00 | 16087.00 | 16171.12 | 16103.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 16106.00 | 16158.09 | 16103.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:30:00 | 16076.00 | 16158.09 | 16103.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 16145.00 | 16155.48 | 16107.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 16186.00 | 16155.48 | 16107.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 16172.00 | 16151.14 | 16113.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 12:15:00 | 16075.00 | 16130.77 | 16110.44 | SL hit (close<static) qty=1.00 sl=16101.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-15 12:15:00 | 16075.00 | 16130.77 | 16110.44 | SL hit (close<static) qty=1.00 sl=16101.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:45:00 | 16162.00 | 16139.42 | 16116.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 09:15:00 | 15690.00 | 16505.67 | 16537.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 15690.00 | 16505.67 | 16537.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 15211.00 | 15720.04 | 16053.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 12:15:00 | 15130.00 | 15124.13 | 15452.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 13:00:00 | 15130.00 | 15124.13 | 15452.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 15027.00 | 15084.53 | 15214.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 10:15:00 | 15009.00 | 15084.53 | 15214.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 14998.00 | 15069.25 | 15146.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 14:15:00 | 14967.00 | 14684.23 | 14655.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-04 14:15:00 | 14967.00 | 14684.23 | 14655.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 14967.00 | 14684.23 | 14655.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 15012.00 | 14895.44 | 14830.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 14928.00 | 14960.86 | 14904.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 14928.00 | 14960.86 | 14904.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 14928.00 | 14960.86 | 14904.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 14926.00 | 14960.86 | 14904.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 14891.00 | 14946.89 | 14903.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 14905.00 | 14946.89 | 14903.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 14954.00 | 14948.31 | 14908.26 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 14788.00 | 14874.57 | 14883.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 14743.00 | 14848.26 | 14870.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 14390.00 | 14316.30 | 14433.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 14390.00 | 14316.30 | 14433.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 14425.00 | 14342.57 | 14408.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 14332.00 | 14327.26 | 14395.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 14333.00 | 14376.80 | 14380.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 14317.00 | 14373.23 | 14377.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 09:15:00 | 13615.40 | 14141.78 | 14184.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 14378.00 | 14141.78 | 14184.95 | SL hit (close>static) qty=0.50 sl=14141.78 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 09:15:00 | 13616.35 | 14141.78 | 14184.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 14378.00 | 14141.78 | 14184.95 | SL hit (close>static) qty=0.50 sl=14141.78 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 09:15:00 | 13601.15 | 14141.78 | 14184.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 14378.00 | 14141.78 | 14184.95 | SL hit (close>static) qty=0.50 sl=14141.78 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 14514.00 | 14216.23 | 14214.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 14:15:00 | 14557.00 | 14401.06 | 14313.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 14484.00 | 14541.06 | 14442.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 14484.00 | 14541.06 | 14442.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 14225.00 | 14471.28 | 14427.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:30:00 | 14281.00 | 14471.28 | 14427.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 14123.00 | 14401.62 | 14400.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:00:00 | 14123.00 | 14401.62 | 14400.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 11:15:00 | 14160.00 | 14353.30 | 14378.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 09:15:00 | 14063.00 | 14189.11 | 14277.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 11:15:00 | 14203.00 | 14184.03 | 14258.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 12:15:00 | 14242.00 | 14184.03 | 14258.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 14150.00 | 14177.22 | 14249.08 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 14823.00 | 14344.21 | 14305.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 13:15:00 | 14982.00 | 14623.63 | 14516.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 14517.00 | 14697.84 | 14585.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 14517.00 | 14697.84 | 14585.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 14517.00 | 14697.84 | 14585.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 14535.00 | 14697.84 | 14585.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 14540.00 | 14666.27 | 14581.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:15:00 | 14604.00 | 14579.98 | 14557.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-16 09:15:00 | 16064.40 | 15925.72 | 15860.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 15950.00 | 16041.37 | 16043.93 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 16180.00 | 16069.10 | 16056.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 16270.00 | 16175.77 | 16119.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 16189.00 | 16198.20 | 16145.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 11:00:00 | 16189.00 | 16198.20 | 16145.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 16140.00 | 16189.70 | 16155.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:00:00 | 16140.00 | 16189.70 | 16155.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 16148.00 | 16181.36 | 16154.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 16148.00 | 16181.36 | 16154.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 16905.00 | 16762.03 | 16662.65 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 16303.00 | 16666.14 | 16697.59 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 16856.00 | 16717.46 | 16708.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 09:15:00 | 17077.00 | 16822.55 | 16761.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 16833.00 | 16907.84 | 16837.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 14:15:00 | 16833.00 | 16907.84 | 16837.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 16833.00 | 16907.84 | 16837.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 16833.00 | 16907.84 | 16837.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 16856.00 | 16897.48 | 16839.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 16956.00 | 16897.48 | 16839.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 11:45:00 | 16890.00 | 16904.85 | 16857.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 11:00:00 | 16888.00 | 16931.10 | 16898.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 14:30:00 | 16882.00 | 16909.54 | 16898.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 16620.00 | 16839.71 | 16867.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 16620.00 | 16839.71 | 16867.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 16620.00 | 16839.71 | 16867.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 16620.00 | 16839.71 | 16867.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 16620.00 | 16839.71 | 16867.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 16599.00 | 16791.56 | 16843.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 16648.00 | 16553.85 | 16640.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 16648.00 | 16553.85 | 16640.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 16648.00 | 16553.85 | 16640.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 16648.00 | 16553.85 | 16640.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 16675.00 | 16578.08 | 16643.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 16305.00 | 16578.08 | 16643.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 15870.00 | 15922.64 | 16031.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 15681.00 | 15922.64 | 16031.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 15997.00 | 15945.41 | 16023.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:45:00 | 16001.00 | 15945.41 | 16023.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 16024.00 | 15966.35 | 16013.64 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 12:15:00 | 16174.00 | 16043.88 | 16040.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 13:15:00 | 16182.00 | 16071.50 | 16052.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 16870.00 | 16912.75 | 16767.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 15:00:00 | 16870.00 | 16912.75 | 16767.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 16818.00 | 16870.87 | 16791.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 16805.00 | 16870.87 | 16791.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 16780.00 | 16852.70 | 16790.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 16919.00 | 16798.95 | 16783.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 14:00:00 | 16874.00 | 16852.53 | 16815.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 11:00:00 | 16839.00 | 16964.50 | 16936.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 16780.00 | 16912.32 | 16916.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 16780.00 | 16912.32 | 16916.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 16780.00 | 16912.32 | 16916.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 16780.00 | 16912.32 | 16916.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 16668.00 | 16835.72 | 16879.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 16794.00 | 16787.52 | 16842.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 10:15:00 | 16794.00 | 16787.52 | 16842.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 16794.00 | 16787.52 | 16842.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 16794.00 | 16787.52 | 16842.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 16799.00 | 16789.82 | 16838.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:45:00 | 16762.00 | 16789.82 | 16838.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 16830.00 | 16749.36 | 16792.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 16830.00 | 16749.36 | 16792.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 16775.00 | 16754.49 | 16791.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 16816.00 | 16754.49 | 16791.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 17072.00 | 16796.10 | 16793.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 17413.00 | 17028.35 | 16910.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 14:15:00 | 17861.00 | 17864.54 | 17772.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 15:00:00 | 17861.00 | 17864.54 | 17772.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 17870.00 | 17959.69 | 17896.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 17881.00 | 17959.69 | 17896.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 17998.00 | 17967.36 | 17905.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 12:30:00 | 18014.00 | 17974.88 | 17914.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 13:15:00 | 18038.00 | 17974.88 | 17914.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:00:00 | 18016.00 | 17994.86 | 17945.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:45:00 | 18010.00 | 17955.53 | 17944.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 18021.00 | 18014.39 | 17985.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 17963.00 | 18014.39 | 17985.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 17998.00 | 18011.12 | 17986.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 11:45:00 | 18052.00 | 18016.29 | 17990.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:00:00 | 18040.00 | 18021.03 | 17995.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:45:00 | 18046.00 | 18050.10 | 18023.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 11:15:00 | 18056.00 | 18050.10 | 18023.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 18040.00 | 18048.87 | 18027.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 18040.00 | 18048.87 | 18027.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 17994.00 | 18037.89 | 18024.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 17994.00 | 18037.89 | 18024.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 17970.00 | 18024.31 | 18019.47 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-15 14:15:00 | 17970.00 | 18024.31 | 18019.47 | SL hit (close<static) qty=1.00 sl=17984.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 14:15:00 | 17970.00 | 18024.31 | 18019.47 | SL hit (close<static) qty=1.00 sl=17984.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 14:15:00 | 17970.00 | 18024.31 | 18019.47 | SL hit (close<static) qty=1.00 sl=17984.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 14:15:00 | 17970.00 | 18024.31 | 18019.47 | SL hit (close<static) qty=1.00 sl=17984.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 17970.00 | 18024.31 | 18019.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 17990.00 | 18017.45 | 18016.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 18030.00 | 18017.45 | 18016.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 18075.00 | 18198.62 | 18206.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 18075.00 | 18198.62 | 18206.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 18075.00 | 18198.62 | 18206.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 18075.00 | 18198.62 | 18206.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 18075.00 | 18198.62 | 18206.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 18075.00 | 18198.62 | 18206.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 18038.00 | 18149.11 | 18181.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 11:15:00 | 18154.00 | 18137.91 | 18169.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 12:00:00 | 18154.00 | 18137.91 | 18169.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 18111.00 | 18132.53 | 18164.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 18068.00 | 18127.92 | 18154.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 18178.00 | 18142.59 | 18151.26 | SL hit (close>static) qty=1.00 sl=18166.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 09:15:00 | 18339.00 | 18188.46 | 18170.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 10:15:00 | 18409.00 | 18232.56 | 18192.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 18175.00 | 18257.87 | 18221.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 14:15:00 | 18175.00 | 18257.87 | 18221.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 18175.00 | 18257.87 | 18221.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 18175.00 | 18257.87 | 18221.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 18187.00 | 18243.69 | 18218.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 18166.00 | 18243.69 | 18218.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 18020.00 | 18171.64 | 18188.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 17833.00 | 18103.92 | 18156.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 16503.00 | 16437.44 | 16801.09 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 10:45:00 | 16355.00 | 16421.95 | 16760.99 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 12:15:00 | 16370.00 | 16429.36 | 16733.54 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 16609.00 | 16527.51 | 16593.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-03 15:15:00 | 16609.00 | 16527.51 | 16593.43 | SL hit (close>ema400) qty=1.00 sl=16593.43 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-03 15:15:00 | 16609.00 | 16527.51 | 16593.43 | SL hit (close>ema400) qty=1.00 sl=16593.43 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 16600.00 | 16527.51 | 16593.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 16625.00 | 16547.01 | 16596.30 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 12:15:00 | 16894.00 | 16665.43 | 16641.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 16997.00 | 16731.74 | 16673.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 16985.00 | 17106.27 | 16980.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 10:15:00 | 16985.00 | 17106.27 | 16980.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 16985.00 | 17106.27 | 16980.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 16985.00 | 17106.27 | 16980.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 17028.00 | 17090.61 | 16985.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:45:00 | 17067.00 | 17085.89 | 16992.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 16848.00 | 17028.09 | 16981.77 | SL hit (close<static) qty=1.00 sl=16960.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:00:00 | 17059.00 | 17001.14 | 16975.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:30:00 | 17054.00 | 17022.29 | 16990.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 12:00:00 | 17055.00 | 17022.29 | 16990.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 17246.00 | 17283.71 | 17181.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 16655.00 | 17111.47 | 17142.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 16655.00 | 17111.47 | 17142.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 16655.00 | 17111.47 | 17142.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 16655.00 | 17111.47 | 17142.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 16549.00 | 16929.78 | 17049.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 16796.00 | 16732.82 | 16886.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:15:00 | 16857.00 | 16732.82 | 16886.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 16666.00 | 16719.45 | 16866.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 16727.00 | 16719.45 | 16866.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 16776.00 | 16722.15 | 16819.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 16792.00 | 16722.15 | 16819.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 16781.00 | 16746.38 | 16814.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:45:00 | 16685.00 | 16719.56 | 16790.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 16835.00 | 16815.78 | 16814.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 16835.00 | 16815.78 | 16814.09 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 16754.00 | 16803.43 | 16808.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 16697.00 | 16770.71 | 16792.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 15465.00 | 15421.88 | 15513.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 11:00:00 | 15465.00 | 15421.88 | 15513.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 15442.00 | 15425.90 | 15507.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:30:00 | 15486.00 | 15425.90 | 15507.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 15484.00 | 15440.62 | 15494.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 15484.00 | 15440.62 | 15494.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 15539.00 | 15460.30 | 15498.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 15562.00 | 15460.30 | 15498.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 15526.00 | 15473.44 | 15500.71 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 12:15:00 | 15656.00 | 15542.18 | 15528.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 12:15:00 | 15670.00 | 15622.54 | 15583.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 15473.00 | 15595.11 | 15577.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 15473.00 | 15595.11 | 15577.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 15473.00 | 15595.11 | 15577.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 15473.00 | 15595.11 | 15577.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 15504.00 | 15576.89 | 15571.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 15384.00 | 15576.89 | 15571.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 15336.00 | 15528.71 | 15549.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 15242.00 | 15395.92 | 15460.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 14911.00 | 14881.54 | 15013.15 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:30:00 | 14790.00 | 14877.43 | 14956.00 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 14991.00 | 14892.55 | 14948.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 14991.00 | 14892.55 | 14948.77 | SL hit (close>ema400) qty=1.00 sl=14948.77 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 14991.00 | 14892.55 | 14948.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 15000.00 | 14914.04 | 14953.43 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 15125.00 | 14984.07 | 14980.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 15252.00 | 15070.27 | 15023.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 15330.00 | 15371.79 | 15254.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 15330.00 | 15371.79 | 15254.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 15475.00 | 15385.75 | 15281.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:45:00 | 15622.00 | 15453.14 | 15369.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 11:00:00 | 15623.00 | 15487.11 | 15392.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 10:00:00 | 15571.00 | 15587.19 | 15493.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 15382.00 | 15567.51 | 15580.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 15382.00 | 15567.51 | 15580.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 15382.00 | 15567.51 | 15580.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 09:15:00 | 15382.00 | 15567.51 | 15580.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 15332.00 | 15445.42 | 15508.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 14688.00 | 14575.35 | 14754.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 10:00:00 | 14688.00 | 14575.35 | 14754.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 14763.00 | 14612.88 | 14755.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 14760.00 | 14612.88 | 14755.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 14827.00 | 14655.70 | 14761.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 14834.00 | 14655.70 | 14761.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 14786.00 | 14704.21 | 14766.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:15:00 | 14750.00 | 14750.90 | 14775.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:45:00 | 14740.00 | 14741.30 | 14766.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 14:15:00 | 14012.50 | 14204.91 | 14369.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 14:15:00 | 14003.00 | 14204.91 | 14369.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 12:15:00 | 13763.00 | 13750.56 | 13944.23 | SL hit (close>ema200) qty=0.50 sl=13750.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 12:15:00 | 13763.00 | 13750.56 | 13944.23 | SL hit (close>ema200) qty=0.50 sl=13750.56 alert=retest2 |

### Cycle 27 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 13350.00 | 13167.08 | 13163.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 13391.00 | 13211.86 | 13184.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 13589.00 | 13608.30 | 13457.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 13589.00 | 13608.30 | 13457.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 13389.00 | 13577.63 | 13521.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 13389.00 | 13577.63 | 13521.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 13319.00 | 13525.90 | 13503.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 13274.00 | 13525.90 | 13503.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 13322.00 | 13485.12 | 13487.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 13271.00 | 13414.60 | 13452.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 13411.00 | 13323.30 | 13389.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 10:15:00 | 13411.00 | 13323.30 | 13389.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 13411.00 | 13323.30 | 13389.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:00:00 | 13411.00 | 13323.30 | 13389.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 13416.00 | 13341.84 | 13392.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 13416.00 | 13341.84 | 13392.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 13384.00 | 13350.27 | 13391.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:00:00 | 13277.00 | 13335.62 | 13380.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:45:00 | 13295.00 | 13304.43 | 13354.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 13267.00 | 13251.69 | 13289.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:15:00 | 12613.15 | 12977.79 | 13121.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:15:00 | 12630.25 | 12977.79 | 13121.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:15:00 | 12603.65 | 12977.79 | 13121.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 13098.00 | 12905.27 | 12995.26 | SL hit (close>ema200) qty=0.50 sl=12905.27 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 13098.00 | 12905.27 | 12995.26 | SL hit (close>ema200) qty=0.50 sl=12905.27 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 13098.00 | 12905.27 | 12995.26 | SL hit (close>ema200) qty=0.50 sl=12905.27 alert=retest2 |

### Cycle 29 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 12231.00 | 12044.34 | 12030.73 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 11993.00 | 12079.86 | 12083.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 12:15:00 | 11866.00 | 11992.97 | 12038.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 10:15:00 | 11885.00 | 11866.93 | 11947.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 10:15:00 | 11885.00 | 11866.93 | 11947.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 11885.00 | 11866.93 | 11947.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 11885.00 | 11866.93 | 11947.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 11800.00 | 11792.07 | 11865.51 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 14:15:00 | 11994.00 | 11891.79 | 11890.41 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 11783.00 | 11876.06 | 11888.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 09:15:00 | 11566.00 | 11784.11 | 11832.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 11427.00 | 11419.60 | 11586.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:45:00 | 11513.00 | 11419.60 | 11586.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 11015.00 | 10912.62 | 11035.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 15:00:00 | 11015.00 | 10912.62 | 11035.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 10975.00 | 10942.27 | 11029.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 11:00:00 | 10883.00 | 10930.42 | 11015.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 11:30:00 | 10877.00 | 10904.54 | 10996.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 10338.85 | 10655.70 | 10822.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 10333.15 | 10655.70 | 10822.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 10680.00 | 10636.93 | 10783.49 | SL hit (close>ema200) qty=0.50 sl=10636.93 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 10680.00 | 10636.93 | 10783.49 | SL hit (close>ema200) qty=0.50 sl=10636.93 alert=retest2 |

### Cycle 33 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 10777.00 | 10344.48 | 10302.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 10846.00 | 10533.61 | 10431.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 10417.00 | 10516.75 | 10441.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 10417.00 | 10516.75 | 10441.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 10417.00 | 10516.75 | 10441.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 10447.00 | 10516.75 | 10441.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 10215.00 | 10456.40 | 10421.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 10215.00 | 10456.40 | 10421.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 10207.00 | 10406.52 | 10401.70 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 10158.00 | 10356.82 | 10379.54 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 10905.00 | 10432.68 | 10391.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 11446.00 | 10983.45 | 10736.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 11196.00 | 11413.99 | 11132.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 11218.00 | 11413.99 | 11132.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 11247.00 | 11349.66 | 11231.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 11385.00 | 11352.93 | 11243.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:15:00 | 11429.00 | 11347.25 | 11267.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 11378.00 | 11601.76 | 11626.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 11378.00 | 11601.76 | 11626.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 11378.00 | 11601.76 | 11626.32 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 11780.00 | 11608.80 | 11589.29 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 11314.00 | 11603.62 | 11628.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 11289.00 | 11464.34 | 11533.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 11199.00 | 11192.72 | 11300.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 13:30:00 | 11175.00 | 11192.72 | 11300.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 10232.00 | 10227.73 | 10376.94 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 15:15:00 | 10520.00 | 10440.31 | 10434.73 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 10355.00 | 10423.25 | 10427.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 9926.00 | 10175.67 | 10283.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 13:15:00 | 10147.00 | 10041.85 | 10171.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 14:00:00 | 10147.00 | 10041.85 | 10171.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 10151.00 | 10073.14 | 10164.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 10190.00 | 10073.14 | 10164.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 10080.00 | 10074.51 | 10156.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 10051.00 | 10074.51 | 10156.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:45:00 | 10056.00 | 10077.85 | 10138.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 10037.00 | 10064.48 | 10126.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 10045.00 | 10126.09 | 10139.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 10196.00 | 9931.21 | 9990.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 10375.00 | 10019.97 | 10025.17 | SL hit (close>static) qty=1.00 sl=10284.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 10375.00 | 10019.97 | 10025.17 | SL hit (close>static) qty=1.00 sl=10284.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 10375.00 | 10019.97 | 10025.17 | SL hit (close>static) qty=1.00 sl=10284.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 10375.00 | 10019.97 | 10025.17 | SL hit (close>static) qty=1.00 sl=10284.00 alert=retest2 |

### Cycle 41 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 10314.00 | 10078.77 | 10051.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 10890.00 | 10327.48 | 10179.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 10585.00 | 10624.26 | 10428.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 14:00:00 | 10585.00 | 10624.26 | 10428.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 10320.00 | 10552.20 | 10443.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:45:00 | 10589.00 | 10571.76 | 10462.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 10:45:00 | 10566.00 | 10629.96 | 10562.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:15:00 | 10536.00 | 10629.96 | 10562.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 14:15:00 | 10357.00 | 10504.27 | 10518.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 14:15:00 | 10357.00 | 10504.27 | 10518.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 14:15:00 | 10357.00 | 10504.27 | 10518.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 14:15:00 | 10357.00 | 10504.27 | 10518.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 10218.00 | 10418.97 | 10475.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 10262.00 | 10253.70 | 10359.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:30:00 | 10302.00 | 10253.70 | 10359.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 10396.00 | 10278.69 | 10351.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 10434.00 | 10278.69 | 10351.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 10310.00 | 10284.95 | 10348.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 10250.00 | 10284.95 | 10348.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:00:00 | 10297.00 | 10282.58 | 10326.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 10472.00 | 10321.65 | 10336.36 | SL hit (close>static) qty=1.00 sl=10411.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 10472.00 | 10321.65 | 10336.36 | SL hit (close>static) qty=1.00 sl=10411.00 alert=retest2 |

### Cycle 43 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 10670.00 | 10391.32 | 10366.69 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 10323.00 | 10427.81 | 10433.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 10264.00 | 10395.05 | 10418.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 10415.00 | 10338.70 | 10381.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 10415.00 | 10338.70 | 10381.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 10415.00 | 10338.70 | 10381.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 10100.00 | 10349.46 | 10370.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 10475.00 | 10184.65 | 10163.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 10475.00 | 10184.65 | 10163.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 10553.00 | 10258.32 | 10199.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 10095.00 | 10309.45 | 10264.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 10095.00 | 10309.45 | 10264.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 10095.00 | 10309.45 | 10264.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 10095.00 | 10309.45 | 10264.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 10115.00 | 10270.56 | 10251.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 10094.00 | 10270.56 | 10251.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 10157.00 | 10232.24 | 10236.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 10134.00 | 10212.59 | 10226.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 10095.00 | 9898.41 | 10007.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 10095.00 | 9898.41 | 10007.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 10095.00 | 9898.41 | 10007.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 10095.00 | 9898.41 | 10007.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 10025.50 | 9923.82 | 10009.19 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 10373.00 | 10065.85 | 10062.16 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 12:15:00 | 9785.00 | 10079.98 | 10090.81 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 13:15:00 | 10100.50 | 10024.82 | 10021.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 10542.00 | 10154.98 | 10083.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 10485.00 | 10491.53 | 10326.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 10435.50 | 10491.53 | 10326.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 10413.00 | 10626.13 | 10564.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 10508.00 | 10595.70 | 10556.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 10515.00 | 10541.20 | 10538.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 10482.50 | 10529.46 | 10533.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 10482.50 | 10529.46 | 10533.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 10482.50 | 10529.46 | 10533.27 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 15:15:00 | 10575.00 | 10538.57 | 10537.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 10860.00 | 10602.85 | 10566.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 11255.00 | 11307.50 | 11173.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 11255.00 | 11307.50 | 11173.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 11255.00 | 11307.50 | 11173.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 11219.00 | 11307.50 | 11173.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 11234.00 | 11295.82 | 11220.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 11234.00 | 11295.82 | 11220.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 11194.00 | 11275.46 | 11218.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 11318.00 | 11275.46 | 11218.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 11242.00 | 11286.65 | 11264.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 11096.00 | 11246.84 | 11258.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 11096.00 | 11246.84 | 11258.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 11096.00 | 11246.84 | 11258.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 10995.50 | 11196.57 | 11234.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 10990.00 | 10986.60 | 11093.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 10:00:00 | 10990.00 | 10986.60 | 11093.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 10866.00 | 10962.48 | 11072.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 11:30:00 | 10845.50 | 10945.88 | 11055.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 12:15:00 | 10821.00 | 10945.88 | 11055.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:45:00 | 10856.00 | 10880.24 | 10994.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 11:15:00 | 11160.00 | 10966.05 | 10998.68 | SL hit (close>static) qty=1.00 sl=11086.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 11:15:00 | 11160.00 | 10966.05 | 10998.68 | SL hit (close>static) qty=1.00 sl=11086.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 11:15:00 | 11160.00 | 10966.05 | 10998.68 | SL hit (close>static) qty=1.00 sl=11086.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 11139.50 | 11037.69 | 11027.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 11306.00 | 11117.88 | 11067.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 11340.00 | 11348.22 | 11230.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 15:00:00 | 11340.00 | 11348.22 | 11230.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 11318.50 | 11383.43 | 11314.09 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 11153.00 | 11278.20 | 11279.21 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 14:15:00 | 11363.00 | 11265.26 | 11254.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 11460.00 | 11304.21 | 11272.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 11310.00 | 11327.73 | 11290.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:45:00 | 11304.00 | 11327.73 | 11290.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 11274.00 | 11316.99 | 11288.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 11258.00 | 11316.99 | 11288.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 11313.00 | 11316.19 | 11291.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 11335.00 | 11293.66 | 11285.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 11:00:00 | 11336.00 | 11309.22 | 11294.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 11267.00 | 11300.78 | 11292.23 | SL hit (close<static) qty=1.00 sl=11268.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 11267.00 | 11300.78 | 11292.23 | SL hit (close<static) qty=1.00 sl=11268.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 11220.00 | 11284.62 | 11285.66 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 11297.00 | 11285.96 | 11285.75 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 11:15:00 | 11204.00 | 11277.26 | 11282.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 12:15:00 | 11130.00 | 11247.81 | 11268.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 09:15:00 | 11195.00 | 11172.82 | 11220.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 11195.00 | 11172.82 | 11220.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 11195.00 | 11172.82 | 11220.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 11216.00 | 11172.82 | 11220.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 11170.00 | 11172.26 | 11215.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 11:15:00 | 11139.00 | 11172.26 | 11215.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 14:15:00 | 10582.05 | 11004.54 | 11118.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 09:15:00 | 16186.00 | 2025-05-15 12:15:00 | 16075.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-05-15 10:30:00 | 16172.00 | 2025-05-15 12:15:00 | 16075.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-05-15 13:45:00 | 16162.00 | 2025-05-21 09:15:00 | 15690.00 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-05-27 10:15:00 | 15009.00 | 2025-06-04 14:15:00 | 14967.00 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2025-05-28 09:30:00 | 14998.00 | 2025-06-04 14:15:00 | 14967.00 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-06-17 11:30:00 | 14332.00 | 2025-06-23 09:15:00 | 13615.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:30:00 | 14332.00 | 2025-06-23 09:15:00 | 14378.00 | STOP_HIT | 0.50 | -0.32% |
| SELL | retest2 | 2025-06-18 14:15:00 | 14333.00 | 2025-06-23 09:15:00 | 13616.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 14:15:00 | 14333.00 | 2025-06-23 09:15:00 | 14378.00 | STOP_HIT | 0.50 | -0.31% |
| SELL | retest2 | 2025-06-19 10:15:00 | 14317.00 | 2025-06-23 09:15:00 | 13601.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-19 10:15:00 | 14317.00 | 2025-06-23 09:15:00 | 14378.00 | STOP_HIT | 0.50 | -0.43% |
| BUY | retest2 | 2025-07-01 14:15:00 | 14604.00 | 2025-07-16 09:15:00 | 16064.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 09:15:00 | 16956.00 | 2025-08-06 09:15:00 | 16620.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-08-04 11:45:00 | 16890.00 | 2025-08-06 09:15:00 | 16620.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-08-05 11:00:00 | 16888.00 | 2025-08-06 09:15:00 | 16620.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-08-05 14:30:00 | 16882.00 | 2025-08-06 09:15:00 | 16620.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-08-22 11:15:00 | 16919.00 | 2025-08-26 12:15:00 | 16780.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-08-22 14:00:00 | 16874.00 | 2025-08-26 12:15:00 | 16780.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-08-26 11:00:00 | 16839.00 | 2025-08-26 12:15:00 | 16780.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-09-09 12:30:00 | 18014.00 | 2025-09-15 14:15:00 | 17970.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-09-09 13:15:00 | 18038.00 | 2025-09-15 14:15:00 | 17970.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-09-10 10:00:00 | 18016.00 | 2025-09-15 14:15:00 | 17970.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-09-11 09:45:00 | 18010.00 | 2025-09-15 14:15:00 | 17970.00 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-09-12 11:45:00 | 18052.00 | 2025-09-22 14:15:00 | 18075.00 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-09-12 13:00:00 | 18040.00 | 2025-09-22 14:15:00 | 18075.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-09-15 10:45:00 | 18046.00 | 2025-09-22 14:15:00 | 18075.00 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2025-09-15 11:15:00 | 18056.00 | 2025-09-22 14:15:00 | 18075.00 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-09-16 09:15:00 | 18030.00 | 2025-09-22 14:15:00 | 18075.00 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-09-24 09:15:00 | 18068.00 | 2025-09-24 13:15:00 | 18178.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2025-10-01 10:45:00 | 16355.00 | 2025-10-03 15:15:00 | 16609.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest1 | 2025-10-01 12:15:00 | 16370.00 | 2025-10-03 15:15:00 | 16609.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-08 12:45:00 | 17067.00 | 2025-10-08 14:15:00 | 16848.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-10-09 10:00:00 | 17059.00 | 2025-10-14 09:15:00 | 16655.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-10-09 11:30:00 | 17054.00 | 2025-10-14 09:15:00 | 16655.00 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-10-09 12:00:00 | 17055.00 | 2025-10-14 09:15:00 | 16655.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-10-16 11:45:00 | 16685.00 | 2025-10-17 11:15:00 | 16835.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest1 | 2025-11-11 10:30:00 | 14790.00 | 2025-11-11 12:15:00 | 14991.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-11-17 09:45:00 | 15622.00 | 2025-11-20 09:15:00 | 15382.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-17 11:00:00 | 15623.00 | 2025-11-20 09:15:00 | 15382.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-18 10:00:00 | 15571.00 | 2025-11-20 09:15:00 | 15382.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-11-27 10:15:00 | 14750.00 | 2025-12-03 14:15:00 | 14012.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 11:45:00 | 14740.00 | 2025-12-03 14:15:00 | 14003.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 10:15:00 | 14750.00 | 2025-12-05 12:15:00 | 13763.00 | STOP_HIT | 0.50 | 6.69% |
| SELL | retest2 | 2025-11-27 11:45:00 | 14740.00 | 2025-12-05 12:15:00 | 13763.00 | STOP_HIT | 0.50 | 6.63% |
| SELL | retest2 | 2025-12-18 14:00:00 | 13277.00 | 2025-12-23 09:15:00 | 12613.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-19 09:45:00 | 13295.00 | 2025-12-23 09:15:00 | 12630.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 10:15:00 | 13267.00 | 2025-12-23 09:15:00 | 12603.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-18 14:00:00 | 13277.00 | 2025-12-24 09:15:00 | 13098.00 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2025-12-19 09:45:00 | 13295.00 | 2025-12-24 09:15:00 | 13098.00 | STOP_HIT | 0.50 | 1.48% |
| SELL | retest2 | 2025-12-22 10:15:00 | 13267.00 | 2025-12-24 09:15:00 | 13098.00 | STOP_HIT | 0.50 | 1.27% |
| SELL | retest2 | 2026-01-20 11:00:00 | 10883.00 | 2026-01-21 10:15:00 | 10338.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 11:30:00 | 10877.00 | 2026-01-21 10:15:00 | 10333.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 11:00:00 | 10883.00 | 2026-01-21 12:15:00 | 10680.00 | STOP_HIT | 0.50 | 1.87% |
| SELL | retest2 | 2026-01-20 11:30:00 | 10877.00 | 2026-01-21 12:15:00 | 10680.00 | STOP_HIT | 0.50 | 1.81% |
| BUY | retest2 | 2026-02-06 11:15:00 | 11385.00 | 2026-02-13 09:15:00 | 11378.00 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2026-02-06 14:15:00 | 11429.00 | 2026-02-13 09:15:00 | 11378.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-03-05 10:15:00 | 10051.00 | 2026-03-10 10:15:00 | 10375.00 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2026-03-05 12:45:00 | 10056.00 | 2026-03-10 10:15:00 | 10375.00 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2026-03-05 13:45:00 | 10037.00 | 2026-03-10 10:15:00 | 10375.00 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2026-03-06 14:45:00 | 10045.00 | 2026-03-10 10:15:00 | 10375.00 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2026-03-12 10:45:00 | 10589.00 | 2026-03-13 14:15:00 | 10357.00 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-03-13 10:45:00 | 10566.00 | 2026-03-13 14:15:00 | 10357.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-03-13 11:15:00 | 10536.00 | 2026-03-13 14:15:00 | 10357.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-03-17 11:15:00 | 10250.00 | 2026-03-18 09:15:00 | 10472.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-03-17 15:00:00 | 10297.00 | 2026-03-18 09:15:00 | 10472.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-03-23 09:15:00 | 10100.00 | 2026-03-25 09:15:00 | 10475.00 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2026-04-13 10:45:00 | 10508.00 | 2026-04-13 14:15:00 | 10482.50 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2026-04-13 13:45:00 | 10515.00 | 2026-04-13 14:15:00 | 10482.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-04-21 09:15:00 | 11318.00 | 2026-04-23 09:15:00 | 11096.00 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-04-22 10:15:00 | 11242.00 | 2026-04-23 09:15:00 | 11096.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-04-24 11:30:00 | 10845.50 | 2026-04-27 11:15:00 | 11160.00 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2026-04-24 12:15:00 | 10821.00 | 2026-04-27 11:15:00 | 11160.00 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2026-04-24 14:45:00 | 10856.00 | 2026-04-27 11:15:00 | 11160.00 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2026-05-06 09:15:00 | 11335.00 | 2026-05-06 11:15:00 | 11267.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-05-06 11:00:00 | 11336.00 | 2026-05-06 11:15:00 | 11267.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-05-08 11:15:00 | 11139.00 | 2026-05-08 14:15:00 | 10582.05 | PARTIAL | 0.50 | 5.00% |
