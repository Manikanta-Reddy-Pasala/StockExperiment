# Bosch Ltd. (BOSCHLTD)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 38050.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 8
- **Target hits / Stop hits / Partials:** 1 / 9 / 2
- **Avg / median % per leg:** 0.37% / -1.30%
- **Sum % (uncompounded):** 4.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.08% | -12.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.08% | -12.5% |
| SELL (all) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.81% | 16.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.81% | 16.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 4 | 33.3% | 1 | 9 | 2 | 0.37% | 4.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 30835.00 | 28660.62 | 28655.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 31195.00 | 28794.86 | 28723.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 39620.00 | 39808.50 | 38107.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 39620.00 | 39808.50 | 38107.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 38425.00 | 39657.57 | 38475.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 38425.00 | 39657.57 | 38475.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 38375.00 | 39644.81 | 38474.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 38560.00 | 39634.22 | 38475.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 38690.00 | 39624.82 | 38476.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:30:00 | 38535.00 | 39624.82 | 38476.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 38495.00 | 39613.58 | 38476.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 12:00:00 | 38495.00 | 39613.58 | 38476.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 38450.00 | 39602.00 | 38476.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:15:00 | 38400.00 | 39602.00 | 38476.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 38250.00 | 39588.55 | 38475.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 38250.00 | 39588.55 | 38475.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 38160.00 | 39574.33 | 38473.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 38160.00 | 39574.33 | 38473.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 38405.00 | 39475.55 | 38461.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 38405.00 | 39475.55 | 38461.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 38250.00 | 39463.35 | 38460.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 38200.00 | 39463.35 | 38460.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 38045.00 | 39449.24 | 38458.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:15:00 | 38010.00 | 39449.24 | 38458.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 38175.00 | 39436.56 | 38456.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:45:00 | 37980.00 | 39436.56 | 38456.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 38270.00 | 39303.24 | 38440.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 38270.00 | 39303.24 | 38440.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 38280.00 | 39284.60 | 38439.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:15:00 | 38400.00 | 39244.82 | 38436.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:30:00 | 38440.00 | 39111.69 | 38461.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 38400.00 | 39095.18 | 38459.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:00:00 | 38460.00 | 39081.22 | 38458.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 38495.00 | 39075.39 | 38458.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 38565.00 | 39075.39 | 38458.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 38790.00 | 39072.55 | 38460.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 37900.00 | 39007.87 | 38468.66 | SL hit (close<static) qty=1.00 sl=38010.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 37900.00 | 39007.87 | 38468.66 | SL hit (close<static) qty=1.00 sl=38010.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 37900.00 | 39007.87 | 38468.66 | SL hit (close<static) qty=1.00 sl=38010.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 37900.00 | 39007.87 | 38468.66 | SL hit (close<static) qty=1.00 sl=38010.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 38995.00 | 38856.56 | 38456.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 14:15:00 | 38945.00 | 38872.76 | 38517.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 37600.00 | 38851.96 | 38524.14 | SL hit (close<static) qty=1.00 sl=38315.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 37600.00 | 38851.96 | 38524.14 | SL hit (close<static) qty=1.00 sl=38315.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 36670.00 | 38258.93 | 38259.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 09:15:00 | 36235.00 | 38000.11 | 38123.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 37035.00 | 37020.23 | 37462.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:45:00 | 37130.00 | 37020.23 | 37462.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 37215.00 | 36345.10 | 36816.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 37215.00 | 36345.10 | 36816.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 38515.00 | 36366.69 | 36824.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 38515.00 | 36366.69 | 36824.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 37230.00 | 37180.42 | 37192.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:45:00 | 37275.00 | 37180.42 | 37192.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 37280.00 | 37181.41 | 37192.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 37280.00 | 37181.41 | 37192.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 37610.00 | 37185.67 | 37194.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 37745.00 | 37185.67 | 37194.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 12:15:00 | 37685.00 | 37207.96 | 37205.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 37895.00 | 37218.66 | 37211.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 37210.00 | 37263.14 | 37234.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 37210.00 | 37263.14 | 37234.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 37210.00 | 37263.14 | 37234.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 37200.00 | 37263.14 | 37234.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 37080.00 | 37261.32 | 37233.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:45:00 | 37180.00 | 37261.32 | 37233.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 36335.00 | 37200.60 | 37204.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 35935.00 | 37188.01 | 37197.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 14:15:00 | 36570.00 | 36536.67 | 36821.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 15:00:00 | 36570.00 | 36536.67 | 36821.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 37260.00 | 36507.66 | 36784.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 12:30:00 | 36695.00 | 36613.66 | 36825.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 34860.25 | 36159.62 | 36495.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 36035.00 | 35992.15 | 36373.69 | SL hit (close>ema200) qty=0.50 sl=35992.15 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:45:00 | 36910.00 | 36012.13 | 36379.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 13:15:00 | 35064.50 | 36051.00 | 36361.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-05 11:15:00 | 33219.00 | 35803.94 | 36217.16 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 11:15:00 | 37100.00 | 32503.70 | 33666.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:15:00 | 37225.00 | 32600.85 | 33703.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 11:15:00 | 38075.00 | 34164.19 | 34358.33 | SL hit (close>static) qty=1.00 sl=37995.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 11:15:00 | 38075.00 | 34164.19 | 34358.33 | SL hit (close>static) qty=1.00 sl=37995.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 15:15:00 | 38200.00 | 34571.80 | 34557.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 38290.00 | 34644.49 | 34593.92 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-03 14:15:00 | 38400.00 | 2025-10-14 09:15:00 | 37900.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-10-09 10:30:00 | 38440.00 | 2025-10-14 09:15:00 | 37900.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-10-09 13:15:00 | 38400.00 | 2025-10-14 09:15:00 | 37900.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-10-09 15:00:00 | 38460.00 | 2025-10-14 09:15:00 | 37900.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-20 10:15:00 | 38995.00 | 2025-10-29 09:15:00 | 37600.00 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-10-27 14:15:00 | 38945.00 | 2025-10-29 09:15:00 | 37600.00 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2026-02-04 12:30:00 | 36695.00 | 2026-02-20 09:15:00 | 34860.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 12:30:00 | 36695.00 | 2026-02-25 09:15:00 | 36035.00 | STOP_HIT | 0.50 | 1.80% |
| SELL | retest2 | 2026-02-25 11:45:00 | 36910.00 | 2026-03-02 13:15:00 | 35064.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:45:00 | 36910.00 | 2026-03-05 11:15:00 | 33219.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-09 11:15:00 | 37100.00 | 2026-04-20 11:15:00 | 38075.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2026-04-09 13:15:00 | 37225.00 | 2026-04-20 11:15:00 | 38075.00 | STOP_HIT | 1.00 | -2.28% |
