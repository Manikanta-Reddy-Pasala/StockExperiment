# Abbott India Ltd. (ABBOTINDIA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 26850.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 87 |
| ALERT1 | 57 |
| ALERT2 | 55 |
| ALERT2_SKIP | 33 |
| ALERT3 | 157 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 71 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 75 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 58
- **Target hits / Stop hits / Partials:** 0 / 73 / 3
- **Avg / median % per leg:** -0.21% / -0.56%
- **Sum % (uncompounded):** -15.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 2 | 6.5% | 0 | 31 | 0 | -0.79% | -24.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 2 | 6.5% | 0 | 31 | 0 | -0.79% | -24.4% |
| SELL (all) | 45 | 16 | 35.6% | 0 | 42 | 3 | 0.19% | 8.7% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.31% | 0.6% |
| SELL @ 3rd Alert (retest2) | 43 | 14 | 32.6% | 0 | 40 | 3 | 0.19% | 8.1% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.31% | 0.6% |
| retest2 (combined) | 74 | 16 | 21.6% | 0 | 71 | 3 | -0.22% | -16.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 30405.00 | 30153.87 | 30126.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 31000.00 | 30482.18 | 30389.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 13:15:00 | 30500.00 | 30544.48 | 30454.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 13:45:00 | 30520.00 | 30544.48 | 30454.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 30835.00 | 31125.06 | 31028.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:00:00 | 30835.00 | 31125.06 | 31028.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 31050.00 | 31110.05 | 31030.46 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 12:15:00 | 30815.00 | 30987.09 | 30995.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 13:15:00 | 30790.00 | 30947.67 | 30977.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 10:15:00 | 30930.00 | 30873.27 | 30926.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 10:15:00 | 30930.00 | 30873.27 | 30926.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 30930.00 | 30873.27 | 30926.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 30930.00 | 30873.27 | 30926.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 30790.00 | 30856.62 | 30913.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 14:45:00 | 30495.00 | 30696.19 | 30824.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 15:15:00 | 30620.00 | 30380.77 | 30372.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 15:15:00 | 30620.00 | 30380.77 | 30372.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 30635.00 | 30431.61 | 30396.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 15:15:00 | 31300.00 | 31674.10 | 31525.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 15:15:00 | 31300.00 | 31674.10 | 31525.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 31300.00 | 31674.10 | 31525.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:30:00 | 31920.00 | 31734.66 | 31590.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 13:00:00 | 31945.00 | 31776.73 | 31622.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:15:00 | 31920.00 | 31772.48 | 31660.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 15:15:00 | 31560.00 | 31645.41 | 31655.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 31560.00 | 31645.41 | 31655.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 14:15:00 | 31525.00 | 31613.04 | 31636.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 31600.00 | 31460.84 | 31514.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 31600.00 | 31460.84 | 31514.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 31600.00 | 31460.84 | 31514.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 31615.00 | 31460.84 | 31514.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 31770.00 | 31522.67 | 31537.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 31770.00 | 31522.67 | 31537.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 31675.00 | 31553.14 | 31550.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 14:15:00 | 31845.00 | 31644.25 | 31608.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 31655.00 | 31675.32 | 31630.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 31655.00 | 31675.32 | 31630.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 31655.00 | 31675.32 | 31630.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:15:00 | 31640.00 | 31675.32 | 31630.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 31655.00 | 31671.25 | 31632.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 12:30:00 | 31800.00 | 31661.20 | 31634.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 15:15:00 | 31555.00 | 31624.42 | 31623.19 | SL hit (close<static) qty=1.00 sl=31565.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 31465.00 | 31632.57 | 31633.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 13:15:00 | 31435.00 | 31593.05 | 31615.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 31520.00 | 31369.92 | 31469.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 11:15:00 | 31520.00 | 31369.92 | 31469.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 31520.00 | 31369.92 | 31469.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 31520.00 | 31369.92 | 31469.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 31425.00 | 31380.94 | 31465.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:15:00 | 31535.00 | 31380.94 | 31465.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 31540.00 | 31412.75 | 31471.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 31540.00 | 31412.75 | 31471.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 31585.00 | 31447.20 | 31482.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:45:00 | 31625.00 | 31447.20 | 31482.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 31410.00 | 31180.88 | 31282.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 31410.00 | 31180.88 | 31282.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 31445.00 | 31233.70 | 31297.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:45:00 | 31455.00 | 31233.70 | 31297.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 31625.00 | 31351.37 | 31342.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 32525.00 | 31807.09 | 31619.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 34225.00 | 34405.42 | 33701.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 34225.00 | 34405.42 | 33701.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 34670.00 | 34812.09 | 34458.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 34860.00 | 34812.09 | 34458.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:00:00 | 34855.00 | 34772.34 | 34498.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 34750.00 | 35036.96 | 35044.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 14:15:00 | 34750.00 | 35036.96 | 35044.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 34690.00 | 34891.16 | 34968.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 34485.00 | 34302.59 | 34475.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 10:15:00 | 34485.00 | 34302.59 | 34475.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 34485.00 | 34302.59 | 34475.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 34475.00 | 34302.59 | 34475.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 34275.00 | 34297.07 | 34457.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:00:00 | 34090.00 | 34255.66 | 34423.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 34690.00 | 34443.50 | 34416.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 34690.00 | 34443.50 | 34416.18 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 34370.00 | 34409.11 | 34414.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 14:15:00 | 34270.00 | 34366.23 | 34392.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 15:15:00 | 34290.00 | 34276.41 | 34321.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 15:15:00 | 34290.00 | 34276.41 | 34321.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 34290.00 | 34276.41 | 34321.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 34435.00 | 34276.41 | 34321.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 34380.00 | 34297.13 | 34326.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:30:00 | 34180.00 | 34264.70 | 34309.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 34210.00 | 34186.96 | 34241.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:45:00 | 34160.00 | 34216.66 | 34246.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 12:15:00 | 34175.00 | 34141.35 | 34139.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 34175.00 | 34141.35 | 34139.49 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 34100.00 | 34143.66 | 34144.98 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 34260.00 | 34166.92 | 34155.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 34750.00 | 34283.54 | 34209.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 34700.00 | 34741.95 | 34538.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:45:00 | 34560.00 | 34741.95 | 34538.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 34640.00 | 34721.56 | 34548.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 34230.00 | 34721.56 | 34548.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 34135.00 | 34604.25 | 34510.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 34205.00 | 34604.25 | 34510.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 34030.00 | 34418.32 | 34437.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 33945.00 | 34266.32 | 34361.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 33325.00 | 33295.52 | 33622.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:15:00 | 33700.00 | 33295.52 | 33622.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 34240.00 | 33484.41 | 33678.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 34240.00 | 33484.41 | 33678.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 34380.00 | 33663.53 | 33742.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 34385.00 | 33663.53 | 33742.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 34720.00 | 33874.82 | 33831.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 34995.00 | 34388.74 | 34128.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 34675.00 | 34746.42 | 34406.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 09:45:00 | 34665.00 | 34746.42 | 34406.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 34370.00 | 34668.70 | 34503.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 34370.00 | 34668.70 | 34503.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 34485.00 | 34631.96 | 34501.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 34560.00 | 34580.57 | 34489.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 33980.00 | 34460.45 | 34443.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 33955.00 | 34460.45 | 34443.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 33925.00 | 34353.36 | 34396.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 33745.00 | 34158.35 | 34295.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 34095.00 | 33912.09 | 34097.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 11:15:00 | 34095.00 | 33912.09 | 34097.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 34095.00 | 33912.09 | 34097.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 34095.00 | 33912.09 | 34097.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 34105.00 | 33950.67 | 34097.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 13:15:00 | 34055.00 | 33950.67 | 34097.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 32352.25 | 32757.99 | 33161.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 32725.00 | 32677.06 | 32962.15 | SL hit (close>ema200) qty=0.50 sl=32677.06 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 33020.00 | 32908.48 | 32898.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 33320.00 | 32990.79 | 32937.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 33005.00 | 33291.97 | 33136.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 14:15:00 | 33005.00 | 33291.97 | 33136.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 33005.00 | 33291.97 | 33136.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 33005.00 | 33291.97 | 33136.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 32250.00 | 33083.58 | 33055.79 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 32680.00 | 33002.86 | 33021.63 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 33650.00 | 33111.03 | 33066.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 12:15:00 | 33720.00 | 33232.83 | 33125.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 32900.00 | 33201.81 | 33132.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 32900.00 | 33201.81 | 33132.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 32900.00 | 33201.81 | 33132.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 32900.00 | 33201.81 | 33132.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 32850.00 | 33131.45 | 33106.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 33195.00 | 33131.45 | 33106.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 33010.00 | 33100.59 | 33101.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 33010.00 | 33100.59 | 33101.49 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 14:15:00 | 33300.00 | 33140.47 | 33119.53 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 32680.00 | 33040.70 | 33082.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 12:15:00 | 32655.00 | 32963.56 | 33043.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 33095.00 | 32780.91 | 32855.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 14:15:00 | 33095.00 | 32780.91 | 32855.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 33095.00 | 32780.91 | 32855.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 33095.00 | 32780.91 | 32855.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 32900.00 | 32804.73 | 32859.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 32920.00 | 32804.73 | 32859.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 32730.00 | 32709.88 | 32776.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:15:00 | 32570.00 | 32664.37 | 32717.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 32275.00 | 31852.85 | 31803.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 32275.00 | 31852.85 | 31803.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 32395.00 | 31961.28 | 31857.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 13:15:00 | 32410.00 | 32421.20 | 32238.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:30:00 | 32420.00 | 32421.20 | 32238.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 32490.00 | 32447.57 | 32283.12 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 32155.00 | 32271.55 | 32275.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 31500.00 | 32117.24 | 32204.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 31510.00 | 31462.02 | 31766.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:30:00 | 31455.00 | 31462.02 | 31766.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 31145.00 | 30915.43 | 31140.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 31145.00 | 30915.43 | 31140.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 31295.00 | 30991.35 | 31154.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:45:00 | 31300.00 | 30991.35 | 31154.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 31320.00 | 31057.08 | 31169.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 31340.00 | 31057.08 | 31169.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 31240.00 | 31081.33 | 31160.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:30:00 | 31265.00 | 31081.33 | 31160.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 30890.00 | 31043.06 | 31135.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 31280.00 | 31043.06 | 31135.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 31330.00 | 31100.45 | 31153.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 31425.00 | 31100.45 | 31153.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 31145.00 | 31114.09 | 31150.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 14:15:00 | 31000.00 | 31107.02 | 31140.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:30:00 | 31025.00 | 31057.61 | 31095.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 31290.00 | 31090.49 | 31092.15 | SL hit (close>static) qty=1.00 sl=31245.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 31340.00 | 31140.39 | 31114.68 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 14:15:00 | 30930.00 | 31094.84 | 31102.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 15:15:00 | 30885.00 | 31052.87 | 31082.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 10:15:00 | 31065.00 | 31046.84 | 31073.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 31065.00 | 31046.84 | 31073.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 31065.00 | 31046.84 | 31073.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 31060.00 | 31046.84 | 31073.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 31120.00 | 31061.47 | 31078.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 31120.00 | 31061.47 | 31078.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 31140.00 | 31077.18 | 31083.72 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 31145.00 | 31090.74 | 31089.29 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 30960.00 | 31064.59 | 31077.54 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 31145.00 | 31079.67 | 31079.63 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 30910.00 | 31054.47 | 31069.86 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 31220.00 | 31078.98 | 31073.49 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 30945.00 | 31052.19 | 31061.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 10:15:00 | 30855.00 | 30966.69 | 31011.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 11:15:00 | 30640.00 | 30627.83 | 30780.60 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 12:15:00 | 30520.00 | 30627.83 | 30780.60 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 13:30:00 | 30480.00 | 30579.61 | 30731.69 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 30340.00 | 30324.94 | 30426.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:00:00 | 30175.00 | 30294.95 | 30403.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 10:15:00 | 30405.00 | 30246.62 | 30330.87 | SL hit (close>ema400) qty=1.00 sl=30330.87 alert=retest1 |

### Cycle 33 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 30015.00 | 29687.45 | 29687.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 30095.00 | 29852.03 | 29772.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 14:15:00 | 29985.00 | 30059.29 | 29922.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 14:15:00 | 29985.00 | 30059.29 | 29922.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 29985.00 | 30059.29 | 29922.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:30:00 | 30410.00 | 30157.57 | 30010.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:00:00 | 30400.00 | 30157.57 | 30010.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 29825.00 | 30023.55 | 30036.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 29825.00 | 30023.55 | 30036.10 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 11:15:00 | 30100.00 | 30048.84 | 30042.64 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 29875.00 | 30030.73 | 30037.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 29865.00 | 29997.58 | 30021.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 11:15:00 | 29830.00 | 29752.97 | 29845.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 12:00:00 | 29830.00 | 29752.97 | 29845.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 29815.00 | 29765.38 | 29842.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:00:00 | 29815.00 | 29765.38 | 29842.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 29905.00 | 29793.30 | 29848.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 29905.00 | 29793.30 | 29848.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 30095.00 | 29853.64 | 29870.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 30095.00 | 29853.64 | 29870.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 15:15:00 | 30255.00 | 29933.91 | 29905.80 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 12:15:00 | 29915.00 | 29973.06 | 29977.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 09:15:00 | 29755.00 | 29879.29 | 29928.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 10:15:00 | 29900.00 | 29883.44 | 29925.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 10:15:00 | 29900.00 | 29883.44 | 29925.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 29900.00 | 29883.44 | 29925.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 29925.00 | 29883.44 | 29925.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 29810.00 | 29854.80 | 29892.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 11:30:00 | 29650.00 | 29768.85 | 29825.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 30125.00 | 29845.06 | 29850.31 | SL hit (close>static) qty=1.00 sl=29995.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 29920.00 | 29860.05 | 29856.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 14:15:00 | 30300.00 | 30008.03 | 29929.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 12:15:00 | 29975.00 | 30035.22 | 29971.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 12:15:00 | 29975.00 | 30035.22 | 29971.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 29975.00 | 30035.22 | 29971.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 30100.00 | 30032.54 | 29981.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 29870.00 | 29955.86 | 29961.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 29870.00 | 29955.86 | 29961.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 29775.00 | 29906.59 | 29935.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 29945.00 | 29839.70 | 29876.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 29945.00 | 29839.70 | 29876.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 29945.00 | 29839.70 | 29876.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 29945.00 | 29839.70 | 29876.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 29935.00 | 29858.76 | 29882.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 29965.00 | 29858.76 | 29882.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 29920.00 | 29871.01 | 29885.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 29920.00 | 29871.01 | 29885.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 29865.00 | 29869.81 | 29883.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:30:00 | 29915.00 | 29869.81 | 29883.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 29900.00 | 29875.84 | 29885.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 29950.00 | 29875.84 | 29885.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 29790.00 | 29858.68 | 29876.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 29770.00 | 29858.68 | 29876.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 29790.00 | 29844.94 | 29868.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 29905.00 | 29844.94 | 29868.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 29730.00 | 29821.95 | 29856.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 29865.00 | 29821.95 | 29856.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 29705.00 | 29750.09 | 29806.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:30:00 | 29630.00 | 29708.16 | 29771.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:00:00 | 29675.00 | 29701.53 | 29762.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 13:15:00 | 29345.00 | 29302.81 | 29300.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 13:15:00 | 29345.00 | 29302.81 | 29300.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 10:15:00 | 29735.00 | 29407.25 | 29350.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 13:15:00 | 29500.00 | 29501.31 | 29415.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 14:00:00 | 29500.00 | 29501.31 | 29415.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 29560.00 | 29513.05 | 29428.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 29560.00 | 29513.05 | 29428.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 29500.00 | 29510.44 | 29434.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:45:00 | 29340.00 | 29473.35 | 29424.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 29345.00 | 29447.68 | 29417.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 29345.00 | 29447.68 | 29417.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 11:15:00 | 29075.00 | 29373.15 | 29386.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 12:15:00 | 29050.00 | 29308.52 | 29355.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 29255.00 | 29087.28 | 29191.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 11:15:00 | 29255.00 | 29087.28 | 29191.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 29255.00 | 29087.28 | 29191.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 29255.00 | 29087.28 | 29191.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 29400.00 | 29149.82 | 29210.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:00:00 | 29400.00 | 29149.82 | 29210.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 29520.00 | 29223.86 | 29238.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:00:00 | 29520.00 | 29223.86 | 29238.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 29530.00 | 29285.09 | 29264.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 10:15:00 | 29600.00 | 29416.36 | 29335.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 15:15:00 | 29450.00 | 29518.01 | 29426.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 15:15:00 | 29450.00 | 29518.01 | 29426.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 29450.00 | 29518.01 | 29426.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 29500.00 | 29518.01 | 29426.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 29410.00 | 29496.40 | 29425.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 29410.00 | 29496.40 | 29425.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 29545.00 | 29506.12 | 29436.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:15:00 | 29290.00 | 29506.12 | 29436.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 29295.00 | 29463.90 | 29423.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 29605.00 | 29459.80 | 29428.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:30:00 | 29550.00 | 29502.27 | 29454.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:00:00 | 29560.00 | 29502.27 | 29454.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 14:15:00 | 29580.00 | 29501.77 | 29471.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 29530.00 | 29507.42 | 29476.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 29530.00 | 29507.42 | 29476.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 29465.00 | 29513.75 | 29486.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:45:00 | 29510.00 | 29513.75 | 29486.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 29360.00 | 29483.00 | 29474.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 29360.00 | 29483.00 | 29474.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 29365.00 | 29459.40 | 29464.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 29365.00 | 29459.40 | 29464.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 29280.00 | 29423.52 | 29447.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 29410.00 | 29401.05 | 29432.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 29410.00 | 29401.05 | 29432.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 29410.00 | 29401.05 | 29432.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 29410.00 | 29401.05 | 29432.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 29210.00 | 29350.67 | 29403.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 29340.00 | 29350.67 | 29403.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 29530.00 | 29378.43 | 29406.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 29530.00 | 29378.43 | 29406.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 29635.00 | 29429.74 | 29426.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 10:15:00 | 29670.00 | 29550.94 | 29493.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 29600.00 | 29631.23 | 29567.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 29600.00 | 29631.23 | 29567.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 29600.00 | 29631.23 | 29567.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 29575.00 | 29631.23 | 29567.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 29705.00 | 29645.98 | 29579.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:30:00 | 29615.00 | 29645.98 | 29579.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 29630.00 | 29642.79 | 29584.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:30:00 | 29475.00 | 29642.79 | 29584.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 29550.00 | 29624.23 | 29581.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:30:00 | 29590.00 | 29624.23 | 29581.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 29540.00 | 29607.38 | 29577.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 15:15:00 | 29620.00 | 29595.91 | 29575.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 29485.00 | 29577.58 | 29570.60 | SL hit (close<static) qty=1.00 sl=29500.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 29535.00 | 29561.45 | 29564.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 29445.00 | 29538.16 | 29553.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 14:15:00 | 29605.00 | 29524.62 | 29542.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 14:15:00 | 29605.00 | 29524.62 | 29542.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 29605.00 | 29524.62 | 29542.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 29605.00 | 29524.62 | 29542.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 29625.00 | 29544.70 | 29550.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 29690.00 | 29544.70 | 29550.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 29305.00 | 29496.81 | 29527.51 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 29615.00 | 29528.57 | 29524.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 15:15:00 | 29780.00 | 29603.43 | 29562.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 29565.00 | 29595.74 | 29562.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 29565.00 | 29595.74 | 29562.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 29565.00 | 29595.74 | 29562.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:30:00 | 29415.00 | 29595.74 | 29562.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 29555.00 | 29587.59 | 29561.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 29555.00 | 29587.59 | 29561.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 29550.00 | 29580.08 | 29560.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:45:00 | 29575.00 | 29580.08 | 29560.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 29650.00 | 29594.06 | 29568.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 13:30:00 | 29810.00 | 29644.25 | 29593.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 12:15:00 | 29550.00 | 29762.45 | 29772.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 29550.00 | 29762.45 | 29772.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 13:15:00 | 29455.00 | 29700.96 | 29743.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 11:15:00 | 29535.00 | 29507.67 | 29617.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 11:45:00 | 29535.00 | 29507.67 | 29617.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 29670.00 | 29550.91 | 29618.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 29670.00 | 29550.91 | 29618.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 30045.00 | 29649.73 | 29657.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 30045.00 | 29649.73 | 29657.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 15:15:00 | 30010.00 | 29721.78 | 29689.51 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 29360.00 | 29674.21 | 29685.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 29270.00 | 29549.50 | 29623.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 29115.00 | 29057.57 | 29233.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 29115.00 | 29057.57 | 29233.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 29000.00 | 28971.59 | 29082.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 13:15:00 | 28885.00 | 28982.42 | 29068.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:00:00 | 28890.00 | 28963.93 | 29052.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 27440.75 | 27729.91 | 27811.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 27445.50 | 27729.91 | 27811.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 10:15:00 | 27850.00 | 27753.93 | 27814.79 | SL hit (close>ema200) qty=0.50 sl=27753.93 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 14:15:00 | 27965.00 | 27862.13 | 27853.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 10:15:00 | 28050.00 | 27923.89 | 27885.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 09:15:00 | 27960.00 | 28010.73 | 27959.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 27960.00 | 28010.73 | 27959.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 27960.00 | 28010.73 | 27959.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 27960.00 | 28010.73 | 27959.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 28020.00 | 28012.58 | 27965.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 11:30:00 | 28090.00 | 28028.06 | 27976.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 14:00:00 | 28055.00 | 28032.16 | 27987.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 14:45:00 | 28090.00 | 28043.73 | 27996.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 15:15:00 | 27950.00 | 28024.98 | 27992.46 | SL hit (close<static) qty=1.00 sl=27960.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 28500.00 | 28679.77 | 28701.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 28290.00 | 28546.20 | 28630.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 28155.00 | 28142.44 | 28275.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 15:00:00 | 28155.00 | 28142.44 | 28275.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 28300.00 | 28175.96 | 28267.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 28300.00 | 28175.96 | 28267.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 28360.00 | 28212.77 | 28275.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 28405.00 | 28212.77 | 28275.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 14:15:00 | 28420.00 | 28325.55 | 28315.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 28600.00 | 28393.15 | 28348.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 10:15:00 | 28365.00 | 28387.52 | 28350.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 11:00:00 | 28365.00 | 28387.52 | 28350.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 28360.00 | 28382.02 | 28351.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 28360.00 | 28382.02 | 28351.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 28230.00 | 28351.61 | 28340.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 28155.00 | 28351.61 | 28340.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 28350.00 | 28351.29 | 28341.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:45:00 | 28470.00 | 28377.03 | 28353.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 28485.00 | 28429.30 | 28383.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 14:15:00 | 28395.00 | 28410.42 | 28389.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 15:15:00 | 28400.00 | 28393.33 | 28383.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 28400.00 | 28394.67 | 28384.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 28155.00 | 28394.67 | 28384.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 28055.00 | 28326.73 | 28354.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 28055.00 | 28326.73 | 28354.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 10:15:00 | 27900.00 | 28241.39 | 28313.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 28140.00 | 28125.75 | 28218.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 15:15:00 | 28140.00 | 28125.75 | 28218.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 28140.00 | 28125.75 | 28218.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 27785.00 | 28125.75 | 28218.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:30:00 | 28035.00 | 28094.63 | 28178.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 28310.00 | 28151.25 | 28184.91 | SL hit (close>static) qty=1.00 sl=28250.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 28360.00 | 28227.20 | 28215.71 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 14:15:00 | 28005.00 | 28201.41 | 28214.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 13:15:00 | 27900.00 | 28119.38 | 28167.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 28030.00 | 28029.36 | 28107.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-16 10:00:00 | 28030.00 | 28029.36 | 28107.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 28040.00 | 28034.79 | 28097.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:30:00 | 27860.00 | 27983.83 | 28068.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 11:15:00 | 27730.00 | 27598.20 | 27590.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 27730.00 | 27598.20 | 27590.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 10:15:00 | 27815.00 | 27705.02 | 27654.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 27685.00 | 27701.02 | 27656.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 12:00:00 | 27685.00 | 27701.02 | 27656.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 27780.00 | 27749.32 | 27693.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 27705.00 | 27749.32 | 27693.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 27465.00 | 27713.36 | 27687.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 27540.00 | 27713.36 | 27687.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 27465.00 | 27663.69 | 27667.67 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 27715.00 | 27650.49 | 27649.26 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 12:15:00 | 27575.00 | 27636.91 | 27643.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 26880.00 | 27476.59 | 27566.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 11:15:00 | 26925.00 | 26874.54 | 27101.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:30:00 | 26910.00 | 26874.54 | 27101.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 27365.00 | 26991.11 | 27116.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 27365.00 | 26991.11 | 27116.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 27725.00 | 27137.89 | 27171.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 27725.00 | 27137.89 | 27171.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 15:15:00 | 27595.00 | 27229.31 | 27209.92 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 27180.00 | 27246.35 | 27247.22 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 10:15:00 | 27305.00 | 27258.08 | 27252.47 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 27130.00 | 27232.47 | 27241.34 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 27340.00 | 27253.42 | 27248.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 27500.00 | 27327.67 | 27285.39 | Break + close above crossover candle high |

### Cycle 66 — SELL (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 12:15:00 | 26870.00 | 27244.51 | 27255.45 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 27185.00 | 27115.89 | 27114.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 27375.00 | 27167.71 | 27138.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 27375.00 | 27414.99 | 27325.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 11:00:00 | 27375.00 | 27414.99 | 27325.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 27310.00 | 27382.47 | 27332.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:30:00 | 27290.00 | 27382.47 | 27332.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 27290.00 | 27363.98 | 27328.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 27290.00 | 27363.98 | 27328.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 27315.00 | 27354.18 | 27327.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 27235.00 | 27354.18 | 27327.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 27085.00 | 27300.35 | 27305.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 26890.00 | 27139.62 | 27223.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 26555.00 | 26536.51 | 26692.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 26555.00 | 26536.51 | 26692.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 26455.00 | 26427.35 | 26544.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 26430.00 | 26427.35 | 26544.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 26635.00 | 26479.13 | 26532.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 26565.00 | 26479.13 | 26532.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 26680.00 | 26519.30 | 26545.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 26715.00 | 26519.30 | 26545.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 26585.00 | 26564.16 | 26562.40 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 14:15:00 | 26450.00 | 26544.45 | 26554.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 26415.00 | 26505.05 | 26534.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 26470.00 | 26442.57 | 26477.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 26470.00 | 26442.57 | 26477.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 26470.00 | 26442.57 | 26477.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 26535.00 | 26442.57 | 26477.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 26530.00 | 26460.05 | 26481.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 26530.00 | 26460.05 | 26481.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 26390.00 | 26446.04 | 26473.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 26440.00 | 26446.04 | 26473.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 26500.00 | 26418.06 | 26448.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 26490.00 | 26418.06 | 26448.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 26500.00 | 26434.45 | 26452.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:15:00 | 26410.00 | 26455.44 | 26459.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 26500.00 | 26464.35 | 26462.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 26500.00 | 26464.35 | 26462.99 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 15:15:00 | 26425.00 | 26456.48 | 26459.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 26300.00 | 26425.18 | 26445.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 26365.00 | 26354.99 | 26397.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 26365.00 | 26354.99 | 26397.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 26365.00 | 26354.99 | 26397.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:45:00 | 26395.00 | 26354.99 | 26397.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 26340.00 | 26353.60 | 26389.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:45:00 | 26290.00 | 26355.90 | 26384.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:00:00 | 26290.00 | 26342.72 | 26375.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:30:00 | 26305.00 | 26347.80 | 26361.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 26205.00 | 26365.03 | 26366.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 26245.00 | 26341.03 | 26355.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-27 13:15:00 | 26500.00 | 26321.19 | 26333.55 | SL hit (close>static) qty=1.00 sl=26485.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 14:15:00 | 26525.00 | 26361.95 | 26350.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-02 12:15:00 | 26720.00 | 26510.07 | 26434.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 13:15:00 | 27560.00 | 27756.77 | 27354.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:30:00 | 27545.00 | 27756.77 | 27354.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 27395.00 | 27635.43 | 27394.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 27275.00 | 27635.43 | 27394.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 27350.00 | 27578.34 | 27390.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 27355.00 | 27578.34 | 27390.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 27375.00 | 27537.67 | 27389.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 12:15:00 | 27420.00 | 27537.67 | 27389.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 14:15:00 | 27420.00 | 27494.71 | 27394.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 26925.00 | 27324.21 | 27337.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 26925.00 | 27324.21 | 27337.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 13:15:00 | 26625.00 | 26855.11 | 26948.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 26750.00 | 26694.33 | 26822.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 11:30:00 | 26695.00 | 26694.33 | 26822.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 26965.00 | 26675.86 | 26775.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:45:00 | 27055.00 | 26675.86 | 26775.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 26990.00 | 26738.69 | 26794.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 26775.00 | 26738.69 | 26794.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 26870.00 | 26764.95 | 26801.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:30:00 | 26560.00 | 26689.21 | 26747.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 09:30:00 | 26625.00 | 26559.75 | 26657.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 13:00:00 | 26550.00 | 26559.79 | 26632.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 26615.00 | 26427.18 | 26490.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 26485.00 | 26438.74 | 26490.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 26590.00 | 26438.74 | 26490.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 26480.00 | 26446.99 | 26489.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 26415.00 | 26446.99 | 26489.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 26620.00 | 26481.59 | 26501.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:00:00 | 26620.00 | 26481.59 | 26501.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-20 12:15:00 | 26730.00 | 26531.28 | 26521.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 26730.00 | 26531.28 | 26521.87 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 25945.00 | 26478.16 | 26507.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 25640.00 | 26310.53 | 26428.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 26180.00 | 25784.82 | 26000.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 26180.00 | 25784.82 | 26000.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 26180.00 | 25784.82 | 26000.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 26180.00 | 25784.82 | 26000.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 26530.00 | 25933.85 | 26048.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 26530.00 | 25933.85 | 26048.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 26785.00 | 26189.87 | 26150.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 27035.00 | 26466.11 | 26289.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 26395.00 | 26637.97 | 26466.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 26395.00 | 26637.97 | 26466.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 26395.00 | 26637.97 | 26466.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 26395.00 | 26637.97 | 26466.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 26505.00 | 26611.38 | 26470.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 26195.00 | 26528.10 | 26445.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 26290.00 | 26480.48 | 26430.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 26225.00 | 26480.48 | 26430.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 26265.00 | 26375.81 | 26390.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 26090.00 | 26318.64 | 26362.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 26490.00 | 26087.52 | 26163.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 26490.00 | 26087.52 | 26163.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 26490.00 | 26087.52 | 26163.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 26375.00 | 26087.52 | 26163.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 26310.00 | 26132.02 | 26176.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:15:00 | 26205.00 | 26132.02 | 26176.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 26455.00 | 26232.29 | 26216.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 26455.00 | 26232.29 | 26216.62 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 09:15:00 | 26025.00 | 26248.72 | 26260.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 12:15:00 | 25900.00 | 26121.74 | 26195.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 26140.00 | 25852.87 | 25952.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 26140.00 | 25852.87 | 25952.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 26140.00 | 25852.87 | 25952.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 26200.00 | 25852.87 | 25952.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 25875.00 | 25857.30 | 25945.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 12:45:00 | 25825.00 | 25865.67 | 25934.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:30:00 | 25825.00 | 25863.23 | 25921.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:45:00 | 25780.00 | 25871.45 | 25910.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:30:00 | 25800.00 | 25852.93 | 25895.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 25790.00 | 25831.88 | 25877.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 25790.00 | 25831.88 | 25877.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 25815.00 | 25817.80 | 25862.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 25575.00 | 25804.70 | 25836.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 25945.00 | 25816.26 | 25814.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 25945.00 | 25816.26 | 25814.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 26090.00 | 25871.01 | 25839.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 13:15:00 | 25780.00 | 25876.65 | 25849.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 13:15:00 | 25780.00 | 25876.65 | 25849.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 25780.00 | 25876.65 | 25849.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:00:00 | 25780.00 | 25876.65 | 25849.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 25990.00 | 25899.32 | 25862.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:30:00 | 25850.00 | 25899.32 | 25862.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 25900.00 | 25899.45 | 25865.55 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 09:15:00 | 25795.00 | 25863.03 | 25868.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 10:15:00 | 25530.00 | 25729.70 | 25791.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 25535.00 | 25486.88 | 25575.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 25535.00 | 25486.88 | 25575.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 25535.00 | 25486.88 | 25575.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:45:00 | 25620.00 | 25486.88 | 25575.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 25500.00 | 25489.50 | 25568.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 25495.00 | 25489.50 | 25568.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 25505.00 | 25492.60 | 25562.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 14:45:00 | 25450.00 | 25480.61 | 25539.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 25465.00 | 25472.59 | 25525.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:30:00 | 25400.00 | 25459.07 | 25514.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 12:45:00 | 25455.00 | 25409.31 | 25449.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 25250.00 | 25377.44 | 25431.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:15:00 | 25220.00 | 25377.44 | 25431.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:45:00 | 25235.00 | 25344.49 | 25371.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 15:15:00 | 25550.00 | 25356.33 | 25362.53 | SL hit (close>static) qty=1.00 sl=25485.00 alert=retest2 |

### Cycle 83 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 25515.00 | 25388.06 | 25376.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 25740.00 | 25458.45 | 25409.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 25485.00 | 25500.40 | 25448.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 14:15:00 | 25485.00 | 25500.40 | 25448.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 25485.00 | 25500.40 | 25448.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:30:00 | 25460.00 | 25500.40 | 25448.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 25510.00 | 25502.32 | 25454.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 25345.00 | 25502.32 | 25454.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 25350.00 | 25471.85 | 25445.01 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 25365.00 | 25427.79 | 25428.23 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 25675.00 | 25473.12 | 25446.50 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 25390.00 | 25454.76 | 25461.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 11:15:00 | 25345.00 | 25432.81 | 25450.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 15:15:00 | 25385.00 | 25379.14 | 25415.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 09:15:00 | 25500.00 | 25379.14 | 25415.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 25550.00 | 25413.31 | 25427.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:45:00 | 25630.00 | 25413.31 | 25427.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 25570.00 | 25444.65 | 25440.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 25835.00 | 25562.18 | 25498.38 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-26 14:45:00 | 30495.00 | 2025-05-30 15:15:00 | 30620.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-06-09 11:30:00 | 31920.00 | 2025-06-11 15:15:00 | 31560.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-06-09 13:00:00 | 31945.00 | 2025-06-11 15:15:00 | 31560.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-06-10 09:15:00 | 31920.00 | 2025-06-11 15:15:00 | 31560.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-06-18 12:30:00 | 31800.00 | 2025-06-18 15:15:00 | 31555.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-06-19 09:15:00 | 31775.00 | 2025-06-19 12:15:00 | 31465.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-06-19 10:00:00 | 31770.00 | 2025-06-19 12:15:00 | 31465.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-07-02 09:15:00 | 34860.00 | 2025-07-04 14:15:00 | 34750.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-07-02 11:00:00 | 34855.00 | 2025-07-04 14:15:00 | 34750.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-07-09 13:00:00 | 34090.00 | 2025-07-11 09:15:00 | 34690.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-07-16 10:30:00 | 34180.00 | 2025-07-22 12:15:00 | 34175.00 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-07-17 09:30:00 | 34210.00 | 2025-07-22 12:15:00 | 34175.00 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-07-17 11:45:00 | 34160.00 | 2025-07-22 12:15:00 | 34175.00 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-08-04 13:15:00 | 34055.00 | 2025-08-07 09:15:00 | 32352.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-04 13:15:00 | 34055.00 | 2025-08-07 14:15:00 | 32725.00 | STOP_HIT | 0.50 | 3.91% |
| BUY | retest2 | 2025-08-14 09:15:00 | 33195.00 | 2025-08-14 13:15:00 | 33010.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-08-22 10:15:00 | 32570.00 | 2025-09-01 15:15:00 | 32275.00 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2025-09-11 14:15:00 | 31000.00 | 2025-09-15 10:15:00 | 31290.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-12 12:30:00 | 31025.00 | 2025-09-15 10:15:00 | 31290.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest1 | 2025-09-22 12:15:00 | 30520.00 | 2025-09-25 10:15:00 | 30405.00 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest1 | 2025-09-22 13:30:00 | 30480.00 | 2025-09-25 10:15:00 | 30405.00 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-09-24 13:00:00 | 30175.00 | 2025-10-01 12:15:00 | 30015.00 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2025-09-25 13:00:00 | 30195.00 | 2025-10-01 12:15:00 | 30015.00 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2025-10-06 11:30:00 | 30410.00 | 2025-10-07 14:15:00 | 29825.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-10-06 12:00:00 | 30400.00 | 2025-10-07 14:15:00 | 29825.00 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-10-20 11:30:00 | 29650.00 | 2025-10-20 13:15:00 | 30125.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-10-23 15:15:00 | 30100.00 | 2025-10-24 12:15:00 | 29870.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-10-30 09:30:00 | 29630.00 | 2025-11-04 13:15:00 | 29345.00 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2025-10-30 11:00:00 | 29675.00 | 2025-11-04 13:15:00 | 29345.00 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2025-11-12 15:00:00 | 29605.00 | 2025-11-14 11:15:00 | 29365.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-13 09:30:00 | 29550.00 | 2025-11-14 11:15:00 | 29365.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-11-13 10:00:00 | 29560.00 | 2025-11-14 11:15:00 | 29365.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-11-13 14:15:00 | 29580.00 | 2025-11-14 11:15:00 | 29365.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-11-19 15:15:00 | 29620.00 | 2025-11-20 09:15:00 | 29485.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-11-20 10:30:00 | 29580.00 | 2025-11-20 11:15:00 | 29535.00 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-11-20 11:15:00 | 29565.00 | 2025-11-20 11:15:00 | 29535.00 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-11-25 13:30:00 | 29810.00 | 2025-11-27 12:15:00 | 29550.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-12-05 13:15:00 | 28885.00 | 2025-12-18 09:15:00 | 27440.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 14:00:00 | 28890.00 | 2025-12-18 09:15:00 | 27445.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 13:15:00 | 28885.00 | 2025-12-18 10:15:00 | 27850.00 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-12-05 14:00:00 | 28890.00 | 2025-12-18 10:15:00 | 27850.00 | STOP_HIT | 0.50 | 3.60% |
| BUY | retest2 | 2025-12-22 11:30:00 | 28090.00 | 2025-12-22 15:15:00 | 27950.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-22 14:00:00 | 28055.00 | 2025-12-22 15:15:00 | 27950.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-12-22 14:45:00 | 28090.00 | 2025-12-22 15:15:00 | 27950.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-23 11:30:00 | 28100.00 | 2026-01-01 13:15:00 | 28500.00 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2025-12-24 14:45:00 | 28255.00 | 2026-01-01 13:15:00 | 28500.00 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2026-01-07 14:45:00 | 28470.00 | 2026-01-09 09:15:00 | 28055.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-01-08 09:30:00 | 28485.00 | 2026-01-09 09:15:00 | 28055.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-01-08 14:15:00 | 28395.00 | 2026-01-09 09:15:00 | 28055.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-01-08 15:15:00 | 28400.00 | 2026-01-09 09:15:00 | 28055.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-01-12 09:15:00 | 27785.00 | 2026-01-12 14:15:00 | 28310.00 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-01-12 11:30:00 | 28035.00 | 2026-01-12 14:15:00 | 28310.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-01-16 12:30:00 | 27860.00 | 2026-01-22 11:15:00 | 27730.00 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2026-02-23 14:15:00 | 26410.00 | 2026-02-23 14:15:00 | 26500.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2026-02-25 11:45:00 | 26290.00 | 2026-02-27 13:15:00 | 26500.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-02-25 13:00:00 | 26290.00 | 2026-02-27 13:15:00 | 26500.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-02-26 12:30:00 | 26305.00 | 2026-02-27 13:15:00 | 26500.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-02-27 09:15:00 | 26205.00 | 2026-02-27 13:15:00 | 26500.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-03-06 12:15:00 | 27420.00 | 2026-03-09 09:15:00 | 26925.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-03-06 14:15:00 | 27420.00 | 2026-03-09 09:15:00 | 26925.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-03-17 11:30:00 | 26560.00 | 2026-03-20 12:15:00 | 26730.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-03-18 09:30:00 | 26625.00 | 2026-03-20 12:15:00 | 26730.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-03-18 13:00:00 | 26550.00 | 2026-03-20 12:15:00 | 26730.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-03-20 09:15:00 | 26615.00 | 2026-03-20 12:15:00 | 26730.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-04-01 11:15:00 | 26205.00 | 2026-04-01 12:15:00 | 26455.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-04-08 12:45:00 | 25825.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-04-08 14:30:00 | 25825.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-04-09 10:45:00 | 25780.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-04-09 12:30:00 | 25800.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-04-13 09:15:00 | 25575.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-04-22 14:45:00 | 25450.00 | 2026-04-28 15:15:00 | 25550.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-04-23 10:00:00 | 25465.00 | 2026-04-28 15:15:00 | 25550.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-04-23 10:30:00 | 25400.00 | 2026-04-29 09:15:00 | 25515.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-04-24 12:45:00 | 25455.00 | 2026-04-29 09:15:00 | 25515.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2026-04-24 14:15:00 | 25220.00 | 2026-04-29 09:15:00 | 25515.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-04-28 10:45:00 | 25235.00 | 2026-04-29 09:15:00 | 25515.00 | STOP_HIT | 1.00 | -1.11% |
