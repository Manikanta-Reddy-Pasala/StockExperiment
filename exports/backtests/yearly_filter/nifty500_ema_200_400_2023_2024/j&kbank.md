# Jammu & Kashmir Bank Ltd. (J&KBANK)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 141.24
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 49 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 32 |
| PARTIAL | 3 |
| TARGET_HIT | 9 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 23
- **Target hits / Stop hits / Partials:** 9 / 23 / 3
- **Avg / median % per leg:** 1.88% / -0.46%
- **Sum % (uncompounded):** 65.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 8 | 47.1% | 8 | 9 | 0 | 4.09% | 69.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 8 | 47.1% | 8 | 9 | 0 | 4.09% | 69.5% |
| SELL (all) | 18 | 4 | 22.2% | 1 | 14 | 3 | -0.20% | -3.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 4 | 22.2% | 1 | 14 | 3 | -0.20% | -3.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 35 | 12 | 34.3% | 9 | 23 | 3 | 1.88% | 65.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 14:15:00 | 129.30 | 133.46 | 133.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 126.45 | 133.35 | 133.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 13:15:00 | 131.60 | 131.52 | 132.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 13:15:00 | 131.60 | 131.52 | 132.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 131.60 | 131.52 | 132.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 09:45:00 | 131.15 | 131.51 | 132.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:00:00 | 131.25 | 131.50 | 132.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:15:00 | 124.69 | 130.85 | 131.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:15:00 | 124.59 | 130.81 | 131.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-31 14:15:00 | 131.75 | 130.71 | 131.78 | SL hit (close>ema200) qty=0.50 sl=130.71 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 13:15:00 | 114.07 | 96.53 | 96.51 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 12:15:00 | 91.75 | 96.75 | 96.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 91.25 | 96.42 | 96.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 12:15:00 | 96.60 | 95.82 | 96.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 12:15:00 | 96.60 | 95.82 | 96.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 12:15:00 | 96.60 | 95.82 | 96.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 13:00:00 | 96.60 | 95.82 | 96.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 13:15:00 | 96.70 | 95.83 | 96.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 13:45:00 | 96.62 | 95.83 | 96.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 14:15:00 | 97.05 | 95.84 | 96.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 14:30:00 | 96.36 | 95.84 | 96.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 15:15:00 | 97.66 | 95.86 | 96.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-13 09:15:00 | 97.00 | 95.86 | 96.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 101.15 | 96.68 | 96.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 103.41 | 96.87 | 96.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 102.99 | 103.33 | 101.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 13:00:00 | 102.99 | 103.33 | 101.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 100.85 | 103.27 | 101.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 100.85 | 103.27 | 101.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 101.40 | 103.25 | 101.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:45:00 | 100.65 | 103.25 | 101.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 106.10 | 110.14 | 107.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 106.10 | 110.14 | 107.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 105.61 | 110.09 | 107.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 14:00:00 | 105.61 | 110.09 | 107.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 105.79 | 106.04 | 105.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:30:00 | 105.70 | 106.04 | 105.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 105.42 | 106.04 | 105.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 105.42 | 106.04 | 105.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 105.55 | 106.03 | 105.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:00:00 | 105.55 | 106.03 | 105.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 106.00 | 106.03 | 105.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 103.86 | 106.03 | 105.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 104.15 | 106.01 | 105.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 12:00:00 | 104.53 | 105.98 | 105.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:30:00 | 104.67 | 105.93 | 105.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:00:00 | 104.71 | 105.91 | 105.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:00:00 | 104.55 | 105.81 | 105.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 103.70 | 105.70 | 105.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 103.70 | 105.70 | 105.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 103.20 | 105.63 | 105.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 10:15:00 | 103.45 | 102.70 | 103.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 10:15:00 | 103.45 | 102.70 | 103.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 103.45 | 102.70 | 103.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:45:00 | 104.30 | 102.70 | 103.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 103.81 | 102.72 | 103.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 103.81 | 102.72 | 103.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 104.40 | 102.74 | 103.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 104.30 | 102.74 | 103.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 103.66 | 102.81 | 103.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:15:00 | 103.92 | 102.81 | 103.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 104.38 | 102.82 | 103.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:15:00 | 104.79 | 102.82 | 103.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 104.79 | 102.84 | 103.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 104.76 | 102.86 | 103.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 104.30 | 103.14 | 103.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 104.25 | 103.14 | 103.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 103.50 | 103.16 | 103.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:30:00 | 103.68 | 103.16 | 103.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 104.38 | 103.17 | 103.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 104.43 | 103.17 | 103.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 105.38 | 103.19 | 103.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:45:00 | 105.32 | 103.19 | 103.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 103.45 | 103.34 | 103.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:30:00 | 103.19 | 103.34 | 103.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:45:00 | 103.23 | 103.34 | 103.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 104.90 | 103.01 | 103.68 | SL hit (close>static) qty=1.00 sl=104.03 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 105.05 | 104.05 | 104.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 10:15:00 | 105.51 | 104.06 | 104.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 12:15:00 | 105.55 | 105.55 | 104.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 13:00:00 | 105.55 | 105.55 | 104.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 103.52 | 105.52 | 104.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 103.52 | 105.52 | 104.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 103.62 | 105.50 | 104.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:15:00 | 102.97 | 105.50 | 104.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 105.41 | 105.30 | 104.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 106.73 | 105.30 | 104.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 11:30:00 | 105.86 | 106.34 | 105.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:45:00 | 105.90 | 106.33 | 105.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 106.11 | 106.30 | 105.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 105.67 | 106.29 | 105.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 105.67 | 106.29 | 105.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 105.52 | 106.28 | 105.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:45:00 | 105.39 | 106.28 | 105.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 105.05 | 106.27 | 105.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:00:00 | 105.05 | 106.27 | 105.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 104.62 | 106.26 | 105.57 | SL hit (close<static) qty=1.00 sl=104.71 alert=retest2 |

### Cycle 7 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 101.37 | 105.26 | 105.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 100.94 | 105.18 | 105.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 11:15:00 | 102.02 | 101.73 | 103.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-02 12:00:00 | 102.02 | 101.73 | 103.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 104.56 | 101.78 | 103.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 103.07 | 101.79 | 103.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 12:30:00 | 103.10 | 101.82 | 103.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 103.02 | 102.23 | 103.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:00:00 | 103.04 | 101.93 | 102.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 102.74 | 101.94 | 102.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:45:00 | 102.85 | 101.94 | 102.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 102.72 | 101.95 | 102.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:45:00 | 102.85 | 101.95 | 102.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 102.80 | 101.96 | 102.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 102.86 | 101.96 | 102.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 102.81 | 101.97 | 102.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 102.20 | 101.97 | 102.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 102.82 | 101.98 | 102.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:00:00 | 101.79 | 101.97 | 102.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:00:00 | 101.44 | 102.00 | 102.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:00:00 | 101.58 | 101.92 | 102.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:30:00 | 101.40 | 101.92 | 102.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 103.17 | 101.93 | 102.74 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 103.17 | 101.93 | 102.74 | SL hit (close>static) qty=1.00 sl=103.00 alert=retest2 |

### Cycle 8 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 105.92 | 103.29 | 103.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 106.45 | 103.60 | 103.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 10:15:00 | 109.35 | 109.52 | 106.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 10:30:00 | 109.12 | 109.52 | 106.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 110.80 | 115.17 | 110.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:00:00 | 110.80 | 115.17 | 110.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 111.61 | 115.14 | 110.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:45:00 | 110.77 | 115.14 | 110.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 108.89 | 115.08 | 110.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:00:00 | 108.89 | 115.08 | 110.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 110.82 | 115.03 | 110.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 112.00 | 114.91 | 110.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 12:30:00 | 111.50 | 114.88 | 111.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 13:00:00 | 111.31 | 114.88 | 111.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 114.01 | 114.75 | 111.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 110.35 | 114.64 | 111.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 110.35 | 114.64 | 111.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 110.51 | 114.60 | 111.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 110.30 | 114.60 | 111.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 113.29 | 114.51 | 111.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 15:00:00 | 114.06 | 114.51 | 111.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 113.53 | 114.49 | 111.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:30:00 | 113.34 | 114.47 | 111.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:45:00 | 113.45 | 114.46 | 111.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 10:15:00 | 122.65 | 114.80 | 111.86 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-28 09:45:00 | 131.15 | 2024-05-31 09:15:00 | 124.69 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2024-05-28 11:00:00 | 131.25 | 2024-05-31 10:15:00 | 124.59 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2024-05-28 09:45:00 | 131.15 | 2024-05-31 14:15:00 | 131.75 | STOP_HIT | 0.50 | -0.46% |
| SELL | retest2 | 2024-05-28 11:00:00 | 131.25 | 2024-05-31 14:15:00 | 131.75 | STOP_HIT | 0.50 | -0.38% |
| SELL | retest2 | 2024-06-04 09:15:00 | 128.40 | 2024-06-04 10:15:00 | 121.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 128.40 | 2024-06-04 11:15:00 | 115.56 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-19 12:00:00 | 104.53 | 2025-08-22 10:15:00 | 103.70 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-08-19 14:30:00 | 104.67 | 2025-08-22 10:15:00 | 103.70 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-08-20 10:00:00 | 104.71 | 2025-08-22 10:15:00 | 103.70 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-08-21 10:00:00 | 104.55 | 2025-08-22 10:15:00 | 103.70 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-09-25 11:30:00 | 103.19 | 2025-09-30 11:15:00 | 104.90 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-09-25 13:45:00 | 103.23 | 2025-09-30 11:15:00 | 104.90 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-10-08 10:00:00 | 103.15 | 2025-10-10 09:15:00 | 105.05 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-11-10 09:15:00 | 106.73 | 2025-11-24 13:15:00 | 104.62 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-11-21 11:30:00 | 105.86 | 2025-11-24 13:15:00 | 104.62 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-11-21 13:45:00 | 105.90 | 2025-11-24 13:15:00 | 104.62 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-11-24 09:15:00 | 106.11 | 2025-11-24 13:15:00 | 104.62 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-11-26 09:15:00 | 106.50 | 2025-12-03 09:15:00 | 105.18 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-01-05 11:15:00 | 103.07 | 2026-01-21 11:15:00 | 103.17 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2026-01-05 12:30:00 | 103.10 | 2026-01-21 11:15:00 | 103.17 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2026-01-08 10:45:00 | 103.02 | 2026-01-21 11:15:00 | 103.17 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2026-01-16 11:00:00 | 103.04 | 2026-01-21 11:15:00 | 103.17 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2026-01-19 11:00:00 | 101.79 | 2026-01-22 09:15:00 | 106.88 | STOP_HIT | 1.00 | -5.00% |
| SELL | retest2 | 2026-01-20 12:00:00 | 101.44 | 2026-01-22 09:15:00 | 106.88 | STOP_HIT | 1.00 | -5.36% |
| SELL | retest2 | 2026-01-21 10:00:00 | 101.58 | 2026-01-22 09:15:00 | 106.88 | STOP_HIT | 1.00 | -5.22% |
| SELL | retest2 | 2026-01-21 10:30:00 | 101.40 | 2026-01-22 09:15:00 | 106.88 | STOP_HIT | 1.00 | -5.40% |
| SELL | retest2 | 2026-02-01 15:00:00 | 102.07 | 2026-02-03 13:15:00 | 103.39 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-03-24 09:15:00 | 112.00 | 2026-04-08 10:15:00 | 122.65 | TARGET_HIT | 1.00 | 9.51% |
| BUY | retest2 | 2026-03-30 12:30:00 | 111.50 | 2026-04-08 10:15:00 | 122.44 | TARGET_HIT | 1.00 | 9.81% |
| BUY | retest2 | 2026-03-30 13:00:00 | 111.31 | 2026-04-08 11:15:00 | 123.20 | TARGET_HIT | 1.00 | 10.68% |
| BUY | retest2 | 2026-04-01 09:15:00 | 114.01 | 2026-04-09 09:15:00 | 125.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 15:00:00 | 114.06 | 2026-04-09 09:15:00 | 125.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:00:00 | 113.53 | 2026-04-09 09:15:00 | 124.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:30:00 | 113.34 | 2026-04-09 09:15:00 | 124.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 11:45:00 | 113.45 | 2026-04-09 09:15:00 | 124.80 | TARGET_HIT | 1.00 | 10.00% |
