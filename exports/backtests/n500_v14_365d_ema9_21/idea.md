# Vodafone Idea Ltd. (IDEA)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 11.24
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 63 |
| ALERT1 | 40 |
| ALERT2 | 39 |
| ALERT2_SKIP | 15 |
| ALERT3 | 107 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 51 |
| PARTIAL | 14 |
| TARGET_HIT | 6 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 37 / 30
- **Target hits / Stop hits / Partials:** 6 / 47 / 14
- **Avg / median % per leg:** 1.14% / 1.37%
- **Sum % (uncompounded):** 76.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 6 | 40.0% | 2 | 13 | 0 | -0.10% | -1.5% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 2.48% | 2.5% |
| BUY @ 3rd Alert (retest2) | 14 | 5 | 35.7% | 2 | 12 | 0 | -0.28% | -4.0% |
| SELL (all) | 52 | 31 | 59.6% | 4 | 34 | 14 | 1.50% | 78.2% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.80% | 1.6% |
| SELL @ 3rd Alert (retest2) | 50 | 29 | 58.0% | 4 | 32 | 14 | 1.53% | 76.6% |
| retest1 (combined) | 3 | 3 | 100.0% | 0 | 3 | 0 | 1.36% | 4.1% |
| retest2 (combined) | 64 | 34 | 53.1% | 6 | 44 | 14 | 1.13% | 72.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 6.99 | 6.84 | 6.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 7.09 | 6.95 | 6.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 6.99 | 6.99 | 6.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:45:00 | 6.99 | 6.99 | 6.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 6.94 | 6.98 | 6.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:30:00 | 6.95 | 6.98 | 6.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 6.96 | 6.98 | 6.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 7.01 | 6.98 | 6.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 6.66 | 7.09 | 7.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 6.66 | 7.09 | 7.13 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 6.78 | 6.74 | 6.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 12:15:00 | 6.81 | 6.76 | 6.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 13:15:00 | 7.13 | 7.14 | 7.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:30:00 | 7.13 | 7.14 | 7.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 6.92 | 7.10 | 7.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 6.92 | 7.10 | 7.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 6.91 | 7.06 | 7.05 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 6.90 | 7.03 | 7.04 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 7.03 | 7.01 | 7.00 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 6.96 | 7.00 | 7.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 6.82 | 6.93 | 6.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 12:15:00 | 6.90 | 6.82 | 6.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 12:15:00 | 6.90 | 6.82 | 6.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 6.90 | 6.82 | 6.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 6.90 | 6.82 | 6.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 6.91 | 6.84 | 6.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 6.91 | 6.84 | 6.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 6.90 | 6.85 | 6.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:30:00 | 6.94 | 6.85 | 6.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 6.88 | 6.87 | 6.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:30:00 | 6.91 | 6.87 | 6.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 6.86 | 6.87 | 6.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:30:00 | 6.88 | 6.87 | 6.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 6.84 | 6.86 | 6.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 14:30:00 | 6.81 | 6.86 | 6.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 6.81 | 6.85 | 6.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 12:15:00 | 6.90 | 6.86 | 6.87 | SL hit (close>static) qty=1.00 sl=6.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-06 12:15:00 | 6.90 | 6.86 | 6.87 | SL hit (close>static) qty=1.00 sl=6.88 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 6.93 | 6.87 | 6.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 7.00 | 6.91 | 6.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 7.00 | 7.01 | 6.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:30:00 | 7.00 | 7.01 | 6.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 6.97 | 7.00 | 6.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:00:00 | 6.97 | 7.00 | 6.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 6.99 | 7.00 | 6.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:30:00 | 6.96 | 7.00 | 6.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 6.98 | 6.99 | 6.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:30:00 | 6.99 | 6.99 | 6.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 6.97 | 6.99 | 6.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 7.04 | 6.99 | 6.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 7.07 | 7.01 | 6.98 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 6.92 | 6.98 | 6.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 6.89 | 6.95 | 6.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 6.71 | 6.67 | 6.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:45:00 | 6.69 | 6.67 | 6.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 6.76 | 6.68 | 6.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 6.74 | 6.68 | 6.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 6.70 | 6.69 | 6.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 15:15:00 | 6.67 | 6.69 | 6.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:45:00 | 6.68 | 6.68 | 6.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:45:00 | 6.68 | 6.68 | 6.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 6.66 | 6.68 | 6.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 6.67 | 6.65 | 6.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 6.66 | 6.65 | 6.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 6.62 | 6.64 | 6.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:45:00 | 6.59 | 6.64 | 6.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:15:00 | 6.59 | 6.63 | 6.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 13:15:00 | 6.34 | 6.48 | 6.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 13:15:00 | 6.35 | 6.48 | 6.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 13:15:00 | 6.35 | 6.48 | 6.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 15:15:00 | 6.33 | 6.43 | 6.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 6.48 | 6.43 | 6.51 | SL hit (close>ema200) qty=0.50 sl=6.43 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 6.48 | 6.43 | 6.51 | SL hit (close>ema200) qty=0.50 sl=6.43 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 6.48 | 6.43 | 6.51 | SL hit (close>ema200) qty=0.50 sl=6.43 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 6.48 | 6.43 | 6.51 | SL hit (close>ema200) qty=0.50 sl=6.43 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 6.60 | 6.53 | 6.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 6.60 | 6.53 | 6.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 6.60 | 6.53 | 6.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 6.76 | 6.59 | 6.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 7.41 | 7.42 | 7.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:00:00 | 7.41 | 7.42 | 7.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 7.40 | 7.41 | 7.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:45:00 | 7.39 | 7.41 | 7.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 7.56 | 7.44 | 7.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 10:15:00 | 7.66 | 7.51 | 7.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 7.37 | 7.48 | 7.48 | SL hit (close<static) qty=1.00 sl=7.39 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 7.37 | 7.46 | 7.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 7.29 | 7.39 | 7.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 09:15:00 | 7.34 | 7.34 | 7.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:45:00 | 7.35 | 7.34 | 7.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 7.35 | 7.30 | 7.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 7.38 | 7.30 | 7.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 7.39 | 7.32 | 7.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:30:00 | 7.31 | 7.32 | 7.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:15:00 | 7.30 | 7.31 | 7.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 13:15:00 | 7.31 | 7.30 | 7.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 7.29 | 7.32 | 7.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 7.18 | 7.29 | 7.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:30:00 | 7.15 | 7.26 | 7.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 7.44 | 7.26 | 7.27 | SL hit (close>static) qty=1.00 sl=7.43 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 7.44 | 7.26 | 7.27 | SL hit (close>static) qty=1.00 sl=7.43 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 7.44 | 7.26 | 7.27 | SL hit (close>static) qty=1.00 sl=7.43 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 7.44 | 7.26 | 7.27 | SL hit (close>static) qty=1.00 sl=7.43 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 7.44 | 7.26 | 7.27 | SL hit (close>static) qty=1.00 sl=7.33 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 7.58 | 7.33 | 7.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 14:15:00 | 7.72 | 7.52 | 7.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 7.60 | 7.62 | 7.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:00:00 | 7.60 | 7.62 | 7.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 7.76 | 7.79 | 7.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 7.74 | 7.79 | 7.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 7.76 | 7.79 | 7.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:15:00 | 7.70 | 7.79 | 7.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 7.71 | 7.77 | 7.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 7.70 | 7.77 | 7.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 7.69 | 7.76 | 7.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 7.70 | 7.76 | 7.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 7.68 | 7.74 | 7.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:45:00 | 7.68 | 7.74 | 7.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 7.73 | 7.73 | 7.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:15:00 | 7.69 | 7.73 | 7.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 7.69 | 7.72 | 7.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 7.65 | 7.72 | 7.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 7.69 | 7.72 | 7.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 7.56 | 7.65 | 7.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 12:15:00 | 7.48 | 7.47 | 7.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 13:00:00 | 7.48 | 7.47 | 7.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 7.45 | 7.47 | 7.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:15:00 | 7.43 | 7.47 | 7.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 10:30:00 | 7.34 | 7.31 | 7.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 10:15:00 | 7.06 | 7.16 | 7.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 6.97 | 7.02 | 7.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-01 14:15:00 | 6.69 | 6.78 | 6.91 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-01 14:15:00 | 6.61 | 6.78 | 6.91 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 13 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 6.94 | 6.87 | 6.86 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 6.77 | 6.84 | 6.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 6.70 | 6.79 | 6.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 6.72 | 6.72 | 6.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:45:00 | 6.71 | 6.72 | 6.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 6.61 | 6.59 | 6.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:45:00 | 6.64 | 6.59 | 6.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 6.59 | 6.59 | 6.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:30:00 | 6.60 | 6.59 | 6.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 6.62 | 6.31 | 6.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 6.62 | 6.31 | 6.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 6.65 | 6.38 | 6.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:00:00 | 6.65 | 6.38 | 6.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 6.50 | 6.41 | 6.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 6.54 | 6.46 | 6.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 6.71 | 6.74 | 6.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 12:15:00 | 6.63 | 6.71 | 6.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 6.63 | 6.71 | 6.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:15:00 | 6.63 | 6.71 | 6.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 6.62 | 6.69 | 6.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 6.63 | 6.69 | 6.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 6.62 | 6.65 | 6.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 6.59 | 6.65 | 6.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 6.76 | 6.67 | 6.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:30:00 | 7.09 | 6.79 | 6.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 13:15:00 | 6.71 | 6.91 | 6.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 6.71 | 6.91 | 6.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 6.57 | 6.79 | 6.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 6.72 | 6.65 | 6.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 6.72 | 6.65 | 6.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 6.72 | 6.65 | 6.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 6.72 | 6.65 | 6.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 6.69 | 6.66 | 6.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 6.62 | 6.66 | 6.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 6.76 | 6.59 | 6.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 6.76 | 6.59 | 6.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 12:15:00 | 6.97 | 6.74 | 6.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 7.26 | 7.26 | 7.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:30:00 | 7.20 | 7.26 | 7.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 8.01 | 8.02 | 7.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:00:00 | 8.05 | 8.03 | 7.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 13:00:00 | 8.04 | 8.03 | 7.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 7.73 | 7.87 | 7.86 | SL hit (close<static) qty=1.00 sl=7.82 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 7.73 | 7.87 | 7.86 | SL hit (close<static) qty=1.00 sl=7.82 alert=retest2 |

### Cycle 18 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 7.78 | 7.85 | 7.86 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 7.91 | 7.85 | 7.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 11:15:00 | 8.58 | 8.00 | 7.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 15:15:00 | 8.38 | 8.41 | 8.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:15:00 | 8.54 | 8.41 | 8.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 8.40 | 8.64 | 8.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 8.38 | 8.64 | 8.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 8.39 | 8.59 | 8.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 8.15 | 8.50 | 8.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 8.23 | 8.21 | 8.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 8.23 | 8.21 | 8.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 8.27 | 8.22 | 8.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:45:00 | 8.19 | 8.22 | 8.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 8.18 | 8.21 | 8.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 8.15 | 8.20 | 8.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 8.39 | 8.29 | 8.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 8.39 | 8.29 | 8.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 8.39 | 8.29 | 8.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 8.39 | 8.29 | 8.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 8.49 | 8.33 | 8.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 8.46 | 8.62 | 8.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 8.46 | 8.62 | 8.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 8.46 | 8.62 | 8.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:45:00 | 8.45 | 8.62 | 8.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 8.67 | 8.63 | 8.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:30:00 | 8.59 | 8.63 | 8.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 8.47 | 8.60 | 8.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 8.47 | 8.60 | 8.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 8.47 | 8.57 | 8.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:15:00 | 8.50 | 8.57 | 8.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 8.48 | 8.55 | 8.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:15:00 | 8.60 | 8.52 | 8.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 12:15:00 | 8.79 | 8.98 | 9.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 8.79 | 8.98 | 9.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 14:15:00 | 8.73 | 8.89 | 8.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 8.65 | 8.54 | 8.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 8.65 | 8.54 | 8.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 8.65 | 8.56 | 8.69 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 8.89 | 8.72 | 8.72 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 8.71 | 8.78 | 8.78 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 8.91 | 8.79 | 8.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 14:15:00 | 9.04 | 8.93 | 8.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 9.30 | 9.32 | 9.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 09:30:00 | 9.33 | 9.32 | 9.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 9.34 | 9.50 | 9.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 9.34 | 9.50 | 9.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 10.09 | 9.62 | 9.42 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 10:15:00 | 9.43 | 9.54 | 9.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 14:15:00 | 9.38 | 9.47 | 9.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 8.88 | 8.87 | 9.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 09:45:00 | 8.92 | 8.87 | 9.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 8.86 | 8.83 | 8.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 12:00:00 | 8.78 | 8.82 | 8.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 12:30:00 | 8.79 | 8.81 | 8.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:00:00 | 8.79 | 8.81 | 8.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 14:15:00 | 9.54 | 8.96 | 8.97 | SL hit (close>static) qty=1.00 sl=9.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 14:15:00 | 9.54 | 8.96 | 8.97 | SL hit (close>static) qty=1.00 sl=9.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 14:15:00 | 9.54 | 8.96 | 8.97 | SL hit (close>static) qty=1.00 sl=9.20 alert=retest2 |

### Cycle 27 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 9.58 | 9.08 | 9.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 12:15:00 | 9.66 | 9.43 | 9.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 14:15:00 | 9.53 | 9.58 | 9.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 14:30:00 | 9.50 | 9.58 | 9.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 9.54 | 9.57 | 9.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 9.71 | 9.57 | 9.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-13 12:15:00 | 10.68 | 10.44 | 10.21 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 10.63 | 10.75 | 10.75 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 10.84 | 10.76 | 10.75 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 14:15:00 | 10.69 | 10.75 | 10.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 15:15:00 | 10.67 | 10.73 | 10.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 10.31 | 10.16 | 10.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 10.31 | 10.16 | 10.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 10.31 | 10.16 | 10.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 12:15:00 | 9.99 | 10.13 | 10.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:45:00 | 9.96 | 10.03 | 10.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 10.19 | 10.14 | 10.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 10.19 | 10.14 | 10.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 11:15:00 | 10.19 | 10.14 | 10.14 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 10.08 | 10.13 | 10.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 10.04 | 10.10 | 10.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 11:15:00 | 10.11 | 10.09 | 10.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 11:15:00 | 10.11 | 10.09 | 10.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 10.11 | 10.09 | 10.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:00:00 | 10.11 | 10.09 | 10.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 10.09 | 10.09 | 10.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:45:00 | 10.09 | 10.09 | 10.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 10.24 | 10.08 | 10.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 10.24 | 10.08 | 10.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 10.11 | 10.08 | 10.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:30:00 | 10.06 | 10.07 | 10.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 10.17 | 10.08 | 10.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 10.17 | 10.08 | 10.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 09:15:00 | 10.36 | 10.16 | 10.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 10:15:00 | 10.32 | 10.43 | 10.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 10:15:00 | 10.32 | 10.43 | 10.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 10.32 | 10.43 | 10.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:45:00 | 10.26 | 10.43 | 10.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 10.56 | 10.46 | 10.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 13:30:00 | 10.62 | 10.48 | 10.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 15:00:00 | 10.67 | 10.52 | 10.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 10:45:00 | 10.64 | 10.57 | 10.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 13:15:00 | 10.29 | 10.51 | 10.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 13:15:00 | 10.29 | 10.51 | 10.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 13:15:00 | 10.29 | 10.51 | 10.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 10.29 | 10.51 | 10.52 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 10.78 | 10.54 | 10.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 09:15:00 | 10.93 | 10.75 | 10.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 12:15:00 | 11.46 | 11.51 | 11.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 13:00:00 | 11.46 | 11.51 | 11.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 11.34 | 11.43 | 11.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 11.33 | 11.43 | 11.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 11.26 | 11.40 | 11.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 11.26 | 11.40 | 11.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 11.30 | 11.38 | 11.31 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 11.18 | 11.28 | 11.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 11.12 | 11.25 | 11.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 11.32 | 11.22 | 11.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 10:15:00 | 11.32 | 11.22 | 11.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 11.32 | 11.22 | 11.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:30:00 | 11.37 | 11.22 | 11.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 11:15:00 | 11.45 | 11.27 | 11.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 11.62 | 11.37 | 11.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 09:15:00 | 11.65 | 11.67 | 11.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 09:30:00 | 11.62 | 11.67 | 11.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 12.07 | 12.08 | 11.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 12.07 | 12.08 | 11.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 12.05 | 12.06 | 11.98 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 11.87 | 11.95 | 11.96 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 11.99 | 11.96 | 11.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 10:15:00 | 12.02 | 11.98 | 11.97 | Break + close above crossover candle high |

### Cycle 40 — SELL (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 13:15:00 | 11.06 | 11.98 | 12.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-31 14:15:00 | 10.71 | 11.73 | 11.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 10:15:00 | 11.70 | 11.51 | 11.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 10:15:00 | 11.70 | 11.51 | 11.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 11.70 | 11.51 | 11.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:45:00 | 11.76 | 11.51 | 11.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 11.52 | 11.52 | 11.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:45:00 | 11.63 | 11.52 | 11.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 11.67 | 11.51 | 11.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 11.67 | 11.51 | 11.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 11.64 | 11.54 | 11.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 11.99 | 11.54 | 11.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 11.87 | 11.60 | 11.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 11.93 | 11.60 | 11.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 11.68 | 11.62 | 11.68 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 11.84 | 11.73 | 11.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 11.93 | 11.77 | 11.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 11:15:00 | 11.67 | 11.76 | 11.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 11:15:00 | 11.67 | 11.76 | 11.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 11.67 | 11.76 | 11.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:30:00 | 11.68 | 11.76 | 11.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 11.58 | 11.72 | 11.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:00:00 | 11.58 | 11.72 | 11.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 11.47 | 11.67 | 11.70 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 09:15:00 | 11.86 | 11.58 | 11.57 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 12:15:00 | 11.42 | 11.55 | 11.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 11.25 | 11.49 | 11.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 11.31 | 11.30 | 11.40 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 13:15:00 | 11.24 | 11.30 | 11.40 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 14:15:00 | 11.24 | 11.30 | 11.39 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 11.23 | 11.28 | 11.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:30:00 | 11.28 | 11.28 | 11.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 11.15 | 10.98 | 11.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 11.15 | 10.98 | 11.11 | SL hit (close>ema400) qty=1.00 sl=11.11 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 11.15 | 10.98 | 11.11 | SL hit (close>ema400) qty=1.00 sl=11.11 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 11.15 | 10.98 | 11.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 11.14 | 11.01 | 11.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 11.05 | 11.05 | 11.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 11.07 | 11.06 | 11.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 10.50 | 10.64 | 10.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 10.52 | 10.64 | 10.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 10:15:00 | 9.95 | 10.26 | 10.48 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-21 10:15:00 | 9.96 | 10.26 | 10.48 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 45 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 11.05 | 10.15 | 10.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 11.32 | 10.39 | 10.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 10.92 | 10.92 | 10.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 10.92 | 10.92 | 10.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 11.06 | 10.95 | 10.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 10.88 | 10.95 | 10.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 10.65 | 10.86 | 10.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 10.65 | 10.86 | 10.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 10.54 | 10.79 | 10.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 10.54 | 10.79 | 10.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 11.20 | 11.30 | 11.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 11.20 | 11.30 | 11.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 11.17 | 11.27 | 11.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 11.12 | 11.27 | 11.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 11.26 | 11.27 | 11.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:45:00 | 11.16 | 11.27 | 11.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 11.26 | 11.27 | 11.19 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 11:15:00 | 11.11 | 11.16 | 11.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 10.94 | 11.12 | 11.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 11.14 | 11.10 | 11.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 11.14 | 11.10 | 11.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 11.14 | 11.10 | 11.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 11.14 | 11.10 | 11.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 11.14 | 11.11 | 11.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 11.51 | 11.11 | 11.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 11.46 | 11.18 | 11.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 11.57 | 11.26 | 11.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 12:15:00 | 11.50 | 11.50 | 11.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 13:00:00 | 11.50 | 11.50 | 11.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 11.47 | 11.49 | 11.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:45:00 | 11.66 | 11.55 | 11.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 11.40 | 11.54 | 11.54 | SL hit (close<static) qty=1.00 sl=11.42 alert=retest2 |

### Cycle 48 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 11.40 | 11.51 | 11.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 11.31 | 11.42 | 11.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 11.44 | 11.40 | 11.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 11:15:00 | 11.44 | 11.40 | 11.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 11.44 | 11.40 | 11.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:00:00 | 11.44 | 11.40 | 11.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 11.45 | 11.41 | 11.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:45:00 | 11.45 | 11.41 | 11.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 11.46 | 11.42 | 11.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:45:00 | 11.47 | 11.42 | 11.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 11.43 | 11.42 | 11.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 11.37 | 11.42 | 11.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 11.36 | 11.41 | 11.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 12:00:00 | 11.31 | 11.38 | 11.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 11.57 | 11.41 | 11.42 | SL hit (close>static) qty=1.00 sl=11.46 alert=retest2 |

### Cycle 49 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 11.58 | 11.45 | 11.43 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 11.31 | 11.44 | 11.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 11.17 | 11.39 | 11.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 10.95 | 10.93 | 11.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 09:15:00 | 10.94 | 10.93 | 11.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 10.92 | 10.93 | 11.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:30:00 | 10.83 | 10.89 | 10.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 11.11 | 10.88 | 10.94 | SL hit (close>static) qty=1.00 sl=11.08 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:15:00 | 10.83 | 10.89 | 10.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:30:00 | 10.82 | 10.85 | 10.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 10.29 | 10.66 | 10.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 10.28 | 10.66 | 10.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 10.31 | 10.08 | 10.23 | SL hit (close>ema200) qty=0.50 sl=10.08 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 10.31 | 10.08 | 10.23 | SL hit (close>ema200) qty=0.50 sl=10.08 alert=retest2 |

### Cycle 51 — BUY (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 15:15:00 | 9.41 | 9.40 | 9.40 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 9.15 | 9.35 | 9.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 9.08 | 9.24 | 9.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 9.23 | 9.13 | 9.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 9.23 | 9.13 | 9.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 9.23 | 9.13 | 9.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 9.23 | 9.13 | 9.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 9.42 | 9.19 | 9.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 9.42 | 9.19 | 9.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 9.35 | 9.22 | 9.26 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 13:15:00 | 9.41 | 9.29 | 9.28 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 8.98 | 9.24 | 9.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 8.92 | 9.18 | 9.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 8.89 | 8.86 | 8.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 8.91 | 8.86 | 8.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 9.14 | 8.94 | 8.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 9.14 | 8.94 | 8.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 9.13 | 8.98 | 9.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 9.14 | 8.98 | 9.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 9.09 | 9.02 | 9.02 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 8.96 | 9.01 | 9.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 8.85 | 8.96 | 8.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 8.80 | 8.71 | 8.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 8.80 | 8.71 | 8.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 8.80 | 8.71 | 8.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 8.74 | 8.71 | 8.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 8.75 | 8.71 | 8.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 8.75 | 8.77 | 8.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 8.75 | 8.77 | 8.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 8.30 | 8.62 | 8.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 8.31 | 8.62 | 8.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 8.31 | 8.62 | 8.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 8.31 | 8.62 | 8.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 8.62 | 8.51 | 8.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 8.62 | 8.51 | 8.62 | SL hit (close>ema200) qty=0.50 sl=8.51 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 8.62 | 8.51 | 8.62 | SL hit (close>ema200) qty=0.50 sl=8.51 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 8.62 | 8.51 | 8.62 | SL hit (close>ema200) qty=0.50 sl=8.51 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 8.62 | 8.51 | 8.62 | SL hit (close>ema200) qty=0.50 sl=8.51 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 8.62 | 8.51 | 8.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 8.54 | 8.51 | 8.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 8.40 | 8.51 | 8.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 8.46 | 8.50 | 8.59 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 8.77 | 8.63 | 8.62 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 8.53 | 8.61 | 8.61 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 15:15:00 | 8.63 | 8.62 | 8.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 9.02 | 8.70 | 8.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 9.11 | 9.12 | 9.01 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 9.28 | 9.12 | 9.01 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 9.13 | 9.20 | 9.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 9.15 | 9.20 | 9.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 9.16 | 9.20 | 9.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 13:15:00 | 9.51 | 9.58 | 9.53 | SL hit (close<ema400) qty=1.00 sl=9.53 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 9.52 | 9.54 | 9.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 9.52 | 9.54 | 9.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 12:15:00 | 9.52 | 9.54 | 9.55 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 14:15:00 | 9.58 | 9.55 | 9.54 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 9.45 | 9.53 | 9.54 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 9.66 | 9.54 | 9.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 10.05 | 9.72 | 9.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 10.04 | 10.19 | 10.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 10.04 | 10.19 | 10.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 10.04 | 10.19 | 10.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:30:00 | 10.13 | 10.19 | 10.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 10.16 | 10.19 | 10.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:30:00 | 10.23 | 10.18 | 10.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 10.77 | 10.19 | 10.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-05 09:15:00 | 11.25 | 10.56 | 10.37 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 10:45:00 | 7.01 | 2025-05-12 11:15:00 | 6.99 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-05-14 09:15:00 | 7.01 | 2025-05-19 13:15:00 | 6.66 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2025-06-05 14:30:00 | 6.81 | 2025-06-06 12:15:00 | 6.90 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-06-06 09:15:00 | 6.81 | 2025-06-06 12:15:00 | 6.90 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-06-16 15:15:00 | 6.67 | 2025-06-19 13:15:00 | 6.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 09:45:00 | 6.68 | 2025-06-19 13:15:00 | 6.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 10:45:00 | 6.68 | 2025-06-19 13:15:00 | 6.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:15:00 | 6.66 | 2025-06-19 15:15:00 | 6.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-16 15:15:00 | 6.67 | 2025-06-20 10:15:00 | 6.48 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2025-06-17 09:45:00 | 6.68 | 2025-06-20 10:15:00 | 6.48 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2025-06-17 10:45:00 | 6.68 | 2025-06-20 10:15:00 | 6.48 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2025-06-17 11:15:00 | 6.66 | 2025-06-20 10:15:00 | 6.48 | STOP_HIT | 0.50 | 2.70% |
| SELL | retest2 | 2025-06-18 11:45:00 | 6.59 | 2025-06-23 11:15:00 | 6.60 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-06-18 13:15:00 | 6.59 | 2025-06-23 11:15:00 | 6.60 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-07-03 10:15:00 | 7.66 | 2025-07-04 11:15:00 | 7.37 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2025-07-09 13:30:00 | 7.31 | 2025-07-14 09:15:00 | 7.44 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-07-10 10:15:00 | 7.30 | 2025-07-14 09:15:00 | 7.44 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-07-10 13:15:00 | 7.31 | 2025-07-14 09:15:00 | 7.44 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-07-11 09:15:00 | 7.29 | 2025-07-14 09:15:00 | 7.44 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-07-11 10:30:00 | 7.15 | 2025-07-14 09:15:00 | 7.44 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest2 | 2025-07-24 11:15:00 | 7.43 | 2025-07-30 10:15:00 | 7.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 10:30:00 | 7.34 | 2025-07-31 09:15:00 | 6.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 11:15:00 | 7.43 | 2025-08-01 14:15:00 | 6.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-28 10:30:00 | 7.34 | 2025-08-01 14:15:00 | 6.61 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-22 13:30:00 | 7.09 | 2025-08-26 13:15:00 | 6.71 | STOP_HIT | 1.00 | -5.36% |
| SELL | retest2 | 2025-08-29 12:15:00 | 6.62 | 2025-09-04 09:15:00 | 6.76 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-09-16 12:00:00 | 8.05 | 2025-09-17 13:15:00 | 7.73 | STOP_HIT | 1.00 | -3.98% |
| BUY | retest2 | 2025-09-16 13:00:00 | 8.04 | 2025-09-17 13:15:00 | 7.73 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-09-30 10:45:00 | 8.19 | 2025-10-01 12:15:00 | 8.39 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-09-30 11:30:00 | 8.18 | 2025-10-01 12:15:00 | 8.39 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-09-30 13:00:00 | 8.15 | 2025-10-01 12:15:00 | 8.39 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-10-07 10:15:00 | 8.60 | 2025-10-13 12:15:00 | 8.79 | STOP_HIT | 1.00 | 2.21% |
| SELL | retest2 | 2025-11-03 12:00:00 | 8.78 | 2025-11-03 14:15:00 | 9.54 | STOP_HIT | 1.00 | -8.66% |
| SELL | retest2 | 2025-11-03 12:30:00 | 8.79 | 2025-11-03 14:15:00 | 9.54 | STOP_HIT | 1.00 | -8.53% |
| SELL | retest2 | 2025-11-03 13:00:00 | 8.79 | 2025-11-03 14:15:00 | 9.54 | STOP_HIT | 1.00 | -8.53% |
| BUY | retest2 | 2025-11-11 09:15:00 | 9.71 | 2025-11-13 12:15:00 | 10.68 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-24 12:15:00 | 9.99 | 2025-11-27 11:15:00 | 10.19 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-11-25 09:45:00 | 9.96 | 2025-11-27 11:15:00 | 10.19 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-12-01 11:30:00 | 10.06 | 2025-12-02 10:15:00 | 10.17 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-12-04 13:30:00 | 10.62 | 2025-12-08 13:15:00 | 10.29 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2025-12-04 15:00:00 | 10.67 | 2025-12-08 13:15:00 | 10.29 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2025-12-05 10:45:00 | 10.64 | 2025-12-08 13:15:00 | 10.29 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest1 | 2026-01-12 13:15:00 | 11.24 | 2026-01-14 10:15:00 | 11.15 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest1 | 2026-01-12 14:15:00 | 11.24 | 2026-01-14 10:15:00 | 11.15 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2026-01-14 14:15:00 | 11.05 | 2026-01-20 09:15:00 | 10.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 11.07 | 2026-01-20 09:15:00 | 10.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 14:15:00 | 11.05 | 2026-01-21 10:15:00 | 9.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 11.07 | 2026-01-21 10:15:00 | 9.96 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-11 12:45:00 | 11.66 | 2026-02-13 09:15:00 | 11.40 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-02-17 12:00:00 | 11.31 | 2026-02-18 09:15:00 | 11.57 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2026-02-25 12:30:00 | 10.83 | 2026-02-26 09:15:00 | 11.11 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2026-02-26 12:15:00 | 10.83 | 2026-03-02 09:15:00 | 10.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:30:00 | 10.82 | 2026-03-02 09:15:00 | 10.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:15:00 | 10.83 | 2026-03-05 11:15:00 | 10.31 | STOP_HIT | 0.50 | 4.80% |
| SELL | retest2 | 2026-02-27 09:30:00 | 10.82 | 2026-03-05 11:15:00 | 10.31 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2026-04-01 10:15:00 | 8.74 | 2026-04-02 09:15:00 | 8.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:45:00 | 8.75 | 2026-04-02 09:15:00 | 8.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 13:30:00 | 8.75 | 2026-04-02 09:15:00 | 8.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 14:15:00 | 8.75 | 2026-04-02 09:15:00 | 8.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 8.74 | 2026-04-02 14:15:00 | 8.62 | STOP_HIT | 0.50 | 1.37% |
| SELL | retest2 | 2026-04-01 10:45:00 | 8.75 | 2026-04-02 14:15:00 | 8.62 | STOP_HIT | 0.50 | 1.49% |
| SELL | retest2 | 2026-04-01 13:30:00 | 8.75 | 2026-04-02 14:15:00 | 8.62 | STOP_HIT | 0.50 | 1.49% |
| SELL | retest2 | 2026-04-01 14:15:00 | 8.75 | 2026-04-02 14:15:00 | 8.62 | STOP_HIT | 0.50 | 1.49% |
| BUY | retest1 | 2026-04-10 09:15:00 | 9.28 | 2026-04-20 13:15:00 | 9.51 | STOP_HIT | 1.00 | 2.48% |
| BUY | retest2 | 2026-04-13 10:15:00 | 9.15 | 2026-04-22 12:15:00 | 9.52 | STOP_HIT | 1.00 | 4.04% |
| BUY | retest2 | 2026-04-13 10:45:00 | 9.16 | 2026-04-22 12:15:00 | 9.52 | STOP_HIT | 1.00 | 3.93% |
| BUY | retest2 | 2026-04-30 13:30:00 | 10.23 | 2026-05-05 09:15:00 | 11.25 | TARGET_HIT | 1.00 | 10.00% |
