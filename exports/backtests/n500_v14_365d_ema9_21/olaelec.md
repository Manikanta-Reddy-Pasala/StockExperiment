# Ola Electric Mobility Ltd. (OLAELEC)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 36.01
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 53 |
| ALERT1 | 37 |
| ALERT2 | 36 |
| ALERT2_SKIP | 20 |
| ALERT3 | 90 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 12 |
| TARGET_HIT | 14 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 24
- **Target hits / Stop hits / Partials:** 14 / 31 / 12
- **Avg / median % per leg:** 2.58% / 4.65%
- **Sum % (uncompounded):** 147.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 9 | 69.2% | 9 | 4 | 0 | 5.53% | 71.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 9 | 69.2% | 9 | 4 | 0 | 5.53% | 71.8% |
| SELL (all) | 44 | 24 | 54.5% | 5 | 27 | 12 | 1.71% | 75.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 44 | 24 | 54.5% | 5 | 27 | 12 | 1.71% | 75.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 57 | 33 | 57.9% | 14 | 31 | 12 | 2.58% | 147.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 49.18 | 47.75 | 47.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 49.66 | 48.75 | 48.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 14:15:00 | 52.15 | 52.28 | 51.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 15:00:00 | 52.15 | 52.28 | 51.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 51.98 | 52.59 | 52.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 51.90 | 52.59 | 52.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 52.14 | 52.50 | 52.30 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 51.35 | 52.06 | 52.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 51.18 | 51.88 | 52.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 51.35 | 51.34 | 51.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 15:00:00 | 51.35 | 51.34 | 51.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 51.60 | 51.40 | 51.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:15:00 | 51.60 | 51.40 | 51.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 51.40 | 51.40 | 51.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:15:00 | 51.26 | 51.40 | 51.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:00:00 | 51.27 | 51.38 | 51.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:45:00 | 51.28 | 51.36 | 51.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:15:00 | 51.15 | 51.40 | 51.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 51.46 | 51.41 | 51.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 51.46 | 51.41 | 51.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-23 11:15:00 | 52.89 | 51.71 | 51.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-23 11:15:00 | 52.89 | 51.71 | 51.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-23 11:15:00 | 52.89 | 51.71 | 51.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-23 11:15:00 | 52.89 | 51.71 | 51.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 52.89 | 51.71 | 51.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 54.30 | 52.23 | 51.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 14:15:00 | 52.55 | 52.68 | 52.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 15:15:00 | 52.45 | 52.68 | 52.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 52.45 | 52.64 | 52.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 52.43 | 52.64 | 52.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 52.41 | 52.59 | 52.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:30:00 | 53.14 | 52.77 | 52.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 13:00:00 | 53.25 | 52.86 | 52.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 12:45:00 | 53.19 | 52.92 | 52.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 50.35 | 52.54 | 52.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 50.35 | 52.54 | 52.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 50.35 | 52.54 | 52.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 50.35 | 52.54 | 52.67 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 53.95 | 51.89 | 51.88 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 49.96 | 51.66 | 51.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 49.80 | 51.04 | 51.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 49.95 | 49.95 | 50.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 13:45:00 | 49.85 | 49.95 | 50.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 49.91 | 49.94 | 50.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 11:15:00 | 49.57 | 49.88 | 50.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:30:00 | 49.58 | 49.76 | 50.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 09:45:00 | 49.42 | 49.29 | 49.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 13:15:00 | 47.10 | 48.04 | 48.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 47.09 | 47.59 | 48.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 46.95 | 47.59 | 48.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 15:15:00 | 47.12 | 47.11 | 47.62 | SL hit (close>ema200) qty=0.50 sl=47.11 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 15:15:00 | 47.12 | 47.11 | 47.62 | SL hit (close>ema200) qty=0.50 sl=47.11 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 15:15:00 | 47.12 | 47.11 | 47.62 | SL hit (close>ema200) qty=0.50 sl=47.11 alert=retest2 |

### Cycle 7 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 43.11 | 40.87 | 40.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 46.70 | 42.04 | 41.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 13:15:00 | 44.90 | 44.98 | 43.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 14:00:00 | 44.90 | 44.98 | 43.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 44.00 | 44.65 | 43.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 43.56 | 44.65 | 43.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 42.73 | 44.27 | 43.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 42.82 | 44.27 | 43.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 42.97 | 44.01 | 43.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:45:00 | 43.31 | 43.53 | 43.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 14:15:00 | 42.39 | 43.30 | 43.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 14:15:00 | 42.39 | 43.30 | 43.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 15:15:00 | 42.13 | 43.07 | 43.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 12:15:00 | 42.76 | 42.50 | 42.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 12:15:00 | 42.76 | 42.50 | 42.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 42.76 | 42.50 | 42.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 41.56 | 42.31 | 42.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 12:15:00 | 41.67 | 42.09 | 42.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 13:15:00 | 42.42 | 41.21 | 41.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 13:15:00 | 42.42 | 41.21 | 41.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 42.42 | 41.21 | 41.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 43.36 | 42.12 | 41.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 42.53 | 42.58 | 42.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 15:00:00 | 42.53 | 42.58 | 42.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 41.75 | 42.40 | 42.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 41.75 | 42.40 | 42.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 41.68 | 42.25 | 42.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 41.70 | 42.25 | 42.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 41.43 | 41.91 | 41.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 14:15:00 | 41.26 | 41.78 | 41.87 | Break + close below crossover candle low |

### Cycle 11 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 43.02 | 41.93 | 41.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 10:15:00 | 43.14 | 42.17 | 42.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 42.00 | 42.28 | 42.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 12:15:00 | 42.00 | 42.28 | 42.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 42.00 | 42.28 | 42.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 42.00 | 42.28 | 42.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 41.25 | 42.07 | 42.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 14:00:00 | 41.25 | 42.07 | 42.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 41.15 | 41.89 | 41.95 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 42.03 | 41.81 | 41.79 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 41.52 | 41.83 | 41.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 41.23 | 41.64 | 41.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 12:15:00 | 41.48 | 41.47 | 41.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 13:00:00 | 41.48 | 41.47 | 41.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 41.11 | 41.11 | 41.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 41.44 | 41.11 | 41.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 40.90 | 40.94 | 41.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:30:00 | 40.48 | 40.80 | 40.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 10:15:00 | 41.60 | 40.26 | 40.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 41.60 | 40.26 | 40.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 10:15:00 | 42.49 | 41.13 | 40.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 14:15:00 | 41.09 | 41.18 | 40.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 15:00:00 | 41.09 | 41.18 | 40.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 41.13 | 41.50 | 41.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 41.13 | 41.50 | 41.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 41.16 | 41.43 | 41.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 13:15:00 | 41.36 | 41.43 | 41.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:15:00 | 41.29 | 41.38 | 41.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 15:15:00 | 41.35 | 41.33 | 41.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-19 13:15:00 | 45.50 | 43.37 | 42.35 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-19 13:15:00 | 45.42 | 43.37 | 42.35 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-19 13:15:00 | 45.49 | 43.37 | 42.35 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 46.99 | 47.96 | 48.01 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 48.51 | 48.07 | 48.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 09:15:00 | 50.82 | 48.91 | 48.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 12:15:00 | 54.74 | 54.95 | 53.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-29 12:45:00 | 55.05 | 54.95 | 53.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 61.34 | 64.25 | 63.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:30:00 | 61.48 | 64.25 | 63.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 60.61 | 63.52 | 63.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 60.61 | 63.52 | 63.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 59.81 | 62.78 | 63.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 09:15:00 | 58.50 | 60.12 | 60.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 15:15:00 | 59.14 | 59.09 | 59.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 15:15:00 | 59.14 | 59.09 | 59.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 59.14 | 59.09 | 59.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 58.54 | 59.09 | 59.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 15:15:00 | 58.45 | 57.60 | 58.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 14:15:00 | 60.75 | 58.53 | 58.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 14:15:00 | 60.75 | 58.53 | 58.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 60.75 | 58.53 | 58.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 15:15:00 | 60.77 | 58.97 | 58.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 14:15:00 | 59.30 | 59.67 | 59.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 14:15:00 | 59.30 | 59.67 | 59.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 59.30 | 59.67 | 59.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:45:00 | 59.40 | 59.67 | 59.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 58.50 | 59.38 | 59.12 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 58.46 | 58.89 | 58.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 09:15:00 | 57.85 | 58.63 | 58.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 57.90 | 57.88 | 58.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 15:00:00 | 57.90 | 57.88 | 58.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 58.40 | 57.77 | 58.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:00:00 | 58.40 | 57.77 | 58.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 58.12 | 57.84 | 58.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 12:30:00 | 57.90 | 57.93 | 58.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 13:30:00 | 57.80 | 57.93 | 58.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:00:00 | 57.92 | 57.93 | 58.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:30:00 | 57.91 | 57.69 | 57.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 57.99 | 57.75 | 57.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 58.00 | 57.75 | 57.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-23 12:15:00 | 58.74 | 57.95 | 57.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 12:15:00 | 58.74 | 57.95 | 57.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 12:15:00 | 58.74 | 57.95 | 57.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 12:15:00 | 58.74 | 57.95 | 57.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 58.74 | 57.95 | 57.88 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 57.30 | 57.84 | 57.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 56.86 | 57.52 | 57.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 54.32 | 54.09 | 54.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 15:00:00 | 54.32 | 54.09 | 54.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 55.76 | 54.48 | 54.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 55.60 | 54.48 | 54.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 56.86 | 54.96 | 55.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 56.86 | 54.96 | 55.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 11:15:00 | 56.67 | 55.30 | 55.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 09:15:00 | 57.30 | 56.48 | 55.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 12:15:00 | 55.99 | 56.50 | 56.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 12:15:00 | 55.99 | 56.50 | 56.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 55.99 | 56.50 | 56.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 55.99 | 56.50 | 56.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 55.40 | 56.28 | 56.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:30:00 | 55.50 | 56.28 | 56.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 56.18 | 56.26 | 56.03 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 09:15:00 | 54.22 | 55.79 | 55.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 11:15:00 | 53.88 | 55.19 | 55.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 09:15:00 | 53.39 | 53.15 | 53.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 53.39 | 53.15 | 53.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 53.39 | 53.15 | 53.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 53.40 | 53.15 | 53.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 50.48 | 49.63 | 50.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 50.48 | 49.63 | 50.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 50.00 | 49.70 | 50.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 49.95 | 49.70 | 50.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 48.75 | 49.51 | 49.98 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 52.45 | 50.12 | 50.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 10:15:00 | 52.75 | 50.65 | 50.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 54.61 | 56.11 | 55.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 54.61 | 56.11 | 55.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 54.61 | 56.11 | 55.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 55.31 | 56.11 | 55.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 54.22 | 55.73 | 54.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:30:00 | 54.47 | 55.73 | 54.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 54.65 | 55.23 | 54.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:00:00 | 54.65 | 55.23 | 54.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 54.85 | 55.15 | 54.87 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 53.43 | 54.57 | 54.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 52.60 | 53.33 | 53.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 10:15:00 | 51.83 | 51.75 | 52.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 11:00:00 | 51.83 | 51.75 | 52.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 52.52 | 51.91 | 52.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 52.52 | 51.91 | 52.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 52.60 | 52.04 | 52.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 52.60 | 52.04 | 52.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 51.98 | 52.03 | 52.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 15:00:00 | 51.55 | 51.94 | 52.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 51.50 | 51.90 | 52.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 51.26 | 51.79 | 51.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 48.97 | 50.53 | 50.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 48.92 | 50.53 | 50.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 48.70 | 50.53 | 50.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 14:15:00 | 50.69 | 50.16 | 50.58 | SL hit (close>ema200) qty=0.50 sl=50.16 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 14:15:00 | 50.69 | 50.16 | 50.58 | SL hit (close>ema200) qty=0.50 sl=50.16 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 14:15:00 | 50.69 | 50.16 | 50.58 | SL hit (close>ema200) qty=0.50 sl=50.16 alert=retest2 |

### Cycle 27 — BUY (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 15:15:00 | 41.70 | 41.25 | 41.20 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 41.19 | 41.25 | 41.25 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 41.74 | 41.25 | 41.24 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 40.66 | 41.13 | 41.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 10:15:00 | 40.42 | 40.89 | 41.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 34.47 | 34.35 | 35.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 14:15:00 | 34.73 | 34.32 | 34.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 34.73 | 34.32 | 34.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 34.73 | 34.32 | 34.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 34.95 | 34.44 | 34.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 34.50 | 34.44 | 34.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 34.82 | 34.52 | 34.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:15:00 | 34.96 | 34.52 | 34.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 34.77 | 34.57 | 34.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 34.30 | 34.50 | 34.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 11:15:00 | 36.70 | 34.92 | 34.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 36.70 | 34.92 | 34.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 12:15:00 | 37.17 | 35.37 | 35.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 36.23 | 36.34 | 35.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 10:00:00 | 36.23 | 36.34 | 35.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 35.94 | 36.55 | 36.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 35.98 | 36.55 | 36.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 35.93 | 36.42 | 36.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 35.93 | 36.42 | 36.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 35.10 | 35.95 | 36.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 34.60 | 35.68 | 35.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 34.33 | 32.39 | 33.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 34.33 | 32.39 | 33.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 34.33 | 32.39 | 33.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 34.33 | 32.39 | 33.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 34.27 | 32.76 | 33.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 34.27 | 32.76 | 33.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 34.38 | 33.72 | 33.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 34.55 | 33.99 | 33.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 12:15:00 | 34.14 | 34.15 | 33.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 13:00:00 | 34.14 | 34.15 | 33.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 35.79 | 34.58 | 34.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 36.53 | 35.18 | 34.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:45:00 | 35.89 | 35.57 | 35.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 11:00:00 | 35.94 | 35.64 | 35.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 12:30:00 | 35.96 | 35.75 | 35.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 36.19 | 36.31 | 36.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:45:00 | 36.69 | 36.31 | 36.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 36.39 | 36.31 | 36.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 13:30:00 | 36.78 | 36.57 | 36.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-02 09:15:00 | 40.18 | 37.67 | 36.89 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-01-02 09:15:00 | 39.48 | 37.67 | 36.89 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-01-02 09:15:00 | 39.53 | 37.67 | 36.89 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-01-02 09:15:00 | 39.56 | 37.67 | 36.89 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-01-02 09:15:00 | 40.46 | 37.67 | 36.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 40.83 | 41.70 | 41.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 40.49 | 41.23 | 41.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 39.38 | 39.26 | 40.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 39.38 | 39.26 | 40.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 39.38 | 39.26 | 40.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 38.47 | 39.04 | 39.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 38.53 | 38.94 | 39.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 38.62 | 38.88 | 39.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:30:00 | 38.50 | 38.75 | 39.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 36.55 | 37.45 | 38.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 36.60 | 37.45 | 38.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 36.69 | 37.45 | 38.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 36.57 | 37.45 | 38.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 09:15:00 | 34.62 | 35.84 | 36.90 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 09:15:00 | 34.68 | 35.84 | 36.90 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 09:15:00 | 34.76 | 35.84 | 36.90 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 09:15:00 | 34.65 | 35.84 | 36.90 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 32.01 | 31.54 | 32.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 32.16 | 31.54 | 32.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 32.06 | 31.65 | 32.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:15:00 | 31.74 | 31.65 | 32.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:45:00 | 31.75 | 31.86 | 32.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 31.77 | 31.84 | 32.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 13:45:00 | 31.82 | 31.69 | 31.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 31.90 | 31.73 | 31.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:30:00 | 31.89 | 31.73 | 31.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 31.88 | 31.76 | 31.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 32.10 | 31.76 | 31.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 32.70 | 31.95 | 31.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 32.70 | 31.95 | 31.96 | SL hit (close>static) qty=1.00 sl=32.53 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 32.70 | 31.95 | 31.96 | SL hit (close>static) qty=1.00 sl=32.53 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 32.70 | 31.95 | 31.96 | SL hit (close>static) qty=1.00 sl=32.53 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 32.70 | 31.95 | 31.96 | SL hit (close>static) qty=1.00 sl=32.53 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 32.70 | 31.95 | 31.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 32.59 | 32.08 | 32.02 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 31.66 | 32.06 | 32.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 31.55 | 31.96 | 32.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 31.93 | 31.38 | 31.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 31.93 | 31.38 | 31.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 31.93 | 31.38 | 31.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 31.93 | 31.38 | 31.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 31.95 | 31.50 | 31.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 32.27 | 31.50 | 31.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 12:15:00 | 31.39 | 31.44 | 31.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 12:30:00 | 31.57 | 31.44 | 31.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 31.97 | 31.42 | 31.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 31.97 | 31.42 | 31.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 31.69 | 31.47 | 31.53 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2026-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 12:15:00 | 31.80 | 31.60 | 31.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 31.94 | 31.67 | 31.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 31.62 | 31.80 | 31.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 31.62 | 31.80 | 31.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 31.62 | 31.80 | 31.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 31.55 | 31.80 | 31.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 31.41 | 31.72 | 31.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 31.41 | 31.72 | 31.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 31.43 | 31.66 | 31.65 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 31.52 | 31.63 | 31.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 31.12 | 31.47 | 31.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 31.47 | 31.43 | 31.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 13:15:00 | 31.47 | 31.43 | 31.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 31.47 | 31.43 | 31.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 31.47 | 31.43 | 31.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 31.69 | 31.48 | 31.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 31.69 | 31.48 | 31.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 31.69 | 31.52 | 31.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 31.47 | 31.52 | 31.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 31.61 | 31.52 | 31.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:45:00 | 31.58 | 31.52 | 31.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 31.51 | 31.52 | 31.53 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 31.73 | 31.56 | 31.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 31.85 | 31.64 | 31.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 12:15:00 | 31.63 | 31.65 | 31.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 12:15:00 | 31.63 | 31.65 | 31.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 31.63 | 31.65 | 31.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:30:00 | 31.61 | 31.65 | 31.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 31.43 | 31.61 | 31.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:30:00 | 31.45 | 31.61 | 31.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2026-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 14:15:00 | 31.17 | 31.52 | 31.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 30.94 | 31.18 | 31.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 31.19 | 31.02 | 31.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 11:15:00 | 31.19 | 31.02 | 31.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 31.19 | 31.02 | 31.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 31.19 | 31.02 | 31.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 31.22 | 31.06 | 31.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:45:00 | 31.21 | 31.06 | 31.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 31.04 | 31.06 | 31.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 30.78 | 31.00 | 31.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 29.24 | 30.61 | 30.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-17 09:15:00 | 27.70 | 29.06 | 29.86 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 41 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 24.70 | 24.05 | 23.97 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 23.52 | 24.05 | 24.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 23.15 | 23.53 | 23.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 24.77 | 23.19 | 23.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 24.77 | 23.19 | 23.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 24.77 | 23.19 | 23.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 24.77 | 23.19 | 23.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 24.30 | 23.41 | 23.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 23.78 | 23.41 | 23.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 23.87 | 23.50 | 23.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 23.87 | 23.50 | 23.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 24.59 | 23.84 | 23.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 24.19 | 24.49 | 24.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 24.19 | 24.49 | 24.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 24.19 | 24.49 | 24.18 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 23.58 | 24.03 | 24.05 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 25.11 | 24.15 | 24.05 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 23.54 | 23.98 | 23.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 14:15:00 | 23.39 | 23.70 | 23.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 23.59 | 23.51 | 23.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 23.60 | 23.51 | 23.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 23.51 | 23.51 | 23.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:45:00 | 23.42 | 23.47 | 23.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 24.15 | 23.56 | 23.64 | SL hit (close>static) qty=1.00 sl=23.69 alert=retest2 |

### Cycle 47 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 23.96 | 23.74 | 23.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 09:15:00 | 24.22 | 23.86 | 23.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 23.68 | 24.00 | 23.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 23.68 | 24.00 | 23.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 23.68 | 24.00 | 23.92 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 23.67 | 23.85 | 23.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 22.88 | 23.55 | 23.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 24.24 | 23.54 | 23.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 24.24 | 23.54 | 23.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 24.24 | 23.54 | 23.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 24.41 | 23.54 | 23.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 24.50 | 23.73 | 23.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 24.63 | 23.73 | 23.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 25.29 | 24.05 | 23.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 25.85 | 24.41 | 24.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 28.65 | 28.94 | 28.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 15:15:00 | 28.65 | 28.94 | 28.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 28.65 | 28.94 | 28.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 29.40 | 28.94 | 28.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 32.34 | 30.63 | 29.64 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 09:15:00 | 38.68 | 39.54 | 39.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 11:15:00 | 38.28 | 39.13 | 39.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 13:15:00 | 38.02 | 37.95 | 38.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 14:00:00 | 38.02 | 37.95 | 38.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 38.48 | 38.06 | 38.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:45:00 | 38.48 | 38.06 | 38.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 38.37 | 38.12 | 38.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 38.04 | 38.12 | 38.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 37.73 | 38.04 | 38.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 37.45 | 38.04 | 38.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 12:15:00 | 35.58 | 36.62 | 37.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 36.80 | 36.32 | 36.93 | SL hit (close>ema200) qty=0.50 sl=36.32 alert=retest2 |

### Cycle 51 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 36.67 | 36.32 | 36.31 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 33.91 | 35.89 | 36.15 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 12:15:00 | 36.12 | 35.06 | 35.04 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-22 11:15:00 | 51.26 | 2025-05-23 11:15:00 | 52.89 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-05-22 13:00:00 | 51.27 | 2025-05-23 11:15:00 | 52.89 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-05-22 13:45:00 | 51.28 | 2025-05-23 11:15:00 | 52.89 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2025-05-23 10:15:00 | 51.15 | 2025-05-23 11:15:00 | 52.89 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2025-05-28 11:30:00 | 53.14 | 2025-05-30 09:15:00 | 50.35 | STOP_HIT | 1.00 | -5.25% |
| BUY | retest2 | 2025-05-28 13:00:00 | 53.25 | 2025-05-30 09:15:00 | 50.35 | STOP_HIT | 1.00 | -5.45% |
| BUY | retest2 | 2025-05-29 12:45:00 | 53.19 | 2025-05-30 09:15:00 | 50.35 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2025-06-05 11:15:00 | 49.57 | 2025-06-12 13:15:00 | 47.10 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2025-06-05 12:30:00 | 49.58 | 2025-06-13 09:15:00 | 47.09 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-06-10 09:45:00 | 49.42 | 2025-06-13 09:15:00 | 46.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-05 11:15:00 | 49.57 | 2025-06-13 15:15:00 | 47.12 | STOP_HIT | 0.50 | 4.94% |
| SELL | retest2 | 2025-06-05 12:30:00 | 49.58 | 2025-06-13 15:15:00 | 47.12 | STOP_HIT | 0.50 | 4.96% |
| SELL | retest2 | 2025-06-10 09:45:00 | 49.42 | 2025-06-13 15:15:00 | 47.12 | STOP_HIT | 0.50 | 4.65% |
| BUY | retest2 | 2025-07-16 13:45:00 | 43.31 | 2025-07-16 14:15:00 | 42.39 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-07-18 10:15:00 | 41.56 | 2025-07-23 13:15:00 | 42.42 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-07-18 12:15:00 | 41.67 | 2025-07-23 13:15:00 | 42.42 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-08-06 09:30:00 | 40.48 | 2025-08-13 10:15:00 | 41.60 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2025-08-18 13:15:00 | 41.36 | 2025-08-19 13:15:00 | 45.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-18 14:15:00 | 41.29 | 2025-08-19 13:15:00 | 45.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-18 15:15:00 | 41.35 | 2025-08-19 13:15:00 | 45.49 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-11 09:15:00 | 58.54 | 2025-09-15 14:15:00 | 60.75 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2025-09-12 15:15:00 | 58.45 | 2025-09-15 14:15:00 | 60.75 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2025-09-19 12:30:00 | 57.90 | 2025-09-23 12:15:00 | 58.74 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-19 13:30:00 | 57.80 | 2025-09-23 12:15:00 | 58.74 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-09-19 14:00:00 | 57.92 | 2025-09-23 12:15:00 | 58.74 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-09-23 10:30:00 | 57.91 | 2025-09-23 12:15:00 | 58.74 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-10-28 15:00:00 | 51.55 | 2025-11-03 09:15:00 | 48.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 09:15:00 | 51.50 | 2025-11-03 09:15:00 | 48.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 09:15:00 | 51.26 | 2025-11-03 09:15:00 | 48.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 15:00:00 | 51.55 | 2025-11-03 14:15:00 | 50.69 | STOP_HIT | 0.50 | 1.67% |
| SELL | retest2 | 2025-10-29 09:15:00 | 51.50 | 2025-11-03 14:15:00 | 50.69 | STOP_HIT | 0.50 | 1.57% |
| SELL | retest2 | 2025-10-30 09:15:00 | 51.26 | 2025-11-03 14:15:00 | 50.69 | STOP_HIT | 0.50 | 1.11% |
| SELL | retest2 | 2025-12-10 12:30:00 | 34.30 | 2025-12-11 11:15:00 | 36.70 | STOP_HIT | 1.00 | -7.00% |
| BUY | retest2 | 2025-12-26 09:15:00 | 36.53 | 2026-01-02 09:15:00 | 40.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-30 09:45:00 | 35.89 | 2026-01-02 09:15:00 | 39.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-30 11:00:00 | 35.94 | 2026-01-02 09:15:00 | 39.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-30 12:30:00 | 35.96 | 2026-01-02 09:15:00 | 39.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-01 13:30:00 | 36.78 | 2026-01-02 09:15:00 | 40.46 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 38.47 | 2026-01-19 09:15:00 | 36.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:00:00 | 38.53 | 2026-01-19 09:15:00 | 36.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 38.62 | 2026-01-19 09:15:00 | 36.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:30:00 | 38.50 | 2026-01-19 09:15:00 | 36.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 38.47 | 2026-01-20 09:15:00 | 34.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 11:00:00 | 38.53 | 2026-01-20 09:15:00 | 34.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 38.62 | 2026-01-20 09:15:00 | 34.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-16 09:30:00 | 38.50 | 2026-01-20 09:15:00 | 34.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-28 10:15:00 | 31.74 | 2026-01-30 09:15:00 | 32.70 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2026-01-28 13:45:00 | 31.75 | 2026-01-30 09:15:00 | 32.70 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2026-01-29 09:30:00 | 31.77 | 2026-01-30 09:15:00 | 32.70 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2026-01-29 13:45:00 | 31.82 | 2026-01-30 09:15:00 | 32.70 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-02-13 15:00:00 | 30.78 | 2026-02-16 09:15:00 | 29.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 15:00:00 | 30.78 | 2026-02-17 09:15:00 | 27.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-17 09:15:00 | 23.78 | 2026-03-17 09:15:00 | 23.87 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-03-24 14:45:00 | 23.42 | 2026-03-25 09:15:00 | 24.15 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2026-04-08 09:15:00 | 29.40 | 2026-04-09 09:15:00 | 32.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-23 10:15:00 | 37.45 | 2026-04-24 12:15:00 | 35.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 10:15:00 | 37.45 | 2026-04-27 09:15:00 | 36.80 | STOP_HIT | 0.50 | 1.74% |
