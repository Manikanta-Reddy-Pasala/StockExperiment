# Aditya Birla Lifestyle Brands Ltd. (ABLBL)

## Backtest Summary

- **Window:** 2025-06-23 09:15:00 → 2026-05-11 15:15:00 (1521 bars)
- **Last close:** 107.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 56 |
| ALERT1 | 43 |
| ALERT2 | 42 |
| ALERT2_SKIP | 27 |
| ALERT3 | 101 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 43 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 30
- **Target hits / Stop hits / Partials:** 2 / 43 / 7
- **Avg / median % per leg:** 0.54% / -0.68%
- **Sum % (uncompounded):** 27.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 4 | 22.2% | 1 | 17 | 0 | -1.12% | -20.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 4 | 22.2% | 1 | 17 | 0 | -1.12% | -20.2% |
| SELL (all) | 34 | 18 | 52.9% | 1 | 26 | 7 | 1.41% | 48.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 18 | 52.9% | 1 | 26 | 7 | 1.41% | 48.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 52 | 22 | 42.3% | 2 | 43 | 7 | 0.54% | 27.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 14:15:00 | 160.03 | 157.54 | 157.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 09:15:00 | 160.13 | 158.46 | 157.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 160.01 | 160.32 | 158.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 160.01 | 160.32 | 158.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 158.00 | 159.85 | 158.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:00:00 | 158.00 | 159.85 | 158.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 157.10 | 159.30 | 158.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:30:00 | 158.50 | 159.30 | 158.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 157.73 | 158.99 | 158.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 159.20 | 158.99 | 158.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:00:00 | 158.98 | 158.67 | 158.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:45:00 | 158.99 | 158.74 | 158.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 14:15:00 | 155.80 | 158.03 | 158.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 155.80 | 158.03 | 158.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 154.98 | 157.09 | 157.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 12:15:00 | 158.10 | 156.81 | 157.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 12:15:00 | 158.10 | 156.81 | 157.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 158.10 | 156.81 | 157.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 158.10 | 156.81 | 157.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 155.60 | 156.57 | 157.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:45:00 | 155.45 | 155.97 | 156.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 161.90 | 157.09 | 156.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 161.90 | 157.09 | 156.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 13:15:00 | 164.00 | 159.83 | 158.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 09:15:00 | 163.17 | 164.45 | 162.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 163.17 | 164.45 | 162.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 163.17 | 164.45 | 162.71 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 160.68 | 162.07 | 162.22 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 163.80 | 162.47 | 162.34 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 160.75 | 162.30 | 162.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 159.20 | 161.18 | 161.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 15:15:00 | 156.00 | 155.96 | 157.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 15:15:00 | 156.00 | 155.96 | 157.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 156.00 | 155.96 | 157.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 157.70 | 155.96 | 157.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 156.53 | 156.07 | 157.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 15:15:00 | 154.50 | 156.11 | 156.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 146.78 | 148.75 | 150.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-29 10:15:00 | 139.05 | 142.75 | 145.31 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 135.01 | 134.38 | 134.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 136.28 | 134.76 | 134.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 135.15 | 135.25 | 134.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 135.15 | 135.25 | 134.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 135.15 | 135.25 | 134.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 135.15 | 135.25 | 134.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 133.46 | 135.01 | 134.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 133.08 | 135.01 | 134.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 132.50 | 134.51 | 134.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 130.83 | 133.77 | 134.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 135.54 | 133.09 | 133.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 135.54 | 133.09 | 133.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 135.54 | 133.09 | 133.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 135.54 | 133.09 | 133.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 137.94 | 134.06 | 133.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 138.00 | 135.40 | 134.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 11:15:00 | 140.54 | 140.83 | 139.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:45:00 | 140.50 | 140.83 | 139.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 138.82 | 140.29 | 139.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:00:00 | 138.82 | 140.29 | 139.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 138.14 | 139.86 | 139.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:15:00 | 137.50 | 139.86 | 139.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 140.69 | 139.48 | 139.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:30:00 | 142.30 | 140.71 | 139.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 10:00:00 | 142.88 | 143.43 | 142.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 15:15:00 | 139.00 | 141.69 | 141.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 139.00 | 141.69 | 141.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 137.20 | 139.58 | 140.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 135.74 | 135.53 | 137.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 11:15:00 | 136.85 | 135.79 | 136.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 136.85 | 135.79 | 136.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:30:00 | 136.73 | 135.79 | 136.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 138.48 | 136.33 | 137.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:45:00 | 138.80 | 136.33 | 137.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 139.31 | 136.92 | 137.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 139.31 | 136.92 | 137.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 139.03 | 137.72 | 137.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 142.21 | 138.62 | 137.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 142.33 | 142.47 | 141.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 13:15:00 | 142.30 | 142.47 | 141.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 142.22 | 142.35 | 141.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 142.79 | 142.35 | 141.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 141.05 | 142.11 | 141.72 | SL hit (close<static) qty=1.00 sl=141.70 alert=retest2 |

### Cycle 12 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 144.92 | 145.49 | 145.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 144.36 | 145.16 | 145.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 145.74 | 145.19 | 145.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 145.74 | 145.19 | 145.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 145.74 | 145.19 | 145.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 146.79 | 145.19 | 145.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 146.41 | 145.43 | 145.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 146.41 | 145.43 | 145.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 145.37 | 145.42 | 145.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:30:00 | 145.77 | 145.42 | 145.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 145.25 | 145.39 | 145.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 14:15:00 | 144.88 | 145.35 | 145.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 15:15:00 | 145.65 | 145.42 | 145.42 | SL hit (close>static) qty=1.00 sl=145.57 alert=retest2 |

### Cycle 13 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 145.55 | 144.40 | 144.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 10:15:00 | 146.49 | 145.28 | 144.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 13:15:00 | 145.12 | 145.41 | 145.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 13:15:00 | 145.12 | 145.41 | 145.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 145.12 | 145.41 | 145.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 146.10 | 145.41 | 145.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 143.43 | 145.24 | 145.09 | SL hit (close<static) qty=1.00 sl=143.78 alert=retest2 |

### Cycle 14 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 142.96 | 144.79 | 144.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 11:15:00 | 141.88 | 144.21 | 144.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 13:15:00 | 142.66 | 141.98 | 142.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 13:15:00 | 142.66 | 141.98 | 142.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 142.66 | 141.98 | 142.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 13:15:00 | 140.87 | 141.97 | 142.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 13:45:00 | 140.81 | 141.67 | 142.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 133.83 | 135.93 | 137.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 133.77 | 135.93 | 137.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 133.70 | 133.28 | 135.12 | SL hit (close>ema200) qty=0.50 sl=133.28 alert=retest2 |

### Cycle 15 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 136.10 | 134.18 | 134.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 136.58 | 135.45 | 134.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 142.12 | 145.83 | 143.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 142.12 | 145.83 | 143.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 142.12 | 145.83 | 143.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 142.12 | 145.83 | 143.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 141.26 | 144.91 | 143.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 141.26 | 144.91 | 143.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 139.39 | 142.20 | 142.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 138.46 | 141.10 | 141.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 143.09 | 140.72 | 141.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 143.09 | 140.72 | 141.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 143.09 | 140.72 | 141.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:45:00 | 143.55 | 140.72 | 141.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 142.90 | 141.15 | 141.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 09:15:00 | 142.41 | 141.15 | 141.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 12:15:00 | 142.48 | 141.63 | 141.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 142.48 | 141.63 | 141.61 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 139.87 | 141.34 | 141.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 13:15:00 | 139.31 | 140.58 | 141.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 12:15:00 | 138.27 | 137.65 | 138.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 13:00:00 | 138.27 | 137.65 | 138.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 137.35 | 137.68 | 138.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:45:00 | 136.82 | 137.32 | 137.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 136.40 | 137.28 | 137.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 11:00:00 | 137.00 | 136.36 | 136.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 13:30:00 | 137.00 | 136.69 | 136.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 137.34 | 136.82 | 136.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 137.34 | 136.82 | 136.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-20 15:15:00 | 138.28 | 137.11 | 137.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 138.28 | 137.11 | 137.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 139.40 | 137.57 | 137.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 10:15:00 | 139.25 | 139.56 | 138.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 13:15:00 | 138.90 | 139.47 | 138.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 138.90 | 139.47 | 138.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 138.62 | 139.47 | 138.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 138.21 | 139.22 | 138.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 138.04 | 139.22 | 138.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 10:15:00 | 136.44 | 138.50 | 138.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 15:15:00 | 136.10 | 137.14 | 137.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 137.15 | 137.14 | 137.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 10:15:00 | 137.72 | 137.26 | 137.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 137.72 | 137.26 | 137.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 137.48 | 137.26 | 137.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 137.22 | 137.25 | 137.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:30:00 | 137.86 | 137.25 | 137.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 137.01 | 136.73 | 137.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:45:00 | 137.20 | 136.73 | 137.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 136.93 | 136.77 | 137.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:15:00 | 136.49 | 136.77 | 137.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 12:15:00 | 136.27 | 135.46 | 135.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 136.27 | 135.46 | 135.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 137.64 | 136.16 | 135.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 14:15:00 | 136.88 | 136.90 | 136.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 14:15:00 | 136.88 | 136.90 | 136.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 136.88 | 136.90 | 136.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 136.75 | 136.90 | 136.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 132.29 | 136.11 | 136.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 132.29 | 136.11 | 136.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 131.68 | 135.22 | 135.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 129.84 | 132.87 | 134.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 12:15:00 | 130.07 | 129.99 | 131.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 13:00:00 | 130.07 | 129.99 | 131.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 128.68 | 128.29 | 129.40 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 12:15:00 | 134.93 | 130.61 | 130.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 13:15:00 | 136.33 | 133.42 | 132.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 10:15:00 | 133.43 | 133.80 | 132.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:45:00 | 133.48 | 133.80 | 132.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 137.11 | 137.72 | 137.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 135.05 | 137.72 | 137.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 134.21 | 137.02 | 136.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 134.21 | 137.02 | 136.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 134.49 | 136.51 | 136.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 11:15:00 | 131.66 | 135.54 | 136.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 15:15:00 | 127.90 | 126.97 | 129.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 09:15:00 | 127.73 | 126.97 | 129.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 128.55 | 127.29 | 128.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:15:00 | 129.88 | 127.29 | 128.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 128.37 | 127.50 | 128.92 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 131.57 | 129.31 | 129.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 132.66 | 130.88 | 130.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 131.46 | 131.76 | 130.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:30:00 | 131.57 | 131.76 | 130.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 131.11 | 131.91 | 131.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 131.73 | 131.91 | 131.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 131.95 | 131.92 | 131.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 10:45:00 | 132.84 | 132.04 | 131.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:45:00 | 133.25 | 132.07 | 131.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 15:15:00 | 129.65 | 131.29 | 131.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 15:15:00 | 129.65 | 131.29 | 131.33 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 131.27 | 130.89 | 130.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 12:15:00 | 131.66 | 131.04 | 130.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 09:15:00 | 131.32 | 131.71 | 131.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 131.32 | 131.71 | 131.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 131.32 | 131.71 | 131.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 10:30:00 | 132.08 | 131.69 | 131.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 11:15:00 | 132.10 | 131.69 | 131.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 132.48 | 132.14 | 131.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 10:45:00 | 132.28 | 132.00 | 131.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 129.21 | 131.44 | 131.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 129.21 | 131.44 | 131.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 12:15:00 | 128.49 | 130.85 | 131.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 128.92 | 128.69 | 129.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 10:45:00 | 128.83 | 128.69 | 129.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 129.20 | 128.76 | 129.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:00:00 | 129.20 | 128.76 | 129.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 129.33 | 128.87 | 129.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 129.33 | 128.87 | 129.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 129.10 | 128.92 | 129.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 130.87 | 128.92 | 129.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 131.22 | 129.38 | 129.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 131.22 | 129.38 | 129.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 130.87 | 129.68 | 129.56 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 127.70 | 129.51 | 129.62 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 130.65 | 129.61 | 129.59 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 14:15:00 | 127.92 | 129.37 | 129.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 10:15:00 | 126.71 | 128.21 | 128.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 11:15:00 | 127.33 | 127.22 | 127.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 11:45:00 | 127.14 | 127.22 | 127.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 127.94 | 127.35 | 127.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 127.94 | 127.35 | 127.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 127.49 | 127.38 | 127.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 125.81 | 127.40 | 127.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 12:15:00 | 119.52 | 122.43 | 124.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 12:15:00 | 119.68 | 118.79 | 121.09 | SL hit (close>ema200) qty=0.50 sl=118.79 alert=retest2 |

### Cycle 33 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 122.70 | 120.99 | 120.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 13:15:00 | 123.77 | 121.55 | 121.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 13:15:00 | 122.07 | 122.52 | 121.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 13:15:00 | 122.07 | 122.52 | 121.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 122.07 | 122.52 | 121.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:15:00 | 124.34 | 122.52 | 121.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 123.27 | 122.67 | 122.09 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 121.17 | 122.10 | 122.20 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 123.27 | 122.34 | 122.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 123.92 | 122.65 | 122.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 125.48 | 126.14 | 124.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 125.48 | 126.14 | 124.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 125.68 | 125.93 | 124.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:45:00 | 124.70 | 125.93 | 124.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 124.93 | 125.69 | 124.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:45:00 | 125.00 | 125.69 | 124.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 126.39 | 125.83 | 125.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:30:00 | 125.13 | 125.83 | 125.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 126.00 | 125.87 | 125.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 125.06 | 125.89 | 125.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 125.10 | 125.74 | 125.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:30:00 | 125.54 | 125.74 | 125.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 125.24 | 125.64 | 125.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:45:00 | 125.18 | 125.64 | 125.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 125.28 | 125.57 | 125.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:30:00 | 125.34 | 125.57 | 125.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 125.25 | 125.51 | 125.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:15:00 | 124.72 | 125.51 | 125.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 124.40 | 125.29 | 125.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 124.40 | 125.29 | 125.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 123.95 | 125.02 | 125.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 14:15:00 | 123.57 | 124.31 | 124.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 118.39 | 117.43 | 118.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 11:15:00 | 118.39 | 117.43 | 118.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 118.39 | 117.43 | 118.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:00:00 | 118.39 | 117.43 | 118.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 118.82 | 117.71 | 118.36 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 120.99 | 119.09 | 118.89 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 15:15:00 | 118.18 | 118.88 | 118.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 09:15:00 | 117.34 | 118.57 | 118.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 118.69 | 118.46 | 118.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 11:15:00 | 118.69 | 118.46 | 118.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 118.69 | 118.46 | 118.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 118.69 | 118.46 | 118.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 118.68 | 118.50 | 118.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:45:00 | 118.76 | 118.50 | 118.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 118.48 | 118.50 | 118.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 117.74 | 118.53 | 118.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 111.85 | 115.31 | 116.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 12:15:00 | 115.11 | 114.41 | 115.57 | SL hit (close>ema200) qty=0.50 sl=114.41 alert=retest2 |

### Cycle 39 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 106.12 | 104.41 | 104.27 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 103.56 | 104.40 | 104.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 102.65 | 103.86 | 104.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 103.67 | 103.44 | 103.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 103.67 | 103.44 | 103.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 104.65 | 103.68 | 103.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 104.65 | 103.68 | 103.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 104.56 | 103.86 | 103.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 121.86 | 103.86 | 103.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 116.12 | 106.31 | 105.07 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 111.43 | 112.62 | 112.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 14:15:00 | 111.14 | 112.11 | 112.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 113.30 | 112.21 | 112.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 113.30 | 112.21 | 112.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 113.30 | 112.21 | 112.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 113.13 | 112.21 | 112.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 114.70 | 112.71 | 112.59 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 112.07 | 113.91 | 114.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 13:15:00 | 111.30 | 112.23 | 112.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 11:15:00 | 107.25 | 106.80 | 107.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 12:00:00 | 107.25 | 106.80 | 107.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 103.91 | 104.76 | 105.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:00:00 | 103.65 | 104.59 | 105.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 13:00:00 | 103.56 | 104.39 | 104.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 12:15:00 | 98.47 | 100.50 | 102.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 12:15:00 | 98.38 | 100.50 | 102.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 14:15:00 | 100.89 | 100.31 | 101.69 | SL hit (close>ema200) qty=0.50 sl=100.31 alert=retest2 |

### Cycle 45 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 103.20 | 101.23 | 101.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 103.35 | 101.66 | 101.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 101.95 | 102.22 | 101.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 101.95 | 102.22 | 101.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 101.95 | 102.22 | 101.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 101.95 | 102.22 | 101.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 102.00 | 102.17 | 101.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 98.56 | 102.17 | 101.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 98.25 | 101.39 | 101.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 97.30 | 99.46 | 100.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 100.00 | 99.00 | 99.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 100.00 | 99.00 | 99.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 100.00 | 99.00 | 99.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 100.00 | 99.00 | 99.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 100.30 | 99.26 | 99.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 100.30 | 99.26 | 99.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 100.80 | 99.57 | 100.02 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 103.06 | 100.68 | 100.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 104.09 | 101.37 | 100.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 102.18 | 102.70 | 101.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 102.18 | 102.70 | 101.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 102.18 | 102.70 | 101.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 102.18 | 102.70 | 101.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 103.00 | 102.76 | 102.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 100.80 | 102.76 | 102.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 100.58 | 102.32 | 101.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:15:00 | 99.93 | 102.32 | 101.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 100.64 | 101.99 | 101.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:45:00 | 101.26 | 101.82 | 101.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 14:15:00 | 101.29 | 101.66 | 101.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 101.29 | 101.66 | 101.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 98.30 | 100.92 | 101.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 12:15:00 | 96.55 | 95.89 | 97.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 13:00:00 | 96.55 | 95.89 | 97.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 97.45 | 96.20 | 97.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 97.45 | 96.20 | 97.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 97.36 | 96.43 | 97.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:00:00 | 97.15 | 97.11 | 97.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 12:15:00 | 98.75 | 97.44 | 97.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 98.75 | 97.44 | 97.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 99.40 | 97.83 | 97.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 96.19 | 97.84 | 97.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 96.19 | 97.84 | 97.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 96.19 | 97.84 | 97.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 96.28 | 97.84 | 97.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 95.45 | 97.36 | 97.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 95.07 | 96.90 | 97.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 11:15:00 | 95.39 | 94.85 | 95.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 11:15:00 | 95.39 | 94.85 | 95.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 95.39 | 94.85 | 95.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:45:00 | 95.01 | 94.85 | 95.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 94.10 | 92.64 | 93.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 93.92 | 92.64 | 93.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 94.10 | 92.93 | 93.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:45:00 | 94.09 | 92.93 | 93.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 93.70 | 93.37 | 93.32 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 91.81 | 93.02 | 93.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 91.19 | 92.47 | 92.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 13:15:00 | 91.09 | 90.63 | 91.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 14:00:00 | 91.09 | 90.63 | 91.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 93.36 | 90.50 | 91.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 93.36 | 90.50 | 91.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 92.40 | 90.88 | 91.26 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 93.10 | 91.72 | 91.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 93.50 | 92.47 | 92.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 90.85 | 92.15 | 91.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 90.85 | 92.15 | 91.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 90.85 | 92.15 | 91.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:15:00 | 90.59 | 92.15 | 91.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 91.40 | 92.00 | 91.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 91.76 | 91.92 | 91.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 100.94 | 97.92 | 96.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 106.12 | 107.15 | 107.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 105.66 | 106.85 | 107.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 106.11 | 105.55 | 106.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 106.11 | 105.55 | 106.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 106.11 | 105.55 | 106.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 106.30 | 105.55 | 106.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 106.07 | 105.65 | 106.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 106.47 | 105.65 | 106.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 105.52 | 105.63 | 106.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 15:15:00 | 105.26 | 105.84 | 106.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:45:00 | 105.33 | 105.64 | 105.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 105.15 | 105.59 | 105.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 15:00:00 | 105.01 | 105.13 | 105.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 103.30 | 104.67 | 105.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:30:00 | 104.25 | 104.67 | 105.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 104.00 | 103.57 | 104.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 104.00 | 103.57 | 104.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 103.90 | 103.64 | 104.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 104.00 | 103.64 | 104.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 102.80 | 103.47 | 104.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:30:00 | 102.72 | 103.32 | 103.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 104.80 | 102.63 | 103.00 | SL hit (close>static) qty=1.00 sl=104.12 alert=retest2 |

### Cycle 55 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 105.10 | 103.38 | 103.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 106.15 | 104.16 | 103.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 111.38 | 114.37 | 110.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 111.38 | 114.37 | 110.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 111.38 | 114.37 | 110.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 11:00:00 | 113.22 | 114.14 | 111.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-11 13:15:00 | 108.75 | 110.91 | 111.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-05-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-11 13:15:00 | 108.75 | 110.91 | 111.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-11 14:15:00 | 107.60 | 110.25 | 110.86 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-01 09:15:00 | 159.20 | 2025-07-01 14:15:00 | 155.80 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-07-01 12:00:00 | 158.98 | 2025-07-01 14:15:00 | 155.80 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-07-01 12:45:00 | 158.99 | 2025-07-01 14:15:00 | 155.80 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-07-03 13:45:00 | 155.45 | 2025-07-04 09:15:00 | 161.90 | STOP_HIT | 1.00 | -4.15% |
| SELL | retest2 | 2025-07-16 15:15:00 | 154.50 | 2025-07-25 11:15:00 | 146.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 15:15:00 | 154.50 | 2025-07-29 10:15:00 | 139.05 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-22 11:30:00 | 142.30 | 2025-08-26 15:15:00 | 139.00 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-08-26 10:00:00 | 142.88 | 2025-08-26 15:15:00 | 139.00 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2025-09-05 09:15:00 | 142.79 | 2025-09-05 10:15:00 | 141.05 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-09-08 09:30:00 | 143.45 | 2025-09-11 12:15:00 | 144.92 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2025-09-08 12:00:00 | 143.27 | 2025-09-11 12:15:00 | 144.92 | STOP_HIT | 1.00 | 1.15% |
| SELL | retest2 | 2025-09-12 14:15:00 | 144.88 | 2025-09-12 15:15:00 | 145.65 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-09-15 09:30:00 | 144.75 | 2025-09-16 09:15:00 | 146.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-09-15 10:45:00 | 144.85 | 2025-09-16 09:15:00 | 146.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-09-15 13:00:00 | 144.11 | 2025-09-16 09:15:00 | 146.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-09-16 11:15:00 | 143.00 | 2025-09-17 12:15:00 | 145.55 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-09-16 12:30:00 | 143.00 | 2025-09-17 12:15:00 | 145.55 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-09-18 14:15:00 | 146.10 | 2025-09-19 09:15:00 | 143.43 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-09-23 13:15:00 | 140.87 | 2025-09-26 09:15:00 | 133.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 13:45:00 | 140.81 | 2025-09-26 09:15:00 | 133.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 13:15:00 | 140.87 | 2025-09-29 09:15:00 | 133.70 | STOP_HIT | 0.50 | 5.09% |
| SELL | retest2 | 2025-09-23 13:45:00 | 140.81 | 2025-09-29 09:15:00 | 133.70 | STOP_HIT | 0.50 | 5.05% |
| SELL | retest2 | 2025-10-10 09:15:00 | 142.41 | 2025-10-10 12:15:00 | 142.48 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-10-16 14:45:00 | 136.82 | 2025-10-20 15:15:00 | 138.28 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-17 09:15:00 | 136.40 | 2025-10-20 15:15:00 | 138.28 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-10-20 11:00:00 | 137.00 | 2025-10-20 15:15:00 | 138.28 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-20 13:30:00 | 137.00 | 2025-10-20 15:15:00 | 138.28 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-29 13:15:00 | 136.49 | 2025-11-03 12:15:00 | 136.27 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2025-11-28 10:45:00 | 132.84 | 2025-11-28 15:15:00 | 129.65 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-11-28 13:45:00 | 133.25 | 2025-11-28 15:15:00 | 129.65 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-12-04 10:30:00 | 132.08 | 2025-12-05 11:15:00 | 129.21 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-12-04 11:15:00 | 132.10 | 2025-12-05 11:15:00 | 129.21 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-12-05 09:15:00 | 132.48 | 2025-12-05 11:15:00 | 129.21 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-12-05 10:45:00 | 132.28 | 2025-12-05 11:15:00 | 129.21 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-12-18 09:15:00 | 125.81 | 2025-12-19 12:15:00 | 119.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-18 09:15:00 | 125.81 | 2025-12-22 12:15:00 | 119.68 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2026-01-16 09:15:00 | 117.74 | 2026-01-20 09:15:00 | 111.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 117.74 | 2026-01-20 12:15:00 | 115.11 | STOP_HIT | 0.50 | 2.23% |
| SELL | retest2 | 2026-02-26 12:00:00 | 103.65 | 2026-03-02 12:15:00 | 98.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 13:00:00 | 103.56 | 2026-03-02 12:15:00 | 98.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:00:00 | 103.65 | 2026-03-02 14:15:00 | 100.89 | STOP_HIT | 0.50 | 2.66% |
| SELL | retest2 | 2026-02-26 13:00:00 | 103.56 | 2026-03-02 14:15:00 | 100.89 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2026-03-05 15:00:00 | 102.50 | 2026-03-05 15:15:00 | 103.20 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2026-03-12 11:45:00 | 101.26 | 2026-03-12 14:15:00 | 101.29 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2026-03-18 12:00:00 | 97.15 | 2026-03-18 12:15:00 | 98.75 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-04-02 11:30:00 | 91.76 | 2026-04-08 09:15:00 | 100.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 15:15:00 | 105.26 | 2026-05-05 14:15:00 | 104.80 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2026-04-28 09:45:00 | 105.33 | 2026-05-06 09:15:00 | 105.10 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2026-04-28 11:15:00 | 105.15 | 2026-05-06 09:15:00 | 105.10 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2026-04-29 15:00:00 | 105.01 | 2026-05-06 09:15:00 | 105.10 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2026-05-04 12:30:00 | 102.72 | 2026-05-06 09:15:00 | 105.10 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2026-05-08 11:00:00 | 113.22 | 2026-05-11 13:15:00 | 108.75 | STOP_HIT | 1.00 | -3.95% |
