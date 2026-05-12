# Adani Power Ltd. (ADANIPOWER)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 225.02
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 213 |
| ALERT1 | 142 |
| ALERT2 | 139 |
| ALERT2_SKIP | 74 |
| ALERT3 | 360 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 184 |
| PARTIAL | 37 |
| TARGET_HIT | 15 |
| STOP_HIT | 175 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 226 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 110 / 116
- **Target hits / Stop hits / Partials:** 15 / 174 / 37
- **Avg / median % per leg:** 1.26% / -0.06%
- **Sum % (uncompounded):** 284.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 21 | 32.3% | 5 | 60 | 0 | -0.13% | -8.2% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.36% | -5.4% |
| BUY @ 3rd Alert (retest2) | 61 | 21 | 34.4% | 5 | 56 | 0 | -0.04% | -2.7% |
| SELL (all) | 161 | 89 | 55.3% | 10 | 114 | 37 | 1.82% | 293.0% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.89% | -1.9% |
| SELL @ 3rd Alert (retest2) | 160 | 89 | 55.6% | 10 | 113 | 37 | 1.84% | 294.9% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.46% | -7.3% |
| retest2 (combined) | 221 | 110 | 49.8% | 15 | 169 | 37 | 1.32% | 292.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 09:15:00 | 46.62 | 47.81 | 47.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 10:15:00 | 46.25 | 46.73 | 47.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 12:15:00 | 44.47 | 44.44 | 45.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-19 13:00:00 | 44.47 | 44.44 | 45.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 13:15:00 | 46.37 | 44.83 | 45.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 13:30:00 | 46.37 | 44.83 | 45.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 47.26 | 45.32 | 45.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 14:30:00 | 47.26 | 45.32 | 45.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 09:15:00 | 49.58 | 46.48 | 46.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 09:15:00 | 52.05 | 49.42 | 48.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 11:15:00 | 51.64 | 51.92 | 50.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-24 12:00:00 | 51.64 | 51.92 | 50.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 51.36 | 51.56 | 50.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 10:00:00 | 51.36 | 51.56 | 50.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 14:15:00 | 51.43 | 51.75 | 51.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 15:00:00 | 51.43 | 51.75 | 51.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 15:15:00 | 51.63 | 51.73 | 51.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 09:15:00 | 51.55 | 51.73 | 51.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 51.65 | 51.71 | 51.50 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 09:15:00 | 50.49 | 51.27 | 51.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 09:15:00 | 49.07 | 50.24 | 50.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 11:15:00 | 50.30 | 50.23 | 50.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 11:15:00 | 50.30 | 50.23 | 50.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 11:15:00 | 50.30 | 50.23 | 50.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-31 12:30:00 | 49.89 | 50.15 | 50.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-31 15:00:00 | 49.76 | 50.14 | 50.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-01 13:15:00 | 51.00 | 50.69 | 50.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-06-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 13:15:00 | 51.00 | 50.69 | 50.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 09:15:00 | 51.50 | 51.07 | 50.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 12:15:00 | 51.17 | 51.19 | 51.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-05 12:45:00 | 51.22 | 51.19 | 51.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 54.77 | 55.23 | 54.91 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 14:15:00 | 54.16 | 54.64 | 54.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-13 11:15:00 | 53.98 | 54.43 | 54.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-14 09:15:00 | 54.08 | 54.02 | 54.29 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 11:15:00 | 53.48 | 53.93 | 54.22 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 09:15:00 | 54.49 | 53.59 | 53.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-06-15 09:15:00 | 54.49 | 53.59 | 53.86 | SL hit (close>ema400) qty=1.00 sl=53.86 alert=retest1 |

### Cycle 6 — BUY (started 2023-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 13:15:00 | 53.99 | 53.92 | 53.92 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 14:15:00 | 53.21 | 53.78 | 53.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-16 15:15:00 | 53.14 | 53.65 | 53.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 09:15:00 | 52.79 | 52.60 | 52.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 09:15:00 | 52.79 | 52.60 | 52.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 52.79 | 52.60 | 52.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 10:00:00 | 52.79 | 52.60 | 52.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 52.40 | 52.56 | 52.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 14:30:00 | 52.22 | 52.41 | 52.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 10:45:00 | 52.22 | 52.35 | 52.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-23 09:15:00 | 49.61 | 51.33 | 51.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-23 09:15:00 | 49.61 | 51.33 | 51.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-06-26 09:15:00 | 47.00 | 49.49 | 50.49 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2023-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 13:15:00 | 50.78 | 50.48 | 50.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 09:15:00 | 51.07 | 50.65 | 50.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 13:15:00 | 50.71 | 50.81 | 50.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-28 14:00:00 | 50.71 | 50.81 | 50.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 50.93 | 50.83 | 50.70 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 11:15:00 | 50.52 | 50.65 | 50.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 14:15:00 | 50.08 | 50.45 | 50.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 09:15:00 | 50.24 | 49.97 | 50.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 09:15:00 | 50.24 | 49.97 | 50.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 50.24 | 49.97 | 50.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-04 10:00:00 | 50.24 | 49.97 | 50.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 10:15:00 | 49.85 | 49.95 | 50.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 12:30:00 | 49.74 | 49.86 | 50.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 14:15:00 | 47.25 | 47.87 | 48.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-07-14 11:15:00 | 47.73 | 47.65 | 47.94 | SL hit (close>ema200) qty=0.50 sl=47.65 alert=retest2 |

### Cycle 10 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 49.36 | 48.19 | 48.10 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 14:15:00 | 48.86 | 49.01 | 49.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 09:15:00 | 48.58 | 48.89 | 48.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 09:15:00 | 48.98 | 48.19 | 48.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 09:15:00 | 48.98 | 48.19 | 48.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 48.98 | 48.19 | 48.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:00:00 | 48.98 | 48.19 | 48.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 48.40 | 48.23 | 48.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 13:30:00 | 48.22 | 48.36 | 48.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 14:15:00 | 52.20 | 49.13 | 48.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 14:15:00 | 52.20 | 49.13 | 48.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 09:15:00 | 53.85 | 51.83 | 51.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 10:15:00 | 54.27 | 54.50 | 53.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 11:00:00 | 54.27 | 54.50 | 53.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 54.01 | 54.40 | 53.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 53.90 | 54.40 | 53.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 52.53 | 53.97 | 53.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:00:00 | 52.53 | 53.97 | 53.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 53.54 | 53.88 | 53.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 10:30:00 | 54.45 | 53.81 | 53.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-08 10:15:00 | 53.96 | 54.76 | 54.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 10:15:00 | 53.96 | 54.76 | 54.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 11:15:00 | 53.56 | 54.52 | 54.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-08 14:15:00 | 55.95 | 54.63 | 54.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 14:15:00 | 55.95 | 54.63 | 54.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 14:15:00 | 55.95 | 54.63 | 54.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 14:45:00 | 56.11 | 54.63 | 54.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2023-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 15:15:00 | 56.03 | 54.91 | 54.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 11:15:00 | 56.14 | 55.45 | 55.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 14:15:00 | 55.40 | 55.61 | 55.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 14:15:00 | 55.40 | 55.61 | 55.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 14:15:00 | 55.40 | 55.61 | 55.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 14:30:00 | 55.55 | 55.61 | 55.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 15:15:00 | 55.50 | 55.59 | 55.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-10 09:15:00 | 55.86 | 55.59 | 55.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-10 10:45:00 | 55.68 | 55.63 | 55.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-16 11:15:00 | 55.82 | 56.53 | 56.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 11:15:00 | 55.82 | 56.53 | 56.56 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 12:15:00 | 56.90 | 56.60 | 56.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 13:15:00 | 57.71 | 56.83 | 56.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-16 14:15:00 | 55.85 | 56.63 | 56.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 14:15:00 | 55.85 | 56.63 | 56.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 14:15:00 | 55.85 | 56.63 | 56.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 15:00:00 | 55.85 | 56.63 | 56.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 15:15:00 | 56.67 | 56.64 | 56.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 09:15:00 | 56.83 | 56.64 | 56.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 11:30:00 | 56.77 | 56.78 | 56.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 13:45:00 | 56.96 | 56.83 | 56.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-08-18 11:15:00 | 62.51 | 58.48 | 57.59 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 64.99 | 66.12 | 66.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 11:15:00 | 64.55 | 65.81 | 66.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 65.31 | 65.29 | 65.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 65.31 | 65.29 | 65.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 65.31 | 65.29 | 65.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 10:15:00 | 64.50 | 65.29 | 65.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 14:15:00 | 64.42 | 64.91 | 65.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 13:15:00 | 65.80 | 65.42 | 65.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 13:15:00 | 65.80 | 65.42 | 65.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 14:15:00 | 66.29 | 65.60 | 65.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 65.69 | 66.08 | 65.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 14:15:00 | 65.69 | 66.08 | 65.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 65.69 | 66.08 | 65.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 15:00:00 | 65.69 | 66.08 | 65.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 65.85 | 66.03 | 65.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:15:00 | 64.13 | 66.03 | 65.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 09:15:00 | 62.57 | 65.34 | 65.55 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 14:15:00 | 65.84 | 65.15 | 65.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 09:15:00 | 67.38 | 65.79 | 65.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 13:15:00 | 67.72 | 68.31 | 67.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 13:15:00 | 67.72 | 68.31 | 67.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 13:15:00 | 67.72 | 68.31 | 67.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 13:45:00 | 67.95 | 68.31 | 67.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 76.60 | 77.53 | 75.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 76.04 | 77.53 | 75.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 74.55 | 76.71 | 75.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-13 10:00:00 | 74.55 | 76.71 | 75.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 74.97 | 76.37 | 75.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 11:15:00 | 75.13 | 76.37 | 75.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 12:00:00 | 75.14 | 76.12 | 75.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 14:45:00 | 75.15 | 75.76 | 75.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-14 09:15:00 | 75.22 | 75.55 | 75.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-14 09:15:00 | 75.22 | 75.55 | 75.58 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 10:15:00 | 75.79 | 75.60 | 75.60 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-09-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-14 15:15:00 | 75.38 | 75.57 | 75.59 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 09:15:00 | 75.85 | 75.63 | 75.62 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 10:15:00 | 75.38 | 75.60 | 75.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 14:15:00 | 75.00 | 75.34 | 75.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 09:15:00 | 75.69 | 74.43 | 74.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 09:15:00 | 75.69 | 74.43 | 74.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 75.69 | 74.43 | 74.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 09:45:00 | 75.90 | 74.43 | 74.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 75.37 | 74.62 | 74.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 11:15:00 | 75.53 | 74.62 | 74.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2023-09-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 12:15:00 | 75.57 | 74.96 | 74.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-21 14:15:00 | 76.46 | 75.32 | 75.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-22 15:15:00 | 76.46 | 76.51 | 76.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-25 09:15:00 | 76.20 | 76.51 | 76.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 76.32 | 76.47 | 76.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 09:45:00 | 75.88 | 76.47 | 76.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 75.89 | 76.28 | 76.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 15:00:00 | 75.89 | 76.28 | 76.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 15:15:00 | 75.99 | 76.22 | 76.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-26 09:15:00 | 76.45 | 76.22 | 76.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-26 10:15:00 | 75.76 | 76.10 | 76.05 | SL hit (close<static) qty=1.00 sl=75.80 alert=retest2 |

### Cycle 27 — SELL (started 2023-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 12:15:00 | 75.74 | 75.97 | 76.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-26 13:15:00 | 75.57 | 75.89 | 75.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-27 15:15:00 | 75.20 | 75.18 | 75.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-28 09:15:00 | 75.54 | 75.18 | 75.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 75.56 | 75.26 | 75.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 11:15:00 | 74.80 | 75.20 | 75.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 12:30:00 | 74.79 | 75.11 | 75.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 13:45:00 | 74.89 | 75.04 | 75.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 15:15:00 | 74.79 | 75.12 | 75.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 74.79 | 75.05 | 75.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 09:15:00 | 75.40 | 75.05 | 75.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 75.58 | 75.16 | 75.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 09:30:00 | 74.97 | 75.16 | 75.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 74.95 | 75.12 | 75.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 09:45:00 | 74.54 | 75.08 | 75.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-04 11:30:00 | 74.25 | 74.13 | 74.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-04 12:45:00 | 74.45 | 74.16 | 74.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-04 14:15:00 | 74.47 | 74.25 | 74.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 14:15:00 | 74.62 | 74.33 | 74.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-04 14:45:00 | 74.15 | 74.33 | 74.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 15:15:00 | 74.18 | 74.30 | 74.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 09:15:00 | 74.69 | 74.30 | 74.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 74.39 | 74.32 | 74.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 10:15:00 | 73.94 | 74.32 | 74.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 11:30:00 | 73.98 | 74.24 | 74.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 12:15:00 | 73.96 | 74.24 | 74.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 14:00:00 | 73.99 | 74.14 | 74.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 73.45 | 73.72 | 74.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 09:30:00 | 73.58 | 73.72 | 74.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 71.06 | 72.61 | 73.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 71.05 | 72.61 | 73.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 71.15 | 72.61 | 73.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 71.05 | 72.61 | 73.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 70.81 | 72.61 | 73.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 70.54 | 72.61 | 73.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 70.73 | 72.61 | 73.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 70.75 | 72.61 | 73.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 70.24 | 72.61 | 73.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 70.28 | 72.61 | 73.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 70.26 | 72.61 | 73.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 70.29 | 72.61 | 73.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-10 09:15:00 | 70.14 | 69.80 | 71.24 | SL hit (close>ema200) qty=0.50 sl=69.80 alert=retest2 |

### Cycle 28 — BUY (started 2023-10-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 12:15:00 | 68.02 | 67.55 | 67.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-19 13:15:00 | 68.28 | 67.69 | 67.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 14:15:00 | 67.70 | 68.24 | 68.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 14:15:00 | 67.70 | 68.24 | 68.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 14:15:00 | 67.70 | 68.24 | 68.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 15:00:00 | 67.70 | 68.24 | 68.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 67.43 | 68.08 | 67.99 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 66.10 | 67.68 | 67.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 10:15:00 | 65.39 | 67.22 | 67.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 65.16 | 65.03 | 66.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-25 09:45:00 | 65.45 | 65.03 | 66.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 64.87 | 62.14 | 63.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 15:00:00 | 64.87 | 62.14 | 63.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 15:15:00 | 66.06 | 62.93 | 63.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:15:00 | 67.49 | 62.93 | 63.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 10:15:00 | 67.40 | 64.55 | 64.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 11:15:00 | 68.98 | 65.44 | 64.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 11:15:00 | 72.75 | 72.79 | 71.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-01 12:00:00 | 72.75 | 72.79 | 71.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 12:15:00 | 76.86 | 77.91 | 76.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:45:00 | 76.94 | 77.91 | 76.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 13:15:00 | 77.16 | 77.76 | 76.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:45:00 | 77.41 | 77.33 | 76.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 12:15:00 | 78.71 | 79.32 | 79.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 12:15:00 | 78.71 | 79.32 | 79.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 14:15:00 | 78.16 | 78.98 | 79.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 09:15:00 | 78.80 | 78.36 | 78.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 09:15:00 | 78.80 | 78.36 | 78.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 78.80 | 78.36 | 78.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 10:00:00 | 78.80 | 78.36 | 78.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 10:15:00 | 78.53 | 78.39 | 78.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 11:45:00 | 78.23 | 78.31 | 78.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-21 10:15:00 | 78.57 | 78.12 | 78.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 10:15:00 | 78.57 | 78.12 | 78.08 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 12:15:00 | 77.68 | 78.16 | 78.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 13:15:00 | 77.20 | 77.97 | 78.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 11:15:00 | 77.10 | 76.83 | 77.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 11:15:00 | 77.10 | 76.83 | 77.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 11:15:00 | 77.10 | 76.83 | 77.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 12:00:00 | 77.10 | 76.83 | 77.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 77.85 | 77.03 | 77.27 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 14:15:00 | 79.30 | 77.61 | 77.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 09:15:00 | 85.86 | 79.61 | 78.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 14:15:00 | 86.44 | 87.87 | 85.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-29 15:00:00 | 86.44 | 87.87 | 85.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 15:15:00 | 86.36 | 87.57 | 85.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:15:00 | 85.00 | 87.57 | 85.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 84.06 | 86.87 | 85.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 10:00:00 | 84.06 | 86.87 | 85.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 84.60 | 86.41 | 85.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 10:30:00 | 84.30 | 86.41 | 85.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 14:15:00 | 86.59 | 85.75 | 85.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 14:45:00 | 84.85 | 85.75 | 85.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 109.25 | 111.40 | 108.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:00:00 | 109.25 | 111.40 | 108.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 107.14 | 110.11 | 108.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 12:45:00 | 107.41 | 110.11 | 108.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 105.60 | 109.21 | 108.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:45:00 | 105.20 | 109.21 | 108.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 10:15:00 | 107.11 | 107.80 | 107.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-11 10:30:00 | 104.63 | 107.80 | 107.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 11:15:00 | 105.94 | 107.43 | 107.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 12:15:00 | 105.60 | 107.06 | 107.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 103.80 | 102.06 | 103.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 103.80 | 102.06 | 103.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 103.80 | 102.06 | 103.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:30:00 | 104.20 | 102.06 | 103.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 103.70 | 102.38 | 103.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 11:00:00 | 103.70 | 102.38 | 103.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 11:15:00 | 104.58 | 102.82 | 103.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 11:45:00 | 104.57 | 102.82 | 103.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 12:15:00 | 104.39 | 103.14 | 103.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 13:00:00 | 104.39 | 103.14 | 103.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-12-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 14:15:00 | 106.19 | 103.88 | 103.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 106.60 | 104.80 | 104.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 11:15:00 | 105.00 | 105.03 | 104.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-15 12:00:00 | 105.00 | 105.03 | 104.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 12:15:00 | 104.88 | 105.00 | 104.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 13:00:00 | 104.88 | 105.00 | 104.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 13:15:00 | 105.36 | 105.07 | 104.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-15 15:00:00 | 108.19 | 105.70 | 104.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 12:00:00 | 106.20 | 106.96 | 106.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 12:15:00 | 105.00 | 106.57 | 106.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 12:15:00 | 105.00 | 106.57 | 106.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 102.10 | 105.67 | 106.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 15:15:00 | 102.92 | 102.59 | 103.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 09:15:00 | 102.76 | 102.59 | 103.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 103.20 | 102.66 | 103.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 13:15:00 | 102.40 | 102.79 | 103.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 14:00:00 | 102.43 | 102.72 | 103.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-27 10:15:00 | 105.07 | 103.19 | 103.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2023-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 10:15:00 | 105.07 | 103.19 | 103.16 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-12-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 13:15:00 | 102.82 | 103.42 | 103.47 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-12-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 14:15:00 | 104.80 | 103.70 | 103.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 14:15:00 | 105.35 | 104.29 | 103.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 09:15:00 | 104.22 | 104.32 | 104.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 09:15:00 | 104.22 | 104.32 | 104.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 104.22 | 104.32 | 104.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 10:15:00 | 104.00 | 104.32 | 104.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 10:15:00 | 105.40 | 104.54 | 104.17 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 11:15:00 | 103.80 | 104.28 | 104.31 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 09:15:00 | 108.90 | 104.93 | 104.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 110.00 | 107.84 | 106.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 12:15:00 | 110.80 | 110.87 | 109.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-05 13:00:00 | 110.80 | 110.87 | 109.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 110.40 | 110.67 | 109.51 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 14:15:00 | 108.24 | 109.21 | 109.23 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 110.70 | 109.38 | 109.30 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 14:15:00 | 108.02 | 109.14 | 109.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 09:15:00 | 107.28 | 108.12 | 108.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 15:15:00 | 105.99 | 104.89 | 105.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 15:15:00 | 105.99 | 104.89 | 105.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 15:15:00 | 105.99 | 104.89 | 105.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 12:00:00 | 104.40 | 104.78 | 105.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 12:30:00 | 104.16 | 104.63 | 105.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-20 09:15:00 | 106.20 | 104.82 | 105.05 | SL hit (close>static) qty=1.00 sl=106.05 alert=retest2 |

### Cycle 46 — BUY (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 11:15:00 | 106.41 | 105.42 | 105.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 15:15:00 | 107.60 | 106.25 | 105.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 09:15:00 | 105.45 | 106.09 | 105.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 105.45 | 106.09 | 105.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 105.45 | 106.09 | 105.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:45:00 | 105.59 | 106.09 | 105.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 104.80 | 105.83 | 105.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:30:00 | 104.94 | 105.83 | 105.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 104.04 | 105.23 | 105.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 103.73 | 104.73 | 105.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 14:15:00 | 105.02 | 104.27 | 104.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 14:15:00 | 105.02 | 104.27 | 104.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 105.02 | 104.27 | 104.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 105.02 | 104.27 | 104.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 105.00 | 104.41 | 104.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 10:00:00 | 105.40 | 104.61 | 104.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 104.57 | 104.60 | 104.71 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2024-01-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 13:15:00 | 106.60 | 105.11 | 104.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 14:15:00 | 108.89 | 105.87 | 105.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 111.20 | 112.59 | 110.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 14:15:00 | 111.20 | 112.59 | 110.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 111.20 | 112.59 | 110.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 15:00:00 | 111.20 | 112.59 | 110.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 113.00 | 112.83 | 111.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 15:00:00 | 113.00 | 112.83 | 111.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 112.08 | 112.70 | 111.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 10:00:00 | 112.08 | 112.70 | 111.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 112.44 | 112.65 | 112.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 10:30:00 | 112.80 | 112.65 | 112.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 111.60 | 112.44 | 111.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 111.60 | 112.44 | 111.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 111.96 | 112.34 | 111.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:30:00 | 111.55 | 112.34 | 111.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 13:15:00 | 112.81 | 112.44 | 112.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 15:15:00 | 113.10 | 112.48 | 112.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-02 15:15:00 | 111.59 | 112.06 | 112.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 15:15:00 | 111.59 | 112.06 | 112.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 10:15:00 | 109.76 | 111.51 | 111.82 | Break + close below crossover candle low |

### Cycle 50 — BUY (started 2024-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 09:15:00 | 114.80 | 110.95 | 110.86 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 11:15:00 | 112.20 | 112.71 | 112.76 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 13:15:00 | 113.60 | 112.91 | 112.85 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 15:15:00 | 112.58 | 112.76 | 112.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 09:15:00 | 109.00 | 112.01 | 112.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 112.94 | 110.57 | 111.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 14:15:00 | 112.94 | 110.57 | 111.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 112.94 | 110.57 | 111.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:00:00 | 112.94 | 110.57 | 111.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 112.80 | 111.01 | 111.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 111.00 | 111.01 | 111.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-14 14:15:00 | 113.98 | 111.85 | 111.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 14:15:00 | 113.98 | 111.85 | 111.72 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 11:15:00 | 111.57 | 111.99 | 111.99 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 14:15:00 | 112.88 | 112.09 | 112.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 09:15:00 | 113.76 | 112.51 | 112.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 13:15:00 | 113.01 | 113.04 | 112.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-19 13:45:00 | 113.39 | 113.04 | 112.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 112.69 | 113.09 | 112.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:15:00 | 112.60 | 113.09 | 112.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 111.79 | 112.83 | 112.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:45:00 | 111.98 | 112.83 | 112.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 11:15:00 | 112.40 | 112.74 | 112.64 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 12:15:00 | 111.60 | 112.52 | 112.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 13:15:00 | 111.40 | 112.29 | 112.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 10:15:00 | 110.53 | 109.68 | 110.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 10:15:00 | 110.53 | 109.68 | 110.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 110.53 | 109.68 | 110.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 10:30:00 | 110.36 | 109.68 | 110.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 110.40 | 109.82 | 110.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 12:00:00 | 110.40 | 109.82 | 110.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 110.60 | 109.98 | 110.59 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 15:15:00 | 113.30 | 111.13 | 111.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 13:15:00 | 114.48 | 112.65 | 112.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 11:15:00 | 113.11 | 113.24 | 112.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-27 12:00:00 | 113.11 | 113.24 | 112.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 112.26 | 113.10 | 112.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 14:00:00 | 112.26 | 113.10 | 112.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 113.40 | 113.16 | 112.72 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 110.80 | 112.46 | 112.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 15:15:00 | 109.80 | 110.98 | 111.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 109.60 | 109.32 | 110.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 15:00:00 | 109.60 | 109.32 | 110.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 110.00 | 109.46 | 110.34 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 11:15:00 | 111.20 | 110.68 | 110.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 09:15:00 | 112.00 | 111.01 | 110.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 113.00 | 113.05 | 112.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 113.00 | 113.05 | 112.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 113.00 | 113.05 | 112.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 113.00 | 113.05 | 112.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 14:15:00 | 113.31 | 114.11 | 113.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-07 15:00:00 | 113.31 | 114.11 | 113.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 15:15:00 | 114.03 | 114.09 | 113.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 09:15:00 | 113.98 | 114.09 | 113.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 113.50 | 113.97 | 113.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-11 13:00:00 | 114.48 | 114.02 | 113.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-12 09:15:00 | 112.09 | 113.35 | 113.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 09:15:00 | 112.09 | 113.35 | 113.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 12:15:00 | 109.79 | 111.73 | 112.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 106.86 | 106.70 | 108.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 11:00:00 | 106.86 | 106.70 | 108.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 108.18 | 107.21 | 108.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 09:30:00 | 106.00 | 106.87 | 107.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 09:15:00 | 100.70 | 103.79 | 104.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 105.23 | 103.42 | 103.95 | SL hit (close>ema200) qty=0.50 sl=103.42 alert=retest2 |

### Cycle 62 — BUY (started 2024-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 14:15:00 | 104.99 | 104.36 | 104.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 105.58 | 104.67 | 104.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 11:15:00 | 105.14 | 105.50 | 105.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 11:15:00 | 105.14 | 105.50 | 105.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 11:15:00 | 105.14 | 105.50 | 105.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 12:00:00 | 105.14 | 105.50 | 105.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 12:15:00 | 105.10 | 105.42 | 105.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 13:00:00 | 105.10 | 105.42 | 105.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 13:15:00 | 105.74 | 105.48 | 105.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 13:45:00 | 105.29 | 105.48 | 105.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 104.61 | 105.31 | 105.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 14:45:00 | 104.90 | 105.31 | 105.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 104.07 | 105.06 | 105.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 107.79 | 105.06 | 105.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-27 14:15:00 | 103.23 | 106.18 | 105.79 | SL hit (close<static) qty=1.00 sl=103.80 alert=retest2 |

### Cycle 63 — SELL (started 2024-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 15:15:00 | 102.70 | 105.48 | 105.51 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 13:15:00 | 106.04 | 105.50 | 105.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 09:15:00 | 112.09 | 107.10 | 106.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 09:15:00 | 125.52 | 126.05 | 122.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-05 09:30:00 | 124.94 | 126.05 | 122.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 123.50 | 125.32 | 123.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 10:15:00 | 123.53 | 125.32 | 123.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 10:15:00 | 123.52 | 124.96 | 123.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 10:00:00 | 124.73 | 123.79 | 123.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 13:15:00 | 123.85 | 124.14 | 123.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 13:15:00 | 122.60 | 123.83 | 123.77 | SL hit (close<static) qty=1.00 sl=123.10 alert=retest2 |

### Cycle 65 — SELL (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 14:15:00 | 122.95 | 123.65 | 123.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 09:15:00 | 122.04 | 122.95 | 123.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 120.16 | 119.71 | 120.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 120.16 | 119.71 | 120.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 120.16 | 119.71 | 120.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:30:00 | 120.05 | 119.71 | 120.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 12:15:00 | 120.30 | 119.95 | 120.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 13:00:00 | 120.30 | 119.95 | 120.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 120.36 | 120.03 | 120.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 14:00:00 | 120.36 | 120.03 | 120.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 120.44 | 120.14 | 120.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:15:00 | 121.45 | 120.14 | 120.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 120.38 | 120.18 | 120.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 10:15:00 | 120.03 | 120.18 | 120.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:15:00 | 120.05 | 120.58 | 120.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 09:30:00 | 119.87 | 119.14 | 119.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 12:00:00 | 120.08 | 119.43 | 119.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 13:15:00 | 120.11 | 119.72 | 119.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 13:15:00 | 120.11 | 119.72 | 119.71 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 14:15:00 | 119.40 | 119.66 | 119.69 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 120.70 | 119.83 | 119.75 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 12:15:00 | 119.49 | 120.01 | 120.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 14:15:00 | 119.29 | 119.81 | 119.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-26 13:15:00 | 118.37 | 118.36 | 118.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-26 14:00:00 | 118.37 | 118.36 | 118.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 118.31 | 118.35 | 118.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 14:30:00 | 118.82 | 118.35 | 118.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 118.49 | 118.42 | 118.72 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 121.74 | 119.33 | 119.03 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 117.71 | 120.44 | 120.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 12:15:00 | 115.51 | 117.53 | 118.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 15:15:00 | 116.87 | 115.89 | 116.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 15:15:00 | 116.87 | 115.89 | 116.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 15:15:00 | 116.87 | 115.89 | 116.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:15:00 | 121.13 | 115.89 | 116.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 123.36 | 117.39 | 117.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:30:00 | 123.91 | 117.39 | 117.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-09 10:15:00 | 119.75 | 117.86 | 117.65 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 117.19 | 119.72 | 119.79 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 124.00 | 120.28 | 119.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 126.26 | 121.48 | 120.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 09:15:00 | 126.69 | 127.03 | 124.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 10:00:00 | 126.69 | 127.03 | 124.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 127.20 | 127.68 | 126.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:30:00 | 126.26 | 127.68 | 126.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 127.05 | 127.39 | 126.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:45:00 | 126.70 | 127.39 | 126.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 126.94 | 127.20 | 126.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 12:00:00 | 126.94 | 127.20 | 126.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 126.95 | 127.15 | 126.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:15:00 | 126.23 | 127.15 | 126.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 129.23 | 127.57 | 127.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 12:00:00 | 137.92 | 130.22 | 128.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 11:15:00 | 137.12 | 139.75 | 140.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 137.12 | 139.75 | 140.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 12:15:00 | 136.47 | 139.09 | 139.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 136.04 | 135.90 | 137.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 13:45:00 | 136.00 | 135.90 | 137.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 138.84 | 136.45 | 137.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:30:00 | 140.88 | 136.45 | 137.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 139.42 | 137.05 | 137.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:45:00 | 139.94 | 137.05 | 137.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 12:15:00 | 139.71 | 137.83 | 137.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 13:15:00 | 141.00 | 138.46 | 138.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 148.24 | 164.97 | 158.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 148.24 | 164.97 | 158.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 148.24 | 164.97 | 158.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 148.24 | 164.97 | 158.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 142.44 | 160.46 | 156.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 142.44 | 160.46 | 156.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 147.98 | 156.19 | 155.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 147.98 | 156.19 | 155.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2024-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 15:15:00 | 144.00 | 153.75 | 154.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 132.00 | 149.40 | 152.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 145.27 | 144.40 | 148.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 15:00:00 | 145.27 | 144.40 | 148.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 157.01 | 146.95 | 148.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 157.01 | 146.95 | 148.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 152.60 | 150.11 | 150.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 12:15:00 | 153.31 | 151.54 | 150.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 153.85 | 154.35 | 153.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 154.40 | 154.35 | 153.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 153.30 | 154.08 | 153.14 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 10:15:00 | 151.93 | 152.85 | 152.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 09:15:00 | 151.12 | 152.11 | 152.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 10:15:00 | 150.07 | 149.93 | 150.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 10:15:00 | 150.07 | 149.93 | 150.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 150.07 | 149.93 | 150.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:00:00 | 150.07 | 149.93 | 150.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 145.38 | 148.90 | 149.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 14:00:00 | 143.98 | 145.01 | 145.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 15:15:00 | 143.92 | 144.82 | 145.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:00:00 | 143.98 | 144.51 | 145.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 10:00:00 | 143.79 | 143.50 | 144.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 143.87 | 143.58 | 144.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 15:15:00 | 143.40 | 143.97 | 144.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 11:15:00 | 143.71 | 143.90 | 144.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 14:15:00 | 143.52 | 143.78 | 144.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 15:00:00 | 143.40 | 143.70 | 143.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 143.72 | 143.62 | 143.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 145.22 | 143.94 | 144.01 | SL hit (close>static) qty=1.00 sl=144.79 alert=retest2 |

### Cycle 80 — BUY (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 10:15:00 | 145.98 | 142.14 | 141.89 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 11:15:00 | 143.17 | 144.11 | 144.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 12:15:00 | 142.73 | 143.84 | 144.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 143.37 | 142.40 | 142.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 143.37 | 142.40 | 142.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 143.37 | 142.40 | 142.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 14:15:00 | 142.07 | 142.56 | 142.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 140.78 | 142.55 | 142.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 09:15:00 | 143.20 | 140.44 | 140.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 143.20 | 140.44 | 140.38 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 15:15:00 | 139.21 | 140.49 | 140.59 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 142.50 | 140.13 | 139.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 144.21 | 141.26 | 140.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 14:15:00 | 143.15 | 143.38 | 142.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 15:00:00 | 143.15 | 143.38 | 142.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 143.42 | 146.60 | 145.61 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 142.87 | 145.43 | 145.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 14:15:00 | 138.40 | 141.29 | 143.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 139.71 | 139.16 | 140.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 139.71 | 139.16 | 140.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 139.71 | 139.16 | 140.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 14:15:00 | 138.47 | 138.94 | 140.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 15:00:00 | 138.33 | 138.82 | 139.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 15:00:00 | 138.16 | 138.78 | 139.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 138.53 | 138.75 | 139.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 138.75 | 138.75 | 139.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 133.22 | 138.80 | 139.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-12 09:15:00 | 124.62 | 137.88 | 138.64 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 139.84 | 136.91 | 136.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 140.38 | 137.61 | 137.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 12:15:00 | 138.83 | 138.98 | 138.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 13:00:00 | 138.83 | 138.98 | 138.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 139.13 | 139.01 | 138.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 10:30:00 | 139.74 | 139.25 | 138.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 11:00:00 | 139.75 | 139.25 | 138.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 09:15:00 | 138.37 | 138.92 | 138.81 | SL hit (close<static) qty=1.00 sl=138.45 alert=retest2 |

### Cycle 87 — SELL (started 2024-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 12:15:00 | 138.11 | 138.63 | 138.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 13:15:00 | 137.84 | 138.47 | 138.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 128.87 | 128.01 | 129.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 14:15:00 | 128.87 | 128.01 | 129.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 128.87 | 128.01 | 129.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 128.87 | 128.01 | 129.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 128.09 | 128.20 | 129.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 11:00:00 | 127.51 | 128.07 | 129.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 14:00:00 | 127.81 | 127.82 | 128.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 14:30:00 | 127.66 | 127.51 | 128.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 13:15:00 | 134.36 | 128.94 | 128.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 13:15:00 | 134.36 | 128.94 | 128.77 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 10:15:00 | 129.64 | 130.47 | 130.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 14:15:00 | 128.79 | 129.78 | 130.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 10:15:00 | 127.34 | 127.18 | 127.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 11:00:00 | 127.34 | 127.18 | 127.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 128.00 | 127.42 | 127.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:30:00 | 127.98 | 127.42 | 127.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 128.65 | 127.67 | 127.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:30:00 | 128.35 | 127.67 | 127.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 127.55 | 127.65 | 127.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 15:15:00 | 126.80 | 127.65 | 127.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:45:00 | 127.37 | 127.43 | 127.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:45:00 | 127.34 | 127.41 | 127.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 13:15:00 | 129.31 | 127.01 | 127.06 | SL hit (close>static) qty=1.00 sl=129.20 alert=retest2 |

### Cycle 90 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 130.02 | 127.61 | 127.33 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 15:15:00 | 126.82 | 127.47 | 127.50 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 09:15:00 | 134.27 | 128.83 | 128.12 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 130.64 | 131.37 | 131.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 128.76 | 130.85 | 131.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 130.34 | 129.48 | 130.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 130.34 | 129.48 | 130.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 130.34 | 129.48 | 130.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 130.34 | 129.48 | 130.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 129.81 | 129.54 | 130.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 131.12 | 129.54 | 130.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 131.07 | 129.85 | 130.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 131.26 | 129.85 | 130.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 131.59 | 130.20 | 130.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:45:00 | 131.41 | 130.20 | 130.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 131.40 | 130.65 | 130.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 133.01 | 131.13 | 130.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 133.74 | 134.31 | 133.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 10:00:00 | 133.74 | 134.31 | 133.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 133.46 | 134.04 | 133.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:30:00 | 133.70 | 134.04 | 133.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 133.31 | 133.90 | 133.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 14:00:00 | 133.60 | 133.84 | 133.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:15:00 | 133.59 | 133.73 | 133.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 132.89 | 133.54 | 133.43 | SL hit (close<static) qty=1.00 sl=133.07 alert=retest2 |

### Cycle 95 — SELL (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 11:15:00 | 132.40 | 133.17 | 133.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 12:15:00 | 132.30 | 133.00 | 133.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 132.86 | 132.86 | 133.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 14:15:00 | 132.86 | 132.86 | 133.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 132.86 | 132.86 | 133.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 132.86 | 132.86 | 133.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 133.04 | 132.89 | 133.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 133.87 | 132.89 | 133.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 134.42 | 133.20 | 133.20 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 131.01 | 132.98 | 133.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 128.60 | 130.47 | 131.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 14:15:00 | 129.62 | 128.82 | 129.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-03 15:00:00 | 129.62 | 128.82 | 129.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 129.80 | 129.02 | 129.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 127.71 | 129.02 | 129.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 10:00:00 | 128.59 | 128.93 | 129.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 11:15:00 | 128.58 | 128.93 | 129.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:15:00 | 128.82 | 128.97 | 129.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 126.03 | 128.04 | 128.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:30:00 | 124.97 | 127.25 | 128.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 15:15:00 | 128.30 | 127.30 | 127.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 128.30 | 127.30 | 127.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 129.00 | 127.64 | 127.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 127.06 | 127.93 | 127.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 14:15:00 | 127.06 | 127.93 | 127.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 127.06 | 127.93 | 127.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 15:00:00 | 127.06 | 127.93 | 127.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 126.94 | 127.73 | 127.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:15:00 | 127.63 | 127.73 | 127.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 10:15:00 | 127.07 | 127.58 | 127.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 127.07 | 127.58 | 127.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 12:15:00 | 126.68 | 127.32 | 127.47 | Break + close below crossover candle low |

### Cycle 100 — BUY (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 14:15:00 | 129.39 | 127.59 | 127.56 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 127.30 | 127.76 | 127.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 13:15:00 | 126.32 | 127.47 | 127.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 122.48 | 121.95 | 123.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 15:00:00 | 122.48 | 121.95 | 123.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 121.76 | 121.92 | 122.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 15:15:00 | 120.48 | 121.64 | 122.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:30:00 | 119.84 | 118.22 | 118.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:00:00 | 120.07 | 118.59 | 119.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 13:15:00 | 120.30 | 119.37 | 119.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 13:15:00 | 120.30 | 119.37 | 119.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 14:15:00 | 121.30 | 119.75 | 119.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 118.15 | 119.57 | 119.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 118.15 | 119.57 | 119.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 118.15 | 119.57 | 119.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 118.15 | 119.57 | 119.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 10:15:00 | 117.73 | 119.20 | 119.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 09:15:00 | 115.23 | 118.00 | 118.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 10:15:00 | 120.98 | 118.59 | 118.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 10:15:00 | 120.98 | 118.59 | 118.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 120.98 | 118.59 | 118.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 120.98 | 118.59 | 118.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 120.90 | 119.06 | 119.06 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 120.78 | 119.40 | 119.22 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 117.48 | 118.88 | 119.06 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 10:15:00 | 119.34 | 118.81 | 118.77 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 117.58 | 118.74 | 118.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 15:15:00 | 116.70 | 117.80 | 118.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 118.18 | 117.88 | 118.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 118.18 | 117.88 | 118.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 118.18 | 117.88 | 118.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:45:00 | 117.95 | 117.88 | 118.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 117.52 | 117.80 | 118.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 11:30:00 | 117.06 | 117.70 | 118.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 14:15:00 | 120.79 | 118.30 | 118.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 120.79 | 118.30 | 118.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 122.28 | 119.44 | 118.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 121.37 | 122.25 | 120.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 121.37 | 122.25 | 120.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 121.40 | 122.08 | 120.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 120.92 | 122.08 | 120.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 121.24 | 121.69 | 121.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:45:00 | 121.20 | 121.69 | 121.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 119.82 | 121.32 | 120.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 119.82 | 121.32 | 120.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 119.80 | 121.01 | 120.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 119.29 | 121.01 | 120.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 118.79 | 120.57 | 120.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 117.92 | 119.67 | 120.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 09:15:00 | 114.14 | 113.74 | 115.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 09:15:00 | 114.14 | 113.74 | 115.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 114.14 | 113.74 | 115.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:30:00 | 115.85 | 113.74 | 115.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 114.55 | 113.90 | 115.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:30:00 | 114.56 | 113.90 | 115.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 92.57 | 89.39 | 91.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:00:00 | 92.57 | 89.39 | 91.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 92.16 | 89.94 | 91.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:15:00 | 93.64 | 89.94 | 91.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 94.81 | 90.91 | 91.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 94.76 | 90.91 | 91.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 12:15:00 | 101.96 | 93.12 | 92.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 14:15:00 | 105.03 | 96.80 | 94.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 10:15:00 | 109.08 | 110.23 | 105.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 11:00:00 | 109.08 | 110.23 | 105.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 109.68 | 109.78 | 108.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:45:00 | 108.80 | 109.78 | 108.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 108.99 | 109.40 | 108.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 14:45:00 | 108.91 | 109.40 | 108.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 108.69 | 109.26 | 108.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:30:00 | 108.81 | 109.12 | 108.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 107.66 | 108.83 | 108.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:00:00 | 107.66 | 108.83 | 108.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 11:15:00 | 107.51 | 108.56 | 108.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 15:15:00 | 107.00 | 107.50 | 107.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 10:15:00 | 107.72 | 107.53 | 107.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 10:15:00 | 107.72 | 107.53 | 107.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 107.72 | 107.53 | 107.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:00:00 | 107.72 | 107.53 | 107.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 107.90 | 107.61 | 107.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:45:00 | 107.91 | 107.61 | 107.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 107.60 | 107.60 | 107.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 13:15:00 | 107.51 | 107.60 | 107.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 10:15:00 | 110.58 | 105.51 | 105.76 | SL hit (close>static) qty=1.00 sl=107.89 alert=retest2 |

### Cycle 112 — BUY (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 11:15:00 | 109.21 | 106.25 | 106.08 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 11:15:00 | 106.23 | 106.83 | 106.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 14:15:00 | 106.10 | 106.55 | 106.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 09:15:00 | 107.56 | 106.59 | 106.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 107.56 | 106.59 | 106.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 107.56 | 106.59 | 106.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:00:00 | 107.56 | 106.59 | 106.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 106.50 | 106.58 | 106.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:45:00 | 107.36 | 106.58 | 106.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 106.38 | 106.54 | 106.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:15:00 | 106.79 | 106.54 | 106.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 106.37 | 106.50 | 106.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 13:45:00 | 105.95 | 106.25 | 106.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 14:15:00 | 100.65 | 101.91 | 102.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-23 13:15:00 | 100.93 | 100.92 | 101.86 | SL hit (close>ema200) qty=0.50 sl=100.92 alert=retest2 |

### Cycle 114 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 101.38 | 101.18 | 101.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 10:15:00 | 102.14 | 101.45 | 101.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 11:15:00 | 101.33 | 101.43 | 101.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 11:15:00 | 101.33 | 101.43 | 101.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 101.33 | 101.43 | 101.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:00:00 | 101.33 | 101.43 | 101.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 101.73 | 101.49 | 101.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:15:00 | 101.25 | 101.49 | 101.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 101.70 | 101.53 | 101.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:30:00 | 101.68 | 101.53 | 101.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 101.51 | 101.53 | 101.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 101.51 | 101.53 | 101.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 101.22 | 101.47 | 101.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 102.11 | 101.47 | 101.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 15:00:00 | 108.80 | 103.42 | 102.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 13:15:00 | 104.36 | 104.78 | 104.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 104.36 | 104.78 | 104.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 104.23 | 104.67 | 104.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 102.60 | 101.51 | 102.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 102.60 | 101.51 | 102.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 102.60 | 101.51 | 102.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 103.81 | 101.51 | 102.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 102.01 | 101.61 | 102.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 101.15 | 102.03 | 102.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:30:00 | 100.80 | 101.83 | 102.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 14:15:00 | 101.30 | 101.45 | 101.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:30:00 | 101.30 | 101.35 | 101.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 96.09 | 99.63 | 100.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 95.76 | 99.63 | 100.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 96.23 | 99.63 | 100.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 96.23 | 99.63 | 100.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 14:15:00 | 91.04 | 93.53 | 96.16 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 116 — BUY (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 11:15:00 | 106.18 | 98.59 | 97.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 15:15:00 | 107.97 | 103.44 | 100.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 11:15:00 | 111.36 | 111.71 | 109.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 12:00:00 | 111.36 | 111.71 | 109.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 109.72 | 111.14 | 109.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 109.90 | 111.14 | 109.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 108.37 | 110.58 | 109.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:00:00 | 108.37 | 110.58 | 109.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 108.45 | 110.16 | 109.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:45:00 | 108.60 | 110.16 | 109.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 109.18 | 109.75 | 109.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:00:00 | 109.18 | 109.75 | 109.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 108.74 | 109.55 | 109.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 15:00:00 | 108.74 | 109.55 | 109.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 15:15:00 | 108.48 | 109.33 | 109.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 108.18 | 109.10 | 109.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 105.32 | 104.82 | 106.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 105.32 | 104.82 | 106.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 106.05 | 105.10 | 106.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:30:00 | 106.47 | 105.10 | 106.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 105.30 | 105.14 | 106.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:30:00 | 104.81 | 105.15 | 105.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:15:00 | 104.97 | 105.25 | 105.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:45:00 | 103.50 | 104.66 | 105.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 99.57 | 102.70 | 103.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 99.72 | 102.70 | 103.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 98.32 | 99.76 | 101.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 100.66 | 99.64 | 101.24 | SL hit (close>ema200) qty=0.50 sl=99.64 alert=retest2 |

### Cycle 118 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 104.42 | 101.16 | 101.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 10:15:00 | 105.85 | 103.63 | 102.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 102.63 | 103.55 | 102.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 102.63 | 103.55 | 102.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 102.63 | 103.55 | 102.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 102.63 | 103.55 | 102.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 104.20 | 103.68 | 103.04 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 100.66 | 102.45 | 102.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 100.26 | 102.01 | 102.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 101.29 | 101.09 | 101.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 101.29 | 101.09 | 101.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 101.29 | 101.09 | 101.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:15:00 | 100.74 | 101.09 | 101.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 13:00:00 | 100.95 | 100.96 | 101.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 14:00:00 | 100.92 | 100.95 | 101.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 15:15:00 | 100.85 | 101.45 | 101.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 101.00 | 101.27 | 101.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 13:45:00 | 100.55 | 100.97 | 101.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 99.68 | 100.88 | 101.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:45:00 | 100.41 | 98.82 | 99.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-12 12:15:00 | 100.47 | 99.28 | 99.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2025-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 12:15:00 | 100.47 | 99.28 | 99.24 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 98.44 | 99.57 | 99.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 97.65 | 99.19 | 99.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 11:15:00 | 98.00 | 97.61 | 98.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 11:15:00 | 98.00 | 97.61 | 98.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 98.00 | 97.61 | 98.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:00:00 | 98.00 | 97.61 | 98.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 98.39 | 97.77 | 98.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:45:00 | 98.64 | 97.77 | 98.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 98.60 | 97.93 | 98.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 98.86 | 97.93 | 98.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 98.10 | 97.97 | 98.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:45:00 | 96.88 | 97.74 | 98.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 13:15:00 | 98.00 | 97.39 | 97.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 98.00 | 97.39 | 97.38 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 14:15:00 | 96.59 | 97.23 | 97.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 96.19 | 96.95 | 97.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 97.77 | 95.30 | 95.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 97.77 | 95.30 | 95.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 97.77 | 95.30 | 95.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 98.07 | 95.30 | 95.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 97.85 | 95.81 | 95.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:30:00 | 97.77 | 95.81 | 95.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 11:15:00 | 97.95 | 96.24 | 96.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 12:15:00 | 101.29 | 97.25 | 96.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-27 13:15:00 | 98.79 | 99.31 | 98.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 13:15:00 | 98.79 | 99.31 | 98.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 98.79 | 99.31 | 98.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:45:00 | 98.75 | 99.31 | 98.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 98.01 | 99.05 | 98.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-28 10:30:00 | 99.18 | 98.93 | 98.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-28 13:15:00 | 97.18 | 98.12 | 98.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2025-02-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 13:15:00 | 97.18 | 98.12 | 98.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 14:15:00 | 95.06 | 97.51 | 97.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 15:15:00 | 98.00 | 97.61 | 97.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 15:15:00 | 98.00 | 97.61 | 97.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 98.00 | 97.61 | 97.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 96.52 | 97.61 | 97.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 94.95 | 97.07 | 97.62 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 100.29 | 97.68 | 97.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 101.37 | 98.90 | 98.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 100.40 | 100.68 | 99.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 100.80 | 100.68 | 99.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 100.70 | 100.69 | 99.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:30:00 | 101.58 | 100.86 | 100.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:45:00 | 101.32 | 100.97 | 100.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 102.40 | 101.01 | 100.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 09:45:00 | 101.51 | 101.74 | 101.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 101.66 | 101.72 | 101.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:30:00 | 101.45 | 101.72 | 101.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 101.70 | 101.72 | 101.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:30:00 | 101.64 | 101.72 | 101.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 100.88 | 101.55 | 101.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:00:00 | 100.88 | 101.55 | 101.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 101.58 | 101.56 | 101.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 15:15:00 | 102.56 | 101.59 | 101.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 13:15:00 | 102.05 | 101.94 | 101.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 14:00:00 | 102.08 | 101.97 | 101.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 102.84 | 102.51 | 102.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 102.78 | 102.61 | 102.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:45:00 | 102.48 | 102.61 | 102.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 102.65 | 102.62 | 102.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:00:00 | 102.65 | 102.62 | 102.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 102.54 | 102.61 | 102.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:45:00 | 102.37 | 102.61 | 102.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 102.80 | 102.64 | 102.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:30:00 | 102.57 | 102.64 | 102.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 103.33 | 102.78 | 102.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 14:30:00 | 102.76 | 102.78 | 102.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 103.20 | 103.78 | 103.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 103.20 | 103.78 | 103.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 103.90 | 103.81 | 103.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 104.55 | 103.92 | 103.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:30:00 | 104.48 | 104.85 | 104.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 10:15:00 | 103.10 | 104.50 | 104.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 10:15:00 | 103.10 | 104.50 | 104.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 11:15:00 | 102.80 | 104.16 | 104.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 102.86 | 102.83 | 103.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-26 10:00:00 | 102.86 | 102.83 | 103.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 101.28 | 100.81 | 101.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 101.28 | 100.81 | 101.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 104.27 | 101.50 | 101.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 104.27 | 101.50 | 101.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 104.69 | 102.14 | 102.11 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 102.05 | 102.32 | 102.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 14:15:00 | 100.87 | 101.44 | 101.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 102.02 | 101.43 | 101.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 102.02 | 101.43 | 101.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 102.02 | 101.43 | 101.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:45:00 | 102.38 | 101.43 | 101.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 102.17 | 101.58 | 101.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 102.17 | 101.58 | 101.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 102.62 | 102.00 | 101.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 15:15:00 | 102.90 | 102.18 | 102.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 104.93 | 105.04 | 103.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 13:15:00 | 105.11 | 105.25 | 104.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 105.11 | 105.25 | 104.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:30:00 | 104.57 | 105.25 | 104.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 97.66 | 104.12 | 104.16 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 103.83 | 102.47 | 102.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 107.73 | 104.07 | 103.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 12:15:00 | 115.29 | 115.52 | 114.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 12:30:00 | 115.40 | 115.52 | 114.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 115.28 | 115.41 | 114.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:45:00 | 114.83 | 115.41 | 114.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 112.65 | 114.82 | 114.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 112.65 | 114.82 | 114.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 110.00 | 113.85 | 114.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 109.07 | 110.06 | 110.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 108.88 | 106.57 | 107.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 108.88 | 106.57 | 107.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 108.88 | 106.57 | 107.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 108.88 | 106.57 | 107.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 112.08 | 107.67 | 108.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 112.08 | 107.67 | 108.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 115.07 | 109.15 | 108.70 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 107.77 | 109.17 | 109.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 107.40 | 108.82 | 109.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 09:15:00 | 106.92 | 106.86 | 107.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 106.92 | 106.86 | 107.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 106.92 | 106.86 | 107.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:00:00 | 106.08 | 106.69 | 107.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 108.87 | 104.08 | 104.82 | SL hit (close>static) qty=1.00 sl=108.38 alert=retest2 |

### Cycle 136 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 108.52 | 105.76 | 105.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 109.47 | 107.34 | 106.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 107.89 | 108.03 | 107.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 107.89 | 108.03 | 107.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 108.13 | 107.97 | 107.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:15:00 | 109.01 | 107.94 | 107.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 111.61 | 108.16 | 107.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 110.04 | 111.36 | 111.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 110.04 | 111.36 | 111.41 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 112.44 | 111.18 | 111.05 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 110.60 | 111.25 | 111.32 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 112.07 | 111.44 | 111.36 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 110.55 | 111.27 | 111.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 15:15:00 | 109.73 | 110.96 | 111.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 111.01 | 110.88 | 111.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 12:15:00 | 111.01 | 110.88 | 111.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 111.01 | 110.88 | 111.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 111.01 | 110.88 | 111.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 110.45 | 110.79 | 111.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:45:00 | 110.24 | 110.54 | 110.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:00:00 | 110.14 | 110.34 | 110.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:30:00 | 110.25 | 109.98 | 110.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 15:00:00 | 108.55 | 109.98 | 110.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 110.88 | 109.97 | 110.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 110.88 | 109.97 | 110.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 110.72 | 110.12 | 110.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:00:00 | 110.72 | 110.12 | 110.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-02 12:15:00 | 111.06 | 110.42 | 110.47 | SL hit (close>static) qty=1.00 sl=111.02 alert=retest2 |

### Cycle 142 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 111.00 | 110.53 | 110.52 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 109.49 | 110.40 | 110.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 109.16 | 110.00 | 110.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 109.00 | 108.91 | 109.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 13:15:00 | 109.00 | 108.91 | 109.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 109.00 | 108.91 | 109.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 109.00 | 108.91 | 109.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 110.28 | 109.21 | 109.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 110.15 | 109.21 | 109.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 110.13 | 109.39 | 109.50 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 110.20 | 109.67 | 109.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 110.76 | 109.85 | 109.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 117.57 | 117.94 | 116.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:45:00 | 117.51 | 117.94 | 116.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 115.55 | 117.31 | 116.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 115.17 | 117.31 | 116.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 115.15 | 116.88 | 116.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 115.15 | 116.88 | 116.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 113.14 | 115.52 | 115.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 111.85 | 113.74 | 114.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 113.42 | 113.39 | 114.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 113.42 | 113.39 | 114.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 112.68 | 113.02 | 113.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 113.42 | 113.02 | 113.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 109.98 | 110.61 | 111.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 109.35 | 110.44 | 111.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:15:00 | 109.52 | 110.44 | 111.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 108.89 | 108.53 | 108.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 108.89 | 108.53 | 108.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 109.39 | 108.72 | 108.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 117.22 | 117.25 | 115.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:45:00 | 117.28 | 117.25 | 115.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 118.19 | 118.53 | 117.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:45:00 | 118.06 | 118.53 | 117.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 118.34 | 118.49 | 117.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 117.85 | 118.49 | 117.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 118.20 | 118.37 | 117.96 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 14:15:00 | 117.56 | 117.79 | 117.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 116.88 | 117.45 | 117.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 12:15:00 | 118.18 | 116.99 | 117.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 12:15:00 | 118.18 | 116.99 | 117.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 118.18 | 116.99 | 117.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:00:00 | 118.18 | 116.99 | 117.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 118.32 | 117.25 | 117.29 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 119.50 | 117.70 | 117.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 15:15:00 | 119.74 | 118.11 | 117.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 13:15:00 | 121.27 | 121.41 | 120.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:45:00 | 121.17 | 121.41 | 120.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 120.78 | 121.16 | 120.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 120.78 | 121.16 | 120.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 121.17 | 121.16 | 120.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 121.88 | 121.06 | 120.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:45:00 | 121.89 | 121.21 | 120.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:15:00 | 123.60 | 121.21 | 120.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 13:15:00 | 121.98 | 121.64 | 121.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 121.37 | 121.58 | 121.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:45:00 | 121.17 | 121.58 | 121.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 121.69 | 121.61 | 121.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 122.88 | 121.67 | 121.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:45:00 | 122.40 | 121.82 | 121.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:45:00 | 122.40 | 121.87 | 121.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 14:15:00 | 121.08 | 121.50 | 121.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 14:15:00 | 121.08 | 121.50 | 121.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 120.53 | 121.22 | 121.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 119.30 | 119.21 | 119.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:15:00 | 119.57 | 119.21 | 119.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 119.66 | 119.30 | 119.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 120.34 | 119.30 | 119.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 118.81 | 119.23 | 119.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:00:00 | 118.70 | 119.13 | 119.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 13:00:00 | 118.70 | 118.96 | 119.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:15:00 | 118.42 | 118.92 | 119.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 15:00:00 | 118.42 | 118.82 | 119.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 118.25 | 118.50 | 118.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:45:00 | 117.85 | 118.37 | 118.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 112.77 | 114.08 | 115.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 112.77 | 114.08 | 115.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 112.50 | 114.08 | 115.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 112.50 | 114.08 | 115.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 14:15:00 | 114.16 | 114.10 | 115.24 | SL hit (close>ema200) qty=0.50 sl=114.10 alert=retest2 |

### Cycle 150 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 117.65 | 115.34 | 115.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 118.60 | 116.00 | 115.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 116.54 | 117.41 | 116.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 116.54 | 117.41 | 116.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 116.54 | 117.41 | 116.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 117.76 | 117.26 | 116.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 10:00:00 | 118.00 | 117.41 | 117.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 113.18 | 116.24 | 116.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 113.18 | 116.24 | 116.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 111.13 | 113.65 | 114.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 11:15:00 | 114.40 | 113.72 | 114.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 11:15:00 | 114.40 | 113.72 | 114.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 114.40 | 113.72 | 114.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:00:00 | 114.40 | 113.72 | 114.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 113.34 | 113.65 | 114.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 112.95 | 113.65 | 114.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 115.97 | 114.13 | 114.45 | SL hit (close>static) qty=1.00 sl=114.43 alert=retest2 |

### Cycle 152 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 116.19 | 114.87 | 114.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 118.52 | 116.10 | 115.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 118.05 | 118.19 | 117.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:45:00 | 118.10 | 118.19 | 117.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 117.68 | 118.09 | 117.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 117.41 | 118.09 | 117.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 117.47 | 117.97 | 117.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 117.47 | 117.97 | 117.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 116.90 | 117.75 | 117.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 116.79 | 117.75 | 117.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 116.72 | 117.36 | 117.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:45:00 | 116.86 | 117.36 | 117.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 116.64 | 117.22 | 117.27 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 119.05 | 117.34 | 117.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 119.76 | 118.31 | 117.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 11:15:00 | 120.34 | 120.45 | 119.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 12:00:00 | 120.34 | 120.45 | 119.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 119.72 | 120.22 | 119.85 | EMA400 retest candle locked (from upside) |

### Cycle 155 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 119.35 | 119.69 | 119.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 119.20 | 119.53 | 119.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 119.65 | 119.56 | 119.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 119.65 | 119.56 | 119.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 119.65 | 119.56 | 119.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 119.64 | 119.56 | 119.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 118.97 | 119.44 | 119.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 118.68 | 119.44 | 119.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:00:00 | 118.91 | 119.33 | 119.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 09:45:00 | 118.78 | 119.00 | 119.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 12:15:00 | 120.26 | 119.20 | 119.27 | SL hit (close>static) qty=1.00 sl=119.69 alert=retest2 |

### Cycle 156 — BUY (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 13:15:00 | 120.64 | 119.49 | 119.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 14:15:00 | 120.90 | 119.77 | 119.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 10:15:00 | 119.50 | 120.01 | 119.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 10:15:00 | 119.50 | 120.01 | 119.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 119.50 | 120.01 | 119.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 119.57 | 120.01 | 119.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 119.48 | 119.90 | 119.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:45:00 | 119.64 | 119.90 | 119.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 119.48 | 119.82 | 119.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:30:00 | 119.52 | 119.82 | 119.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 119.07 | 119.67 | 119.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:00:00 | 119.07 | 119.67 | 119.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 118.53 | 119.44 | 119.53 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 09:15:00 | 121.09 | 119.67 | 119.61 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 118.81 | 119.54 | 119.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 117.30 | 118.99 | 119.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 118.86 | 118.80 | 119.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 12:00:00 | 118.86 | 118.80 | 119.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 119.08 | 118.86 | 119.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:00:00 | 119.08 | 118.86 | 119.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 119.45 | 118.98 | 119.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:15:00 | 119.52 | 118.98 | 119.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 120.04 | 119.19 | 119.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 120.04 | 119.19 | 119.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 15:15:00 | 120.00 | 119.35 | 119.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 121.30 | 119.74 | 119.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 121.01 | 121.13 | 120.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 12:30:00 | 121.14 | 121.13 | 120.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 120.51 | 121.01 | 120.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:45:00 | 120.37 | 121.01 | 120.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 121.73 | 121.15 | 120.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 122.47 | 121.20 | 120.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:45:00 | 122.00 | 121.48 | 121.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 122.70 | 121.61 | 121.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 10:45:00 | 122.54 | 121.70 | 121.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 121.90 | 121.81 | 121.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:15:00 | 121.37 | 121.81 | 121.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 121.37 | 121.72 | 121.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 121.29 | 121.72 | 121.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 121.06 | 121.59 | 121.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:15:00 | 121.00 | 121.59 | 121.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 120.76 | 121.42 | 121.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:15:00 | 120.69 | 121.42 | 121.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 120.76 | 121.29 | 121.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 120.76 | 121.29 | 121.33 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 12:15:00 | 121.98 | 121.43 | 121.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 13:15:00 | 122.28 | 121.60 | 121.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 13:15:00 | 127.62 | 128.92 | 127.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 13:15:00 | 127.62 | 128.92 | 127.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 127.62 | 128.92 | 127.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:00:00 | 127.62 | 128.92 | 127.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 126.75 | 128.49 | 127.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 126.75 | 128.49 | 127.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 127.00 | 128.19 | 127.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 127.88 | 128.19 | 127.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 127.80 | 128.37 | 127.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 127.90 | 128.37 | 127.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 125.64 | 127.83 | 127.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 125.05 | 127.83 | 127.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 124.73 | 127.21 | 127.32 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 128.75 | 127.26 | 127.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 14:15:00 | 129.77 | 128.36 | 127.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 13:15:00 | 129.56 | 129.76 | 128.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 13:45:00 | 129.40 | 129.76 | 128.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 129.27 | 129.66 | 128.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 129.27 | 129.66 | 128.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 129.24 | 129.57 | 128.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 128.35 | 129.57 | 128.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 128.30 | 129.32 | 128.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 127.82 | 129.32 | 128.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 127.97 | 129.05 | 128.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 127.87 | 129.05 | 128.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 12:15:00 | 128.00 | 128.67 | 128.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 14:15:00 | 127.63 | 128.36 | 128.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 15:15:00 | 126.70 | 126.40 | 126.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-19 09:15:00 | 135.25 | 126.40 | 126.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 166 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 134.97 | 128.12 | 127.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 12:15:00 | 142.64 | 133.32 | 130.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 10:15:00 | 163.40 | 163.89 | 153.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 11:00:00 | 163.40 | 163.89 | 153.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 153.70 | 161.53 | 156.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 153.70 | 161.53 | 156.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 153.90 | 160.00 | 156.65 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 144.65 | 153.63 | 154.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 144.30 | 146.34 | 148.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 146.40 | 146.18 | 147.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 15:00:00 | 146.40 | 146.18 | 147.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 147.60 | 146.46 | 147.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 144.90 | 146.46 | 147.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 144.85 | 146.14 | 147.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:45:00 | 144.20 | 145.57 | 146.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:00:00 | 144.10 | 145.08 | 146.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 147.98 | 146.73 | 146.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 147.98 | 146.73 | 146.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 151.35 | 147.65 | 147.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 09:15:00 | 147.16 | 149.00 | 148.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 09:15:00 | 147.16 | 149.00 | 148.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 147.16 | 149.00 | 148.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:30:00 | 154.09 | 151.09 | 149.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:45:00 | 153.07 | 152.46 | 150.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 148.92 | 150.24 | 150.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 148.92 | 150.24 | 150.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 147.78 | 149.50 | 150.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 149.20 | 148.97 | 149.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 149.20 | 148.97 | 149.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 149.20 | 148.97 | 149.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 149.20 | 148.97 | 149.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 149.00 | 148.98 | 149.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:00:00 | 148.59 | 148.90 | 149.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 14:15:00 | 149.50 | 149.03 | 149.27 | SL hit (close>static) qty=1.00 sl=149.48 alert=retest2 |

### Cycle 170 — BUY (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 09:15:00 | 150.92 | 149.56 | 149.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 10:15:00 | 152.93 | 150.24 | 149.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 15:15:00 | 153.80 | 154.29 | 152.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 09:15:00 | 153.79 | 154.29 | 152.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 152.02 | 153.84 | 152.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 151.89 | 153.84 | 152.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 151.85 | 153.44 | 152.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 11:15:00 | 153.25 | 153.44 | 152.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-20 09:15:00 | 168.58 | 164.57 | 161.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 11:15:00 | 164.22 | 166.54 | 166.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 163.30 | 165.12 | 165.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 163.55 | 163.14 | 164.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 163.55 | 163.14 | 164.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 163.55 | 163.14 | 164.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 164.24 | 163.14 | 164.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 166.96 | 163.90 | 164.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 166.96 | 163.90 | 164.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 164.03 | 163.93 | 164.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:45:00 | 163.80 | 163.96 | 164.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:30:00 | 162.50 | 163.59 | 164.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:15:00 | 163.64 | 162.83 | 163.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:15:00 | 155.61 | 157.71 | 159.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:15:00 | 154.38 | 157.71 | 159.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:15:00 | 155.46 | 157.71 | 159.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 157.84 | 157.74 | 158.92 | SL hit (close>ema200) qty=0.50 sl=157.74 alert=retest2 |

### Cycle 172 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 152.24 | 151.86 | 151.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 10:15:00 | 153.19 | 152.17 | 151.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 152.25 | 152.52 | 152.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 13:15:00 | 152.25 | 152.52 | 152.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 152.25 | 152.52 | 152.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 152.25 | 152.52 | 152.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 151.79 | 152.38 | 152.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 151.79 | 152.38 | 152.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 152.10 | 152.32 | 152.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 154.70 | 152.32 | 152.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 150.94 | 152.98 | 153.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 150.94 | 152.98 | 153.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 12:15:00 | 150.79 | 152.04 | 152.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 151.71 | 151.19 | 151.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 151.71 | 151.19 | 151.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 151.71 | 151.19 | 151.93 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 152.47 | 152.04 | 152.00 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 151.58 | 151.97 | 151.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 15:15:00 | 151.15 | 151.80 | 151.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 15:15:00 | 147.29 | 147.07 | 148.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:15:00 | 147.08 | 147.07 | 148.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 147.59 | 147.17 | 147.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 147.70 | 147.17 | 147.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 147.97 | 147.33 | 147.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 148.12 | 147.33 | 147.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 147.80 | 147.43 | 147.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:30:00 | 147.47 | 147.52 | 147.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 148.44 | 147.77 | 148.00 | SL hit (close>static) qty=1.00 sl=148.19 alert=retest2 |

### Cycle 176 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 149.80 | 148.33 | 148.22 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 147.65 | 148.31 | 148.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 147.52 | 148.15 | 148.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 13:15:00 | 147.96 | 147.94 | 148.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 13:45:00 | 147.89 | 147.94 | 148.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 146.91 | 147.61 | 147.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:30:00 | 146.25 | 147.07 | 147.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 146.26 | 146.77 | 147.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:45:00 | 146.17 | 146.56 | 147.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 138.94 | 141.05 | 142.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 138.95 | 141.05 | 142.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 138.86 | 141.05 | 142.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 140.16 | 139.65 | 141.28 | SL hit (close>ema200) qty=0.50 sl=139.65 alert=retest2 |

### Cycle 178 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 142.27 | 141.42 | 141.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 144.62 | 142.53 | 141.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 144.01 | 144.33 | 143.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 13:00:00 | 144.01 | 144.33 | 143.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 143.22 | 144.43 | 143.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 143.22 | 144.43 | 143.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 143.18 | 144.18 | 143.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 142.45 | 144.18 | 143.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 142.60 | 143.67 | 143.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 142.24 | 143.38 | 143.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 141.77 | 141.74 | 142.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:00:00 | 141.77 | 141.74 | 142.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 142.89 | 141.92 | 142.15 | EMA400 retest candle locked (from downside) |

### Cycle 180 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 143.36 | 142.46 | 142.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 143.75 | 143.05 | 142.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 143.28 | 143.56 | 143.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 143.28 | 143.56 | 143.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 143.28 | 143.56 | 143.18 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 142.48 | 143.06 | 143.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 142.16 | 142.88 | 143.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 15:15:00 | 141.09 | 140.99 | 141.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 15:15:00 | 141.09 | 140.99 | 141.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 141.09 | 140.99 | 141.75 | EMA400 retest candle locked (from downside) |

### Cycle 182 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 144.29 | 142.06 | 142.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 145.03 | 142.65 | 142.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 143.29 | 143.52 | 143.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 15:00:00 | 143.29 | 143.52 | 143.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 142.89 | 143.39 | 143.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 146.85 | 145.03 | 143.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 145.75 | 146.79 | 146.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 145.75 | 146.79 | 146.81 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 148.30 | 146.92 | 146.85 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 146.11 | 146.95 | 146.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 145.21 | 146.28 | 146.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 141.61 | 141.55 | 142.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 141.59 | 141.55 | 142.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 142.84 | 141.98 | 142.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 142.85 | 141.98 | 142.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 142.55 | 142.09 | 142.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 141.94 | 142.55 | 142.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 143.20 | 142.78 | 142.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 143.20 | 142.78 | 142.76 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 142.30 | 142.73 | 142.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 141.40 | 142.46 | 142.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 138.79 | 138.31 | 139.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:45:00 | 139.19 | 138.31 | 139.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 136.87 | 138.02 | 139.31 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 141.45 | 139.76 | 139.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 141.72 | 140.15 | 139.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 139.92 | 140.13 | 139.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 11:15:00 | 139.92 | 140.13 | 139.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 139.92 | 140.13 | 139.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:45:00 | 140.00 | 140.13 | 139.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 136.09 | 139.32 | 139.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 133.38 | 138.13 | 138.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 136.60 | 136.48 | 137.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 15:15:00 | 135.90 | 134.78 | 135.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 135.90 | 134.78 | 135.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 134.15 | 134.78 | 135.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 136.20 | 134.79 | 135.21 | SL hit (close>static) qty=1.00 sl=135.90 alert=retest2 |

### Cycle 190 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 136.10 | 135.07 | 134.93 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 134.00 | 134.76 | 134.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 131.28 | 133.45 | 134.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 133.06 | 133.02 | 133.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 133.06 | 133.02 | 133.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 134.48 | 133.31 | 133.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 134.48 | 133.31 | 133.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 134.50 | 133.55 | 133.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 141.22 | 133.55 | 133.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 142.20 | 135.28 | 134.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 146.00 | 138.86 | 136.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 152.00 | 152.57 | 149.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 11:00:00 | 153.18 | 152.69 | 149.95 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 15:00:00 | 152.95 | 152.71 | 150.83 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:15:00 | 153.45 | 152.62 | 150.96 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:45:00 | 153.14 | 152.50 | 151.19 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 151.10 | 151.95 | 151.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-09 15:15:00 | 151.10 | 151.95 | 151.40 | SL hit (close<ema400) qty=1.00 sl=151.40 alert=retest1 |

### Cycle 193 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 148.98 | 150.73 | 150.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 13:15:00 | 148.02 | 149.87 | 150.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 11:15:00 | 149.76 | 149.30 | 149.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 11:15:00 | 149.76 | 149.30 | 149.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 149.76 | 149.30 | 149.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 149.30 | 149.30 | 149.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 149.50 | 149.34 | 149.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:30:00 | 149.78 | 149.34 | 149.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 149.89 | 149.45 | 149.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:30:00 | 150.32 | 149.45 | 149.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 150.95 | 149.75 | 149.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:00:00 | 150.95 | 149.75 | 149.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 150.94 | 149.99 | 150.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 149.25 | 149.99 | 150.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:15:00 | 141.79 | 146.24 | 148.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 12:15:00 | 143.45 | 142.51 | 144.56 | SL hit (close>ema200) qty=0.50 sl=142.51 alert=retest2 |

### Cycle 194 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 144.23 | 142.78 | 142.77 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 142.06 | 143.27 | 143.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 13:15:00 | 140.39 | 142.69 | 143.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 14:15:00 | 141.44 | 141.31 | 141.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-26 15:00:00 | 141.44 | 141.31 | 141.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 141.33 | 141.13 | 141.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:30:00 | 141.31 | 141.13 | 141.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 139.14 | 137.62 | 138.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:00:00 | 139.14 | 137.62 | 138.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 137.73 | 137.64 | 138.48 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 140.07 | 138.82 | 138.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 141.40 | 139.58 | 139.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 12:15:00 | 139.56 | 139.71 | 139.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 13:00:00 | 139.56 | 139.71 | 139.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 139.69 | 139.70 | 139.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:45:00 | 139.48 | 139.70 | 139.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 138.98 | 139.56 | 139.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 138.98 | 139.56 | 139.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 139.30 | 139.51 | 139.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 136.99 | 139.51 | 139.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 138.47 | 139.30 | 139.22 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 138.79 | 139.15 | 139.17 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 15:15:00 | 139.50 | 139.17 | 139.16 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 09:15:00 | 138.90 | 139.12 | 139.14 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 139.23 | 139.16 | 139.15 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 12:15:00 | 138.90 | 139.10 | 139.13 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 140.50 | 139.36 | 139.24 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 138.94 | 139.21 | 139.22 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 09:15:00 | 143.27 | 140.02 | 139.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 147.22 | 142.08 | 140.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 14:15:00 | 146.84 | 147.59 | 145.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 15:00:00 | 146.84 | 147.59 | 145.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 147.60 | 147.59 | 145.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 09:30:00 | 150.27 | 148.25 | 146.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 150.50 | 152.37 | 152.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 150.50 | 152.37 | 152.51 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 154.85 | 152.72 | 152.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 10:15:00 | 156.40 | 153.45 | 152.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 13:15:00 | 152.82 | 153.49 | 153.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 13:15:00 | 152.82 | 153.49 | 153.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 152.82 | 153.49 | 153.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 152.82 | 153.49 | 153.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 151.55 | 153.10 | 152.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 151.55 | 153.10 | 152.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 151.75 | 152.83 | 152.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 145.80 | 151.43 | 152.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 148.26 | 148.12 | 149.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 10:45:00 | 148.06 | 148.12 | 149.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 151.07 | 148.71 | 149.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 151.12 | 148.71 | 149.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 151.68 | 149.31 | 149.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 152.80 | 149.31 | 149.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 150.20 | 149.84 | 150.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 154.27 | 149.84 | 150.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 154.79 | 150.83 | 150.46 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 151.19 | 152.57 | 152.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 150.63 | 152.18 | 152.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 154.35 | 152.32 | 152.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 154.35 | 152.32 | 152.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 154.35 | 152.32 | 152.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 155.30 | 152.32 | 152.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 155.23 | 152.90 | 152.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 157.69 | 153.86 | 153.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 152.30 | 154.84 | 154.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 152.30 | 154.84 | 154.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 152.30 | 154.84 | 154.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:15:00 | 151.38 | 154.84 | 154.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 151.06 | 154.09 | 153.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:45:00 | 152.54 | 153.91 | 153.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 167.79 | 164.20 | 162.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 217.90 | 218.95 | 219.07 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 221.38 | 219.27 | 219.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 15:15:00 | 222.69 | 219.96 | 219.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 226.18 | 228.78 | 226.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 226.18 | 228.78 | 226.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 226.18 | 228.78 | 226.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 10:15:00 | 232.84 | 228.79 | 227.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 11:00:00 | 232.75 | 229.58 | 228.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 226.25 | 228.58 | 228.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 226.25 | 228.58 | 228.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 225.26 | 227.58 | 228.12 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-31 12:30:00 | 49.89 | 2023-06-01 13:15:00 | 51.00 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2023-05-31 15:00:00 | 49.76 | 2023-06-01 13:15:00 | 51.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest1 | 2023-06-14 11:15:00 | 53.48 | 2023-06-15 09:15:00 | 54.49 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2023-06-15 11:30:00 | 53.80 | 2023-06-16 13:15:00 | 53.99 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2023-06-15 14:00:00 | 53.74 | 2023-06-16 13:15:00 | 53.99 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2023-06-16 12:30:00 | 53.87 | 2023-06-16 13:15:00 | 53.99 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2023-06-21 14:30:00 | 52.22 | 2023-06-23 09:15:00 | 49.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-22 10:45:00 | 52.22 | 2023-06-23 09:15:00 | 49.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-21 14:30:00 | 52.22 | 2023-06-26 09:15:00 | 47.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-06-22 10:45:00 | 52.22 | 2023-06-26 09:15:00 | 47.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-07-04 12:30:00 | 49.74 | 2023-07-13 14:15:00 | 47.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-04 12:30:00 | 49.74 | 2023-07-14 11:15:00 | 47.73 | STOP_HIT | 0.50 | 4.04% |
| SELL | retest2 | 2023-07-25 13:30:00 | 48.22 | 2023-07-25 14:15:00 | 52.20 | STOP_HIT | 1.00 | -8.25% |
| BUY | retest2 | 2023-08-03 10:30:00 | 54.45 | 2023-08-08 10:15:00 | 53.96 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2023-08-10 09:15:00 | 55.86 | 2023-08-16 11:15:00 | 55.82 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2023-08-10 10:45:00 | 55.68 | 2023-08-16 11:15:00 | 55.82 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2023-08-17 09:15:00 | 56.83 | 2023-08-18 11:15:00 | 62.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-17 11:30:00 | 56.77 | 2023-08-18 11:15:00 | 62.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-17 13:45:00 | 56.96 | 2023-08-18 11:15:00 | 62.66 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-08-28 10:15:00 | 64.50 | 2023-08-29 13:15:00 | 65.80 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2023-08-28 14:15:00 | 64.42 | 2023-08-29 13:15:00 | 65.80 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2023-09-13 11:15:00 | 75.13 | 2023-09-14 09:15:00 | 75.22 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2023-09-13 12:00:00 | 75.14 | 2023-09-14 09:15:00 | 75.22 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2023-09-13 14:45:00 | 75.15 | 2023-09-14 09:15:00 | 75.22 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2023-09-26 09:15:00 | 76.45 | 2023-09-26 10:15:00 | 75.76 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2023-09-28 11:15:00 | 74.80 | 2023-10-09 09:15:00 | 71.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-28 12:30:00 | 74.79 | 2023-10-09 09:15:00 | 71.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-28 13:45:00 | 74.89 | 2023-10-09 09:15:00 | 71.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-28 15:15:00 | 74.79 | 2023-10-09 09:15:00 | 71.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-03 09:45:00 | 74.54 | 2023-10-09 09:15:00 | 70.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-04 11:30:00 | 74.25 | 2023-10-09 09:15:00 | 70.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-04 12:45:00 | 74.45 | 2023-10-09 09:15:00 | 70.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-04 14:15:00 | 74.47 | 2023-10-09 09:15:00 | 70.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-05 10:15:00 | 73.94 | 2023-10-09 09:15:00 | 70.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-05 11:30:00 | 73.98 | 2023-10-09 09:15:00 | 70.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-05 12:15:00 | 73.96 | 2023-10-09 09:15:00 | 70.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-05 14:00:00 | 73.99 | 2023-10-09 09:15:00 | 70.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-28 11:15:00 | 74.80 | 2023-10-10 09:15:00 | 70.14 | STOP_HIT | 0.50 | 6.23% |
| SELL | retest2 | 2023-09-28 12:30:00 | 74.79 | 2023-10-10 09:15:00 | 70.14 | STOP_HIT | 0.50 | 6.22% |
| SELL | retest2 | 2023-09-28 13:45:00 | 74.89 | 2023-10-10 09:15:00 | 70.14 | STOP_HIT | 0.50 | 6.34% |
| SELL | retest2 | 2023-09-28 15:15:00 | 74.79 | 2023-10-10 09:15:00 | 70.14 | STOP_HIT | 0.50 | 6.22% |
| SELL | retest2 | 2023-10-03 09:45:00 | 74.54 | 2023-10-10 09:15:00 | 70.14 | STOP_HIT | 0.50 | 5.90% |
| SELL | retest2 | 2023-10-04 11:30:00 | 74.25 | 2023-10-10 09:15:00 | 70.14 | STOP_HIT | 0.50 | 5.54% |
| SELL | retest2 | 2023-10-04 12:45:00 | 74.45 | 2023-10-10 09:15:00 | 70.14 | STOP_HIT | 0.50 | 5.79% |
| SELL | retest2 | 2023-10-04 14:15:00 | 74.47 | 2023-10-10 09:15:00 | 70.14 | STOP_HIT | 0.50 | 5.81% |
| SELL | retest2 | 2023-10-05 10:15:00 | 73.94 | 2023-10-10 09:15:00 | 70.14 | STOP_HIT | 0.50 | 5.14% |
| SELL | retest2 | 2023-10-05 11:30:00 | 73.98 | 2023-10-10 09:15:00 | 70.14 | STOP_HIT | 0.50 | 5.19% |
| SELL | retest2 | 2023-10-05 12:15:00 | 73.96 | 2023-10-10 09:15:00 | 70.14 | STOP_HIT | 0.50 | 5.16% |
| SELL | retest2 | 2023-10-05 14:00:00 | 73.99 | 2023-10-10 09:15:00 | 70.14 | STOP_HIT | 0.50 | 5.20% |
| SELL | retest2 | 2023-10-11 12:15:00 | 70.77 | 2023-10-16 09:15:00 | 67.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-11 12:15:00 | 70.77 | 2023-10-17 09:15:00 | 68.19 | STOP_HIT | 0.50 | 3.65% |
| BUY | retest2 | 2023-11-08 09:45:00 | 77.41 | 2023-11-13 12:15:00 | 78.71 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2023-11-16 11:45:00 | 78.23 | 2023-11-21 10:15:00 | 78.57 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-12-15 15:00:00 | 108.19 | 2023-12-20 12:15:00 | 105.00 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2023-12-20 12:00:00 | 106.20 | 2023-12-20 12:15:00 | 105.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2023-12-26 13:15:00 | 102.40 | 2023-12-27 10:15:00 | 105.07 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2023-12-26 14:00:00 | 102.43 | 2023-12-27 10:15:00 | 105.07 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-01-19 12:00:00 | 104.40 | 2024-01-20 09:15:00 | 106.20 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-01-19 12:30:00 | 104.16 | 2024-01-20 09:15:00 | 106.20 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-02-01 15:15:00 | 113.10 | 2024-02-02 15:15:00 | 111.59 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-02-14 09:15:00 | 111.00 | 2024-02-14 14:15:00 | 113.98 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2024-03-11 13:00:00 | 114.48 | 2024-03-12 09:15:00 | 112.09 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-03-15 09:30:00 | 106.00 | 2024-03-20 09:15:00 | 100.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-15 09:30:00 | 106.00 | 2024-03-21 09:15:00 | 105.23 | STOP_HIT | 0.50 | 0.73% |
| BUY | retest2 | 2024-03-27 09:15:00 | 107.79 | 2024-03-27 14:15:00 | 103.23 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest2 | 2024-04-09 10:00:00 | 124.73 | 2024-04-09 13:15:00 | 122.60 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-04-09 13:15:00 | 123.85 | 2024-04-09 13:15:00 | 122.60 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-04-18 10:15:00 | 120.03 | 2024-04-22 13:15:00 | 120.11 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-04-18 14:15:00 | 120.05 | 2024-04-22 13:15:00 | 120.11 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-04-22 09:30:00 | 119.87 | 2024-04-22 13:15:00 | 120.11 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-04-22 12:00:00 | 120.08 | 2024-04-22 13:15:00 | 120.11 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-05-21 12:00:00 | 137.92 | 2024-05-28 11:15:00 | 137.12 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-06-26 14:00:00 | 143.98 | 2024-07-02 10:15:00 | 145.22 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-06-26 15:15:00 | 143.92 | 2024-07-02 10:15:00 | 145.22 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-06-27 10:00:00 | 143.98 | 2024-07-02 10:15:00 | 145.22 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-06-28 10:00:00 | 143.79 | 2024-07-02 10:15:00 | 145.22 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-06-28 15:15:00 | 143.40 | 2024-07-09 09:15:00 | 145.58 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-07-01 11:15:00 | 143.71 | 2024-07-09 09:15:00 | 145.58 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-07-01 14:15:00 | 143.52 | 2024-07-09 09:15:00 | 145.58 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-07-01 15:00:00 | 143.40 | 2024-07-09 09:15:00 | 145.58 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-07-02 12:45:00 | 143.00 | 2024-07-09 09:15:00 | 145.58 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-07-02 13:30:00 | 142.66 | 2024-07-09 09:15:00 | 145.58 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-07-03 09:45:00 | 142.68 | 2024-07-09 10:15:00 | 145.98 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-07-03 10:15:00 | 142.96 | 2024-07-09 10:15:00 | 145.98 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-07-08 11:00:00 | 139.46 | 2024-07-09 10:15:00 | 145.98 | STOP_HIT | 1.00 | -4.68% |
| SELL | retest2 | 2024-07-08 15:15:00 | 139.54 | 2024-07-09 10:15:00 | 145.98 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2024-07-16 14:15:00 | 142.07 | 2024-07-23 09:15:00 | 143.20 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-07-18 09:15:00 | 140.78 | 2024-07-23 09:15:00 | 143.20 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-08-07 14:15:00 | 138.47 | 2024-08-12 09:15:00 | 124.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-07 15:00:00 | 138.33 | 2024-08-12 09:15:00 | 124.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-08 15:00:00 | 138.16 | 2024-08-12 09:15:00 | 124.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-09 10:15:00 | 138.53 | 2024-08-12 09:15:00 | 124.68 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-12 09:15:00 | 133.22 | 2024-08-12 09:15:00 | 126.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-12 09:15:00 | 133.22 | 2024-08-12 13:15:00 | 138.48 | STOP_HIT | 0.50 | -3.95% |
| SELL | retest2 | 2024-08-12 14:15:00 | 138.16 | 2024-08-16 14:15:00 | 139.45 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-08-12 14:45:00 | 137.88 | 2024-08-16 15:15:00 | 139.84 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-08-13 12:15:00 | 138.15 | 2024-08-16 15:15:00 | 139.84 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-08-13 15:15:00 | 137.20 | 2024-08-16 15:15:00 | 139.84 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-08-21 10:30:00 | 139.74 | 2024-08-22 09:15:00 | 138.37 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-08-21 11:00:00 | 139.75 | 2024-08-22 09:15:00 | 138.37 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-08-30 11:00:00 | 127.51 | 2024-09-02 13:15:00 | 134.36 | STOP_HIT | 1.00 | -5.37% |
| SELL | retest2 | 2024-08-30 14:00:00 | 127.81 | 2024-09-02 13:15:00 | 134.36 | STOP_HIT | 1.00 | -5.12% |
| SELL | retest2 | 2024-08-30 14:30:00 | 127.66 | 2024-09-02 13:15:00 | 134.36 | STOP_HIT | 1.00 | -5.25% |
| SELL | retest2 | 2024-09-10 15:15:00 | 126.80 | 2024-09-12 13:15:00 | 129.31 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-09-11 09:45:00 | 127.37 | 2024-09-12 13:15:00 | 129.31 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-09-11 11:45:00 | 127.34 | 2024-09-12 13:15:00 | 129.31 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-09-25 14:00:00 | 133.60 | 2024-09-26 09:15:00 | 132.89 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-09-25 15:15:00 | 133.59 | 2024-09-26 09:15:00 | 132.89 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-10-04 09:15:00 | 127.71 | 2024-10-08 15:15:00 | 128.30 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-10-04 10:00:00 | 128.59 | 2024-10-08 15:15:00 | 128.30 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2024-10-04 11:15:00 | 128.58 | 2024-10-08 15:15:00 | 128.30 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2024-10-04 12:15:00 | 128.82 | 2024-10-08 15:15:00 | 128.30 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2024-10-07 10:30:00 | 124.97 | 2024-10-08 15:15:00 | 128.30 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-10-10 09:15:00 | 127.63 | 2024-10-10 10:15:00 | 127.07 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-10-21 15:15:00 | 120.48 | 2024-10-24 13:15:00 | 120.30 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2024-10-24 09:30:00 | 119.84 | 2024-10-24 13:15:00 | 120.30 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-10-24 11:00:00 | 120.07 | 2024-10-24 13:15:00 | 120.30 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-11-05 11:30:00 | 117.06 | 2024-11-05 14:15:00 | 120.79 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2024-12-09 13:15:00 | 107.51 | 2024-12-12 10:15:00 | 110.58 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2024-12-17 13:45:00 | 105.95 | 2024-12-20 14:15:00 | 100.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 13:45:00 | 105.95 | 2024-12-23 13:15:00 | 100.93 | STOP_HIT | 0.50 | 4.74% |
| BUY | retest2 | 2024-12-30 09:15:00 | 102.11 | 2025-01-03 13:15:00 | 104.36 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2024-12-30 15:00:00 | 108.80 | 2025-01-03 13:15:00 | 104.36 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2025-01-08 09:15:00 | 101.15 | 2025-01-10 09:15:00 | 96.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 10:30:00 | 100.80 | 2025-01-10 09:15:00 | 95.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 14:15:00 | 101.30 | 2025-01-10 09:15:00 | 96.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:30:00 | 101.30 | 2025-01-10 09:15:00 | 96.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 101.15 | 2025-01-13 14:15:00 | 91.04 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-08 10:30:00 | 100.80 | 2025-01-13 14:15:00 | 90.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-08 14:15:00 | 101.30 | 2025-01-13 14:15:00 | 91.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 09:30:00 | 101.30 | 2025-01-13 14:15:00 | 91.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 104.81 | 2025-01-27 09:15:00 | 99.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:15:00 | 104.97 | 2025-01-27 09:15:00 | 99.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 103.50 | 2025-01-28 09:15:00 | 98.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 104.81 | 2025-01-28 11:15:00 | 100.66 | STOP_HIT | 0.50 | 3.96% |
| SELL | retest2 | 2025-01-23 14:15:00 | 104.97 | 2025-01-28 11:15:00 | 100.66 | STOP_HIT | 0.50 | 4.11% |
| SELL | retest2 | 2025-01-24 09:45:00 | 103.50 | 2025-01-28 11:15:00 | 100.66 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2025-02-04 10:15:00 | 100.74 | 2025-02-12 12:15:00 | 100.47 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-02-04 13:00:00 | 100.95 | 2025-02-12 12:15:00 | 100.47 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2025-02-04 14:00:00 | 100.92 | 2025-02-12 12:15:00 | 100.47 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-02-05 15:15:00 | 100.85 | 2025-02-12 12:15:00 | 100.47 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-02-06 13:45:00 | 100.55 | 2025-02-12 12:15:00 | 100.47 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-02-07 09:15:00 | 99.68 | 2025-02-12 12:15:00 | 100.47 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-02-11 09:45:00 | 100.41 | 2025-02-12 12:15:00 | 100.47 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-02-18 12:45:00 | 96.88 | 2025-02-20 13:15:00 | 98.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-02-28 10:30:00 | 99.18 | 2025-02-28 13:15:00 | 97.18 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-03-07 10:30:00 | 101.58 | 2025-03-25 10:15:00 | 103.10 | STOP_HIT | 1.00 | 1.50% |
| BUY | retest2 | 2025-03-07 14:45:00 | 101.32 | 2025-03-25 10:15:00 | 103.10 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2025-03-10 09:15:00 | 102.40 | 2025-03-25 10:15:00 | 103.10 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2025-03-11 09:45:00 | 101.51 | 2025-03-25 10:15:00 | 103.10 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-03-11 15:15:00 | 102.56 | 2025-03-25 10:15:00 | 103.10 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2025-03-12 13:15:00 | 102.05 | 2025-03-25 10:15:00 | 103.10 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2025-03-12 14:00:00 | 102.08 | 2025-03-25 10:15:00 | 103.10 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2025-03-18 09:15:00 | 102.84 | 2025-03-25 10:15:00 | 103.10 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2025-03-21 09:15:00 | 104.55 | 2025-03-25 10:15:00 | 103.10 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-03-25 09:30:00 | 104.48 | 2025-03-25 10:15:00 | 103.10 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-05-08 13:00:00 | 106.08 | 2025-05-12 09:15:00 | 108.87 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-05-15 14:15:00 | 109.01 | 2025-05-21 11:15:00 | 110.04 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2025-05-16 09:15:00 | 111.61 | 2025-05-21 11:15:00 | 110.04 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-05-30 09:45:00 | 110.24 | 2025-06-02 12:15:00 | 111.06 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-05-30 14:00:00 | 110.14 | 2025-06-02 12:15:00 | 111.06 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-05-30 14:30:00 | 110.25 | 2025-06-02 12:15:00 | 111.06 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-05-30 15:00:00 | 108.55 | 2025-06-02 12:15:00 | 111.06 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-06-19 10:30:00 | 109.35 | 2025-06-24 12:15:00 | 108.89 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-06-19 11:15:00 | 109.52 | 2025-06-24 12:15:00 | 108.89 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-07-14 09:15:00 | 121.88 | 2025-07-16 14:15:00 | 121.08 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-07-14 09:45:00 | 121.89 | 2025-07-16 14:15:00 | 121.08 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-07-14 10:15:00 | 123.60 | 2025-07-16 14:15:00 | 121.08 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-07-14 13:15:00 | 121.98 | 2025-07-16 14:15:00 | 121.08 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-07-15 09:15:00 | 122.88 | 2025-07-16 14:15:00 | 121.08 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-07-15 09:45:00 | 122.40 | 2025-07-16 14:15:00 | 121.08 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-15 10:45:00 | 122.40 | 2025-07-16 14:15:00 | 121.08 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-07-22 11:00:00 | 118.70 | 2025-07-28 13:15:00 | 112.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 13:00:00 | 118.70 | 2025-07-28 13:15:00 | 112.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 14:15:00 | 118.42 | 2025-07-28 13:15:00 | 112.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 15:00:00 | 118.42 | 2025-07-28 13:15:00 | 112.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 11:00:00 | 118.70 | 2025-07-28 14:15:00 | 114.16 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2025-07-22 13:00:00 | 118.70 | 2025-07-28 14:15:00 | 114.16 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2025-07-22 14:15:00 | 118.42 | 2025-07-28 14:15:00 | 114.16 | STOP_HIT | 0.50 | 3.60% |
| SELL | retest2 | 2025-07-22 15:00:00 | 118.42 | 2025-07-28 14:15:00 | 114.16 | STOP_HIT | 0.50 | 3.60% |
| SELL | retest2 | 2025-07-23 10:45:00 | 117.85 | 2025-07-29 13:15:00 | 117.65 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-08-01 09:15:00 | 117.76 | 2025-08-01 14:15:00 | 113.18 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-08-01 10:00:00 | 118.00 | 2025-08-01 14:15:00 | 113.18 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2025-08-07 13:15:00 | 112.95 | 2025-08-07 14:15:00 | 115.97 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-08-22 11:15:00 | 118.68 | 2025-08-25 12:15:00 | 120.26 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-08-22 12:00:00 | 118.91 | 2025-08-25 12:15:00 | 120.26 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-08-25 09:45:00 | 118.78 | 2025-08-25 12:15:00 | 120.26 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-09-03 09:15:00 | 122.47 | 2025-09-05 11:15:00 | 120.76 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-09-03 13:45:00 | 122.00 | 2025-09-05 11:15:00 | 120.76 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-09-04 09:15:00 | 122.70 | 2025-09-05 11:15:00 | 120.76 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-04 10:45:00 | 122.54 | 2025-09-05 11:15:00 | 120.76 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-30 11:45:00 | 144.20 | 2025-10-01 12:15:00 | 147.98 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-09-30 14:00:00 | 144.10 | 2025-10-01 12:15:00 | 147.98 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-10-07 09:30:00 | 154.09 | 2025-10-08 14:15:00 | 148.92 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2025-10-07 13:45:00 | 153.07 | 2025-10-08 14:15:00 | 148.92 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-10-10 12:00:00 | 148.59 | 2025-10-10 14:15:00 | 149.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-10-13 09:15:00 | 148.53 | 2025-10-13 09:15:00 | 150.92 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-10-15 11:15:00 | 153.25 | 2025-10-20 09:15:00 | 168.58 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-29 13:45:00 | 163.80 | 2025-11-04 09:15:00 | 155.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 14:30:00 | 162.50 | 2025-11-04 09:15:00 | 154.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 13:15:00 | 163.64 | 2025-11-04 09:15:00 | 155.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 13:45:00 | 163.80 | 2025-11-04 10:15:00 | 157.84 | STOP_HIT | 0.50 | 3.64% |
| SELL | retest2 | 2025-10-29 14:30:00 | 162.50 | 2025-11-04 10:15:00 | 157.84 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2025-10-30 13:15:00 | 163.64 | 2025-11-04 10:15:00 | 157.84 | STOP_HIT | 0.50 | 3.54% |
| BUY | retest2 | 2025-11-14 09:15:00 | 154.70 | 2025-11-18 09:15:00 | 150.94 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-11-26 12:30:00 | 147.47 | 2025-11-26 14:15:00 | 148.44 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-12-02 13:30:00 | 146.25 | 2025-12-08 13:15:00 | 138.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 10:30:00 | 146.26 | 2025-12-08 13:15:00 | 138.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 12:45:00 | 146.17 | 2025-12-08 13:15:00 | 138.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 13:30:00 | 146.25 | 2025-12-09 11:15:00 | 140.16 | STOP_HIT | 0.50 | 4.16% |
| SELL | retest2 | 2025-12-03 10:30:00 | 146.26 | 2025-12-09 11:15:00 | 140.16 | STOP_HIT | 0.50 | 4.17% |
| SELL | retest2 | 2025-12-03 12:45:00 | 146.17 | 2025-12-09 11:15:00 | 140.16 | STOP_HIT | 0.50 | 4.11% |
| BUY | retest2 | 2026-01-01 09:30:00 | 146.85 | 2026-01-05 13:15:00 | 145.75 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-01-14 10:00:00 | 141.94 | 2026-01-16 09:15:00 | 143.20 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-01-29 09:15:00 | 134.15 | 2026-01-29 13:15:00 | 136.20 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-01-29 14:15:00 | 133.91 | 2026-02-01 10:15:00 | 136.10 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest1 | 2026-02-06 11:00:00 | 153.18 | 2026-02-09 15:15:00 | 151.10 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest1 | 2026-02-06 15:00:00 | 152.95 | 2026-02-09 15:15:00 | 151.10 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest1 | 2026-02-09 09:15:00 | 153.45 | 2026-02-09 15:15:00 | 151.10 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest1 | 2026-02-09 10:45:00 | 153.14 | 2026-02-09 15:15:00 | 151.10 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-02-12 09:15:00 | 149.25 | 2026-02-13 11:15:00 | 141.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 149.25 | 2026-02-16 12:15:00 | 143.45 | STOP_HIT | 0.50 | 3.89% |
| BUY | retest2 | 2026-03-16 09:30:00 | 150.27 | 2026-03-19 14:15:00 | 150.50 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2026-04-02 11:45:00 | 152.54 | 2026-04-08 09:15:00 | 167.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-07 10:15:00 | 232.84 | 2026-05-08 12:15:00 | 226.25 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-05-07 11:00:00 | 232.75 | 2026-05-08 12:15:00 | 226.25 | STOP_HIT | 1.00 | -2.79% |
