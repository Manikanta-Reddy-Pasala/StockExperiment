# Mangalore Refinery & Petrochemicals Ltd. (MRPL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 167.97
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 77 |
| ALERT1 | 56 |
| ALERT2 | 53 |
| ALERT2_SKIP | 26 |
| ALERT3 | 116 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 70 |
| PARTIAL | 14 |
| TARGET_HIT | 11 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 86 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 43
- **Target hits / Stop hits / Partials:** 11 / 61 / 14
- **Avg / median % per leg:** 1.96% / 0.68%
- **Sum % (uncompounded):** 168.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 13 | 38.2% | 8 | 26 | 0 | 1.53% | 52.0% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -5.25% | -10.5% |
| BUY @ 3rd Alert (retest2) | 32 | 13 | 40.6% | 8 | 24 | 0 | 1.95% | 62.5% |
| SELL (all) | 52 | 30 | 57.7% | 3 | 35 | 14 | 2.24% | 116.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 52 | 30 | 57.7% | 3 | 35 | 14 | 2.24% | 116.3% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -5.25% | -10.5% |
| retest2 (combined) | 84 | 43 | 51.2% | 11 | 59 | 14 | 2.13% | 178.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 131.23 | 128.49 | 128.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 131.99 | 130.42 | 129.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 12:15:00 | 138.32 | 138.36 | 136.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 13:00:00 | 138.32 | 138.36 | 136.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 141.42 | 138.75 | 137.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 137.70 | 138.75 | 137.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 139.67 | 139.69 | 138.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 139.62 | 139.69 | 138.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 138.30 | 139.42 | 138.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 138.30 | 139.42 | 138.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 138.99 | 139.33 | 138.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:45:00 | 139.37 | 139.34 | 138.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 10:15:00 | 139.17 | 139.32 | 138.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 137.69 | 138.70 | 138.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 137.69 | 138.70 | 138.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 136.71 | 138.30 | 138.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 138.18 | 137.93 | 138.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 138.18 | 137.93 | 138.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 138.18 | 137.93 | 138.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 138.18 | 137.93 | 138.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 138.88 | 138.12 | 138.33 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 139.89 | 138.47 | 138.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 142.50 | 139.28 | 138.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 140.06 | 141.12 | 140.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 140.06 | 141.12 | 140.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 140.06 | 141.12 | 140.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 139.39 | 141.12 | 140.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 140.58 | 141.02 | 140.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 140.82 | 141.02 | 140.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:30:00 | 140.85 | 140.91 | 140.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 15:00:00 | 140.74 | 140.76 | 140.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 15:15:00 | 143.82 | 145.64 | 145.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 143.82 | 145.64 | 145.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 12:15:00 | 143.03 | 144.33 | 145.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 143.88 | 143.84 | 144.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 143.88 | 143.84 | 144.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 143.88 | 143.84 | 144.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:30:00 | 146.94 | 143.84 | 144.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 142.65 | 143.61 | 144.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:15:00 | 141.90 | 143.61 | 144.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 09:30:00 | 142.04 | 141.65 | 142.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 10:15:00 | 143.24 | 141.97 | 141.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 143.24 | 141.97 | 141.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 11:15:00 | 144.25 | 142.43 | 142.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 142.46 | 142.89 | 142.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 142.46 | 142.89 | 142.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 142.46 | 142.89 | 142.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 142.46 | 142.89 | 142.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 142.35 | 142.78 | 142.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 142.35 | 142.78 | 142.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 143.30 | 142.88 | 142.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 147.35 | 143.29 | 142.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 12:15:00 | 142.75 | 144.27 | 144.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 142.75 | 144.27 | 144.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 141.06 | 143.63 | 144.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 138.87 | 137.38 | 138.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 138.87 | 137.38 | 138.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 138.87 | 137.38 | 138.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 139.65 | 137.38 | 138.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 138.17 | 137.54 | 138.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 137.58 | 137.54 | 138.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 137.29 | 137.45 | 138.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 138.76 | 135.54 | 135.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 138.76 | 135.54 | 135.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 143.92 | 137.22 | 136.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 10:15:00 | 142.51 | 142.54 | 140.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 10:45:00 | 142.50 | 142.54 | 140.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 141.05 | 141.91 | 140.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:00:00 | 141.05 | 141.91 | 140.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 140.50 | 141.63 | 140.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:45:00 | 140.65 | 141.63 | 140.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 140.24 | 141.35 | 140.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 140.24 | 141.35 | 140.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 140.45 | 141.17 | 140.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:30:00 | 140.45 | 141.17 | 140.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 141.85 | 141.22 | 140.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 142.94 | 141.22 | 140.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 145.70 | 147.58 | 147.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 145.70 | 147.58 | 147.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 143.86 | 146.47 | 147.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 145.00 | 144.97 | 145.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 10:45:00 | 145.08 | 144.97 | 145.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 146.38 | 145.25 | 145.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 146.38 | 145.25 | 145.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 150.07 | 146.22 | 146.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:00:00 | 150.07 | 146.22 | 146.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 13:15:00 | 148.05 | 146.58 | 146.45 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 145.56 | 146.33 | 146.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 143.29 | 145.30 | 145.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 143.05 | 142.38 | 143.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 143.05 | 142.38 | 143.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 143.05 | 142.38 | 143.22 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 13:15:00 | 145.77 | 144.02 | 143.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 10:15:00 | 146.56 | 145.06 | 144.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 15:15:00 | 147.00 | 147.05 | 146.27 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 09:15:00 | 148.15 | 147.05 | 146.27 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 10:45:00 | 147.83 | 147.41 | 146.58 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 140.22 | 146.77 | 146.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 12 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 140.22 | 146.77 | 146.77 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 148.75 | 144.30 | 144.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 10:15:00 | 152.18 | 145.88 | 144.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 12:15:00 | 153.04 | 153.82 | 150.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 13:00:00 | 153.04 | 153.82 | 150.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 151.08 | 153.04 | 150.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 150.84 | 153.04 | 150.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 147.10 | 151.62 | 150.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 147.10 | 151.62 | 150.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 145.04 | 150.30 | 150.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 145.10 | 150.30 | 150.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 144.84 | 149.21 | 149.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 142.20 | 147.10 | 148.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 125.39 | 124.90 | 126.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:45:00 | 125.40 | 124.90 | 126.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 126.88 | 125.52 | 126.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 126.88 | 125.52 | 126.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 126.95 | 125.81 | 126.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 127.99 | 125.81 | 126.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 126.62 | 125.97 | 126.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:00:00 | 126.23 | 126.02 | 126.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:15:00 | 126.20 | 126.33 | 126.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:30:00 | 124.88 | 126.09 | 126.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:45:00 | 126.11 | 124.02 | 124.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 126.81 | 124.58 | 124.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 126.81 | 124.58 | 124.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-08 11:15:00 | 126.55 | 124.97 | 124.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 11:15:00 | 126.55 | 124.97 | 124.89 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 124.29 | 125.04 | 125.05 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 125.93 | 125.01 | 125.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 15:15:00 | 126.40 | 125.29 | 125.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 126.10 | 126.52 | 125.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 15:00:00 | 126.10 | 126.52 | 125.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 126.80 | 126.57 | 126.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:15:00 | 125.60 | 126.57 | 126.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 124.83 | 126.23 | 125.93 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 11:15:00 | 124.70 | 125.68 | 125.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 13:15:00 | 124.50 | 125.29 | 125.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 14:15:00 | 124.00 | 123.79 | 124.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-14 15:00:00 | 124.00 | 123.79 | 124.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 124.21 | 123.78 | 124.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:00:00 | 123.05 | 123.71 | 124.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:30:00 | 123.25 | 123.51 | 124.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:45:00 | 123.12 | 123.49 | 123.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 11:15:00 | 125.41 | 124.20 | 124.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 125.41 | 124.20 | 124.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 126.20 | 125.02 | 124.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 125.38 | 125.43 | 125.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 15:00:00 | 125.38 | 125.43 | 125.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 125.08 | 125.36 | 125.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 127.70 | 125.36 | 125.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:00:00 | 125.60 | 126.44 | 125.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 125.44 | 126.22 | 125.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 14:15:00 | 125.24 | 125.73 | 125.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 125.24 | 125.73 | 125.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 124.50 | 125.48 | 125.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 12:15:00 | 123.79 | 123.58 | 124.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 13:15:00 | 123.70 | 123.58 | 124.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 123.96 | 123.79 | 124.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 122.43 | 123.79 | 124.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 10:15:00 | 123.37 | 123.75 | 124.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 125.02 | 123.08 | 123.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 125.02 | 123.08 | 123.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 125.96 | 124.44 | 123.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 126.53 | 126.93 | 126.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 126.04 | 126.75 | 126.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 126.04 | 126.75 | 126.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 126.04 | 126.75 | 126.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 125.90 | 126.58 | 126.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:30:00 | 125.98 | 126.58 | 126.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 125.05 | 125.97 | 125.94 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 125.00 | 125.78 | 125.86 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 127.13 | 126.03 | 125.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 127.65 | 126.53 | 126.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 126.72 | 126.92 | 126.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 14:15:00 | 126.72 | 126.92 | 126.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 126.72 | 126.92 | 126.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 126.72 | 126.92 | 126.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 126.81 | 126.90 | 126.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 126.06 | 126.90 | 126.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 127.52 | 127.02 | 126.66 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 125.76 | 126.40 | 126.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 15:15:00 | 125.12 | 126.01 | 126.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 127.75 | 126.36 | 126.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 127.75 | 126.36 | 126.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 127.75 | 126.36 | 126.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 127.85 | 126.36 | 126.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 127.26 | 126.54 | 126.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 128.85 | 127.25 | 126.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 13:15:00 | 128.12 | 128.23 | 127.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:00:00 | 128.12 | 128.23 | 127.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 127.65 | 128.11 | 127.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 127.65 | 128.11 | 127.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 127.65 | 128.02 | 127.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 126.85 | 128.02 | 127.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 126.89 | 127.79 | 127.51 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 126.31 | 127.19 | 127.28 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 127.48 | 127.24 | 127.23 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 14:15:00 | 126.95 | 127.17 | 127.20 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 128.00 | 127.33 | 127.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 129.25 | 128.01 | 127.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 10:15:00 | 130.85 | 130.86 | 129.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 11:00:00 | 130.85 | 130.86 | 129.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 130.02 | 130.59 | 130.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 130.02 | 130.59 | 130.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 129.75 | 130.42 | 130.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 130.90 | 130.42 | 130.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 130.29 | 130.69 | 130.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:00:00 | 130.29 | 130.69 | 130.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 129.95 | 130.54 | 130.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 129.94 | 130.54 | 130.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 129.50 | 130.21 | 130.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 128.96 | 129.96 | 130.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 128.98 | 128.40 | 128.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 128.98 | 128.40 | 128.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 128.98 | 128.40 | 128.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 129.70 | 128.40 | 128.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 128.63 | 128.45 | 128.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:45:00 | 128.35 | 128.42 | 128.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 13:15:00 | 129.17 | 128.57 | 128.85 | SL hit (close>static) qty=1.00 sl=128.98 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 11:15:00 | 129.32 | 129.02 | 129.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-26 12:15:00 | 129.80 | 129.18 | 129.07 | Break + close above crossover candle high |

### Cycle 32 — SELL (started 2025-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 13:15:00 | 127.53 | 128.85 | 128.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 15:15:00 | 127.30 | 128.32 | 128.67 | Break + close below crossover candle low |

### Cycle 33 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 132.57 | 129.17 | 129.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 10:15:00 | 134.66 | 130.27 | 129.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 10:15:00 | 132.85 | 132.92 | 131.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-30 10:30:00 | 132.71 | 132.92 | 131.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 131.95 | 132.63 | 131.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:00:00 | 131.95 | 132.63 | 131.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 132.58 | 132.62 | 131.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 134.00 | 132.57 | 131.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 15:00:00 | 133.26 | 133.33 | 132.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 12:15:00 | 133.37 | 132.93 | 132.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-03 14:15:00 | 146.59 | 136.38 | 134.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 145.28 | 145.79 | 145.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 11:15:00 | 143.84 | 145.19 | 145.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 141.11 | 140.76 | 142.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 141.11 | 140.76 | 142.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 141.11 | 140.76 | 142.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:30:00 | 141.99 | 140.76 | 142.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 141.90 | 140.82 | 141.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 142.20 | 140.82 | 141.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 143.85 | 141.43 | 141.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 143.85 | 141.43 | 141.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 142.27 | 141.60 | 141.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 15:00:00 | 141.77 | 141.63 | 141.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 145.97 | 142.56 | 142.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 145.97 | 142.56 | 142.30 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 140.95 | 142.59 | 142.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 140.20 | 142.11 | 142.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 143.23 | 141.79 | 142.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 10:15:00 | 143.23 | 141.79 | 142.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 143.23 | 141.79 | 142.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 143.23 | 141.79 | 142.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 145.65 | 142.56 | 142.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 13:15:00 | 151.00 | 145.05 | 143.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 145.87 | 147.89 | 145.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 145.87 | 147.89 | 145.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 145.87 | 147.89 | 145.90 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 145.25 | 145.63 | 145.66 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 150.40 | 146.58 | 146.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 12:15:00 | 150.80 | 148.48 | 147.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 15:15:00 | 162.00 | 162.36 | 159.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 09:15:00 | 162.44 | 162.36 | 159.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 167.70 | 163.43 | 159.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 11:15:00 | 171.85 | 166.70 | 163.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:30:00 | 169.50 | 168.27 | 165.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 171.72 | 168.22 | 165.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 13:15:00 | 169.48 | 173.02 | 171.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 170.85 | 172.58 | 171.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:15:00 | 171.29 | 172.58 | 171.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 10:45:00 | 171.25 | 171.21 | 170.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 169.13 | 171.26 | 171.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 169.13 | 171.26 | 171.27 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 171.67 | 171.27 | 171.27 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 170.83 | 171.21 | 171.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 15:15:00 | 170.00 | 170.97 | 171.13 | Break + close below crossover candle low |

### Cycle 43 — BUY (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 09:15:00 | 172.30 | 171.24 | 171.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 176.35 | 173.05 | 172.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 12:15:00 | 174.86 | 175.29 | 173.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 12:45:00 | 174.50 | 175.29 | 173.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 173.36 | 174.82 | 174.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 176.65 | 174.82 | 174.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:00:00 | 176.24 | 175.10 | 174.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 172.17 | 174.24 | 174.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 172.17 | 174.24 | 174.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 11:15:00 | 171.70 | 173.37 | 173.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 175.76 | 173.20 | 173.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 175.76 | 173.20 | 173.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 175.76 | 173.20 | 173.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 175.76 | 173.20 | 173.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 175.00 | 173.56 | 173.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 175.14 | 173.56 | 173.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 174.55 | 173.76 | 173.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 180.49 | 175.42 | 174.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 11:15:00 | 178.46 | 178.79 | 176.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 12:00:00 | 178.46 | 178.79 | 176.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 176.10 | 178.28 | 177.40 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 175.39 | 176.74 | 176.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 14:15:00 | 174.53 | 176.04 | 176.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 178.56 | 176.07 | 176.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 178.56 | 176.07 | 176.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 178.56 | 176.07 | 176.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 178.56 | 176.07 | 176.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 179.55 | 176.77 | 176.68 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 174.32 | 176.62 | 176.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 173.51 | 175.65 | 176.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 11:15:00 | 161.60 | 161.51 | 163.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:30:00 | 161.52 | 161.51 | 163.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 162.51 | 159.71 | 160.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:30:00 | 158.63 | 159.85 | 160.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 15:15:00 | 158.74 | 159.73 | 160.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 158.31 | 158.98 | 159.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:30:00 | 158.80 | 159.17 | 159.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 150.80 | 153.64 | 155.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 150.86 | 153.64 | 155.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 150.70 | 152.99 | 154.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 150.39 | 152.99 | 154.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 151.46 | 151.25 | 153.19 | SL hit (close>ema200) qty=0.50 sl=151.25 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 154.70 | 153.86 | 153.77 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 153.05 | 153.71 | 153.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 151.38 | 153.24 | 153.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 15:15:00 | 149.61 | 149.50 | 150.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:15:00 | 150.20 | 149.50 | 150.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 151.38 | 149.87 | 150.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 151.81 | 149.87 | 150.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 150.95 | 150.09 | 150.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 11:30:00 | 150.48 | 150.15 | 150.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 150.00 | 150.22 | 150.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 148.98 | 147.27 | 147.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 148.98 | 147.27 | 147.19 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 146.59 | 147.54 | 147.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 15:15:00 | 146.45 | 147.20 | 147.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 146.36 | 146.35 | 146.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 10:15:00 | 146.36 | 146.35 | 146.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 146.36 | 146.35 | 146.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:45:00 | 146.59 | 146.35 | 146.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 146.77 | 146.44 | 146.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 146.77 | 146.44 | 146.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 145.61 | 146.27 | 146.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 13:30:00 | 145.49 | 146.13 | 146.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:00:00 | 145.53 | 145.69 | 146.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 145.60 | 145.67 | 146.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 146.81 | 143.76 | 144.35 | SL hit (close>static) qty=1.00 sl=146.77 alert=retest2 |

### Cycle 53 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 152.25 | 146.16 | 145.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 155.48 | 148.02 | 146.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 09:15:00 | 149.76 | 152.10 | 150.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 149.76 | 152.10 | 150.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 149.76 | 152.10 | 150.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 149.76 | 152.10 | 150.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 151.16 | 151.91 | 150.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 13:00:00 | 151.33 | 151.68 | 150.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 15:15:00 | 151.40 | 151.47 | 150.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:00:00 | 151.61 | 151.30 | 150.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 149.31 | 150.60 | 150.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 149.31 | 150.60 | 150.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 15:15:00 | 148.51 | 150.00 | 150.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 147.99 | 147.74 | 148.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 147.99 | 147.74 | 148.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 147.99 | 147.74 | 148.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 148.72 | 147.74 | 148.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 148.28 | 147.86 | 148.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:45:00 | 147.15 | 147.78 | 148.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 14:00:00 | 147.15 | 147.65 | 148.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 15:00:00 | 146.77 | 147.48 | 148.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 139.79 | 143.22 | 145.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 139.79 | 143.22 | 145.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 139.43 | 143.22 | 145.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 12:15:00 | 139.89 | 139.18 | 141.29 | SL hit (close>ema200) qty=0.50 sl=139.18 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 145.10 | 142.43 | 142.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 09:15:00 | 147.14 | 144.58 | 143.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 152.30 | 152.37 | 149.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 13:30:00 | 152.35 | 152.37 | 149.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 149.96 | 151.79 | 149.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 146.73 | 151.79 | 149.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 145.75 | 150.58 | 149.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 145.75 | 150.58 | 149.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 145.60 | 149.59 | 149.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 145.13 | 149.59 | 149.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 144.46 | 148.56 | 148.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 14:15:00 | 143.51 | 146.27 | 147.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 146.17 | 142.57 | 144.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 146.17 | 142.57 | 144.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 146.17 | 142.57 | 144.26 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 153.07 | 145.76 | 145.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 156.24 | 150.31 | 147.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 153.63 | 154.00 | 151.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:45:00 | 153.69 | 154.00 | 151.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 155.46 | 154.51 | 152.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 10:00:00 | 156.16 | 154.84 | 153.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 11:45:00 | 156.71 | 155.28 | 153.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 13:00:00 | 156.05 | 155.44 | 153.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 14:30:00 | 156.03 | 155.68 | 154.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 166.20 | 157.87 | 155.46 | EMA400 retest candle locked (from upside) |
| Target hit | 2026-01-29 09:15:00 | 171.78 | 164.45 | 160.58 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 165.12 | 169.29 | 169.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 162.88 | 168.01 | 169.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 169.06 | 168.22 | 169.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 169.06 | 168.22 | 169.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 169.06 | 168.22 | 169.06 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 172.69 | 169.91 | 169.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 177.26 | 171.38 | 170.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 182.49 | 182.69 | 179.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:15:00 | 187.77 | 182.69 | 179.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 182.00 | 183.56 | 182.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:30:00 | 182.55 | 183.56 | 182.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 183.27 | 183.50 | 182.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 184.12 | 183.84 | 182.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 185.69 | 190.67 | 191.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 185.69 | 190.67 | 191.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 183.51 | 186.38 | 188.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 185.93 | 185.69 | 187.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 185.93 | 185.69 | 187.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 187.14 | 186.02 | 187.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:00:00 | 185.14 | 185.84 | 187.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 191.33 | 187.00 | 187.37 | SL hit (close>static) qty=1.00 sl=188.35 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 193.20 | 188.24 | 187.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 193.60 | 189.31 | 188.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 11:15:00 | 191.43 | 191.51 | 189.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 12:00:00 | 191.43 | 191.51 | 189.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 190.20 | 191.35 | 190.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 190.20 | 191.35 | 190.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 189.93 | 191.06 | 190.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:15:00 | 189.90 | 191.06 | 190.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 189.90 | 190.83 | 190.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 192.27 | 190.83 | 190.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 15:15:00 | 191.00 | 191.96 | 192.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 191.00 | 191.96 | 192.01 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 192.42 | 192.05 | 192.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 10:15:00 | 195.81 | 193.48 | 192.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 198.09 | 198.78 | 197.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 198.09 | 198.78 | 197.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 198.09 | 198.78 | 197.01 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 194.03 | 196.42 | 196.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 191.73 | 194.99 | 195.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 10:15:00 | 189.47 | 189.45 | 191.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:45:00 | 188.90 | 189.45 | 191.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 191.52 | 189.86 | 191.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:00:00 | 191.52 | 189.86 | 191.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 191.79 | 190.25 | 191.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 13:00:00 | 191.79 | 190.25 | 191.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 191.47 | 190.49 | 191.75 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 10:15:00 | 197.23 | 193.09 | 192.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 201.00 | 196.12 | 194.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 192.30 | 201.73 | 199.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 192.30 | 201.73 | 199.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 192.30 | 201.73 | 199.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:45:00 | 193.12 | 201.73 | 199.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 189.65 | 199.31 | 198.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:45:00 | 191.28 | 199.31 | 198.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 190.40 | 197.53 | 197.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 189.07 | 194.65 | 196.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 10:15:00 | 194.20 | 192.82 | 194.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 11:00:00 | 194.20 | 192.82 | 194.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 194.18 | 193.09 | 194.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:45:00 | 195.37 | 193.09 | 194.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 194.00 | 193.28 | 194.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:45:00 | 195.75 | 193.28 | 194.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 195.02 | 193.62 | 194.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:30:00 | 194.90 | 193.62 | 194.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 193.00 | 193.50 | 194.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 191.52 | 193.78 | 194.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 181.94 | 188.65 | 191.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 10:15:00 | 189.01 | 188.72 | 191.03 | SL hit (close>ema200) qty=0.50 sl=188.72 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 12:15:00 | 195.81 | 188.09 | 187.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-16 13:15:00 | 200.55 | 190.58 | 188.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 09:15:00 | 195.59 | 196.24 | 192.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-17 09:30:00 | 195.93 | 196.24 | 192.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 202.50 | 197.49 | 193.20 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 15:15:00 | 190.81 | 194.18 | 194.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 188.29 | 193.00 | 194.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 194.75 | 188.77 | 190.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 194.75 | 188.77 | 190.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 194.75 | 188.77 | 190.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 197.37 | 188.77 | 190.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 196.90 | 190.40 | 191.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 196.90 | 190.40 | 191.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 192.30 | 191.46 | 191.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:00:00 | 191.76 | 191.52 | 191.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 15:15:00 | 182.17 | 185.53 | 188.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 15:15:00 | 181.65 | 181.37 | 184.04 | SL hit (close>ema200) qty=0.50 sl=181.37 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 187.22 | 185.21 | 185.18 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 179.10 | 184.33 | 184.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 177.47 | 180.52 | 182.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 09:15:00 | 182.33 | 180.25 | 182.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 182.33 | 180.25 | 182.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 182.33 | 180.25 | 182.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:30:00 | 183.17 | 180.25 | 182.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 181.82 | 180.57 | 182.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:15:00 | 181.37 | 180.57 | 182.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 183.60 | 181.17 | 182.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:30:00 | 182.76 | 181.17 | 182.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 182.36 | 181.41 | 182.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 13:45:00 | 182.09 | 181.56 | 182.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 14:15:00 | 181.89 | 181.56 | 182.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 11:15:00 | 184.20 | 182.14 | 182.15 | SL hit (close>static) qty=1.00 sl=184.16 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 185.87 | 182.89 | 182.49 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 176.93 | 182.10 | 182.32 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 182.65 | 180.68 | 180.52 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 15:15:00 | 180.00 | 180.54 | 180.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-09 09:15:00 | 178.09 | 180.05 | 180.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 172.88 | 172.26 | 174.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 172.88 | 172.26 | 174.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 172.88 | 172.26 | 174.12 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 15:15:00 | 176.20 | 174.97 | 174.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 178.19 | 175.62 | 175.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 14:15:00 | 175.25 | 176.24 | 175.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 14:15:00 | 175.25 | 176.24 | 175.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 175.25 | 176.24 | 175.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 15:00:00 | 175.25 | 176.24 | 175.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 175.00 | 175.99 | 175.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 175.84 | 175.99 | 175.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-24 09:15:00 | 193.42 | 188.80 | 186.62 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 09:15:00 | 173.27 | 184.63 | 185.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 14:15:00 | 172.32 | 176.92 | 180.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 15:15:00 | 174.31 | 174.15 | 177.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 09:15:00 | 173.87 | 174.15 | 177.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 174.17 | 174.15 | 176.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:45:00 | 172.61 | 174.32 | 176.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 173.38 | 174.06 | 175.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:30:00 | 172.89 | 172.93 | 174.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:15:00 | 163.98 | 167.81 | 170.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:15:00 | 164.71 | 167.81 | 170.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:15:00 | 164.25 | 167.81 | 170.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-05-05 10:15:00 | 156.04 | 160.29 | 164.73 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 77 — BUY (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 09:15:00 | 167.88 | 158.10 | 157.94 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 13:45:00 | 139.37 | 2025-05-22 12:15:00 | 137.69 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-05-22 10:15:00 | 139.17 | 2025-05-22 12:15:00 | 137.69 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-05-27 11:15:00 | 140.82 | 2025-05-30 15:15:00 | 143.82 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2025-05-27 12:30:00 | 140.85 | 2025-05-30 15:15:00 | 143.82 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2025-05-27 15:00:00 | 140.74 | 2025-05-30 15:15:00 | 143.82 | STOP_HIT | 1.00 | 2.19% |
| SELL | retest2 | 2025-06-03 11:15:00 | 141.90 | 2025-06-09 10:15:00 | 143.24 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-06-05 09:30:00 | 142.04 | 2025-06-09 10:15:00 | 143.24 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-06-11 09:15:00 | 147.35 | 2025-06-12 12:15:00 | 142.75 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-06-17 11:45:00 | 137.58 | 2025-06-23 09:15:00 | 138.76 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-06-17 15:15:00 | 137.29 | 2025-06-23 09:15:00 | 138.76 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-06-26 09:15:00 | 142.94 | 2025-07-07 15:15:00 | 145.70 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest1 | 2025-07-18 09:15:00 | 148.15 | 2025-07-21 09:15:00 | 140.22 | STOP_HIT | 1.00 | -5.35% |
| BUY | retest1 | 2025-07-18 10:45:00 | 147.83 | 2025-07-21 09:15:00 | 140.22 | STOP_HIT | 1.00 | -5.15% |
| SELL | retest2 | 2025-08-05 11:00:00 | 126.23 | 2025-08-08 11:15:00 | 126.55 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-08-05 14:15:00 | 126.20 | 2025-08-08 11:15:00 | 126.55 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-08-06 09:30:00 | 124.88 | 2025-08-08 11:15:00 | 126.55 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-08-08 09:45:00 | 126.11 | 2025-08-08 11:15:00 | 126.55 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-08-18 12:00:00 | 123.05 | 2025-08-19 11:15:00 | 125.41 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-08-18 13:30:00 | 123.25 | 2025-08-19 11:15:00 | 125.41 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-08-18 14:45:00 | 123.12 | 2025-08-19 11:15:00 | 125.41 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-08-21 09:15:00 | 127.70 | 2025-08-22 14:15:00 | 125.24 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-08-22 10:00:00 | 125.60 | 2025-08-22 14:15:00 | 125.24 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-08-22 11:15:00 | 125.44 | 2025-08-22 14:15:00 | 125.24 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-08-28 09:15:00 | 122.43 | 2025-09-01 11:15:00 | 125.02 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-08-28 10:15:00 | 123.37 | 2025-09-01 11:15:00 | 125.02 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-09-25 11:45:00 | 128.35 | 2025-09-25 13:15:00 | 129.17 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-09-26 09:15:00 | 126.70 | 2025-09-26 10:15:00 | 129.45 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-10-01 09:15:00 | 134.00 | 2025-10-03 14:15:00 | 146.59 | TARGET_HIT | 1.00 | 9.39% |
| BUY | retest2 | 2025-10-01 15:00:00 | 133.26 | 2025-10-03 14:15:00 | 146.71 | TARGET_HIT | 1.00 | 10.09% |
| BUY | retest2 | 2025-10-03 12:15:00 | 133.37 | 2025-10-06 09:15:00 | 147.40 | TARGET_HIT | 1.00 | 10.52% |
| SELL | retest2 | 2025-10-15 15:00:00 | 141.77 | 2025-10-16 09:15:00 | 145.97 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-11-03 11:15:00 | 171.85 | 2025-11-10 10:15:00 | 169.13 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-11-03 13:30:00 | 169.50 | 2025-11-10 10:15:00 | 169.13 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-11-04 09:15:00 | 171.72 | 2025-11-10 10:15:00 | 169.13 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-11-06 13:15:00 | 169.48 | 2025-11-10 10:15:00 | 169.13 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-11-06 14:15:00 | 171.29 | 2025-11-10 10:15:00 | 169.13 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-11-07 10:45:00 | 171.25 | 2025-11-10 10:15:00 | 169.13 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-13 09:15:00 | 176.65 | 2025-11-14 09:15:00 | 172.17 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-11-13 10:00:00 | 176.24 | 2025-11-14 09:15:00 | 172.17 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-12-02 13:30:00 | 158.63 | 2025-12-08 12:15:00 | 150.80 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2025-12-02 15:15:00 | 158.74 | 2025-12-08 12:15:00 | 150.86 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2025-12-04 09:15:00 | 158.31 | 2025-12-08 13:15:00 | 150.70 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2025-12-04 11:30:00 | 158.80 | 2025-12-08 13:15:00 | 150.39 | PARTIAL | 0.50 | 5.29% |
| SELL | retest2 | 2025-12-02 13:30:00 | 158.63 | 2025-12-09 11:15:00 | 151.46 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2025-12-02 15:15:00 | 158.74 | 2025-12-09 11:15:00 | 151.46 | STOP_HIT | 0.50 | 4.59% |
| SELL | retest2 | 2025-12-04 09:15:00 | 158.31 | 2025-12-09 11:15:00 | 151.46 | STOP_HIT | 0.50 | 4.33% |
| SELL | retest2 | 2025-12-04 11:30:00 | 158.80 | 2025-12-09 11:15:00 | 151.46 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2025-12-15 11:30:00 | 150.48 | 2025-12-22 09:15:00 | 148.98 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2025-12-16 10:15:00 | 150.00 | 2025-12-22 09:15:00 | 148.98 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2025-12-26 13:30:00 | 145.49 | 2025-12-31 09:15:00 | 146.81 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-29 10:00:00 | 145.53 | 2025-12-31 09:15:00 | 146.81 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-29 11:00:00 | 145.60 | 2025-12-31 09:15:00 | 146.81 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-01-02 13:00:00 | 151.33 | 2026-01-05 13:15:00 | 149.31 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-01-02 15:15:00 | 151.40 | 2026-01-05 13:15:00 | 149.31 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-01-05 11:00:00 | 151.61 | 2026-01-05 13:15:00 | 149.31 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-01-07 12:45:00 | 147.15 | 2026-01-09 09:15:00 | 139.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 14:00:00 | 147.15 | 2026-01-09 09:15:00 | 139.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 15:00:00 | 146.77 | 2026-01-09 09:15:00 | 139.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 12:45:00 | 147.15 | 2026-01-12 12:15:00 | 139.89 | STOP_HIT | 0.50 | 4.93% |
| SELL | retest2 | 2026-01-07 14:00:00 | 147.15 | 2026-01-12 12:15:00 | 139.89 | STOP_HIT | 0.50 | 4.93% |
| SELL | retest2 | 2026-01-07 15:00:00 | 146.77 | 2026-01-12 12:15:00 | 139.89 | STOP_HIT | 0.50 | 4.69% |
| BUY | retest2 | 2026-01-27 10:00:00 | 156.16 | 2026-01-29 09:15:00 | 171.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-27 11:45:00 | 156.71 | 2026-01-29 09:15:00 | 171.66 | TARGET_HIT | 1.00 | 9.54% |
| BUY | retest2 | 2026-01-27 13:00:00 | 156.05 | 2026-01-29 09:15:00 | 171.63 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2026-01-27 14:30:00 | 156.03 | 2026-01-29 10:15:00 | 172.38 | TARGET_HIT | 1.00 | 10.48% |
| BUY | retest2 | 2026-01-29 09:45:00 | 170.50 | 2026-02-01 14:15:00 | 165.12 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2026-02-06 14:30:00 | 184.12 | 2026-02-13 09:15:00 | 185.69 | STOP_HIT | 1.00 | 0.85% |
| SELL | retest2 | 2026-02-17 11:00:00 | 185.14 | 2026-02-17 12:15:00 | 191.33 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2026-02-19 09:15:00 | 192.27 | 2026-02-20 15:15:00 | 191.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-03-11 10:15:00 | 191.52 | 2026-03-12 09:15:00 | 181.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:15:00 | 191.52 | 2026-03-12 10:15:00 | 189.01 | STOP_HIT | 0.50 | 1.31% |
| SELL | retest2 | 2026-03-12 12:15:00 | 192.19 | 2026-03-13 09:15:00 | 182.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:30:00 | 190.85 | 2026-03-13 10:15:00 | 181.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 12:15:00 | 192.19 | 2026-03-16 09:15:00 | 187.23 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2026-03-12 14:30:00 | 190.85 | 2026-03-16 09:15:00 | 187.23 | STOP_HIT | 0.50 | 1.90% |
| SELL | retest2 | 2026-03-20 15:00:00 | 191.76 | 2026-03-23 15:15:00 | 182.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 15:00:00 | 191.76 | 2026-03-24 15:15:00 | 181.65 | STOP_HIT | 0.50 | 5.27% |
| SELL | retest2 | 2026-03-30 13:45:00 | 182.09 | 2026-04-01 11:15:00 | 184.20 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-03-30 14:15:00 | 181.89 | 2026-04-01 11:15:00 | 184.20 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-04-17 09:15:00 | 175.84 | 2026-04-24 09:15:00 | 193.42 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-29 13:45:00 | 172.61 | 2026-05-04 09:15:00 | 163.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-29 14:45:00 | 173.38 | 2026-05-04 09:15:00 | 164.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-30 09:30:00 | 172.89 | 2026-05-04 09:15:00 | 164.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-29 13:45:00 | 172.61 | 2026-05-05 10:15:00 | 156.04 | TARGET_HIT | 0.50 | 9.60% |
| SELL | retest2 | 2026-04-29 14:45:00 | 173.38 | 2026-05-05 14:15:00 | 155.60 | TARGET_HIT | 0.50 | 10.25% |
| SELL | retest2 | 2026-04-30 09:30:00 | 172.89 | 2026-05-06 09:15:00 | 155.35 | TARGET_HIT | 0.50 | 10.15% |
