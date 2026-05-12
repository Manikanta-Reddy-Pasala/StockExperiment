# Welspun Living Ltd. (WELSPUNLIV)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-11 15:15:00 (3716 bars)
- **Last close:** 135.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 141 |
| ALERT1 | 98 |
| ALERT2 | 97 |
| ALERT2_SKIP | 47 |
| ALERT3 | 238 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 85 |
| PARTIAL | 16 |
| TARGET_HIT | 13 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 101 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 48 / 53
- **Target hits / Stop hits / Partials:** 13 / 72 / 16
- **Avg / median % per leg:** 1.55% / -0.54%
- **Sum % (uncompounded):** 156.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 12 | 42.9% | 8 | 20 | 0 | 2.28% | 63.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 28 | 12 | 42.9% | 8 | 20 | 0 | 2.28% | 63.7% |
| SELL (all) | 73 | 36 | 49.3% | 5 | 52 | 16 | 1.27% | 92.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.77% | -2.8% |
| SELL @ 3rd Alert (retest2) | 72 | 36 | 50.0% | 5 | 51 | 16 | 1.32% | 95.1% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.77% | -2.8% |
| retest2 (combined) | 100 | 48 | 48.0% | 13 | 71 | 16 | 1.59% | 158.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 141.45 | 139.70 | 139.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 11:15:00 | 147.95 | 142.89 | 141.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 09:15:00 | 144.70 | 144.79 | 143.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 09:30:00 | 144.10 | 144.79 | 143.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 143.90 | 145.06 | 144.39 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 143.35 | 144.13 | 144.13 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 145.45 | 144.24 | 144.17 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 12:15:00 | 143.90 | 144.11 | 144.13 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 144.40 | 144.12 | 144.12 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 11:15:00 | 143.85 | 144.10 | 144.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 13:15:00 | 143.80 | 144.03 | 144.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 10:15:00 | 144.00 | 143.49 | 143.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 10:15:00 | 144.00 | 143.49 | 143.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 144.00 | 143.49 | 143.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:00:00 | 144.00 | 143.49 | 143.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 144.10 | 143.61 | 143.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 12:15:00 | 143.95 | 143.61 | 143.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 14:00:00 | 143.90 | 143.70 | 143.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 14:15:00 | 136.75 | 138.04 | 139.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 14:15:00 | 136.71 | 138.04 | 139.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 138.45 | 137.96 | 139.05 | SL hit (close>ema200) qty=0.50 sl=137.96 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 14:15:00 | 135.65 | 132.75 | 132.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 136.20 | 133.72 | 132.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 138.26 | 138.85 | 137.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 09:15:00 | 138.26 | 138.85 | 137.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 138.26 | 138.85 | 137.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 137.87 | 138.85 | 137.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 138.60 | 139.53 | 138.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:45:00 | 138.61 | 139.53 | 138.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 137.20 | 139.06 | 138.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:00:00 | 137.20 | 139.06 | 138.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 138.20 | 138.89 | 138.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 15:15:00 | 138.78 | 138.89 | 138.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 12:15:00 | 144.24 | 144.96 | 145.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 12:15:00 | 144.24 | 144.96 | 145.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 143.29 | 144.42 | 144.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 10:15:00 | 145.09 | 144.56 | 144.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 10:15:00 | 145.09 | 144.56 | 144.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 145.09 | 144.56 | 144.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 145.09 | 144.56 | 144.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 144.80 | 144.61 | 144.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:45:00 | 145.26 | 144.61 | 144.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 148.53 | 145.39 | 145.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 150.94 | 147.92 | 146.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 15:15:00 | 153.50 | 153.54 | 151.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 09:15:00 | 154.20 | 153.54 | 151.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 153.79 | 154.58 | 153.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 153.65 | 154.58 | 153.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 153.31 | 154.33 | 153.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:45:00 | 153.50 | 154.33 | 153.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 152.70 | 154.00 | 153.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 11:30:00 | 152.24 | 154.00 | 153.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 151.20 | 153.44 | 153.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 151.20 | 153.44 | 153.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 14:15:00 | 150.99 | 152.64 | 152.83 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 15:15:00 | 153.10 | 152.72 | 152.72 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 09:15:00 | 151.93 | 152.56 | 152.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 10:15:00 | 151.40 | 152.33 | 152.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 14:15:00 | 151.53 | 150.70 | 151.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 14:15:00 | 151.53 | 150.70 | 151.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 151.53 | 150.70 | 151.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 15:00:00 | 151.53 | 150.70 | 151.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 151.60 | 150.88 | 151.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:15:00 | 151.30 | 150.88 | 151.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 151.92 | 151.09 | 151.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:15:00 | 153.40 | 151.09 | 151.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 152.19 | 151.31 | 151.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 12:00:00 | 151.58 | 151.36 | 151.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 12:15:00 | 152.12 | 151.51 | 151.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 12:15:00 | 152.12 | 151.51 | 151.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 13:15:00 | 153.43 | 151.90 | 151.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 13:15:00 | 171.00 | 171.23 | 167.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 14:00:00 | 171.00 | 171.23 | 167.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 169.34 | 171.41 | 169.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:45:00 | 169.60 | 171.41 | 169.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 169.90 | 171.11 | 169.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 171.52 | 171.11 | 169.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 10:30:00 | 171.92 | 172.02 | 171.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 12:30:00 | 171.48 | 174.97 | 174.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-29 09:15:00 | 188.67 | 183.58 | 181.44 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 14:15:00 | 183.40 | 185.33 | 185.35 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 10:15:00 | 186.10 | 185.41 | 185.36 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 184.65 | 185.26 | 185.30 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 12:15:00 | 185.80 | 185.37 | 185.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 13:15:00 | 187.30 | 185.76 | 185.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 184.85 | 185.72 | 185.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 184.85 | 185.72 | 185.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 184.85 | 185.72 | 185.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:45:00 | 183.89 | 185.72 | 185.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 186.50 | 185.87 | 185.66 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 14:15:00 | 185.00 | 185.52 | 185.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 181.37 | 184.63 | 185.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 189.74 | 179.89 | 181.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 189.74 | 179.89 | 181.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 189.74 | 179.89 | 181.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 189.74 | 179.89 | 181.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 190.33 | 181.98 | 182.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:45:00 | 191.57 | 181.98 | 182.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 188.80 | 183.34 | 182.89 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 13:15:00 | 181.57 | 183.02 | 183.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 14:15:00 | 179.13 | 182.24 | 182.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 12:15:00 | 178.79 | 177.72 | 179.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 12:15:00 | 178.79 | 177.72 | 179.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 178.79 | 177.72 | 179.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 12:45:00 | 179.10 | 177.72 | 179.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 180.30 | 178.24 | 179.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:00:00 | 180.30 | 178.24 | 179.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 178.01 | 178.19 | 179.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 10:15:00 | 176.90 | 178.19 | 178.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:30:00 | 177.24 | 178.10 | 178.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 14:15:00 | 177.00 | 177.97 | 178.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 14:15:00 | 176.98 | 175.72 | 176.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 176.00 | 175.77 | 176.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-19 10:15:00 | 179.06 | 176.77 | 176.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 179.06 | 176.77 | 176.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 11:15:00 | 180.42 | 177.50 | 176.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 11:15:00 | 193.55 | 193.99 | 191.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 12:00:00 | 193.55 | 193.99 | 191.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 201.36 | 205.13 | 203.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:00:00 | 201.36 | 205.13 | 203.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 199.51 | 204.01 | 202.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:45:00 | 198.86 | 204.01 | 202.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 13:15:00 | 195.80 | 201.17 | 201.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 14:15:00 | 191.60 | 195.91 | 198.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 192.10 | 188.73 | 190.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 192.10 | 188.73 | 190.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 192.10 | 188.73 | 190.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:00:00 | 192.10 | 188.73 | 190.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 188.55 | 188.69 | 190.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 11:30:00 | 187.78 | 188.74 | 190.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 09:45:00 | 185.86 | 188.61 | 189.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 13:15:00 | 178.39 | 184.26 | 185.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-12 11:15:00 | 181.64 | 181.40 | 183.05 | SL hit (close>ema200) qty=0.50 sl=181.40 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 10:15:00 | 183.20 | 179.61 | 179.59 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 177.82 | 179.64 | 179.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 176.31 | 178.95 | 179.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 177.50 | 176.58 | 177.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 177.50 | 176.58 | 177.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 177.00 | 176.66 | 177.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 177.59 | 176.66 | 177.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 176.73 | 176.68 | 177.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 10:15:00 | 176.51 | 176.68 | 177.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:00:00 | 176.06 | 176.70 | 177.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 12:15:00 | 167.68 | 169.82 | 171.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 09:15:00 | 167.26 | 169.13 | 170.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-27 11:15:00 | 166.72 | 166.60 | 168.19 | SL hit (close>ema200) qty=0.50 sl=166.60 alert=retest2 |

### Cycle 25 — BUY (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 10:15:00 | 172.17 | 167.35 | 166.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 14:15:00 | 175.05 | 171.45 | 169.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 11:15:00 | 171.67 | 172.62 | 170.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 12:00:00 | 171.67 | 172.62 | 170.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 169.75 | 171.80 | 170.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 169.75 | 171.80 | 170.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 170.00 | 171.44 | 170.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 15:15:00 | 168.80 | 171.44 | 170.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 168.80 | 170.91 | 170.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 165.40 | 170.91 | 170.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 10:15:00 | 167.33 | 169.60 | 169.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 161.19 | 166.09 | 167.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 160.60 | 160.21 | 163.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:45:00 | 160.51 | 160.21 | 163.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 163.31 | 160.81 | 162.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 163.31 | 160.81 | 162.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 163.59 | 161.37 | 162.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 163.59 | 161.37 | 162.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 163.95 | 161.89 | 162.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 164.84 | 161.89 | 162.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 166.10 | 163.68 | 163.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 12:15:00 | 167.91 | 164.53 | 163.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 11:15:00 | 167.82 | 168.07 | 166.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 11:45:00 | 168.31 | 168.07 | 166.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 167.10 | 167.93 | 166.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 166.15 | 167.93 | 166.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 165.44 | 167.43 | 166.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:45:00 | 165.39 | 167.43 | 166.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 166.25 | 167.19 | 166.76 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 14:15:00 | 165.56 | 166.41 | 166.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 09:15:00 | 163.86 | 165.73 | 166.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 10:15:00 | 164.07 | 163.52 | 164.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 10:15:00 | 164.07 | 163.52 | 164.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 164.07 | 163.52 | 164.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 164.07 | 163.52 | 164.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 162.56 | 163.33 | 164.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:15:00 | 162.11 | 163.33 | 164.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 12:15:00 | 166.20 | 163.88 | 163.96 | SL hit (close>static) qty=1.00 sl=164.95 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 13:15:00 | 166.47 | 164.40 | 164.18 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 161.87 | 163.95 | 164.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 161.17 | 163.07 | 163.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 162.92 | 162.58 | 163.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 10:15:00 | 162.92 | 162.58 | 163.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 162.92 | 162.58 | 163.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:45:00 | 163.00 | 162.58 | 163.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 164.52 | 162.97 | 163.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 164.52 | 162.97 | 163.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 165.00 | 163.38 | 163.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:30:00 | 164.76 | 163.38 | 163.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 13:15:00 | 167.19 | 164.14 | 163.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 14:15:00 | 167.80 | 164.87 | 164.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 09:15:00 | 163.08 | 164.85 | 164.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 163.08 | 164.85 | 164.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 163.08 | 164.85 | 164.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:45:00 | 162.41 | 164.85 | 164.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 162.80 | 164.44 | 164.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:00:00 | 162.80 | 164.44 | 164.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 11:15:00 | 161.23 | 163.80 | 163.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 14:15:00 | 159.93 | 162.33 | 163.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 157.32 | 155.41 | 158.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 09:45:00 | 157.00 | 155.41 | 158.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 158.00 | 156.28 | 157.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:30:00 | 158.16 | 156.28 | 157.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 158.88 | 156.80 | 158.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 158.88 | 156.80 | 158.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 159.00 | 157.24 | 158.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:30:00 | 159.09 | 157.24 | 158.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 155.80 | 157.38 | 158.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:30:00 | 157.57 | 157.38 | 158.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 148.79 | 152.63 | 154.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 13:15:00 | 147.23 | 151.73 | 154.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:30:00 | 147.62 | 149.35 | 152.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:45:00 | 147.77 | 150.44 | 151.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 13:30:00 | 147.80 | 149.72 | 150.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 151.41 | 149.87 | 150.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:45:00 | 151.80 | 149.87 | 150.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 151.80 | 150.26 | 150.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-30 12:15:00 | 154.00 | 151.45 | 151.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 154.00 | 151.45 | 151.24 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 149.79 | 151.08 | 151.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 13:15:00 | 149.12 | 150.69 | 151.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 151.90 | 150.93 | 151.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 14:15:00 | 151.90 | 150.93 | 151.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 151.90 | 150.93 | 151.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 15:00:00 | 151.90 | 150.93 | 151.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 151.90 | 151.13 | 151.19 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 154.88 | 151.88 | 151.53 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 148.49 | 150.94 | 151.20 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 11:15:00 | 153.17 | 151.39 | 151.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-04 12:15:00 | 153.68 | 151.84 | 151.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 12:15:00 | 153.55 | 153.98 | 153.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-05 13:00:00 | 153.55 | 153.98 | 153.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 159.74 | 160.76 | 158.86 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 154.29 | 158.09 | 158.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 150.42 | 154.82 | 156.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 12:15:00 | 151.55 | 150.82 | 152.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 12:15:00 | 151.55 | 150.82 | 152.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 151.55 | 150.82 | 152.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 151.55 | 150.82 | 152.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 150.24 | 150.23 | 151.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:45:00 | 147.28 | 149.45 | 150.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 146.79 | 149.52 | 150.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 10:30:00 | 147.02 | 148.90 | 149.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 151.65 | 149.58 | 149.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 151.65 | 149.58 | 149.46 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 10:15:00 | 148.20 | 149.59 | 149.67 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 12:15:00 | 151.15 | 149.56 | 149.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 13:15:00 | 151.78 | 150.00 | 149.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 151.66 | 151.97 | 151.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 13:00:00 | 151.66 | 151.97 | 151.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 151.09 | 151.73 | 151.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 151.09 | 151.73 | 151.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 152.49 | 151.88 | 151.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:15:00 | 153.10 | 151.79 | 151.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-05 09:15:00 | 168.41 | 162.66 | 159.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 13:15:00 | 169.86 | 170.93 | 170.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 169.00 | 170.36 | 170.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 10:15:00 | 169.92 | 169.85 | 170.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 10:15:00 | 169.92 | 169.85 | 170.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 169.92 | 169.85 | 170.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 11:00:00 | 169.92 | 169.85 | 170.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 170.69 | 170.02 | 170.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 170.69 | 170.02 | 170.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 171.22 | 170.26 | 170.49 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 175.80 | 171.67 | 171.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 10:15:00 | 178.80 | 174.34 | 172.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 176.79 | 177.14 | 175.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 11:15:00 | 176.20 | 177.14 | 175.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 174.71 | 176.65 | 175.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:00:00 | 174.71 | 176.65 | 175.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 174.73 | 176.27 | 175.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:15:00 | 174.65 | 176.27 | 175.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 172.59 | 175.53 | 174.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:45:00 | 173.03 | 175.53 | 174.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 171.97 | 174.82 | 174.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 15:00:00 | 171.97 | 174.82 | 174.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 15:15:00 | 172.11 | 174.28 | 174.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 168.01 | 171.73 | 172.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 170.85 | 170.66 | 171.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 170.85 | 170.66 | 171.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 168.49 | 170.28 | 171.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:30:00 | 168.08 | 169.10 | 170.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 09:15:00 | 159.68 | 162.35 | 165.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 10:15:00 | 162.40 | 162.36 | 164.86 | SL hit (close>ema200) qty=0.50 sl=162.36 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 161.58 | 157.14 | 156.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 162.76 | 160.66 | 159.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 163.55 | 163.84 | 162.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 15:00:00 | 163.55 | 163.84 | 162.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 160.75 | 163.08 | 162.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 160.81 | 163.08 | 162.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 158.27 | 162.12 | 161.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 158.27 | 162.12 | 161.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 158.21 | 161.34 | 161.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 156.11 | 159.68 | 160.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 158.79 | 158.50 | 159.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 159.36 | 158.50 | 159.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 158.14 | 158.43 | 159.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:30:00 | 159.18 | 158.43 | 159.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 159.28 | 158.54 | 159.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:00:00 | 159.28 | 158.54 | 159.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 159.60 | 158.75 | 159.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 159.60 | 158.75 | 159.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 159.98 | 159.00 | 159.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 156.40 | 159.18 | 159.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 148.58 | 151.49 | 154.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 146.50 | 145.61 | 147.83 | SL hit (close>ema200) qty=0.50 sl=145.61 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 154.40 | 148.96 | 148.49 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 151.94 | 152.61 | 152.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 15:15:00 | 150.95 | 152.11 | 152.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 149.67 | 148.68 | 150.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 149.67 | 148.68 | 150.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 150.57 | 149.15 | 150.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 150.57 | 149.15 | 150.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 149.54 | 149.23 | 150.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:30:00 | 148.68 | 149.16 | 149.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 147.87 | 149.02 | 149.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 14:15:00 | 151.60 | 148.62 | 148.98 | SL hit (close>static) qty=1.00 sl=150.83 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 15:15:00 | 154.08 | 149.71 | 149.45 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 145.64 | 148.89 | 149.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 145.24 | 148.16 | 148.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 145.41 | 142.59 | 144.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 12:15:00 | 145.41 | 142.59 | 144.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 145.41 | 142.59 | 144.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 145.41 | 142.59 | 144.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 146.99 | 143.47 | 145.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 146.99 | 143.47 | 145.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 147.44 | 144.26 | 145.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 15:15:00 | 146.82 | 144.26 | 145.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 147.80 | 145.94 | 145.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 149.64 | 147.50 | 146.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 12:15:00 | 147.65 | 148.26 | 147.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 12:15:00 | 147.65 | 148.26 | 147.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 147.65 | 148.26 | 147.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:45:00 | 147.62 | 148.26 | 147.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 146.35 | 147.88 | 147.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 146.35 | 147.88 | 147.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 14:15:00 | 138.32 | 145.96 | 146.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 135.35 | 139.46 | 141.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 131.91 | 131.52 | 133.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 131.91 | 131.52 | 133.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 131.91 | 131.52 | 133.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:30:00 | 133.66 | 131.52 | 133.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 132.42 | 131.79 | 132.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 11:00:00 | 131.14 | 131.66 | 132.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 124.58 | 126.00 | 127.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-11 15:15:00 | 124.20 | 123.93 | 125.73 | SL hit (close>ema200) qty=0.50 sl=123.93 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 120.97 | 119.35 | 119.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 122.26 | 120.23 | 119.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 122.36 | 122.56 | 121.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 122.36 | 122.56 | 121.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 122.40 | 122.54 | 121.75 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 14:15:00 | 119.12 | 121.28 | 121.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 117.01 | 120.08 | 120.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 116.50 | 116.40 | 118.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 09:45:00 | 116.60 | 116.40 | 118.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 109.46 | 108.29 | 110.68 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 116.88 | 112.25 | 111.74 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 119.00 | 120.13 | 120.26 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 13:15:00 | 121.20 | 120.46 | 120.39 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 11:15:00 | 119.90 | 120.81 | 120.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 11:15:00 | 117.90 | 119.57 | 120.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 124.15 | 119.22 | 119.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 124.15 | 119.22 | 119.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 124.15 | 119.22 | 119.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 124.15 | 119.22 | 119.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 127.37 | 120.85 | 120.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 128.28 | 124.84 | 122.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 137.30 | 138.52 | 135.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 137.98 | 138.37 | 136.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 137.98 | 138.37 | 136.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 137.54 | 138.37 | 136.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 136.76 | 138.05 | 136.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 136.76 | 138.05 | 136.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 135.73 | 137.58 | 136.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 135.73 | 137.58 | 136.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 137.00 | 136.69 | 136.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 136.07 | 136.69 | 136.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 136.50 | 136.65 | 136.56 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 12:15:00 | 136.22 | 136.48 | 136.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 14:15:00 | 136.00 | 136.32 | 136.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 11:15:00 | 132.10 | 131.31 | 132.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 11:15:00 | 132.10 | 131.31 | 132.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 132.10 | 131.31 | 132.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 12:00:00 | 132.10 | 131.31 | 132.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 132.29 | 131.51 | 132.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:30:00 | 131.20 | 131.64 | 132.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 14:15:00 | 124.64 | 128.07 | 130.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 118.08 | 124.89 | 128.27 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 61 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 122.03 | 119.04 | 118.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 123.88 | 120.01 | 119.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 15:15:00 | 131.44 | 131.50 | 129.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 09:15:00 | 131.63 | 131.50 | 129.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 130.77 | 132.49 | 131.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 130.77 | 132.49 | 131.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 131.21 | 132.24 | 131.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 130.29 | 132.24 | 131.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 132.19 | 132.21 | 131.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:15:00 | 132.19 | 132.21 | 131.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 132.75 | 132.32 | 132.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:15:00 | 133.29 | 132.32 | 132.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 12:15:00 | 131.69 | 132.68 | 132.42 | SL hit (close<static) qty=1.00 sl=131.81 alert=retest2 |

### Cycle 62 — SELL (started 2025-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 14:15:00 | 130.79 | 132.05 | 132.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 126.20 | 130.79 | 131.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 127.59 | 126.51 | 128.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 127.59 | 126.51 | 128.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 127.85 | 127.05 | 128.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:45:00 | 128.25 | 127.05 | 128.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 128.60 | 127.47 | 128.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 128.60 | 127.47 | 128.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 128.23 | 127.62 | 128.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 128.68 | 127.62 | 128.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 130.30 | 128.16 | 128.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:00:00 | 130.30 | 128.16 | 128.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 130.19 | 128.56 | 128.59 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 11:15:00 | 130.35 | 128.92 | 128.75 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 127.25 | 128.85 | 129.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 126.10 | 128.30 | 128.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 11:15:00 | 126.57 | 126.25 | 127.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 11:15:00 | 126.57 | 126.25 | 127.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 126.57 | 126.25 | 127.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:30:00 | 126.00 | 126.25 | 127.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 133.13 | 124.62 | 125.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:00:00 | 133.13 | 124.62 | 125.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 135.25 | 126.74 | 126.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 139.14 | 134.23 | 130.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 138.28 | 139.57 | 135.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 15:00:00 | 138.28 | 139.57 | 135.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 139.37 | 139.23 | 135.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 14:45:00 | 144.79 | 140.71 | 137.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:00:00 | 144.73 | 146.70 | 145.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 11:30:00 | 144.88 | 145.01 | 144.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 14:15:00 | 143.95 | 144.50 | 144.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 14:15:00 | 143.95 | 144.50 | 144.51 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 146.37 | 144.75 | 144.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 149.17 | 146.67 | 145.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 12:15:00 | 147.95 | 148.27 | 147.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 13:00:00 | 147.95 | 148.27 | 147.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 146.61 | 147.94 | 147.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 146.61 | 147.94 | 147.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 146.85 | 147.72 | 147.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:45:00 | 146.40 | 147.72 | 147.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 146.40 | 147.45 | 146.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 144.51 | 147.45 | 146.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 145.61 | 146.66 | 146.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 144.29 | 145.90 | 146.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 144.70 | 144.46 | 145.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 09:45:00 | 144.11 | 144.46 | 145.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 148.57 | 144.75 | 145.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 148.57 | 144.75 | 145.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 148.46 | 145.49 | 145.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 152.26 | 147.47 | 146.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 09:15:00 | 150.86 | 151.53 | 150.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 150.86 | 151.53 | 150.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 150.86 | 151.53 | 150.07 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 148.65 | 149.86 | 149.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 13:15:00 | 147.79 | 149.05 | 149.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 148.70 | 148.70 | 149.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 148.70 | 148.70 | 149.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 148.70 | 148.70 | 149.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 13:15:00 | 148.01 | 148.64 | 149.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 15:15:00 | 146.50 | 148.47 | 148.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:30:00 | 147.78 | 147.54 | 148.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 10:15:00 | 140.61 | 144.67 | 146.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 10:15:00 | 139.17 | 144.67 | 146.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 10:15:00 | 140.39 | 144.67 | 146.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-05-30 14:15:00 | 133.21 | 139.39 | 143.21 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 71 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 139.07 | 135.56 | 135.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 140.38 | 137.97 | 136.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 11:15:00 | 136.98 | 138.31 | 137.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 11:15:00 | 136.98 | 138.31 | 137.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 136.98 | 138.31 | 137.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:45:00 | 137.00 | 138.31 | 137.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 136.51 | 137.95 | 137.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:45:00 | 136.21 | 137.95 | 137.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 137.88 | 137.64 | 137.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 137.65 | 137.64 | 137.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 137.76 | 137.66 | 137.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:15:00 | 138.78 | 137.67 | 137.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 11:00:00 | 138.75 | 137.95 | 137.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 13:00:00 | 140.28 | 138.43 | 137.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 14:15:00 | 138.73 | 139.30 | 138.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 139.74 | 139.39 | 138.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 137.13 | 138.43 | 138.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 137.13 | 138.43 | 138.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 135.65 | 137.64 | 138.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 133.22 | 133.02 | 134.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 14:15:00 | 133.22 | 133.02 | 134.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 133.22 | 133.02 | 134.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 133.22 | 133.02 | 134.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 133.40 | 133.08 | 134.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 134.76 | 133.08 | 134.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 133.55 | 133.18 | 134.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:00:00 | 133.10 | 133.16 | 133.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:45:00 | 132.89 | 133.16 | 133.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 132.73 | 132.93 | 133.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 130.50 | 129.68 | 129.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 130.50 | 129.68 | 129.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 131.26 | 129.98 | 129.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 09:15:00 | 140.44 | 141.71 | 138.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 10:00:00 | 140.44 | 141.71 | 138.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 141.47 | 142.29 | 141.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 141.47 | 142.29 | 141.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 141.14 | 142.06 | 141.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:30:00 | 141.26 | 142.06 | 141.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 141.04 | 141.86 | 141.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 140.90 | 141.86 | 141.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 140.22 | 141.30 | 141.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:00:00 | 140.22 | 141.30 | 141.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 140.00 | 141.04 | 141.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 139.80 | 140.79 | 140.95 | Break + close below crossover candle low |

### Cycle 75 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 143.90 | 141.31 | 141.16 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 139.92 | 140.92 | 141.01 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 142.59 | 141.20 | 141.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 14:15:00 | 144.17 | 142.76 | 142.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 12:15:00 | 143.37 | 143.45 | 142.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 13:00:00 | 143.37 | 143.45 | 142.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 143.35 | 143.47 | 142.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 145.68 | 143.47 | 142.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 15:15:00 | 144.13 | 144.74 | 144.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 142.28 | 143.54 | 143.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 12:15:00 | 142.28 | 143.54 | 143.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 13:15:00 | 142.01 | 143.24 | 143.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 15:15:00 | 143.50 | 143.08 | 143.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 15:15:00 | 143.50 | 143.08 | 143.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 143.50 | 143.08 | 143.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 143.79 | 143.08 | 143.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 142.81 | 143.03 | 143.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:30:00 | 143.00 | 143.03 | 143.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 142.82 | 142.99 | 143.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:30:00 | 143.18 | 142.99 | 143.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 142.89 | 142.97 | 143.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:45:00 | 143.13 | 142.97 | 143.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 142.91 | 142.96 | 143.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:30:00 | 143.08 | 142.96 | 143.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 142.96 | 142.96 | 143.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:00:00 | 142.96 | 142.96 | 143.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 143.17 | 143.00 | 143.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:45:00 | 143.25 | 143.00 | 143.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 143.36 | 143.07 | 143.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:30:00 | 142.61 | 142.86 | 143.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 141.70 | 139.95 | 139.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 09:15:00 | 141.70 | 139.95 | 139.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 11:15:00 | 142.43 | 140.79 | 140.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 10:15:00 | 141.70 | 141.99 | 141.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 11:00:00 | 141.70 | 141.99 | 141.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 141.25 | 141.87 | 141.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 141.25 | 141.87 | 141.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 140.15 | 141.52 | 141.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:30:00 | 140.20 | 141.52 | 141.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 140.13 | 141.02 | 141.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 139.48 | 140.38 | 140.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 10:15:00 | 138.68 | 138.53 | 139.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:45:00 | 138.60 | 138.53 | 139.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 140.87 | 139.05 | 139.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 140.87 | 139.05 | 139.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 15:15:00 | 141.60 | 139.56 | 139.55 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 138.51 | 139.49 | 139.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 137.23 | 139.03 | 139.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 135.98 | 135.60 | 136.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 135.98 | 135.60 | 136.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 136.75 | 135.94 | 136.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 136.75 | 135.94 | 136.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 136.60 | 136.07 | 136.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 137.14 | 136.07 | 136.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 136.99 | 136.25 | 136.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:00:00 | 136.99 | 136.25 | 136.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 137.60 | 136.52 | 136.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:00:00 | 137.60 | 136.52 | 136.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 138.00 | 136.82 | 136.79 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 132.65 | 136.19 | 136.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 125.25 | 133.24 | 135.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 123.50 | 123.36 | 126.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:45:00 | 123.60 | 123.36 | 126.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 126.90 | 124.65 | 126.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 126.90 | 124.65 | 126.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 126.89 | 125.10 | 126.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 126.17 | 125.10 | 126.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:15:00 | 119.86 | 121.91 | 123.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-07 09:15:00 | 113.55 | 117.35 | 120.18 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 85 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 117.48 | 115.11 | 114.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 121.87 | 117.51 | 116.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 13:15:00 | 121.10 | 121.12 | 119.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 14:00:00 | 121.10 | 121.12 | 119.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 119.98 | 120.81 | 119.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 119.61 | 120.81 | 119.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 120.14 | 120.68 | 119.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 120.30 | 120.68 | 119.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 119.98 | 120.54 | 119.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 119.98 | 120.54 | 119.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 119.76 | 120.38 | 119.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 119.76 | 120.38 | 119.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 119.92 | 120.29 | 119.94 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 119.24 | 119.73 | 119.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 118.81 | 119.43 | 119.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 110.75 | 110.54 | 111.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 13:30:00 | 110.95 | 110.54 | 111.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 109.87 | 110.64 | 111.51 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 14:15:00 | 112.60 | 111.90 | 111.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 114.90 | 112.61 | 112.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 113.75 | 113.83 | 113.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:00:00 | 113.75 | 113.83 | 113.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 113.23 | 113.71 | 113.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:15:00 | 113.06 | 113.71 | 113.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 112.98 | 113.56 | 113.21 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 112.49 | 112.98 | 113.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 111.58 | 112.44 | 112.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 112.56 | 112.39 | 112.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 112.56 | 112.39 | 112.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 112.56 | 112.39 | 112.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 112.56 | 112.39 | 112.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 113.61 | 112.63 | 112.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 113.61 | 112.63 | 112.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 113.50 | 112.80 | 112.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 113.27 | 112.80 | 112.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 114.15 | 113.07 | 112.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 115.50 | 113.72 | 113.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 15:15:00 | 114.11 | 114.17 | 113.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:15:00 | 114.28 | 114.17 | 113.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 114.10 | 114.16 | 113.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:15:00 | 113.66 | 114.16 | 113.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 113.93 | 114.11 | 113.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 11:45:00 | 114.14 | 114.12 | 113.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 14:30:00 | 114.12 | 114.10 | 113.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 15:00:00 | 114.25 | 114.10 | 113.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 123.00 | 114.08 | 113.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-10 09:15:00 | 125.55 | 116.17 | 114.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 15:15:00 | 124.65 | 125.90 | 125.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 122.38 | 123.66 | 124.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 12:15:00 | 122.30 | 122.15 | 122.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 12:15:00 | 122.30 | 122.15 | 122.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 122.30 | 122.15 | 122.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 121.18 | 122.05 | 122.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 115.12 | 116.69 | 118.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 114.98 | 114.83 | 115.99 | SL hit (close>ema200) qty=0.50 sl=114.83 alert=retest2 |

### Cycle 91 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 118.14 | 116.13 | 116.03 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 114.95 | 116.19 | 116.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 12:15:00 | 114.64 | 115.88 | 116.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 12:15:00 | 115.09 | 115.01 | 115.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 13:00:00 | 115.09 | 115.01 | 115.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 115.58 | 115.18 | 115.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 115.58 | 115.18 | 115.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 115.40 | 115.23 | 115.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 114.93 | 115.23 | 115.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 15:15:00 | 117.39 | 114.94 | 115.07 | SL hit (close>static) qty=1.00 sl=115.64 alert=retest2 |

### Cycle 93 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 119.07 | 115.77 | 115.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 13:15:00 | 120.48 | 119.31 | 118.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 122.84 | 122.92 | 121.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:45:00 | 122.99 | 122.92 | 121.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 122.81 | 122.85 | 122.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 121.41 | 122.85 | 122.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 123.08 | 125.61 | 124.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:30:00 | 123.10 | 125.61 | 124.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 122.75 | 125.04 | 124.54 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 121.46 | 123.87 | 124.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 118.86 | 121.99 | 123.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 13:15:00 | 121.18 | 121.12 | 122.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 14:00:00 | 121.18 | 121.12 | 122.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 122.30 | 121.35 | 122.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 122.30 | 121.35 | 122.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 121.69 | 121.42 | 122.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 122.88 | 121.67 | 122.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 122.50 | 121.83 | 122.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 129.16 | 121.83 | 122.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 129.99 | 123.47 | 122.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 133.08 | 130.57 | 129.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 128.45 | 131.61 | 131.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 128.45 | 131.61 | 131.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 128.45 | 131.61 | 131.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 128.45 | 131.61 | 131.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 129.10 | 131.11 | 131.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 131.49 | 131.11 | 131.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 130.60 | 131.20 | 131.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 130.60 | 131.20 | 131.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 130.35 | 131.03 | 131.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 128.80 | 127.91 | 128.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 128.80 | 127.91 | 128.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 128.80 | 127.91 | 128.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 128.80 | 127.91 | 128.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 128.19 | 127.96 | 128.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 128.75 | 127.96 | 128.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 129.18 | 128.21 | 128.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 129.18 | 128.21 | 128.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 129.15 | 128.40 | 128.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 128.60 | 128.79 | 128.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 130.10 | 128.92 | 128.98 | SL hit (close>static) qty=1.00 sl=129.88 alert=retest2 |

### Cycle 97 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 131.01 | 129.34 | 129.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 132.00 | 130.14 | 129.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 137.36 | 137.89 | 135.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 137.36 | 137.89 | 135.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 137.36 | 137.89 | 135.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 139.20 | 137.60 | 137.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 13:15:00 | 137.20 | 138.32 | 138.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 13:15:00 | 137.20 | 138.32 | 138.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 15:15:00 | 136.54 | 137.72 | 138.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 10:15:00 | 137.95 | 137.59 | 138.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 10:15:00 | 137.95 | 137.59 | 138.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 137.95 | 137.59 | 138.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:45:00 | 138.18 | 137.59 | 138.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 138.69 | 137.81 | 138.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 138.69 | 137.81 | 138.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 138.54 | 137.96 | 138.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:30:00 | 137.97 | 138.08 | 138.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 14:15:00 | 140.00 | 138.47 | 138.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 140.00 | 138.47 | 138.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 15:15:00 | 141.40 | 139.05 | 138.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 137.42 | 138.73 | 138.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 137.42 | 138.73 | 138.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 137.42 | 138.73 | 138.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 137.42 | 138.73 | 138.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 136.44 | 138.27 | 138.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 136.01 | 137.26 | 137.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 132.65 | 131.51 | 133.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 10:00:00 | 132.65 | 131.51 | 133.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 132.69 | 131.74 | 133.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 132.45 | 131.74 | 133.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 132.86 | 131.97 | 133.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 132.49 | 132.75 | 133.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:00:00 | 132.48 | 132.75 | 133.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:45:00 | 132.05 | 132.71 | 133.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 132.13 | 132.41 | 132.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 132.13 | 132.35 | 132.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 138.02 | 132.35 | 132.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 136.85 | 133.25 | 133.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 136.85 | 133.25 | 133.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 11:15:00 | 141.61 | 135.69 | 134.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 142.06 | 142.29 | 138.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 09:30:00 | 142.05 | 142.29 | 138.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 136.85 | 140.74 | 139.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 137.32 | 140.74 | 139.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 136.81 | 139.95 | 139.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:15:00 | 136.78 | 139.95 | 139.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 136.87 | 138.86 | 138.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 136.21 | 138.33 | 138.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 09:15:00 | 141.65 | 138.42 | 138.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 141.65 | 138.42 | 138.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 141.65 | 138.42 | 138.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 141.65 | 138.42 | 138.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 138.67 | 138.47 | 138.57 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 15:15:00 | 139.80 | 138.72 | 138.61 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 12:15:00 | 136.95 | 138.27 | 138.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 136.49 | 137.56 | 138.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 132.17 | 132.12 | 133.53 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 14:30:00 | 131.06 | 132.09 | 133.39 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 134.69 | 132.66 | 133.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 134.69 | 132.66 | 133.43 | SL hit (close>ema400) qty=1.00 sl=133.43 alert=retest1 |

### Cycle 105 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 140.54 | 134.24 | 134.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 12:15:00 | 141.95 | 136.85 | 135.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 12:15:00 | 137.68 | 138.36 | 137.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 13:00:00 | 137.68 | 138.36 | 137.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 137.45 | 138.08 | 137.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 137.45 | 138.08 | 137.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 137.70 | 137.87 | 137.24 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 135.85 | 136.92 | 137.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 11:15:00 | 135.40 | 136.62 | 136.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 134.82 | 134.67 | 135.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 15:00:00 | 134.82 | 134.67 | 135.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 136.99 | 135.13 | 135.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 134.73 | 135.05 | 135.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 137.00 | 133.92 | 134.10 | SL hit (close>static) qty=1.00 sl=136.99 alert=retest2 |

### Cycle 107 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 135.75 | 134.28 | 134.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 137.18 | 135.57 | 134.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 136.01 | 136.13 | 135.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 14:30:00 | 136.08 | 136.13 | 135.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 136.04 | 136.62 | 136.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:30:00 | 135.59 | 136.62 | 136.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 135.63 | 136.42 | 136.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 135.63 | 136.42 | 136.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 134.65 | 135.87 | 135.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 133.20 | 134.75 | 135.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 132.86 | 131.49 | 132.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 132.86 | 131.49 | 132.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 132.86 | 131.49 | 132.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:45:00 | 132.70 | 131.49 | 132.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 133.29 | 131.85 | 132.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 133.29 | 131.85 | 132.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 133.39 | 132.16 | 132.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 132.44 | 132.16 | 132.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 133.15 | 132.11 | 132.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 133.15 | 132.11 | 132.02 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 130.00 | 131.89 | 132.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 12:15:00 | 129.64 | 130.94 | 131.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 129.56 | 129.27 | 130.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 13:45:00 | 129.46 | 129.27 | 130.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 130.67 | 129.58 | 130.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:45:00 | 131.60 | 129.58 | 130.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 128.78 | 129.42 | 129.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 128.56 | 129.42 | 129.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 122.13 | 123.80 | 125.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 122.40 | 121.89 | 123.73 | SL hit (close>ema200) qty=0.50 sl=121.89 alert=retest2 |

### Cycle 111 — BUY (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 14:15:00 | 125.93 | 123.24 | 123.16 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 12:15:00 | 121.83 | 123.22 | 123.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 120.84 | 122.74 | 123.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 115.93 | 115.47 | 117.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 10:00:00 | 115.93 | 115.47 | 117.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 117.04 | 115.80 | 117.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 117.04 | 115.80 | 117.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 116.38 | 115.92 | 117.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 14:15:00 | 115.98 | 115.92 | 117.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 14:45:00 | 116.01 | 116.10 | 117.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 117.99 | 116.48 | 117.25 | SL hit (close>static) qty=1.00 sl=117.58 alert=retest2 |

### Cycle 113 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 121.14 | 118.42 | 118.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 11:15:00 | 125.38 | 119.82 | 118.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 122.26 | 122.98 | 120.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 11:15:00 | 120.92 | 122.38 | 121.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 120.92 | 122.38 | 121.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 120.92 | 122.38 | 121.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 120.68 | 122.04 | 121.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:30:00 | 120.90 | 122.04 | 121.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 120.60 | 121.75 | 120.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 14:30:00 | 121.37 | 121.63 | 120.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 124.07 | 121.38 | 120.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 12:15:00 | 123.37 | 124.38 | 124.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 123.37 | 124.38 | 124.42 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 126.84 | 124.72 | 124.49 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 123.56 | 124.73 | 124.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 123.30 | 124.44 | 124.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 122.92 | 122.16 | 123.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 122.92 | 122.16 | 123.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 123.60 | 122.45 | 123.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 145.22 | 122.45 | 123.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 143.59 | 126.68 | 125.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 15:15:00 | 146.94 | 141.11 | 134.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 14:15:00 | 145.07 | 145.29 | 139.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 15:00:00 | 145.07 | 145.29 | 139.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 142.02 | 144.42 | 140.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:30:00 | 140.86 | 144.42 | 140.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 140.02 | 143.04 | 140.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 140.02 | 143.04 | 140.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 140.51 | 142.53 | 140.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:30:00 | 141.37 | 142.48 | 140.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 138.70 | 141.14 | 140.53 | SL hit (close<static) qty=1.00 sl=139.36 alert=retest2 |

### Cycle 118 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 139.50 | 141.83 | 141.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 14:15:00 | 137.04 | 140.05 | 141.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 09:15:00 | 140.25 | 139.56 | 140.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 140.25 | 139.56 | 140.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 140.25 | 139.56 | 140.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:45:00 | 140.75 | 139.56 | 140.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 139.88 | 139.45 | 140.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:30:00 | 140.05 | 139.45 | 140.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 140.34 | 139.54 | 140.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:45:00 | 140.38 | 139.54 | 140.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 139.90 | 139.61 | 140.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 137.84 | 139.61 | 140.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 13:15:00 | 145.82 | 140.26 | 140.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 145.82 | 140.26 | 140.15 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 11:15:00 | 139.14 | 140.67 | 140.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 14:15:00 | 138.26 | 139.71 | 140.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 13:15:00 | 139.22 | 138.92 | 139.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 13:15:00 | 139.22 | 138.92 | 139.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 139.22 | 138.92 | 139.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:30:00 | 139.75 | 138.92 | 139.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 139.16 | 138.97 | 139.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 139.11 | 138.97 | 139.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 139.28 | 139.03 | 139.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 139.33 | 139.03 | 139.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 138.28 | 138.88 | 139.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 12:30:00 | 137.69 | 138.53 | 139.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 13:30:00 | 137.85 | 138.39 | 138.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 14:30:00 | 137.85 | 138.44 | 138.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:30:00 | 137.80 | 138.28 | 138.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 140.46 | 138.71 | 138.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 140.46 | 138.71 | 138.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-19 11:15:00 | 141.40 | 139.25 | 139.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 11:15:00 | 141.40 | 139.25 | 139.16 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 137.90 | 138.96 | 139.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 137.21 | 138.61 | 138.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 139.40 | 138.48 | 138.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 139.40 | 138.48 | 138.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 139.40 | 138.48 | 138.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 13:30:00 | 135.35 | 137.95 | 138.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 141.44 | 138.40 | 138.50 | SL hit (close>static) qty=1.00 sl=139.75 alert=retest2 |

### Cycle 123 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 140.75 | 138.87 | 138.70 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 136.34 | 138.64 | 138.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 132.69 | 137.00 | 137.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 135.30 | 135.13 | 136.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 09:15:00 | 133.70 | 135.13 | 136.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 121.05 | 119.37 | 121.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 121.99 | 119.37 | 121.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 121.40 | 119.77 | 121.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 115.10 | 120.59 | 121.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 11:15:00 | 119.85 | 118.35 | 118.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 119.85 | 118.35 | 118.21 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 115.97 | 118.02 | 118.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 113.77 | 116.47 | 117.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 111.16 | 110.43 | 112.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 111.16 | 110.43 | 112.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 113.69 | 110.81 | 111.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 113.80 | 110.81 | 111.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 114.71 | 111.59 | 111.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 114.71 | 111.59 | 111.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 114.69 | 112.21 | 112.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 116.49 | 113.06 | 112.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 113.41 | 114.02 | 113.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 113.41 | 114.02 | 113.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 113.41 | 114.02 | 113.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 115.40 | 113.32 | 113.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 11:15:00 | 112.12 | 113.10 | 113.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 112.12 | 113.10 | 113.12 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 114.87 | 113.34 | 113.20 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 109.95 | 112.84 | 113.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 109.20 | 112.11 | 112.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 112.39 | 110.67 | 111.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 112.39 | 110.67 | 111.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 112.39 | 110.67 | 111.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 11:30:00 | 110.99 | 110.97 | 111.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 13:15:00 | 113.42 | 111.75 | 111.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 113.42 | 111.75 | 111.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 114.88 | 112.73 | 112.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 113.88 | 113.99 | 113.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 113.88 | 113.99 | 113.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 113.19 | 113.83 | 113.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 111.50 | 113.83 | 113.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 110.39 | 113.14 | 112.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 110.39 | 113.14 | 112.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 109.62 | 112.44 | 112.61 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 13:15:00 | 114.55 | 112.88 | 112.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 14:15:00 | 116.85 | 113.67 | 113.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 111.06 | 113.65 | 113.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 111.06 | 113.65 | 113.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 111.06 | 113.65 | 113.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 111.06 | 113.65 | 113.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 110.79 | 113.08 | 113.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:45:00 | 110.40 | 113.08 | 113.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 111.13 | 112.69 | 112.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 109.31 | 111.46 | 112.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 113.45 | 111.22 | 111.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 113.45 | 111.22 | 111.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 113.45 | 111.22 | 111.93 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 115.40 | 112.62 | 112.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 116.68 | 114.19 | 113.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 122.01 | 122.17 | 120.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 122.01 | 122.17 | 120.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 121.01 | 122.79 | 122.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 121.78 | 122.79 | 122.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 09:15:00 | 133.96 | 131.20 | 128.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 129.15 | 131.55 | 131.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 128.82 | 131.00 | 131.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 133.50 | 131.22 | 131.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 133.50 | 131.22 | 131.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 133.50 | 131.22 | 131.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 133.50 | 131.22 | 131.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 134.39 | 131.85 | 131.65 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 131.60 | 132.03 | 132.05 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 133.00 | 132.23 | 132.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 133.79 | 132.54 | 132.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 131.33 | 132.41 | 132.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 131.33 | 132.41 | 132.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 131.33 | 132.41 | 132.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 131.33 | 132.41 | 132.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 130.07 | 131.94 | 132.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 128.24 | 131.02 | 131.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 130.20 | 129.57 | 130.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 130.20 | 129.57 | 130.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 130.20 | 129.57 | 130.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 130.53 | 129.57 | 130.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 129.50 | 129.56 | 130.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 129.51 | 129.56 | 130.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 130.00 | 129.45 | 130.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:00:00 | 130.00 | 129.45 | 130.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 129.34 | 129.43 | 129.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 15:15:00 | 129.11 | 129.43 | 129.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:00:00 | 128.86 | 129.26 | 129.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 130.26 | 128.85 | 129.32 | SL hit (close>static) qty=1.00 sl=130.21 alert=retest2 |

### Cycle 141 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 130.36 | 129.67 | 129.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 132.21 | 130.18 | 129.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 11:15:00 | 131.36 | 132.11 | 131.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 11:15:00 | 131.36 | 132.11 | 131.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 131.36 | 132.11 | 131.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:45:00 | 131.38 | 132.11 | 131.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 133.81 | 134.32 | 133.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:45:00 | 133.21 | 134.32 | 133.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 09:15:00 | 134.25 | 134.27 | 133.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-11 10:45:00 | 135.72 | 134.57 | 133.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-27 12:15:00 | 143.95 | 2024-05-31 14:15:00 | 136.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 14:00:00 | 143.90 | 2024-05-31 14:15:00 | 136.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 12:15:00 | 143.95 | 2024-06-03 09:15:00 | 138.45 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2024-05-27 14:00:00 | 143.90 | 2024-06-03 09:15:00 | 138.45 | STOP_HIT | 0.50 | 3.79% |
| BUY | retest2 | 2024-06-12 15:15:00 | 138.78 | 2024-06-21 12:15:00 | 144.24 | STOP_HIT | 1.00 | 3.93% |
| SELL | retest2 | 2024-07-08 12:00:00 | 151.58 | 2024-07-08 12:15:00 | 152.12 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-07-16 09:15:00 | 171.52 | 2024-07-29 09:15:00 | 188.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-18 10:30:00 | 171.92 | 2024-07-29 09:15:00 | 188.63 | TARGET_HIT | 1.00 | 9.72% |
| BUY | retest2 | 2024-07-23 12:30:00 | 171.48 | 2024-07-29 14:15:00 | 189.11 | TARGET_HIT | 1.00 | 10.28% |
| SELL | retest2 | 2024-08-13 10:15:00 | 176.90 | 2024-08-19 10:15:00 | 179.06 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-08-13 12:30:00 | 177.24 | 2024-08-19 10:15:00 | 179.06 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-08-13 14:15:00 | 177.00 | 2024-08-19 10:15:00 | 179.06 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-08-16 14:15:00 | 176.98 | 2024-08-19 10:15:00 | 179.06 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-09-04 11:30:00 | 187.78 | 2024-09-11 13:15:00 | 178.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-04 11:30:00 | 187.78 | 2024-09-12 11:15:00 | 181.64 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2024-09-06 09:45:00 | 185.86 | 2024-09-17 09:15:00 | 176.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-06 09:45:00 | 185.86 | 2024-09-17 13:15:00 | 177.66 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2024-09-20 10:15:00 | 176.51 | 2024-09-25 12:15:00 | 167.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-20 13:00:00 | 176.06 | 2024-09-26 09:15:00 | 167.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-20 10:15:00 | 176.51 | 2024-09-27 11:15:00 | 166.72 | STOP_HIT | 0.50 | 5.55% |
| SELL | retest2 | 2024-09-20 13:00:00 | 176.06 | 2024-09-27 11:15:00 | 166.72 | STOP_HIT | 0.50 | 5.31% |
| SELL | retest2 | 2024-10-15 12:15:00 | 162.11 | 2024-10-16 12:15:00 | 166.20 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2024-10-25 13:15:00 | 147.23 | 2024-10-30 12:15:00 | 154.00 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2024-10-28 09:30:00 | 147.62 | 2024-10-30 12:15:00 | 154.00 | STOP_HIT | 1.00 | -4.32% |
| SELL | retest2 | 2024-10-29 10:45:00 | 147.77 | 2024-10-30 12:15:00 | 154.00 | STOP_HIT | 1.00 | -4.22% |
| SELL | retest2 | 2024-10-29 13:30:00 | 147.80 | 2024-10-30 12:15:00 | 154.00 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2024-11-18 14:45:00 | 147.28 | 2024-11-25 09:15:00 | 151.65 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2024-11-21 09:15:00 | 146.79 | 2024-11-25 09:15:00 | 151.65 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2024-11-21 10:30:00 | 147.02 | 2024-11-25 09:15:00 | 151.65 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2024-11-29 14:15:00 | 153.10 | 2024-12-05 09:15:00 | 168.41 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-20 12:30:00 | 168.08 | 2024-12-24 09:15:00 | 159.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 12:30:00 | 168.08 | 2024-12-24 10:15:00 | 162.40 | STOP_HIT | 0.50 | 3.38% |
| SELL | retest2 | 2025-01-08 09:15:00 | 156.40 | 2025-01-10 09:15:00 | 148.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 156.40 | 2025-01-14 10:15:00 | 146.50 | STOP_HIT | 0.50 | 6.33% |
| SELL | retest2 | 2025-01-23 11:30:00 | 148.68 | 2025-01-24 14:15:00 | 151.60 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-01-24 09:15:00 | 147.87 | 2025-01-24 14:15:00 | 151.60 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-02-06 11:00:00 | 131.14 | 2025-02-11 09:15:00 | 124.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 11:00:00 | 131.14 | 2025-02-11 15:15:00 | 124.20 | STOP_HIT | 0.50 | 5.29% |
| SELL | retest2 | 2025-04-04 09:30:00 | 131.20 | 2025-04-04 14:15:00 | 124.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:30:00 | 131.20 | 2025-04-07 09:15:00 | 118.08 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-23 14:15:00 | 133.29 | 2025-04-24 12:15:00 | 131.69 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-05-09 14:45:00 | 144.79 | 2025-05-14 14:15:00 | 143.95 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-05-13 14:00:00 | 144.73 | 2025-05-14 14:15:00 | 143.95 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-05-14 11:30:00 | 144.88 | 2025-05-14 14:15:00 | 143.95 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-05-28 13:15:00 | 148.01 | 2025-05-30 10:15:00 | 140.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-28 15:15:00 | 146.50 | 2025-05-30 10:15:00 | 139.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-29 11:30:00 | 147.78 | 2025-05-30 10:15:00 | 140.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-28 13:15:00 | 148.01 | 2025-05-30 14:15:00 | 133.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-28 15:15:00 | 146.50 | 2025-05-30 14:15:00 | 133.00 | TARGET_HIT | 0.50 | 9.21% |
| SELL | retest2 | 2025-05-29 11:30:00 | 147.78 | 2025-05-30 15:15:00 | 131.85 | TARGET_HIT | 0.50 | 10.78% |
| BUY | retest2 | 2025-06-10 09:15:00 | 138.78 | 2025-06-12 11:15:00 | 137.13 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-10 11:00:00 | 138.75 | 2025-06-12 11:15:00 | 137.13 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-06-10 13:00:00 | 140.28 | 2025-06-12 11:15:00 | 137.13 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-06-11 14:15:00 | 138.73 | 2025-06-12 11:15:00 | 137.13 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-06-17 12:00:00 | 133.10 | 2025-06-24 12:15:00 | 130.50 | STOP_HIT | 1.00 | 1.95% |
| SELL | retest2 | 2025-06-17 12:45:00 | 132.89 | 2025-06-24 12:15:00 | 130.50 | STOP_HIT | 1.00 | 1.80% |
| SELL | retest2 | 2025-06-18 09:45:00 | 132.73 | 2025-06-24 12:15:00 | 130.50 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2025-07-08 09:15:00 | 145.68 | 2025-07-09 12:15:00 | 142.28 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-07-08 15:15:00 | 144.13 | 2025-07-09 12:15:00 | 142.28 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-07-11 09:30:00 | 142.61 | 2025-07-18 09:15:00 | 141.70 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-08-05 09:15:00 | 126.17 | 2025-08-06 09:15:00 | 119.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 09:15:00 | 126.17 | 2025-08-07 09:15:00 | 113.55 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-09 11:45:00 | 114.14 | 2025-09-10 09:15:00 | 125.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-09 14:30:00 | 114.12 | 2025-09-10 09:15:00 | 125.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-09 15:00:00 | 114.25 | 2025-09-10 09:15:00 | 125.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-10 09:15:00 | 123.00 | 2025-09-19 15:15:00 | 124.65 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2025-09-16 09:15:00 | 123.03 | 2025-09-19 15:15:00 | 124.65 | STOP_HIT | 1.00 | 1.32% |
| SELL | retest2 | 2025-09-25 09:30:00 | 121.18 | 2025-09-29 11:15:00 | 115.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:30:00 | 121.18 | 2025-09-30 15:15:00 | 114.98 | STOP_HIT | 0.50 | 5.12% |
| SELL | retest2 | 2025-10-08 09:15:00 | 114.93 | 2025-10-08 15:15:00 | 117.39 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-11-03 09:15:00 | 131.49 | 2025-11-04 09:15:00 | 130.60 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-11-10 09:15:00 | 128.60 | 2025-11-10 10:15:00 | 130.10 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-11-18 09:15:00 | 139.20 | 2025-11-19 13:15:00 | 137.20 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-11-20 13:30:00 | 137.97 | 2025-11-20 14:15:00 | 140.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-11-27 09:30:00 | 132.49 | 2025-11-28 09:15:00 | 136.85 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-11-27 10:00:00 | 132.48 | 2025-11-28 09:15:00 | 136.85 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-11-27 10:45:00 | 132.05 | 2025-11-28 09:15:00 | 136.85 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2025-11-27 15:15:00 | 132.13 | 2025-11-28 09:15:00 | 136.85 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest1 | 2025-12-09 14:30:00 | 131.06 | 2025-12-10 09:15:00 | 134.69 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-12-17 09:45:00 | 134.73 | 2025-12-19 09:15:00 | 137.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-12-31 09:15:00 | 132.44 | 2026-01-02 15:15:00 | 133.15 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-08 11:15:00 | 128.56 | 2026-01-12 09:15:00 | 122.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:15:00 | 128.56 | 2026-01-12 15:15:00 | 122.40 | STOP_HIT | 0.50 | 4.79% |
| SELL | retest2 | 2026-01-21 14:15:00 | 115.98 | 2026-01-21 15:15:00 | 117.99 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-01-21 14:45:00 | 116.01 | 2026-01-21 15:15:00 | 117.99 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2026-01-23 14:30:00 | 121.37 | 2026-01-29 12:15:00 | 123.37 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2026-01-27 09:15:00 | 124.07 | 2026-01-29 12:15:00 | 123.37 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2026-02-05 13:30:00 | 141.37 | 2026-02-06 10:15:00 | 138.70 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-02-09 09:15:00 | 145.86 | 2026-02-10 10:15:00 | 138.90 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2026-02-12 09:15:00 | 137.84 | 2026-02-12 13:15:00 | 145.82 | STOP_HIT | 1.00 | -5.79% |
| SELL | retest2 | 2026-02-18 12:30:00 | 137.69 | 2026-02-19 11:15:00 | 141.40 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-02-18 13:30:00 | 137.85 | 2026-02-19 11:15:00 | 141.40 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-02-18 14:30:00 | 137.85 | 2026-02-19 11:15:00 | 141.40 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-02-19 09:30:00 | 137.80 | 2026-02-19 11:15:00 | 141.40 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-02-20 13:30:00 | 135.35 | 2026-02-23 09:15:00 | 141.44 | STOP_HIT | 1.00 | -4.50% |
| SELL | retest2 | 2026-03-09 09:15:00 | 115.10 | 2026-03-11 11:15:00 | 119.85 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2026-03-20 09:15:00 | 115.40 | 2026-03-20 11:15:00 | 112.12 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-03-24 11:30:00 | 110.99 | 2026-03-24 13:15:00 | 113.42 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-04-13 10:15:00 | 121.78 | 2026-04-22 09:15:00 | 133.96 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 15:15:00 | 129.11 | 2026-05-05 14:15:00 | 130.26 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-05-05 10:00:00 | 128.86 | 2026-05-05 14:15:00 | 130.26 | STOP_HIT | 1.00 | -1.09% |
