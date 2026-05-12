# Biocon Ltd. (BIOCON)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 378.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 158 |
| ALERT1 | 107 |
| ALERT2 | 106 |
| ALERT2_SKIP | 62 |
| ALERT3 | 284 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 95 |
| PARTIAL | 20 |
| TARGET_HIT | 6 |
| STOP_HIT | 88 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 114 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 57 / 57
- **Target hits / Stop hits / Partials:** 6 / 88 / 20
- **Avg / median % per leg:** 1.43% / 0.23%
- **Sum % (uncompounded):** 163.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 15 | 31.2% | 1 | 47 | 0 | -0.15% | -7.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 48 | 15 | 31.2% | 1 | 47 | 0 | -0.15% | -7.1% |
| SELL (all) | 66 | 42 | 63.6% | 5 | 41 | 20 | 2.58% | 170.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 66 | 42 | 63.6% | 5 | 41 | 20 | 2.58% | 170.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 114 | 57 | 50.0% | 6 | 88 | 20 | 1.43% | 163.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 297.65 | 301.98 | 302.53 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 304.55 | 302.38 | 302.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 307.00 | 303.30 | 302.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 15:15:00 | 311.50 | 313.31 | 310.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 15:15:00 | 311.50 | 313.31 | 310.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 311.50 | 313.31 | 310.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:15:00 | 308.20 | 313.31 | 310.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 307.35 | 312.12 | 309.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:45:00 | 306.40 | 312.12 | 309.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 308.40 | 311.38 | 309.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:15:00 | 308.30 | 311.38 | 309.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 13:15:00 | 304.70 | 308.65 | 308.76 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 310.80 | 308.78 | 308.75 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 14:15:00 | 306.00 | 308.42 | 308.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-18 11:15:00 | 304.95 | 307.17 | 307.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-21 09:15:00 | 306.95 | 306.75 | 307.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 306.95 | 306.75 | 307.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 306.95 | 306.75 | 307.62 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 10:15:00 | 309.15 | 308.14 | 308.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 11:15:00 | 314.00 | 309.36 | 308.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 15:15:00 | 318.50 | 319.88 | 316.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-27 09:15:00 | 312.90 | 319.88 | 316.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 309.10 | 317.72 | 315.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:45:00 | 306.40 | 317.72 | 315.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 311.65 | 316.51 | 315.36 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 12:15:00 | 311.15 | 314.65 | 314.67 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 13:15:00 | 316.00 | 314.92 | 314.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 09:15:00 | 317.35 | 315.39 | 315.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 13:15:00 | 315.45 | 315.80 | 315.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 13:15:00 | 315.45 | 315.80 | 315.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 315.45 | 315.80 | 315.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 315.45 | 315.80 | 315.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 317.50 | 316.14 | 315.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 13:45:00 | 320.10 | 317.89 | 316.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 11:15:00 | 314.00 | 317.57 | 317.22 | SL hit (close<static) qty=1.00 sl=315.25 alert=retest2 |

### Cycle 9 — SELL (started 2024-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 13:15:00 | 313.10 | 316.26 | 316.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 14:15:00 | 310.30 | 315.07 | 316.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 310.85 | 310.63 | 312.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 310.85 | 310.63 | 312.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 310.85 | 310.63 | 312.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:15:00 | 310.00 | 310.63 | 312.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 13:30:00 | 310.40 | 310.67 | 311.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:15:00 | 310.25 | 310.67 | 311.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 294.45 | 310.46 | 311.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 294.50 | 309.23 | 310.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 294.88 | 309.23 | 310.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 294.74 | 309.23 | 310.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 279.00 | 298.68 | 305.20 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 10 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 314.05 | 303.90 | 303.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 11:15:00 | 316.05 | 307.81 | 305.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 09:15:00 | 343.05 | 343.55 | 338.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 09:15:00 | 343.05 | 343.55 | 338.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 343.05 | 343.55 | 338.59 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 09:15:00 | 337.50 | 338.58 | 338.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 09:15:00 | 327.50 | 334.40 | 336.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 09:15:00 | 333.50 | 330.50 | 332.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 333.50 | 330.50 | 332.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 333.50 | 330.50 | 332.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:45:00 | 338.00 | 330.50 | 332.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 335.10 | 331.42 | 333.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:00:00 | 335.10 | 331.42 | 333.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 336.65 | 332.47 | 333.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:00:00 | 336.65 | 332.47 | 333.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 333.10 | 333.11 | 333.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 337.75 | 333.11 | 333.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 336.80 | 333.85 | 333.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 10:15:00 | 341.40 | 335.36 | 334.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 11:15:00 | 343.80 | 343.95 | 341.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-24 11:45:00 | 343.80 | 343.95 | 341.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 342.85 | 343.71 | 342.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 342.85 | 343.71 | 342.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 342.70 | 343.51 | 342.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 09:15:00 | 343.10 | 343.51 | 342.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 10:15:00 | 340.35 | 342.89 | 342.20 | SL hit (close<static) qty=1.00 sl=341.75 alert=retest2 |

### Cycle 13 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 11:15:00 | 337.10 | 341.73 | 341.74 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 350.00 | 342.63 | 341.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 351.80 | 344.47 | 342.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 346.20 | 347.52 | 345.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 13:00:00 | 346.20 | 347.52 | 345.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 344.20 | 346.85 | 345.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 344.20 | 346.85 | 345.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 345.55 | 346.59 | 345.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 344.60 | 346.59 | 345.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 344.70 | 346.21 | 345.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 349.35 | 346.21 | 345.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 12:15:00 | 363.20 | 365.32 | 365.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 12:15:00 | 363.20 | 365.32 | 365.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 358.55 | 363.29 | 364.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 15:15:00 | 355.25 | 355.09 | 357.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-12 09:15:00 | 352.95 | 355.09 | 357.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 354.70 | 352.52 | 354.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:45:00 | 353.60 | 352.52 | 354.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 357.80 | 353.58 | 354.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:00:00 | 357.80 | 353.58 | 354.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 360.00 | 354.86 | 355.35 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2024-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 12:15:00 | 359.15 | 355.72 | 355.70 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 14:15:00 | 353.00 | 356.06 | 356.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 350.70 | 354.61 | 355.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 13:15:00 | 336.65 | 334.76 | 337.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 13:15:00 | 336.65 | 334.76 | 337.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 336.65 | 334.76 | 337.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:00:00 | 336.65 | 334.76 | 337.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 336.85 | 335.18 | 337.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 336.85 | 335.18 | 337.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 338.65 | 336.16 | 337.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:15:00 | 339.50 | 336.16 | 337.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 340.60 | 337.05 | 338.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 340.60 | 337.05 | 338.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 347.80 | 340.15 | 339.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 14:15:00 | 348.60 | 341.84 | 340.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 362.80 | 366.08 | 359.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 09:30:00 | 363.50 | 366.08 | 359.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 360.15 | 364.23 | 360.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 13:00:00 | 360.15 | 364.23 | 360.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 361.60 | 363.71 | 360.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:30:00 | 364.80 | 361.60 | 360.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 12:00:00 | 364.00 | 362.78 | 361.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:30:00 | 366.55 | 363.72 | 362.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 11:00:00 | 364.00 | 363.78 | 362.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 359.40 | 362.90 | 362.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 359.40 | 362.90 | 362.35 | SL hit (close<static) qty=1.00 sl=360.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 13:15:00 | 357.10 | 361.24 | 361.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 354.50 | 359.89 | 361.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 358.60 | 357.78 | 359.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 10:15:00 | 358.60 | 357.78 | 359.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 358.60 | 357.78 | 359.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:45:00 | 358.20 | 357.78 | 359.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 358.80 | 357.98 | 359.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:30:00 | 360.40 | 357.98 | 359.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 360.25 | 358.44 | 359.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 13:00:00 | 360.25 | 358.44 | 359.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 359.60 | 358.67 | 359.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:00:00 | 359.60 | 358.67 | 359.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 355.25 | 357.99 | 359.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 343.00 | 357.59 | 358.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 11:15:00 | 346.30 | 345.49 | 345.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 346.30 | 345.49 | 345.41 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 12:15:00 | 342.65 | 344.92 | 345.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 14:15:00 | 339.70 | 343.54 | 344.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 10:15:00 | 340.10 | 338.24 | 340.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 10:15:00 | 340.10 | 338.24 | 340.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 340.10 | 338.24 | 340.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:00:00 | 340.10 | 338.24 | 340.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 340.45 | 338.68 | 340.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 12:00:00 | 340.45 | 338.68 | 340.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 340.40 | 339.03 | 340.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 12:45:00 | 340.35 | 339.03 | 340.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 340.25 | 339.27 | 340.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:00:00 | 340.25 | 339.27 | 340.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 337.45 | 338.91 | 339.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 09:30:00 | 335.20 | 338.69 | 339.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 10:15:00 | 341.60 | 339.27 | 339.79 | SL hit (close>static) qty=1.00 sl=340.60 alert=retest2 |

### Cycle 22 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 341.40 | 334.39 | 334.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 344.95 | 337.54 | 335.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 353.10 | 355.04 | 352.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 09:30:00 | 353.75 | 355.04 | 352.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 352.65 | 353.99 | 352.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:30:00 | 352.85 | 353.99 | 352.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 352.10 | 353.61 | 352.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:00:00 | 352.10 | 353.61 | 352.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 351.80 | 353.25 | 352.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 351.80 | 353.25 | 352.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 352.00 | 353.00 | 352.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 353.55 | 353.00 | 352.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 15:00:00 | 352.85 | 353.63 | 353.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 13:15:00 | 352.10 | 354.89 | 355.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 13:15:00 | 352.10 | 354.89 | 355.20 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 360.50 | 356.05 | 355.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 09:15:00 | 363.35 | 359.79 | 358.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 361.25 | 363.04 | 361.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 14:15:00 | 361.25 | 363.04 | 361.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 361.25 | 363.04 | 361.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:45:00 | 362.10 | 363.04 | 361.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 361.50 | 362.73 | 361.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 360.30 | 362.73 | 361.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 359.20 | 362.02 | 361.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:15:00 | 356.90 | 362.02 | 361.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 11:15:00 | 360.85 | 361.03 | 361.04 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 12:15:00 | 368.55 | 362.54 | 361.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 13:15:00 | 375.40 | 365.11 | 362.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 372.15 | 380.32 | 375.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 372.15 | 380.32 | 375.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 372.15 | 380.32 | 375.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 372.15 | 380.32 | 375.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 370.10 | 378.28 | 374.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:00:00 | 370.10 | 378.28 | 374.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 374.20 | 376.94 | 374.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:00:00 | 374.20 | 376.94 | 374.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 373.40 | 376.23 | 374.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:30:00 | 374.70 | 376.23 | 374.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 373.90 | 375.76 | 374.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:30:00 | 371.85 | 375.76 | 374.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 373.00 | 375.21 | 374.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 368.00 | 375.21 | 374.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 374.50 | 374.64 | 374.37 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 371.50 | 374.01 | 374.11 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 13:15:00 | 378.20 | 374.85 | 374.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 380.05 | 376.53 | 375.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 385.00 | 387.33 | 383.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 14:15:00 | 385.00 | 387.33 | 383.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 385.00 | 387.33 | 383.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 385.00 | 387.33 | 383.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 383.90 | 386.50 | 384.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 383.90 | 386.50 | 384.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 380.45 | 385.29 | 384.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:45:00 | 379.15 | 385.29 | 384.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 376.40 | 383.51 | 383.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:45:00 | 376.10 | 383.51 | 383.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 15:15:00 | 377.00 | 382.21 | 382.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 10:15:00 | 375.55 | 380.20 | 381.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 12:15:00 | 381.95 | 380.05 | 381.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 12:15:00 | 381.95 | 380.05 | 381.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 381.95 | 380.05 | 381.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 13:00:00 | 381.95 | 380.05 | 381.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 381.75 | 380.39 | 381.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 14:15:00 | 382.55 | 380.39 | 381.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 379.00 | 380.11 | 381.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 14:30:00 | 380.10 | 380.11 | 381.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 382.70 | 380.52 | 381.22 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 10:15:00 | 390.70 | 382.55 | 382.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 12:15:00 | 392.70 | 385.53 | 383.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 382.50 | 387.23 | 385.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 382.50 | 387.23 | 385.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 382.50 | 387.23 | 385.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 382.50 | 387.23 | 385.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 379.25 | 385.63 | 384.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 379.25 | 385.63 | 384.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 12:15:00 | 382.00 | 383.99 | 384.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 13:15:00 | 378.65 | 382.92 | 383.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 362.45 | 361.40 | 367.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 362.45 | 361.40 | 367.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 362.70 | 361.52 | 366.30 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 372.20 | 367.18 | 366.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 375.90 | 371.20 | 369.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 10:15:00 | 372.65 | 372.67 | 370.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 10:45:00 | 372.30 | 372.67 | 370.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 369.05 | 371.95 | 370.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:45:00 | 368.95 | 371.95 | 370.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 369.15 | 371.39 | 370.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:00:00 | 371.50 | 371.28 | 370.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 14:45:00 | 372.95 | 371.50 | 371.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 13:45:00 | 371.15 | 371.17 | 371.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 369.15 | 370.77 | 370.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 369.15 | 370.77 | 370.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 364.60 | 369.40 | 370.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 12:15:00 | 366.60 | 365.01 | 366.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 12:15:00 | 366.60 | 365.01 | 366.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 366.60 | 365.01 | 366.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:15:00 | 367.40 | 365.01 | 366.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 368.00 | 365.61 | 366.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:00:00 | 368.00 | 365.61 | 366.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 371.65 | 366.82 | 367.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 371.65 | 366.82 | 367.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 15:15:00 | 370.00 | 367.45 | 367.45 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 364.50 | 366.86 | 367.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 360.65 | 365.04 | 366.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 344.65 | 342.49 | 347.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:30:00 | 343.45 | 342.49 | 347.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 348.60 | 343.91 | 346.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:30:00 | 348.75 | 343.91 | 346.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 348.65 | 344.86 | 346.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 349.30 | 344.86 | 346.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 12:15:00 | 349.00 | 346.25 | 346.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 13:00:00 | 349.00 | 346.25 | 346.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 351.15 | 347.23 | 347.16 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 346.45 | 347.08 | 347.09 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 15:15:00 | 347.30 | 347.12 | 347.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 350.25 | 347.75 | 347.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 347.60 | 348.07 | 347.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 12:15:00 | 347.60 | 348.07 | 347.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 347.60 | 348.07 | 347.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 347.60 | 348.07 | 347.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 345.95 | 347.65 | 347.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:00:00 | 345.95 | 347.65 | 347.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 345.80 | 347.28 | 347.35 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 09:15:00 | 348.60 | 346.84 | 346.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 12:15:00 | 351.75 | 348.35 | 347.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 349.55 | 350.22 | 348.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 10:00:00 | 349.55 | 350.22 | 348.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 347.50 | 349.67 | 348.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:00:00 | 347.50 | 349.67 | 348.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 344.70 | 348.68 | 348.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 345.50 | 348.68 | 348.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 345.30 | 348.00 | 348.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 342.55 | 346.24 | 347.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 341.50 | 340.13 | 342.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 13:15:00 | 341.50 | 340.13 | 342.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 341.50 | 340.13 | 342.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:00:00 | 341.50 | 340.13 | 342.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 340.65 | 340.23 | 341.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 341.00 | 340.23 | 341.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 339.80 | 340.08 | 341.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 13:15:00 | 336.60 | 340.09 | 341.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 336.90 | 339.46 | 340.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:45:00 | 337.00 | 339.04 | 340.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 15:15:00 | 336.05 | 339.04 | 340.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 319.77 | 321.94 | 326.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 320.05 | 321.94 | 326.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 320.15 | 321.94 | 326.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 319.25 | 321.94 | 326.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 316.80 | 315.35 | 319.88 | SL hit (close>ema200) qty=0.50 sl=315.35 alert=retest2 |

### Cycle 42 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 325.20 | 321.46 | 321.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 325.95 | 322.36 | 321.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 11:15:00 | 322.35 | 324.01 | 322.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 11:15:00 | 322.35 | 324.01 | 322.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 322.35 | 324.01 | 322.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 12:00:00 | 322.35 | 324.01 | 322.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 12:15:00 | 326.00 | 324.41 | 323.21 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 09:15:00 | 321.45 | 322.43 | 322.53 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 319.45 | 315.12 | 315.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 320.55 | 317.48 | 316.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 11:15:00 | 323.65 | 326.37 | 323.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 11:15:00 | 323.65 | 326.37 | 323.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 323.65 | 326.37 | 323.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 323.65 | 326.37 | 323.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 324.40 | 325.97 | 323.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:45:00 | 320.30 | 325.97 | 323.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 322.05 | 325.19 | 323.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:30:00 | 321.85 | 325.19 | 323.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 321.30 | 324.41 | 323.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 15:00:00 | 321.30 | 324.41 | 323.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 320.60 | 323.65 | 323.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 09:15:00 | 335.85 | 323.65 | 323.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 13:15:00 | 331.30 | 337.83 | 338.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 13:15:00 | 331.30 | 337.83 | 338.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 09:15:00 | 327.30 | 334.15 | 335.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 329.90 | 328.96 | 331.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 10:00:00 | 329.90 | 328.96 | 331.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 331.90 | 329.55 | 331.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:45:00 | 332.20 | 329.55 | 331.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 332.15 | 330.07 | 331.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:30:00 | 333.85 | 330.07 | 331.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 330.60 | 330.42 | 331.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:45:00 | 332.55 | 330.42 | 331.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 326.45 | 329.63 | 331.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:45:00 | 328.60 | 329.63 | 331.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 328.30 | 325.43 | 326.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:00:00 | 328.30 | 325.43 | 326.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 329.50 | 326.25 | 327.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 329.50 | 326.25 | 327.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 331.55 | 328.21 | 327.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 335.95 | 330.30 | 328.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 12:15:00 | 362.70 | 363.12 | 358.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 12:45:00 | 362.75 | 363.12 | 358.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 371.30 | 373.73 | 371.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 14:15:00 | 375.00 | 372.96 | 371.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 13:30:00 | 375.00 | 375.74 | 374.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 376.30 | 375.85 | 374.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:15:00 | 376.55 | 375.59 | 374.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 375.65 | 375.60 | 374.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 13:45:00 | 378.80 | 376.73 | 375.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 364.30 | 375.11 | 375.05 | SL hit (close<static) qty=1.00 sl=369.10 alert=retest2 |

### Cycle 47 — SELL (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 10:15:00 | 369.20 | 373.93 | 374.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 363.10 | 367.27 | 368.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 14:15:00 | 353.40 | 353.04 | 355.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 14:45:00 | 353.60 | 353.04 | 355.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 352.70 | 351.57 | 353.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:30:00 | 354.00 | 351.57 | 353.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 356.90 | 352.64 | 353.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:00:00 | 356.90 | 352.64 | 353.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 356.00 | 353.31 | 353.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:15:00 | 353.95 | 353.31 | 353.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 350.80 | 352.81 | 353.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:45:00 | 346.60 | 351.25 | 352.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 345.40 | 350.67 | 352.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 10:00:00 | 345.55 | 349.64 | 351.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:45:00 | 345.95 | 348.92 | 350.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 14:15:00 | 329.27 | 334.47 | 339.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 335.35 | 333.92 | 338.66 | SL hit (close>ema200) qty=0.50 sl=333.92 alert=retest2 |

### Cycle 48 — BUY (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 15:15:00 | 343.15 | 340.51 | 340.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 349.75 | 345.33 | 343.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 362.70 | 362.95 | 356.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 10:00:00 | 362.70 | 362.95 | 356.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 370.20 | 370.93 | 368.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:00:00 | 370.20 | 370.93 | 368.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 368.00 | 369.88 | 368.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 366.35 | 369.88 | 368.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 369.50 | 369.81 | 368.91 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 363.55 | 367.52 | 367.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 359.00 | 364.77 | 366.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 373.35 | 364.63 | 365.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 373.35 | 364.63 | 365.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 373.35 | 364.63 | 365.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 372.25 | 364.63 | 365.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 10:15:00 | 385.60 | 368.82 | 367.67 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 09:15:00 | 370.80 | 372.79 | 372.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 364.80 | 370.52 | 371.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 09:15:00 | 375.70 | 365.77 | 367.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-13 09:15:00 | 375.70 | 365.77 | 367.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 375.70 | 365.77 | 367.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:30:00 | 375.70 | 365.77 | 367.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 371.75 | 366.97 | 368.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 11:15:00 | 369.85 | 366.97 | 368.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 380.20 | 370.27 | 369.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 10:15:00 | 380.20 | 370.27 | 369.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 11:15:00 | 385.00 | 373.21 | 370.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 09:15:00 | 381.70 | 382.01 | 376.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-15 10:00:00 | 381.70 | 382.01 | 376.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 394.20 | 397.55 | 394.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 394.20 | 397.55 | 394.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 395.30 | 397.10 | 394.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 396.00 | 397.10 | 394.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 394.40 | 396.56 | 394.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 394.25 | 396.56 | 394.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 394.05 | 396.06 | 394.51 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 386.05 | 392.30 | 393.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 381.55 | 388.61 | 391.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 388.70 | 386.95 | 389.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 388.70 | 386.95 | 389.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 395.70 | 388.93 | 390.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 395.70 | 388.93 | 390.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 395.25 | 390.20 | 390.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 396.50 | 390.20 | 390.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 396.35 | 391.43 | 391.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 12:15:00 | 397.30 | 392.60 | 391.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 390.00 | 393.98 | 392.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 390.00 | 393.98 | 392.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 390.00 | 393.98 | 392.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 390.00 | 393.98 | 392.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 392.35 | 393.65 | 392.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:15:00 | 393.35 | 393.65 | 392.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 12:00:00 | 393.70 | 393.66 | 392.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 14:15:00 | 388.05 | 391.60 | 392.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 388.05 | 391.60 | 392.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 373.85 | 387.55 | 390.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 367.75 | 364.05 | 370.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 10:00:00 | 367.75 | 364.05 | 370.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 370.45 | 365.33 | 370.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 370.45 | 365.33 | 370.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 369.55 | 366.17 | 370.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:15:00 | 365.00 | 366.17 | 370.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 10:15:00 | 365.20 | 362.95 | 362.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 10:15:00 | 365.20 | 362.95 | 362.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 13:15:00 | 372.85 | 364.93 | 363.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 366.60 | 367.07 | 365.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 366.60 | 367.07 | 365.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 366.60 | 367.07 | 365.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 370.50 | 364.42 | 364.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 10:15:00 | 377.75 | 385.71 | 386.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 377.75 | 385.71 | 386.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 375.20 | 383.61 | 385.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 365.85 | 358.94 | 364.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 365.85 | 358.94 | 364.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 365.85 | 358.94 | 364.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:30:00 | 365.55 | 358.94 | 364.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 364.60 | 360.07 | 364.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:30:00 | 362.50 | 360.55 | 363.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:00:00 | 362.50 | 360.55 | 363.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:30:00 | 362.65 | 360.77 | 363.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 344.38 | 352.42 | 357.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 344.38 | 352.42 | 357.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 344.52 | 352.42 | 357.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 15:15:00 | 347.70 | 347.69 | 351.47 | SL hit (close>ema200) qty=0.50 sl=347.69 alert=retest2 |

### Cycle 58 — BUY (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 12:15:00 | 312.95 | 308.03 | 307.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 313.60 | 309.15 | 308.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 15:15:00 | 332.60 | 332.98 | 328.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 09:15:00 | 332.55 | 332.98 | 328.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 331.25 | 332.17 | 329.45 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 326.60 | 328.64 | 328.65 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 13:15:00 | 329.05 | 328.72 | 328.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 15:15:00 | 330.00 | 329.01 | 328.82 | Break + close above crossover candle high |

### Cycle 61 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 325.25 | 328.26 | 328.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 10:15:00 | 325.00 | 327.60 | 328.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 327.70 | 324.65 | 325.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 327.70 | 324.65 | 325.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 327.70 | 324.65 | 325.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 328.15 | 324.65 | 325.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 331.60 | 326.04 | 326.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 11:15:00 | 333.10 | 327.45 | 326.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 339.00 | 339.53 | 336.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 339.00 | 339.53 | 336.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 350.40 | 348.65 | 345.78 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 12:15:00 | 344.25 | 347.54 | 347.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 13:15:00 | 343.30 | 346.69 | 347.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 345.70 | 342.91 | 344.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 345.70 | 342.91 | 344.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 345.70 | 342.91 | 344.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:30:00 | 348.00 | 342.91 | 344.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 342.75 | 342.88 | 344.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 340.85 | 342.46 | 343.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 12:30:00 | 341.75 | 340.11 | 341.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 15:00:00 | 341.15 | 340.67 | 341.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 348.00 | 342.59 | 342.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 348.00 | 342.59 | 342.31 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 324.95 | 341.25 | 342.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 306.35 | 326.49 | 333.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 315.00 | 311.93 | 319.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 315.00 | 311.93 | 319.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 325.85 | 315.90 | 320.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:45:00 | 326.70 | 315.90 | 320.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 320.00 | 316.72 | 320.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 311.15 | 318.62 | 320.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 326.80 | 317.46 | 316.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 326.80 | 317.46 | 316.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 328.55 | 319.68 | 317.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 15:15:00 | 330.35 | 330.62 | 328.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 09:15:00 | 330.25 | 330.62 | 328.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 330.15 | 331.48 | 330.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 12:15:00 | 336.40 | 331.27 | 330.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 11:15:00 | 327.60 | 331.85 | 332.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 11:15:00 | 327.60 | 331.85 | 332.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 13:15:00 | 326.95 | 330.13 | 331.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 318.85 | 316.67 | 321.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 318.85 | 316.67 | 321.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 320.55 | 318.10 | 321.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:45:00 | 321.70 | 318.10 | 321.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 320.60 | 318.60 | 321.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:30:00 | 321.35 | 318.60 | 321.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 320.65 | 319.01 | 321.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 320.65 | 319.01 | 321.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 320.90 | 319.39 | 321.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 322.25 | 319.39 | 321.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 318.55 | 319.22 | 320.82 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 323.40 | 320.70 | 320.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 11:15:00 | 324.60 | 321.48 | 321.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 321.90 | 321.99 | 321.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 14:15:00 | 321.90 | 321.99 | 321.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 321.90 | 321.99 | 321.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:30:00 | 322.05 | 321.99 | 321.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 321.30 | 321.85 | 321.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:45:00 | 322.80 | 322.08 | 321.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 12:15:00 | 319.20 | 321.08 | 321.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 12:15:00 | 319.20 | 321.08 | 321.20 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 333.05 | 322.65 | 321.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 12:15:00 | 333.65 | 327.85 | 324.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 12:15:00 | 342.45 | 344.40 | 341.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 12:45:00 | 343.15 | 344.40 | 341.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 338.75 | 343.27 | 340.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 338.75 | 343.27 | 340.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 335.20 | 341.66 | 340.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 335.20 | 341.66 | 340.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 333.50 | 338.79 | 339.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-12 09:15:00 | 326.50 | 331.49 | 334.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-13 09:15:00 | 334.40 | 329.80 | 331.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 334.40 | 329.80 | 331.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 334.40 | 329.80 | 331.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 12:30:00 | 330.05 | 330.61 | 331.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 14:15:00 | 329.60 | 330.63 | 331.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 10:15:00 | 337.85 | 332.97 | 332.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 337.85 | 332.97 | 332.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 12:15:00 | 339.35 | 335.14 | 333.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 335.05 | 336.38 | 334.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 335.05 | 336.38 | 334.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 335.05 | 336.38 | 334.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:45:00 | 336.10 | 336.38 | 334.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 334.95 | 336.09 | 334.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:30:00 | 335.40 | 336.09 | 334.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 335.40 | 335.95 | 334.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:30:00 | 335.40 | 335.95 | 334.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 335.40 | 335.84 | 334.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:45:00 | 334.40 | 335.84 | 334.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 337.20 | 336.11 | 335.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:15:00 | 337.95 | 336.11 | 335.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 341.50 | 336.86 | 335.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:30:00 | 338.75 | 341.42 | 340.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 334.60 | 338.55 | 339.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 334.60 | 338.55 | 339.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 333.15 | 337.47 | 338.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 339.10 | 337.08 | 338.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 339.10 | 337.08 | 338.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 339.10 | 337.08 | 338.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 337.65 | 337.08 | 338.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 339.85 | 337.63 | 338.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:15:00 | 339.85 | 337.63 | 338.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 339.80 | 338.20 | 338.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 13:00:00 | 339.80 | 338.20 | 338.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 339.40 | 338.44 | 338.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:15:00 | 340.20 | 338.44 | 338.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 340.85 | 338.92 | 338.74 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 335.45 | 338.46 | 338.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 11:15:00 | 333.85 | 336.83 | 337.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 334.40 | 332.08 | 333.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 334.40 | 332.08 | 333.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 334.40 | 332.08 | 333.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:00:00 | 334.40 | 332.08 | 333.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 333.70 | 332.40 | 333.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:00:00 | 332.30 | 332.38 | 333.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 12:15:00 | 334.45 | 333.54 | 333.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 334.45 | 333.54 | 333.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 337.00 | 334.75 | 334.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 12:15:00 | 334.95 | 335.21 | 334.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 13:00:00 | 334.95 | 335.21 | 334.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 334.25 | 335.02 | 334.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:45:00 | 334.20 | 335.02 | 334.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 334.65 | 334.95 | 334.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 335.75 | 334.77 | 334.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 332.85 | 334.30 | 334.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 332.85 | 334.30 | 334.31 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 336.20 | 334.62 | 334.45 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 332.45 | 334.14 | 334.31 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 14:15:00 | 335.55 | 334.45 | 334.39 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 332.40 | 334.00 | 334.21 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 336.85 | 334.60 | 334.41 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 10:15:00 | 331.90 | 334.89 | 335.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 13:15:00 | 330.70 | 333.05 | 334.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 12:15:00 | 331.10 | 330.80 | 332.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-06 13:00:00 | 331.10 | 330.80 | 332.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 332.70 | 330.92 | 331.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:15:00 | 334.10 | 330.92 | 331.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 333.80 | 331.50 | 332.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:30:00 | 333.80 | 331.50 | 332.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 12:15:00 | 338.65 | 333.30 | 332.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 346.80 | 337.90 | 335.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 341.00 | 341.04 | 338.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 341.00 | 341.04 | 338.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 341.00 | 341.04 | 338.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 10:45:00 | 344.15 | 342.02 | 339.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 347.70 | 353.05 | 353.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 347.70 | 353.05 | 353.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 345.00 | 351.44 | 352.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 346.25 | 346.13 | 349.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-19 09:45:00 | 346.40 | 346.13 | 349.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 349.80 | 346.86 | 349.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:15:00 | 350.80 | 346.86 | 349.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 350.15 | 347.52 | 349.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:45:00 | 350.80 | 347.52 | 349.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 348.75 | 347.87 | 349.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 14:45:00 | 348.85 | 347.87 | 349.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 347.00 | 347.69 | 348.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 346.30 | 347.69 | 348.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 348.55 | 347.87 | 348.85 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 352.25 | 349.52 | 349.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 353.10 | 351.11 | 350.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 351.95 | 352.23 | 351.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 15:00:00 | 351.95 | 352.23 | 351.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 351.65 | 352.11 | 351.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 346.15 | 352.11 | 351.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 350.25 | 351.74 | 351.05 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 349.50 | 350.53 | 350.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 11:15:00 | 348.35 | 349.74 | 350.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 13:15:00 | 351.55 | 350.06 | 350.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 13:15:00 | 351.55 | 350.06 | 350.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 351.55 | 350.06 | 350.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:00:00 | 351.55 | 350.06 | 350.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 351.75 | 350.40 | 350.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 353.55 | 351.19 | 350.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 11:15:00 | 351.50 | 351.64 | 351.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 11:15:00 | 351.50 | 351.64 | 351.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 351.50 | 351.64 | 351.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 351.50 | 351.64 | 351.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 349.90 | 351.29 | 350.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:00:00 | 349.90 | 351.29 | 350.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 350.25 | 351.08 | 350.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:45:00 | 349.65 | 351.08 | 350.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 350.60 | 350.99 | 350.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 15:00:00 | 350.60 | 350.99 | 350.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 350.00 | 350.79 | 350.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 352.85 | 350.79 | 350.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 11:00:00 | 351.00 | 352.21 | 351.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:30:00 | 352.15 | 351.74 | 351.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 368.00 | 372.33 | 372.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 368.00 | 372.33 | 372.54 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 376.40 | 372.56 | 372.16 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 370.40 | 372.61 | 372.64 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 376.80 | 373.21 | 372.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 379.65 | 374.49 | 373.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 393.45 | 394.03 | 389.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 393.45 | 394.03 | 389.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 397.65 | 398.70 | 396.08 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 390.10 | 395.12 | 395.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 387.25 | 391.39 | 393.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 392.20 | 390.86 | 392.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 10:00:00 | 392.20 | 390.86 | 392.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 393.80 | 391.45 | 392.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 394.10 | 391.45 | 392.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 395.00 | 392.16 | 393.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:45:00 | 395.80 | 392.16 | 393.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 395.85 | 393.63 | 393.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 401.35 | 396.10 | 394.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 396.05 | 397.31 | 396.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 396.05 | 397.31 | 396.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 396.05 | 397.31 | 396.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:15:00 | 394.85 | 397.31 | 396.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 394.95 | 396.84 | 396.16 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 393.00 | 395.67 | 395.72 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 399.45 | 396.03 | 395.83 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 11:15:00 | 389.95 | 394.57 | 395.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 388.25 | 392.62 | 394.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 391.90 | 391.14 | 392.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 391.90 | 391.14 | 392.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 391.90 | 391.14 | 392.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 391.90 | 391.14 | 392.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 393.85 | 391.68 | 393.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:30:00 | 394.50 | 391.68 | 393.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 394.35 | 392.21 | 393.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:15:00 | 395.50 | 392.21 | 393.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 397.00 | 393.17 | 393.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 397.00 | 393.17 | 393.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 397.20 | 393.98 | 393.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 398.35 | 394.85 | 394.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 12:15:00 | 396.40 | 396.81 | 395.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 12:15:00 | 396.40 | 396.81 | 395.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 396.40 | 396.81 | 395.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:30:00 | 395.30 | 396.81 | 395.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 397.60 | 396.97 | 395.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:30:00 | 396.85 | 396.97 | 395.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 395.50 | 396.73 | 395.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 393.65 | 396.73 | 395.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 390.30 | 395.45 | 395.40 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 390.00 | 394.36 | 394.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 387.65 | 391.99 | 393.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 383.55 | 383.23 | 386.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:45:00 | 383.75 | 383.23 | 386.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 374.70 | 381.22 | 384.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 374.15 | 381.22 | 384.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 12:30:00 | 371.90 | 377.99 | 382.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 10:15:00 | 355.44 | 359.97 | 364.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 10:15:00 | 353.30 | 359.97 | 364.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-11 09:15:00 | 336.74 | 348.54 | 356.33 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 100 — BUY (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 15:15:00 | 354.35 | 351.82 | 351.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 355.85 | 352.62 | 351.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 11:15:00 | 358.75 | 358.84 | 356.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 12:15:00 | 358.90 | 358.84 | 356.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 359.45 | 362.78 | 361.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 359.45 | 362.78 | 361.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 361.25 | 362.48 | 361.91 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 359.85 | 361.31 | 361.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 359.65 | 360.98 | 361.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 360.50 | 360.13 | 360.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 360.50 | 360.13 | 360.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 360.50 | 360.13 | 360.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 359.15 | 360.13 | 360.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 359.35 | 359.98 | 360.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 359.90 | 359.98 | 360.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 358.65 | 358.94 | 359.82 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 362.65 | 360.12 | 359.92 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 357.05 | 359.67 | 359.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 356.70 | 359.20 | 359.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 353.10 | 350.49 | 352.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 13:15:00 | 353.10 | 350.49 | 352.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 353.10 | 350.49 | 352.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 353.10 | 350.49 | 352.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 352.80 | 350.95 | 352.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 352.90 | 350.95 | 352.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 352.00 | 351.16 | 352.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 350.15 | 351.16 | 352.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 351.85 | 351.30 | 352.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 349.55 | 351.26 | 352.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 14:15:00 | 356.15 | 352.55 | 352.64 | SL hit (close>static) qty=1.00 sl=354.10 alert=retest2 |

### Cycle 104 — BUY (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 15:15:00 | 354.30 | 352.90 | 352.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 357.80 | 353.88 | 353.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 359.75 | 359.77 | 357.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:00:00 | 359.75 | 359.77 | 357.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 358.20 | 359.12 | 357.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 362.15 | 358.09 | 357.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 363.20 | 364.46 | 364.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 363.20 | 364.46 | 364.51 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 366.30 | 364.58 | 364.53 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 361.90 | 364.43 | 364.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 10:15:00 | 360.40 | 363.62 | 364.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 10:15:00 | 360.80 | 360.38 | 361.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 11:00:00 | 360.80 | 360.38 | 361.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 361.00 | 360.64 | 361.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:30:00 | 361.50 | 360.64 | 361.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 361.00 | 360.40 | 361.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:15:00 | 361.50 | 360.40 | 361.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 359.90 | 360.30 | 361.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:45:00 | 359.15 | 360.01 | 360.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 362.35 | 358.06 | 359.34 | SL hit (close>static) qty=1.00 sl=362.30 alert=retest2 |

### Cycle 108 — BUY (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 13:15:00 | 366.50 | 361.12 | 360.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 14:15:00 | 368.35 | 362.56 | 361.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 364.75 | 367.90 | 365.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 364.75 | 367.90 | 365.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 364.75 | 367.90 | 365.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:45:00 | 364.70 | 367.90 | 365.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 364.25 | 367.17 | 365.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 364.25 | 367.17 | 365.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 364.00 | 366.54 | 365.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:30:00 | 364.25 | 366.54 | 365.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 361.30 | 364.90 | 364.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 358.60 | 362.35 | 363.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 360.10 | 359.53 | 360.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 360.10 | 359.53 | 360.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 360.10 | 359.53 | 360.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:00:00 | 356.45 | 358.89 | 360.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:00:00 | 357.05 | 358.19 | 359.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:00:00 | 355.10 | 357.57 | 359.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 338.63 | 345.11 | 351.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 339.20 | 345.11 | 351.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 13:15:00 | 343.15 | 342.34 | 346.79 | SL hit (close>ema200) qty=0.50 sl=342.34 alert=retest2 |

### Cycle 110 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 346.65 | 344.20 | 344.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 346.80 | 344.99 | 344.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 347.55 | 349.02 | 347.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 347.55 | 349.02 | 347.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 347.55 | 349.02 | 347.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:00:00 | 347.55 | 349.02 | 347.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 345.85 | 348.39 | 347.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 345.85 | 348.39 | 347.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 347.30 | 348.17 | 347.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 346.60 | 348.17 | 347.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 344.90 | 347.51 | 347.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 344.90 | 347.51 | 347.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 346.80 | 347.37 | 346.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:00:00 | 347.50 | 347.40 | 347.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:45:00 | 347.30 | 348.17 | 347.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 348.30 | 350.64 | 350.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 348.30 | 350.64 | 350.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 347.25 | 349.24 | 350.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 13:15:00 | 352.40 | 348.75 | 349.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 13:15:00 | 352.40 | 348.75 | 349.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 352.40 | 348.75 | 349.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:00:00 | 352.40 | 348.75 | 349.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 351.90 | 349.38 | 349.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 353.30 | 349.38 | 349.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 354.65 | 350.66 | 350.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 355.10 | 352.55 | 351.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 356.65 | 357.47 | 355.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 10:15:00 | 356.65 | 357.47 | 355.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 356.65 | 357.47 | 355.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 356.65 | 357.47 | 355.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 358.40 | 357.65 | 355.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:30:00 | 359.60 | 357.62 | 356.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 13:15:00 | 359.15 | 361.53 | 361.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 359.15 | 361.53 | 361.55 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 362.85 | 360.95 | 360.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 12:15:00 | 364.00 | 361.56 | 361.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 11:15:00 | 374.75 | 374.95 | 372.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 12:00:00 | 374.75 | 374.95 | 372.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 371.80 | 374.54 | 372.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 371.80 | 374.54 | 372.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 372.50 | 374.13 | 372.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 374.60 | 374.13 | 372.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-13 09:15:00 | 412.06 | 399.92 | 392.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 13:15:00 | 409.80 | 415.12 | 415.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 09:15:00 | 394.35 | 409.47 | 412.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 10:15:00 | 399.85 | 399.12 | 404.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 11:00:00 | 399.85 | 399.12 | 404.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 396.30 | 394.48 | 396.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 397.70 | 394.48 | 396.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 397.10 | 395.00 | 396.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 397.20 | 395.00 | 396.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 397.75 | 395.55 | 396.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 397.75 | 395.55 | 396.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 398.70 | 396.74 | 396.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 398.70 | 396.74 | 396.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 398.50 | 397.09 | 397.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 402.80 | 397.09 | 397.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 399.45 | 397.56 | 397.34 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 396.60 | 398.02 | 398.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 13:15:00 | 394.40 | 396.99 | 397.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 11:15:00 | 395.95 | 395.53 | 396.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 11:15:00 | 395.95 | 395.53 | 396.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 395.95 | 395.53 | 396.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:15:00 | 399.00 | 395.53 | 396.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 395.20 | 395.46 | 396.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:15:00 | 394.80 | 395.46 | 396.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 400.40 | 397.21 | 397.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 400.40 | 397.21 | 397.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 09:15:00 | 403.75 | 398.52 | 397.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 09:15:00 | 394.85 | 404.23 | 401.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 394.85 | 404.23 | 401.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 394.85 | 404.23 | 401.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 394.85 | 404.23 | 401.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 394.00 | 402.19 | 401.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 394.00 | 402.19 | 401.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 388.75 | 399.50 | 400.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 386.65 | 389.48 | 391.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 385.15 | 384.54 | 387.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 385.15 | 384.54 | 387.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 382.45 | 381.06 | 382.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 383.65 | 381.06 | 382.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 385.70 | 381.99 | 382.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 387.60 | 381.99 | 382.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 385.20 | 382.63 | 382.78 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 386.00 | 383.30 | 383.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 387.75 | 384.80 | 383.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 385.30 | 385.39 | 384.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 385.30 | 385.37 | 384.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 385.30 | 385.37 | 384.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 384.60 | 385.37 | 384.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 384.55 | 385.26 | 384.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:00:00 | 384.55 | 385.26 | 384.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 386.85 | 385.58 | 384.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:30:00 | 385.00 | 385.58 | 384.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 386.15 | 386.89 | 385.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 385.85 | 386.89 | 385.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 385.75 | 386.62 | 385.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 385.75 | 386.62 | 385.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 385.30 | 386.36 | 385.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 384.80 | 386.36 | 385.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 384.75 | 386.04 | 385.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 386.70 | 386.04 | 385.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:00:00 | 386.60 | 386.12 | 385.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 11:15:00 | 397.85 | 399.25 | 399.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 397.85 | 399.25 | 399.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 12:15:00 | 396.50 | 398.70 | 399.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 392.00 | 391.62 | 393.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 13:30:00 | 391.70 | 391.62 | 393.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 388.60 | 390.81 | 392.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:30:00 | 386.25 | 391.00 | 391.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 15:00:00 | 388.00 | 389.38 | 390.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 387.55 | 389.17 | 390.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 12:15:00 | 393.25 | 390.74 | 390.87 | SL hit (close>static) qty=1.00 sl=393.10 alert=retest2 |

### Cycle 122 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 394.15 | 391.42 | 391.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 396.55 | 392.96 | 391.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 389.55 | 393.56 | 392.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 389.55 | 393.56 | 392.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 389.55 | 393.56 | 392.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 389.55 | 393.56 | 392.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 391.55 | 393.16 | 392.62 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 391.00 | 392.27 | 392.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 389.85 | 391.78 | 392.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 391.90 | 389.43 | 390.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 391.90 | 389.43 | 390.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 391.90 | 389.43 | 390.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 391.90 | 389.43 | 390.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 391.55 | 389.85 | 390.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:15:00 | 387.45 | 389.92 | 390.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 368.08 | 376.34 | 380.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 12:15:00 | 374.45 | 372.62 | 375.69 | SL hit (close>ema200) qty=0.50 sl=372.62 alert=retest2 |

### Cycle 124 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 382.00 | 377.02 | 376.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 11:15:00 | 383.60 | 378.34 | 377.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 14:15:00 | 379.25 | 379.35 | 378.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 14:30:00 | 379.70 | 379.35 | 378.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 378.90 | 379.26 | 378.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 381.20 | 379.26 | 378.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 15:15:00 | 376.50 | 380.05 | 379.59 | SL hit (close<static) qty=1.00 sl=378.00 alert=retest2 |

### Cycle 125 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 376.15 | 378.94 | 379.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 374.00 | 377.96 | 378.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 372.85 | 366.78 | 369.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 372.85 | 366.78 | 369.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 372.85 | 366.78 | 369.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 373.75 | 366.78 | 369.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 371.00 | 367.62 | 369.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 370.00 | 368.09 | 369.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 372.80 | 370.16 | 370.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 372.80 | 370.16 | 370.05 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 367.95 | 370.07 | 370.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 10:15:00 | 363.30 | 367.49 | 368.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 365.80 | 364.78 | 366.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 365.80 | 364.78 | 366.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 364.45 | 364.46 | 366.38 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 371.15 | 367.52 | 367.31 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 361.50 | 366.72 | 367.01 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 14:15:00 | 367.50 | 365.81 | 365.77 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 363.25 | 365.49 | 365.64 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 375.70 | 367.53 | 366.56 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 361.90 | 366.71 | 366.97 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 371.00 | 367.20 | 366.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 374.00 | 368.56 | 367.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 371.60 | 372.40 | 370.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 371.60 | 372.40 | 370.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 373.10 | 372.54 | 370.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 12:15:00 | 373.80 | 372.54 | 370.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 13:15:00 | 369.15 | 371.96 | 370.92 | SL hit (close<static) qty=1.00 sl=370.50 alert=retest2 |

### Cycle 135 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 366.15 | 369.72 | 370.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 362.50 | 366.54 | 368.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 365.60 | 364.55 | 366.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 365.60 | 364.55 | 366.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 371.70 | 365.96 | 366.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 371.70 | 365.96 | 366.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 373.70 | 367.51 | 367.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 15:15:00 | 375.35 | 372.91 | 371.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 11:15:00 | 374.95 | 376.33 | 373.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 11:15:00 | 374.95 | 376.33 | 373.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 374.95 | 376.33 | 373.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 374.95 | 376.33 | 373.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 375.65 | 376.19 | 373.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 380.50 | 375.72 | 374.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 13:00:00 | 381.20 | 376.70 | 375.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:30:00 | 382.85 | 378.59 | 376.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 09:30:00 | 380.00 | 380.20 | 378.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 378.85 | 379.93 | 378.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:45:00 | 378.55 | 379.93 | 378.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 379.45 | 379.67 | 378.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 377.80 | 379.67 | 378.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 379.00 | 379.53 | 378.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 381.95 | 379.53 | 378.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 379.10 | 379.45 | 378.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:15:00 | 377.90 | 379.45 | 378.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 377.25 | 379.01 | 378.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 377.25 | 379.01 | 378.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 378.35 | 378.88 | 378.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:30:00 | 377.05 | 378.88 | 378.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 377.60 | 378.62 | 378.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 12:15:00 | 377.60 | 378.62 | 378.64 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 378.90 | 378.69 | 378.67 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 378.10 | 378.57 | 378.61 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 10:15:00 | 383.65 | 379.58 | 379.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 12:15:00 | 386.65 | 381.42 | 380.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 380.85 | 381.95 | 380.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 15:15:00 | 380.85 | 381.95 | 380.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 380.85 | 381.95 | 380.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 388.10 | 382.53 | 381.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 15:15:00 | 389.75 | 391.12 | 391.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 389.75 | 391.12 | 391.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 387.80 | 390.46 | 390.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 389.70 | 388.48 | 389.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 15:15:00 | 389.70 | 388.48 | 389.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 389.70 | 388.48 | 389.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 381.90 | 388.48 | 389.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 389.45 | 385.05 | 384.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 389.45 | 385.05 | 384.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 392.30 | 387.22 | 385.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 382.80 | 388.51 | 387.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 382.80 | 388.51 | 387.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 382.80 | 388.51 | 387.31 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 384.30 | 386.47 | 386.51 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 14:15:00 | 389.75 | 387.08 | 386.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 397.00 | 389.55 | 387.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 394.65 | 397.62 | 394.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 394.65 | 397.62 | 394.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 394.65 | 397.62 | 394.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 394.65 | 397.62 | 394.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 394.55 | 397.01 | 394.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 391.15 | 397.01 | 394.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 390.35 | 395.68 | 394.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 391.45 | 395.68 | 394.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 395.55 | 395.37 | 394.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:15:00 | 393.80 | 395.37 | 394.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 394.60 | 395.21 | 394.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:15:00 | 396.15 | 395.21 | 394.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 15:15:00 | 391.40 | 394.22 | 394.14 | SL hit (close<static) qty=1.00 sl=392.60 alert=retest2 |

### Cycle 145 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 388.10 | 393.00 | 393.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 384.05 | 391.21 | 392.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 377.75 | 377.39 | 383.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 15:00:00 | 377.75 | 377.39 | 383.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 379.90 | 378.38 | 382.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 375.70 | 378.38 | 382.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 378.85 | 378.65 | 381.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:00:00 | 378.30 | 378.58 | 381.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 383.10 | 379.17 | 380.86 | SL hit (close>static) qty=1.00 sl=383.05 alert=retest2 |

### Cycle 146 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 383.90 | 382.15 | 381.93 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 374.50 | 380.73 | 381.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 372.35 | 379.05 | 380.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 375.60 | 373.02 | 376.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 375.60 | 373.02 | 376.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 375.60 | 373.02 | 376.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 377.15 | 373.02 | 376.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 378.45 | 374.11 | 376.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 378.45 | 374.11 | 376.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 377.05 | 374.70 | 376.48 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 380.00 | 377.85 | 377.57 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 371.25 | 376.53 | 377.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 369.60 | 374.18 | 375.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 379.55 | 371.41 | 372.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 379.55 | 371.41 | 372.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 379.55 | 371.41 | 372.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 379.55 | 371.41 | 372.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 381.05 | 373.34 | 372.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 381.80 | 375.03 | 373.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 374.35 | 377.15 | 375.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 374.35 | 377.15 | 375.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 374.35 | 377.15 | 375.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:15:00 | 374.35 | 377.15 | 375.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 371.45 | 376.01 | 375.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 371.45 | 376.01 | 375.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 371.80 | 375.17 | 374.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:45:00 | 371.75 | 375.17 | 374.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 371.55 | 374.45 | 374.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 362.85 | 370.93 | 372.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 367.45 | 364.28 | 367.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 367.45 | 364.28 | 367.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 367.45 | 364.28 | 367.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:30:00 | 364.75 | 364.00 | 367.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 363.80 | 365.78 | 367.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 346.51 | 361.17 | 364.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 345.61 | 361.17 | 364.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 353.90 | 352.57 | 356.56 | SL hit (close>ema200) qty=0.50 sl=352.57 alert=retest2 |

### Cycle 152 — BUY (started 2026-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 14:15:00 | 348.70 | 348.41 | 348.40 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 15:15:00 | 348.00 | 348.33 | 348.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 340.70 | 346.81 | 347.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 14:15:00 | 344.55 | 344.45 | 345.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-13 14:45:00 | 345.05 | 344.45 | 345.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 346.25 | 344.81 | 345.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:15:00 | 349.05 | 344.81 | 345.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 349.80 | 345.81 | 346.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:45:00 | 350.85 | 345.81 | 346.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 349.85 | 347.30 | 346.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 351.70 | 349.46 | 348.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 12:15:00 | 347.80 | 349.40 | 348.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 12:15:00 | 347.80 | 349.40 | 348.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 347.80 | 349.40 | 348.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:00:00 | 347.80 | 349.40 | 348.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 349.70 | 349.46 | 348.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 350.75 | 349.77 | 348.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:45:00 | 351.05 | 350.46 | 349.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 15:15:00 | 356.60 | 358.78 | 358.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 356.60 | 358.78 | 358.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 355.60 | 358.14 | 358.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 352.85 | 352.81 | 355.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 352.85 | 352.81 | 355.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 352.85 | 352.81 | 355.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 355.10 | 352.81 | 355.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 355.20 | 353.39 | 354.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 355.20 | 353.39 | 354.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 360.00 | 354.71 | 355.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:30:00 | 358.90 | 354.71 | 355.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 361.90 | 356.15 | 355.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 11:15:00 | 363.40 | 359.79 | 357.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 362.55 | 363.31 | 361.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:00:00 | 362.55 | 363.31 | 361.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 363.15 | 363.15 | 361.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 358.55 | 363.15 | 361.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 358.05 | 362.13 | 361.21 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 357.60 | 360.48 | 360.58 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 364.20 | 360.69 | 360.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 365.40 | 361.63 | 361.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 14:15:00 | 360.00 | 362.36 | 361.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 14:15:00 | 360.00 | 362.36 | 361.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 360.00 | 362.36 | 361.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:45:00 | 360.20 | 362.36 | 361.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 363.00 | 362.49 | 361.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:15:00 | 360.50 | 362.49 | 361.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 362.80 | 362.55 | 361.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 363.55 | 362.55 | 361.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 365.25 | 363.09 | 362.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:30:00 | 366.20 | 364.23 | 362.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-29 13:45:00 | 320.10 | 2024-05-30 11:15:00 | 314.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-06-03 10:15:00 | 310.00 | 2024-06-04 09:15:00 | 294.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 13:30:00 | 310.40 | 2024-06-04 09:15:00 | 294.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 14:15:00 | 310.25 | 2024-06-04 09:15:00 | 294.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 10:15:00 | 310.00 | 2024-06-04 12:15:00 | 279.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 13:30:00 | 310.40 | 2024-06-04 12:15:00 | 279.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 14:15:00 | 310.25 | 2024-06-04 12:15:00 | 279.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 294.45 | 2024-06-04 12:15:00 | 279.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 294.45 | 2024-06-05 10:15:00 | 299.30 | STOP_HIT | 0.50 | -1.65% |
| BUY | retest2 | 2024-06-25 09:15:00 | 343.10 | 2024-06-25 10:15:00 | 340.35 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-06-28 09:15:00 | 349.35 | 2024-07-09 12:15:00 | 363.20 | STOP_HIT | 1.00 | 3.96% |
| BUY | retest2 | 2024-07-31 09:30:00 | 364.80 | 2024-08-01 11:15:00 | 359.40 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-07-31 12:00:00 | 364.00 | 2024-08-01 11:15:00 | 359.40 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-08-01 09:30:00 | 366.55 | 2024-08-01 11:15:00 | 359.40 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-08-01 11:00:00 | 364.00 | 2024-08-01 11:15:00 | 359.40 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-08-05 09:15:00 | 343.00 | 2024-08-08 11:15:00 | 346.30 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-08-13 09:30:00 | 335.20 | 2024-08-13 10:15:00 | 341.60 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-08-13 15:15:00 | 335.60 | 2024-08-16 14:15:00 | 341.40 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-08-26 09:15:00 | 353.55 | 2024-08-29 13:15:00 | 352.10 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-08-27 15:00:00 | 352.85 | 2024-08-29 13:15:00 | 352.10 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2024-09-25 15:00:00 | 371.50 | 2024-09-27 14:15:00 | 369.15 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-09-26 14:45:00 | 372.95 | 2024-09-27 14:15:00 | 369.15 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-09-27 13:45:00 | 371.15 | 2024-09-27 14:15:00 | 369.15 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-10-21 13:15:00 | 336.60 | 2024-10-25 09:15:00 | 319.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 336.90 | 2024-10-25 09:15:00 | 320.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:45:00 | 337.00 | 2024-10-25 09:15:00 | 320.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 15:15:00 | 336.05 | 2024-10-25 09:15:00 | 319.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 13:15:00 | 336.60 | 2024-10-28 09:15:00 | 316.80 | STOP_HIT | 0.50 | 5.88% |
| SELL | retest2 | 2024-10-21 14:00:00 | 336.90 | 2024-10-28 09:15:00 | 316.80 | STOP_HIT | 0.50 | 5.97% |
| SELL | retest2 | 2024-10-21 14:45:00 | 337.00 | 2024-10-28 09:15:00 | 316.80 | STOP_HIT | 0.50 | 5.99% |
| SELL | retest2 | 2024-10-21 15:15:00 | 336.05 | 2024-10-28 09:15:00 | 316.80 | STOP_HIT | 0.50 | 5.73% |
| BUY | retest2 | 2024-11-11 09:15:00 | 335.85 | 2024-11-13 13:15:00 | 331.30 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-12-04 14:15:00 | 375.00 | 2024-12-09 09:15:00 | 364.30 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-12-05 13:30:00 | 375.00 | 2024-12-09 09:15:00 | 364.30 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-12-05 15:00:00 | 376.30 | 2024-12-09 09:15:00 | 364.30 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-12-06 10:15:00 | 376.55 | 2024-12-09 09:15:00 | 364.30 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2024-12-06 13:45:00 | 378.80 | 2024-12-09 09:15:00 | 364.30 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2024-12-18 14:45:00 | 346.60 | 2024-12-23 14:15:00 | 329.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 14:45:00 | 346.60 | 2024-12-24 09:15:00 | 335.35 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2024-12-19 09:15:00 | 345.40 | 2024-12-24 15:15:00 | 343.15 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2024-12-19 10:00:00 | 345.55 | 2024-12-24 15:15:00 | 343.15 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2024-12-20 09:45:00 | 345.95 | 2024-12-24 15:15:00 | 343.15 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2025-01-13 11:15:00 | 369.85 | 2025-01-14 10:15:00 | 380.20 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-01-24 11:15:00 | 393.35 | 2025-01-24 14:15:00 | 388.05 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-01-24 12:00:00 | 393.70 | 2025-01-24 14:15:00 | 388.05 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-01-29 12:15:00 | 365.00 | 2025-02-01 10:15:00 | 365.20 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-02-04 09:15:00 | 370.50 | 2025-02-10 10:15:00 | 377.75 | STOP_HIT | 1.00 | 1.96% |
| SELL | retest2 | 2025-02-13 11:30:00 | 362.50 | 2025-02-14 13:15:00 | 344.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 12:00:00 | 362.50 | 2025-02-14 13:15:00 | 344.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 12:30:00 | 362.65 | 2025-02-14 13:15:00 | 344.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 11:30:00 | 362.50 | 2025-02-17 15:15:00 | 347.70 | STOP_HIT | 0.50 | 4.08% |
| SELL | retest2 | 2025-02-13 12:00:00 | 362.50 | 2025-02-17 15:15:00 | 347.70 | STOP_HIT | 0.50 | 4.08% |
| SELL | retest2 | 2025-02-13 12:30:00 | 362.65 | 2025-02-17 15:15:00 | 347.70 | STOP_HIT | 0.50 | 4.12% |
| SELL | retest2 | 2025-04-01 11:30:00 | 340.85 | 2025-04-03 09:15:00 | 348.00 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-04-02 12:30:00 | 341.75 | 2025-04-03 09:15:00 | 348.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-04-02 15:00:00 | 341.15 | 2025-04-03 09:15:00 | 348.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-04-09 09:15:00 | 311.15 | 2025-04-15 09:15:00 | 326.80 | STOP_HIT | 1.00 | -5.03% |
| BUY | retest2 | 2025-04-22 12:15:00 | 336.40 | 2025-04-24 11:15:00 | 327.60 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-05-02 09:45:00 | 322.80 | 2025-05-02 12:15:00 | 319.20 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-05-13 12:30:00 | 330.05 | 2025-05-14 10:15:00 | 337.85 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-05-13 14:15:00 | 329.60 | 2025-05-14 10:15:00 | 337.85 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-05-15 14:15:00 | 337.95 | 2025-05-20 13:15:00 | 334.60 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-05-16 09:15:00 | 341.50 | 2025-05-20 13:15:00 | 334.60 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-05-20 09:30:00 | 338.75 | 2025-05-20 13:15:00 | 334.60 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-05-26 12:00:00 | 332.30 | 2025-05-27 12:15:00 | 334.45 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-05-29 09:15:00 | 335.75 | 2025-05-29 10:15:00 | 332.85 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-06-11 10:45:00 | 344.15 | 2025-06-18 10:15:00 | 347.70 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2025-06-27 09:15:00 | 352.85 | 2025-07-08 10:15:00 | 368.00 | STOP_HIT | 1.00 | 4.29% |
| BUY | retest2 | 2025-06-30 11:00:00 | 351.00 | 2025-07-08 10:15:00 | 368.00 | STOP_HIT | 1.00 | 4.84% |
| BUY | retest2 | 2025-06-30 12:30:00 | 352.15 | 2025-07-08 10:15:00 | 368.00 | STOP_HIT | 1.00 | 4.50% |
| SELL | retest2 | 2025-08-05 10:15:00 | 374.15 | 2025-08-08 10:15:00 | 355.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 12:30:00 | 371.90 | 2025-08-08 10:15:00 | 353.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 10:15:00 | 374.15 | 2025-08-11 09:15:00 | 336.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-05 12:30:00 | 371.90 | 2025-08-11 09:15:00 | 334.71 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-02 13:15:00 | 349.55 | 2025-09-02 14:15:00 | 356.15 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-09-05 09:15:00 | 362.15 | 2025-09-11 13:15:00 | 363.20 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-09-17 11:45:00 | 359.15 | 2025-09-18 09:15:00 | 362.35 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-09-25 12:00:00 | 356.45 | 2025-09-26 14:15:00 | 338.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 14:00:00 | 357.05 | 2025-09-26 14:15:00 | 339.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 12:00:00 | 356.45 | 2025-09-29 13:15:00 | 343.15 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-09-25 14:00:00 | 357.05 | 2025-09-29 13:15:00 | 343.15 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2025-09-25 15:00:00 | 355.10 | 2025-09-30 13:15:00 | 337.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 15:00:00 | 355.10 | 2025-09-30 14:15:00 | 341.00 | STOP_HIT | 0.50 | 3.97% |
| BUY | retest2 | 2025-10-06 15:00:00 | 347.50 | 2025-10-13 11:15:00 | 348.30 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-10-07 13:45:00 | 347.30 | 2025-10-13 11:15:00 | 348.30 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-10-20 09:30:00 | 359.60 | 2025-10-24 13:15:00 | 359.15 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-11-03 09:15:00 | 374.60 | 2025-11-13 09:15:00 | 412.06 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-02 13:15:00 | 394.80 | 2025-12-02 15:15:00 | 400.40 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-12-17 09:15:00 | 386.70 | 2025-12-26 11:15:00 | 397.85 | STOP_HIT | 1.00 | 2.88% |
| BUY | retest2 | 2025-12-17 11:00:00 | 386.60 | 2025-12-26 11:15:00 | 397.85 | STOP_HIT | 1.00 | 2.91% |
| SELL | retest2 | 2026-01-01 10:30:00 | 386.25 | 2026-01-02 12:15:00 | 393.25 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-01-01 15:00:00 | 388.00 | 2026-01-02 12:15:00 | 393.25 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-01-02 09:15:00 | 387.55 | 2026-01-02 12:15:00 | 393.25 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-01-07 12:15:00 | 387.45 | 2026-01-12 11:15:00 | 368.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 12:15:00 | 387.45 | 2026-01-13 12:15:00 | 374.45 | STOP_HIT | 0.50 | 3.36% |
| BUY | retest2 | 2026-01-16 09:15:00 | 381.20 | 2026-01-16 15:15:00 | 376.50 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-01-22 11:30:00 | 370.00 | 2026-01-22 14:15:00 | 372.80 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-02-04 12:15:00 | 373.80 | 2026-02-04 13:15:00 | 369.15 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-02-13 09:15:00 | 380.50 | 2026-02-18 12:15:00 | 377.60 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-02-13 13:00:00 | 381.20 | 2026-02-18 12:15:00 | 377.60 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-02-16 09:30:00 | 382.85 | 2026-02-18 12:15:00 | 377.60 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-17 09:30:00 | 380.00 | 2026-02-18 12:15:00 | 377.60 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-02-20 09:30:00 | 388.10 | 2026-02-27 15:15:00 | 389.75 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2026-03-04 09:15:00 | 381.90 | 2026-03-06 09:15:00 | 389.45 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-03-12 13:15:00 | 396.15 | 2026-03-12 15:15:00 | 391.40 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-03-17 11:15:00 | 375.70 | 2026-03-18 09:15:00 | 383.10 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-03-17 12:15:00 | 378.85 | 2026-03-18 09:15:00 | 383.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-03-17 13:00:00 | 378.30 | 2026-03-18 09:15:00 | 383.10 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-04-01 10:30:00 | 364.75 | 2026-04-02 09:15:00 | 346.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 14:30:00 | 363.80 | 2026-04-02 09:15:00 | 345.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:30:00 | 364.75 | 2026-04-06 12:15:00 | 353.90 | STOP_HIT | 0.50 | 2.97% |
| SELL | retest2 | 2026-04-01 14:30:00 | 363.80 | 2026-04-06 12:15:00 | 353.90 | STOP_HIT | 0.50 | 2.72% |
| BUY | retest2 | 2026-04-16 14:30:00 | 350.75 | 2026-04-23 15:15:00 | 356.60 | STOP_HIT | 1.00 | 1.67% |
| BUY | retest2 | 2026-04-17 09:45:00 | 351.05 | 2026-04-23 15:15:00 | 356.60 | STOP_HIT | 1.00 | 1.58% |
