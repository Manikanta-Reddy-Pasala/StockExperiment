# Bank of India (BANKINDIA)

## Backtest Summary

- **Window:** 2025-12-22 09:15:00 → 2026-05-08 15:15:00 (644 bars)
- **Last close:** 139.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 30 |
| ALERT1 | 19 |
| ALERT2 | 19 |
| ALERT2_SKIP | 8 |
| ALERT3 | 40 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 28 |
| PARTIAL | 7 |
| TARGET_HIT | 1 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 19
- **Target hits / Stop hits / Partials:** 1 / 29 / 7
- **Avg / median % per leg:** 0.86% / -0.08%
- **Sum % (uncompounded):** 31.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 3 | 17.6% | 0 | 17 | 0 | -1.30% | -22.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 3 | 17.6% | 0 | 17 | 0 | -1.30% | -22.0% |
| SELL (all) | 20 | 15 | 75.0% | 1 | 12 | 7 | 2.69% | 53.7% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 1 | 1 | 2 | 6.49% | 25.9% |
| SELL @ 3rd Alert (retest2) | 16 | 11 | 68.8% | 0 | 11 | 5 | 1.74% | 27.8% |
| retest1 (combined) | 4 | 4 | 100.0% | 1 | 1 | 2 | 6.49% | 25.9% |
| retest2 (combined) | 33 | 14 | 42.4% | 0 | 28 | 5 | 0.17% | 5.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 141.97 | 140.44 | 140.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 142.15 | 140.78 | 140.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 148.54 | 149.22 | 147.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 148.54 | 149.22 | 147.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 150.80 | 150.96 | 149.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 150.43 | 150.96 | 149.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 150.32 | 150.81 | 149.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 151.21 | 150.79 | 149.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:00:00 | 151.05 | 150.96 | 150.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 147.33 | 150.18 | 150.10 | SL hit (close<static) qty=1.00 sl=149.71 alert=retest2 |

### Cycle 2 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 148.12 | 149.77 | 149.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 145.75 | 147.55 | 148.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 145.59 | 145.42 | 146.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 145.91 | 145.42 | 146.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 146.49 | 145.64 | 146.59 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 148.72 | 147.11 | 146.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 151.70 | 148.08 | 147.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 10:15:00 | 160.85 | 161.17 | 158.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 11:00:00 | 160.85 | 161.17 | 158.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 158.24 | 159.94 | 158.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 158.13 | 159.94 | 158.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 158.49 | 159.65 | 158.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 11:45:00 | 157.25 | 159.65 | 158.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 156.65 | 159.03 | 158.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:00:00 | 156.65 | 159.03 | 158.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 157.37 | 158.70 | 158.59 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 15:15:00 | 157.45 | 158.45 | 158.49 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 165.10 | 159.78 | 159.09 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 159.64 | 161.62 | 161.63 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 161.90 | 161.68 | 161.65 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 161.09 | 161.56 | 161.60 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 162.18 | 161.69 | 161.65 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 12:15:00 | 160.60 | 161.47 | 161.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 160.24 | 161.22 | 161.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 162.98 | 161.57 | 161.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 162.98 | 161.57 | 161.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 162.98 | 161.57 | 161.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 162.98 | 161.57 | 161.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 163.45 | 161.95 | 161.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 164.33 | 162.54 | 162.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 164.84 | 166.03 | 164.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 12:00:00 | 164.84 | 166.03 | 164.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 163.60 | 165.54 | 164.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:00:00 | 163.60 | 165.54 | 164.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 164.91 | 165.42 | 164.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:45:00 | 165.76 | 165.33 | 164.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 165.84 | 165.43 | 164.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 159.78 | 163.83 | 164.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 159.78 | 163.83 | 164.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 154.52 | 161.38 | 163.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 152.00 | 151.68 | 155.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 156.65 | 151.68 | 155.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 156.65 | 152.67 | 155.23 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 157.70 | 156.33 | 156.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 158.56 | 156.78 | 156.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 12:15:00 | 161.70 | 162.24 | 161.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 13:00:00 | 161.70 | 162.24 | 161.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 163.93 | 166.60 | 165.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:45:00 | 163.45 | 166.60 | 165.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 165.87 | 166.46 | 165.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:00:00 | 166.35 | 166.32 | 165.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 163.74 | 165.68 | 165.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 163.74 | 165.68 | 165.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 161.79 | 164.66 | 165.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 163.91 | 162.94 | 163.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 11:15:00 | 163.91 | 162.94 | 163.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 163.91 | 162.94 | 163.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:00:00 | 163.91 | 162.94 | 163.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 164.93 | 163.34 | 163.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 164.93 | 163.34 | 163.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 165.71 | 163.81 | 164.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 165.71 | 163.81 | 164.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 165.71 | 164.46 | 164.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 166.91 | 164.95 | 164.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 171.68 | 171.81 | 170.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 171.68 | 171.81 | 170.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 169.08 | 171.16 | 170.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 169.08 | 171.16 | 170.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 168.89 | 170.71 | 170.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 170.08 | 170.71 | 170.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 170.64 | 170.69 | 170.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:15:00 | 171.80 | 170.69 | 170.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 13:15:00 | 171.50 | 171.00 | 170.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 14:45:00 | 171.59 | 171.23 | 170.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 174.23 | 171.18 | 170.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 174.15 | 174.79 | 173.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:00:00 | 174.15 | 174.79 | 173.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 174.91 | 174.82 | 174.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 12:15:00 | 175.27 | 174.82 | 174.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 15:00:00 | 175.50 | 174.81 | 174.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 176.71 | 174.89 | 174.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 10:30:00 | 175.36 | 175.11 | 174.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 176.16 | 176.48 | 175.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 176.16 | 176.48 | 175.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 176.02 | 176.39 | 175.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 173.23 | 176.39 | 175.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 172.84 | 175.68 | 175.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 172.84 | 175.68 | 175.56 | SL hit (close<static) qty=1.00 sl=173.72 alert=retest2 |

### Cycle 16 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 173.11 | 175.17 | 175.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 163.94 | 171.43 | 173.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 165.57 | 165.28 | 168.46 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:30:00 | 164.46 | 164.98 | 167.77 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 15:00:00 | 163.43 | 164.11 | 166.64 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 156.24 | 159.39 | 162.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 155.26 | 159.39 | 162.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-03-09 10:15:00 | 148.01 | 157.50 | 161.57 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 17 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 152.89 | 151.53 | 151.34 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 148.15 | 151.34 | 151.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 145.85 | 148.99 | 150.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 149.59 | 148.28 | 149.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 149.59 | 148.28 | 149.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 149.59 | 148.28 | 149.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 151.22 | 148.28 | 149.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 151.66 | 148.96 | 149.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 151.44 | 148.96 | 149.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 150.53 | 149.27 | 149.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 149.85 | 149.27 | 149.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 150.99 | 150.04 | 150.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 13:15:00 | 150.99 | 150.04 | 150.04 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 144.43 | 149.02 | 149.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 143.81 | 147.18 | 148.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 145.75 | 145.31 | 146.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 145.75 | 145.31 | 146.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 145.75 | 145.31 | 146.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 145.11 | 145.37 | 146.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 147.68 | 145.91 | 146.84 | SL hit (close>static) qty=1.00 sl=147.60 alert=retest2 |

### Cycle 21 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 150.15 | 147.59 | 147.34 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 143.73 | 147.37 | 147.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 141.73 | 144.82 | 146.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 140.51 | 139.97 | 142.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 140.51 | 139.97 | 142.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 140.51 | 139.97 | 142.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 140.31 | 139.97 | 142.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:45:00 | 140.34 | 140.50 | 142.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 143.40 | 141.08 | 142.42 | SL hit (close>static) qty=1.00 sl=142.81 alert=retest2 |

### Cycle 23 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 142.75 | 140.95 | 140.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 143.02 | 141.37 | 140.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 138.08 | 141.00 | 140.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 138.08 | 141.00 | 140.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 138.08 | 141.00 | 140.88 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 137.54 | 140.31 | 140.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 11:15:00 | 137.24 | 139.70 | 140.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 145.86 | 139.80 | 139.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 145.86 | 139.80 | 139.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 145.86 | 139.80 | 139.91 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 147.25 | 141.29 | 140.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 147.99 | 145.45 | 144.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 142.17 | 145.97 | 145.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 142.17 | 145.97 | 145.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 142.17 | 145.97 | 145.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 144.62 | 145.03 | 144.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 144.50 | 144.84 | 144.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 144.50 | 144.84 | 144.86 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 15:15:00 | 145.35 | 144.96 | 144.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 148.02 | 145.57 | 145.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 12:15:00 | 148.41 | 148.54 | 147.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 13:00:00 | 148.41 | 148.54 | 147.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 149.20 | 148.80 | 147.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:30:00 | 149.89 | 148.57 | 148.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 149.77 | 148.79 | 148.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 13:00:00 | 150.72 | 151.85 | 151.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 148.38 | 150.77 | 150.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 148.38 | 150.77 | 150.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 146.76 | 149.97 | 150.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 148.22 | 147.93 | 148.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 13:00:00 | 148.22 | 147.93 | 148.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 145.55 | 145.06 | 146.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:15:00 | 144.41 | 145.50 | 146.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:15:00 | 137.19 | 138.98 | 140.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 141.06 | 139.19 | 140.00 | SL hit (close>ema200) qty=0.50 sl=139.19 alert=retest2 |

### Cycle 29 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 142.35 | 140.37 | 140.25 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 10:15:00 | 139.95 | 140.32 | 140.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 11:15:00 | 139.17 | 140.09 | 140.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 15:15:00 | 139.85 | 139.76 | 140.01 | EMA200 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-01-07 11:15:00 | 151.21 | 2026-01-08 10:15:00 | 147.33 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-01-07 14:00:00 | 151.05 | 2026-01-08 10:15:00 | 147.33 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-01-30 09:45:00 | 165.76 | 2026-02-01 09:15:00 | 159.78 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2026-01-30 10:30:00 | 165.84 | 2026-02-01 09:15:00 | 159.78 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2026-02-11 13:00:00 | 166.35 | 2026-02-12 11:15:00 | 163.74 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2026-02-20 11:15:00 | 171.80 | 2026-03-02 09:15:00 | 172.84 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2026-02-20 13:15:00 | 171.50 | 2026-03-02 09:15:00 | 172.84 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2026-02-20 14:45:00 | 171.59 | 2026-03-02 09:15:00 | 172.84 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2026-02-23 09:15:00 | 174.23 | 2026-03-02 09:15:00 | 172.84 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-02-25 12:15:00 | 175.27 | 2026-03-02 10:15:00 | 173.11 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-02-25 15:00:00 | 175.50 | 2026-03-02 10:15:00 | 173.11 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-02-26 09:15:00 | 176.71 | 2026-03-02 10:15:00 | 173.11 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-02-26 10:30:00 | 175.36 | 2026-03-02 10:15:00 | 173.11 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest1 | 2026-03-05 11:30:00 | 164.46 | 2026-03-09 09:15:00 | 156.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-05 15:00:00 | 163.43 | 2026-03-09 09:15:00 | 155.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-05 11:30:00 | 164.46 | 2026-03-09 10:15:00 | 148.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-03-05 15:00:00 | 163.43 | 2026-03-10 10:15:00 | 153.71 | STOP_HIT | 0.50 | 5.95% |
| SELL | retest2 | 2026-03-11 10:15:00 | 155.14 | 2026-03-16 10:15:00 | 147.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:30:00 | 155.00 | 2026-03-16 10:15:00 | 147.58 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2026-03-11 13:30:00 | 155.35 | 2026-03-16 10:15:00 | 147.37 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2026-03-11 14:30:00 | 155.13 | 2026-03-16 12:15:00 | 147.25 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2026-03-11 10:15:00 | 155.14 | 2026-03-16 14:15:00 | 150.73 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2026-03-11 11:30:00 | 155.00 | 2026-03-16 14:15:00 | 150.73 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2026-03-11 13:30:00 | 155.35 | 2026-03-16 14:15:00 | 150.73 | STOP_HIT | 0.50 | 2.97% |
| SELL | retest2 | 2026-03-11 14:30:00 | 155.13 | 2026-03-16 14:15:00 | 150.73 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2026-03-13 09:15:00 | 153.54 | 2026-03-18 11:15:00 | 152.89 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2026-03-20 12:15:00 | 149.85 | 2026-03-20 13:15:00 | 150.99 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-03-24 10:30:00 | 145.11 | 2026-03-24 12:15:00 | 147.68 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-04-01 10:15:00 | 140.31 | 2026-04-01 12:15:00 | 143.40 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-04-01 11:45:00 | 140.34 | 2026-04-01 12:15:00 | 143.40 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-04-02 09:15:00 | 136.65 | 2026-04-06 13:15:00 | 142.75 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2026-04-13 12:15:00 | 144.62 | 2026-04-13 13:15:00 | 144.50 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2026-04-20 10:30:00 | 149.89 | 2026-04-24 09:15:00 | 148.38 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-04-21 09:15:00 | 149.77 | 2026-04-24 09:15:00 | 148.38 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-04-23 13:00:00 | 150.72 | 2026-04-24 09:15:00 | 148.38 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-29 14:15:00 | 144.41 | 2026-05-05 11:15:00 | 137.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-29 14:15:00 | 144.41 | 2026-05-06 09:15:00 | 141.06 | STOP_HIT | 0.50 | 2.32% |
