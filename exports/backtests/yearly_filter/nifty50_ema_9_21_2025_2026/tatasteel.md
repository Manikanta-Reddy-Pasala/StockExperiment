# TATASTEEL (TATASTEEL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 214.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 61 |
| ALERT1 | 49 |
| ALERT2 | 48 |
| ALERT2_SKIP | 27 |
| ALERT3 | 134 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 60 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 36
- **Target hits / Stop hits / Partials:** 0 / 60 / 2
- **Avg / median % per leg:** 0.28% / -0.30%
- **Sum % (uncompounded):** 17.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 5 | 19.2% | 0 | 26 | 0 | -0.26% | -6.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 5 | 19.2% | 0 | 26 | 0 | -0.26% | -6.9% |
| SELL (all) | 36 | 21 | 58.3% | 0 | 34 | 2 | 0.67% | 24.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 21 | 58.3% | 0 | 34 | 2 | 0.67% | 24.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 62 | 26 | 41.9% | 0 | 60 | 2 | 0.28% | 17.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 147.74 | 143.97 | 143.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 150.40 | 146.28 | 145.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 149.17 | 149.50 | 147.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:45:00 | 149.10 | 149.50 | 147.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 162.92 | 162.94 | 162.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:45:00 | 162.31 | 162.94 | 162.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 162.70 | 162.80 | 162.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:00:00 | 163.11 | 162.67 | 162.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 13:15:00 | 161.08 | 162.37 | 162.26 | SL hit (close<static) qty=1.00 sl=162.07 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 161.76 | 162.14 | 162.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 161.40 | 161.99 | 162.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 162.39 | 161.59 | 161.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 162.39 | 161.59 | 161.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 162.39 | 161.59 | 161.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 10:30:00 | 161.69 | 161.70 | 161.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 11:15:00 | 162.60 | 161.88 | 161.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 162.60 | 161.88 | 161.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 163.77 | 162.26 | 162.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 161.95 | 162.48 | 162.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 161.95 | 162.48 | 162.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 161.95 | 162.48 | 162.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 162.07 | 162.48 | 162.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 161.57 | 162.30 | 162.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:45:00 | 161.33 | 162.30 | 162.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 12:15:00 | 161.17 | 161.96 | 162.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 160.83 | 161.59 | 161.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 158.19 | 158.11 | 158.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 12:00:00 | 158.19 | 158.11 | 158.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 158.59 | 158.19 | 158.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 158.59 | 158.19 | 158.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 158.08 | 158.17 | 158.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:45:00 | 158.65 | 158.17 | 158.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 158.18 | 158.21 | 158.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:45:00 | 157.30 | 158.03 | 158.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 11:30:00 | 157.55 | 158.17 | 158.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 157.27 | 157.95 | 158.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 155.67 | 157.97 | 158.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 157.43 | 157.01 | 157.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:00:00 | 157.43 | 157.01 | 157.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 157.70 | 157.14 | 157.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 157.70 | 157.14 | 157.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 157.59 | 157.23 | 157.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 157.00 | 157.23 | 157.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 157.37 | 157.26 | 157.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:15:00 | 156.70 | 157.31 | 157.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 11:00:00 | 156.35 | 156.42 | 156.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:15:00 | 156.78 | 156.70 | 156.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 14:00:00 | 156.59 | 156.68 | 156.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 153.14 | 152.39 | 153.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:45:00 | 153.14 | 152.39 | 153.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 153.69 | 152.65 | 153.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 153.69 | 152.65 | 153.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 153.86 | 152.89 | 153.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:45:00 | 153.64 | 152.89 | 153.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 154.22 | 153.16 | 153.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 154.22 | 153.16 | 153.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 154.10 | 153.57 | 153.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:00:00 | 154.10 | 153.57 | 153.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 153.36 | 153.53 | 153.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:30:00 | 154.03 | 153.53 | 153.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 151.07 | 151.41 | 151.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 15:15:00 | 150.51 | 151.41 | 151.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 09:45:00 | 150.90 | 151.23 | 151.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 153.05 | 151.59 | 151.91 | SL hit (close>static) qty=1.00 sl=152.45 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 152.94 | 152.06 | 152.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 154.21 | 152.58 | 152.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 10:15:00 | 160.68 | 161.00 | 159.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 11:00:00 | 160.68 | 161.00 | 159.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 159.89 | 160.48 | 159.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:15:00 | 159.80 | 160.48 | 159.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 159.80 | 160.35 | 159.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 159.25 | 160.35 | 159.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 158.20 | 159.92 | 159.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 158.20 | 159.92 | 159.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 157.89 | 159.51 | 159.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 158.10 | 159.51 | 159.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 159.08 | 159.41 | 159.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 159.08 | 159.41 | 159.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 159.79 | 159.49 | 159.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:45:00 | 159.41 | 159.49 | 159.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 164.11 | 165.38 | 164.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 163.58 | 165.38 | 164.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 164.39 | 165.18 | 164.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:30:00 | 164.55 | 165.18 | 164.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 162.80 | 164.71 | 163.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:00:00 | 162.80 | 164.71 | 163.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 162.68 | 164.30 | 163.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:15:00 | 162.38 | 164.30 | 163.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 162.80 | 163.43 | 163.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 09:15:00 | 159.27 | 161.49 | 162.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 159.94 | 159.59 | 160.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 159.94 | 159.59 | 160.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 159.94 | 159.59 | 160.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:30:00 | 159.66 | 159.58 | 160.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 13:15:00 | 161.00 | 160.02 | 160.49 | SL hit (close>static) qty=1.00 sl=160.94 alert=retest2 |

### Cycle 7 — BUY (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 14:15:00 | 159.96 | 158.73 | 158.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 161.88 | 159.56 | 159.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 14:15:00 | 163.13 | 163.36 | 162.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 15:00:00 | 163.13 | 163.36 | 162.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 163.50 | 163.33 | 162.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:30:00 | 164.31 | 163.24 | 162.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 161.78 | 163.09 | 163.00 | SL hit (close<static) qty=1.00 sl=162.60 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 161.74 | 162.82 | 162.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 14:15:00 | 161.45 | 162.14 | 162.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 160.44 | 160.24 | 160.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 160.44 | 160.24 | 160.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 161.68 | 160.53 | 160.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 161.68 | 160.53 | 160.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 161.54 | 160.73 | 161.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 161.54 | 160.73 | 161.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 162.49 | 161.38 | 161.28 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 160.65 | 161.32 | 161.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 10:15:00 | 159.89 | 161.03 | 161.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 11:15:00 | 161.20 | 161.07 | 161.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 11:15:00 | 161.20 | 161.07 | 161.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 161.20 | 161.07 | 161.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:30:00 | 161.18 | 161.07 | 161.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 160.67 | 160.99 | 161.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 14:15:00 | 159.94 | 160.87 | 161.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 159.25 | 157.93 | 157.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 159.25 | 157.93 | 157.83 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 157.35 | 158.40 | 158.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 156.61 | 158.04 | 158.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 158.37 | 157.72 | 158.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 13:15:00 | 158.37 | 157.72 | 158.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 158.37 | 157.72 | 158.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 158.37 | 157.72 | 158.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 159.88 | 158.16 | 158.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 159.88 | 158.16 | 158.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 159.83 | 158.49 | 158.36 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 157.51 | 158.24 | 158.27 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 158.49 | 158.23 | 158.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 161.05 | 158.83 | 158.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 12:15:00 | 160.70 | 160.70 | 159.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 13:00:00 | 160.70 | 160.70 | 159.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 160.35 | 160.50 | 160.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 157.57 | 160.50 | 160.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 157.15 | 159.83 | 159.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 157.15 | 159.83 | 159.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 157.05 | 159.28 | 159.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 13:15:00 | 155.95 | 158.01 | 158.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 157.30 | 157.11 | 158.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 157.30 | 157.11 | 158.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 157.30 | 157.11 | 158.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:00:00 | 156.17 | 156.94 | 157.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 13:15:00 | 158.69 | 158.00 | 157.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 13:15:00 | 158.69 | 158.00 | 157.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 159.29 | 158.26 | 158.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 14:15:00 | 161.46 | 161.65 | 160.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 15:00:00 | 161.46 | 161.65 | 160.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 160.30 | 161.32 | 160.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 160.30 | 161.32 | 160.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 160.11 | 161.08 | 160.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 160.11 | 161.08 | 160.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 159.27 | 160.39 | 160.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 158.67 | 160.04 | 160.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 159.74 | 159.58 | 159.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:30:00 | 159.23 | 159.58 | 159.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 159.85 | 159.64 | 159.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:00:00 | 159.53 | 159.68 | 159.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 156.20 | 155.54 | 155.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 156.20 | 155.54 | 155.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 157.29 | 155.89 | 155.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 09:15:00 | 166.00 | 166.45 | 164.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 10:00:00 | 166.00 | 166.45 | 164.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 169.50 | 169.49 | 168.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 10:00:00 | 169.70 | 169.52 | 169.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 12:00:00 | 169.98 | 169.68 | 169.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 168.65 | 169.47 | 169.17 | SL hit (close<static) qty=1.00 sl=168.75 alert=retest2 |

### Cycle 20 — SELL (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 15:15:00 | 169.12 | 169.31 | 169.33 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 169.96 | 169.44 | 169.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 12:15:00 | 171.12 | 169.98 | 169.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 170.90 | 171.00 | 170.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 09:45:00 | 171.43 | 171.00 | 170.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 169.98 | 170.83 | 170.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 169.98 | 170.83 | 170.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 171.11 | 170.89 | 170.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 12:45:00 | 171.39 | 171.00 | 170.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 13:15:00 | 171.84 | 171.00 | 170.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 171.47 | 171.40 | 171.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:45:00 | 171.53 | 171.45 | 171.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 171.28 | 171.47 | 171.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 171.28 | 171.47 | 171.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 171.98 | 171.57 | 171.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 171.14 | 171.57 | 171.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 171.60 | 171.58 | 171.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 172.84 | 171.46 | 171.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 12:45:00 | 172.50 | 171.84 | 171.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 09:45:00 | 173.60 | 173.08 | 172.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 170.33 | 172.28 | 172.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 170.33 | 172.28 | 172.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 10:15:00 | 168.66 | 171.56 | 172.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 169.53 | 169.16 | 170.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:30:00 | 169.71 | 169.16 | 170.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 169.91 | 169.31 | 170.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 169.91 | 169.31 | 170.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 169.07 | 169.26 | 170.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:00:00 | 168.44 | 169.10 | 169.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:15:00 | 168.70 | 169.23 | 169.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 15:00:00 | 168.70 | 169.12 | 169.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 168.76 | 169.11 | 169.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 168.20 | 168.93 | 169.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 167.66 | 168.93 | 169.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 173.00 | 168.83 | 168.92 | SL hit (close>static) qty=1.00 sl=170.28 alert=retest2 |

### Cycle 23 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 174.13 | 169.89 | 169.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 09:15:00 | 176.01 | 172.79 | 171.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 09:15:00 | 173.84 | 174.94 | 173.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 173.84 | 174.94 | 173.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 173.84 | 174.94 | 173.76 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 172.46 | 173.43 | 173.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 170.80 | 172.52 | 172.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 171.95 | 171.11 | 171.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 171.95 | 171.11 | 171.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 171.95 | 171.11 | 171.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 171.95 | 171.11 | 171.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 173.35 | 171.55 | 172.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 173.35 | 171.55 | 172.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 173.76 | 172.00 | 172.22 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 173.41 | 172.49 | 172.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 174.03 | 173.38 | 172.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 11:15:00 | 173.21 | 173.47 | 173.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 11:15:00 | 173.21 | 173.47 | 173.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 173.21 | 173.47 | 173.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 173.21 | 173.47 | 173.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 171.55 | 173.09 | 173.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 171.55 | 173.09 | 173.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 171.88 | 172.85 | 172.91 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 175.06 | 172.88 | 172.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 176.60 | 174.44 | 173.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 13:15:00 | 174.61 | 174.66 | 174.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 13:15:00 | 174.61 | 174.66 | 174.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 174.61 | 174.66 | 174.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 174.12 | 174.66 | 174.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 174.39 | 174.60 | 174.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 174.39 | 174.60 | 174.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 183.25 | 183.76 | 182.72 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 181.55 | 182.70 | 182.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 181.06 | 182.37 | 182.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 177.50 | 177.48 | 178.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 177.50 | 177.48 | 178.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 180.55 | 178.23 | 179.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 180.55 | 178.23 | 179.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 180.50 | 178.69 | 179.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 180.51 | 178.69 | 179.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 181.50 | 179.70 | 179.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 182.02 | 180.67 | 180.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 179.11 | 180.59 | 180.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 179.11 | 180.59 | 180.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 179.11 | 180.59 | 180.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 179.11 | 180.59 | 180.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 179.84 | 180.44 | 180.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 179.33 | 180.44 | 180.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 181.20 | 180.95 | 180.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:45:00 | 180.97 | 180.95 | 180.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 179.90 | 180.74 | 180.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:45:00 | 179.67 | 180.74 | 180.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 179.18 | 180.43 | 180.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:00:00 | 179.18 | 180.43 | 180.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 12:15:00 | 179.13 | 180.17 | 180.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 13:15:00 | 178.71 | 179.88 | 180.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 09:15:00 | 182.04 | 179.96 | 180.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 182.04 | 179.96 | 180.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 182.04 | 179.96 | 180.09 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 181.95 | 180.36 | 180.26 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 177.50 | 180.05 | 180.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 174.18 | 177.60 | 178.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 13:15:00 | 172.90 | 172.65 | 173.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 10:15:00 | 172.76 | 172.63 | 173.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 172.76 | 172.63 | 173.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:30:00 | 173.31 | 172.63 | 173.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 173.40 | 172.90 | 173.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:45:00 | 173.30 | 172.90 | 173.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 173.32 | 172.99 | 173.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:30:00 | 173.53 | 172.99 | 173.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 173.20 | 173.03 | 173.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 173.55 | 173.03 | 173.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 173.36 | 173.10 | 173.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:30:00 | 172.65 | 172.95 | 173.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:15:00 | 172.58 | 172.93 | 173.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:45:00 | 172.40 | 172.87 | 173.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:15:00 | 172.45 | 172.87 | 173.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 169.09 | 166.87 | 167.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 169.28 | 168.23 | 168.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 169.28 | 168.23 | 168.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 169.90 | 168.56 | 168.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 168.40 | 168.81 | 168.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 11:15:00 | 168.40 | 168.81 | 168.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 168.40 | 168.81 | 168.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 168.38 | 168.81 | 168.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 167.60 | 168.57 | 168.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 167.60 | 168.57 | 168.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 167.83 | 168.42 | 168.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 167.47 | 168.42 | 168.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 168.18 | 168.35 | 168.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 167.85 | 168.25 | 168.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 10:15:00 | 168.75 | 168.35 | 168.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 10:15:00 | 168.75 | 168.35 | 168.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 168.75 | 168.35 | 168.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:00:00 | 168.75 | 168.35 | 168.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 168.41 | 168.36 | 168.36 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 12:15:00 | 168.55 | 168.33 | 168.32 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 167.61 | 168.32 | 168.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 165.90 | 167.52 | 167.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 167.02 | 166.86 | 167.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 15:00:00 | 167.02 | 166.86 | 167.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 167.00 | 166.89 | 167.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 165.75 | 166.75 | 167.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:00:00 | 165.96 | 166.81 | 166.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:45:00 | 166.16 | 166.65 | 166.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 165.57 | 163.82 | 163.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 165.57 | 163.82 | 163.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 166.28 | 164.31 | 163.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 170.45 | 171.84 | 170.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 170.45 | 171.84 | 170.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 170.05 | 171.26 | 170.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 170.05 | 171.26 | 170.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 170.01 | 171.01 | 170.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:45:00 | 169.72 | 171.01 | 170.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 169.92 | 170.79 | 170.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 169.92 | 170.79 | 170.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 169.85 | 170.60 | 170.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 171.04 | 170.44 | 170.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:00:00 | 170.19 | 170.41 | 170.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 15:00:00 | 170.39 | 170.21 | 170.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 169.22 | 170.03 | 170.03 | SL hit (close<static) qty=1.00 sl=169.36 alert=retest2 |

### Cycle 38 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 169.79 | 169.98 | 170.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 168.52 | 169.62 | 169.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 168.63 | 168.57 | 169.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 170.65 | 169.02 | 169.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 170.65 | 169.02 | 169.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 170.85 | 169.02 | 169.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 170.59 | 169.33 | 169.28 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 168.64 | 170.07 | 170.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 168.29 | 169.71 | 169.94 | Break + close below crossover candle low |

### Cycle 41 — BUY (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 09:15:00 | 172.70 | 169.96 | 169.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 12:15:00 | 174.06 | 172.46 | 171.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 181.84 | 182.09 | 180.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 13:45:00 | 182.00 | 182.09 | 180.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 186.15 | 186.11 | 184.57 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 181.39 | 183.83 | 184.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 180.24 | 183.11 | 183.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 10:15:00 | 179.60 | 179.42 | 180.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 10:15:00 | 179.60 | 179.42 | 180.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 179.60 | 179.42 | 180.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:45:00 | 180.05 | 179.42 | 180.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 181.96 | 179.91 | 180.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:00:00 | 181.96 | 179.91 | 180.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 182.54 | 180.44 | 180.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 182.54 | 180.44 | 180.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 183.30 | 181.44 | 181.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 10:15:00 | 183.93 | 182.06 | 181.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 12:15:00 | 181.94 | 182.05 | 181.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 12:15:00 | 181.94 | 182.05 | 181.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 181.94 | 182.05 | 181.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:00:00 | 181.94 | 182.05 | 181.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 181.43 | 181.92 | 181.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:00:00 | 181.43 | 181.92 | 181.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 182.52 | 182.04 | 181.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 184.30 | 182.12 | 181.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 11:15:00 | 185.27 | 186.94 | 187.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 185.27 | 186.94 | 187.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 184.38 | 186.16 | 186.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 184.30 | 183.90 | 184.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 184.30 | 183.90 | 184.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 187.97 | 184.81 | 185.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 189.45 | 184.81 | 185.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 188.60 | 185.57 | 185.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 189.40 | 187.66 | 186.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 188.91 | 189.10 | 187.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 13:00:00 | 188.91 | 189.10 | 187.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 187.91 | 188.86 | 187.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 187.91 | 188.86 | 187.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 187.55 | 188.60 | 187.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 187.55 | 188.60 | 187.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 187.88 | 188.45 | 187.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 189.00 | 188.45 | 187.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 14:15:00 | 193.25 | 194.37 | 194.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 14:15:00 | 193.25 | 194.37 | 194.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 09:15:00 | 191.59 | 193.53 | 194.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 187.02 | 186.45 | 188.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:45:00 | 186.88 | 186.45 | 188.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 188.72 | 187.17 | 188.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 191.95 | 187.17 | 188.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 192.55 | 188.25 | 189.23 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 193.76 | 190.09 | 189.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 195.72 | 192.80 | 191.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 193.41 | 194.41 | 193.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 193.41 | 194.41 | 193.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 193.41 | 194.41 | 193.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 10:30:00 | 194.40 | 194.57 | 193.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 204.69 | 205.90 | 206.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 204.69 | 205.90 | 206.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 203.23 | 205.36 | 205.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 205.00 | 204.43 | 205.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 12:15:00 | 205.00 | 204.43 | 205.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 205.00 | 204.43 | 205.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 205.00 | 204.43 | 205.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 204.99 | 204.54 | 205.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:45:00 | 205.03 | 204.54 | 205.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 205.86 | 204.80 | 205.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 205.86 | 204.80 | 205.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 205.33 | 204.91 | 205.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 203.79 | 204.91 | 205.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 206.97 | 204.09 | 204.33 | SL hit (close>static) qty=1.00 sl=205.90 alert=retest2 |

### Cycle 49 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 209.10 | 205.09 | 204.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 209.53 | 207.18 | 205.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 207.80 | 207.81 | 206.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:00:00 | 207.80 | 207.81 | 206.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 207.00 | 207.61 | 206.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:30:00 | 207.42 | 207.61 | 206.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 205.41 | 207.17 | 206.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 205.41 | 207.17 | 206.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 205.20 | 206.77 | 206.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 206.15 | 206.58 | 206.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 209.96 | 212.66 | 212.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 209.96 | 212.66 | 212.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 209.32 | 211.99 | 212.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 211.60 | 211.23 | 211.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 14:15:00 | 211.60 | 211.23 | 211.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 211.60 | 211.23 | 211.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:30:00 | 211.51 | 211.23 | 211.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 199.97 | 200.06 | 202.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:15:00 | 199.39 | 200.06 | 201.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 199.05 | 200.04 | 201.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 189.42 | 198.01 | 200.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 189.10 | 198.01 | 200.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 193.12 | 192.76 | 195.82 | SL hit (close>ema200) qty=0.50 sl=192.76 alert=retest2 |

### Cycle 51 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 191.17 | 188.52 | 188.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 193.56 | 190.12 | 189.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 192.61 | 194.17 | 192.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 192.61 | 194.17 | 192.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 192.61 | 194.17 | 192.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 197.67 | 192.28 | 192.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 187.50 | 194.38 | 194.05 | SL hit (close<static) qty=1.00 sl=189.37 alert=retest2 |

### Cycle 52 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 188.55 | 193.21 | 193.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 15:15:00 | 187.00 | 189.42 | 191.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 191.30 | 189.40 | 190.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 191.30 | 189.40 | 190.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 191.30 | 189.40 | 190.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 191.30 | 189.40 | 190.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 191.24 | 189.77 | 190.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 191.58 | 189.77 | 190.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 191.20 | 190.20 | 190.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 194.97 | 190.20 | 190.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 194.80 | 191.12 | 191.13 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 195.77 | 192.05 | 191.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 196.60 | 193.90 | 192.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 194.31 | 194.76 | 193.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 194.31 | 194.76 | 193.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 194.31 | 194.76 | 193.39 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 190.00 | 193.20 | 193.25 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 197.79 | 193.42 | 193.17 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 188.82 | 193.81 | 193.83 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 194.85 | 193.26 | 193.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 195.97 | 194.09 | 193.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 194.71 | 195.16 | 194.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 194.71 | 195.16 | 194.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 194.86 | 195.10 | 194.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:00:00 | 194.86 | 195.10 | 194.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 195.85 | 195.25 | 194.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:00:00 | 196.48 | 195.50 | 194.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 12:15:00 | 211.37 | 211.84 | 211.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 211.37 | 211.84 | 211.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 209.58 | 210.96 | 211.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 213.55 | 210.94 | 211.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 213.55 | 210.94 | 211.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 213.55 | 210.94 | 211.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:30:00 | 213.52 | 210.94 | 211.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 214.54 | 211.66 | 211.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 217.60 | 213.81 | 212.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 214.85 | 214.92 | 213.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 214.85 | 214.92 | 213.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 214.05 | 214.87 | 213.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 13:45:00 | 216.57 | 215.27 | 214.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 10:15:00 | 211.02 | 214.03 | 214.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 211.02 | 214.03 | 214.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 12:15:00 | 210.29 | 212.80 | 213.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 213.89 | 212.38 | 213.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 213.89 | 212.38 | 213.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 213.89 | 212.38 | 213.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 214.11 | 212.38 | 213.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 213.94 | 212.69 | 213.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 214.43 | 212.69 | 213.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 212.43 | 212.48 | 212.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:30:00 | 212.98 | 212.48 | 212.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 213.60 | 212.01 | 212.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:00:00 | 212.71 | 212.15 | 212.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 213.73 | 212.47 | 212.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 213.73 | 212.47 | 212.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 214.83 | 213.12 | 212.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 215.35 | 216.31 | 215.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 215.35 | 216.31 | 215.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 215.35 | 216.31 | 215.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 215.34 | 216.31 | 215.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 216.39 | 216.33 | 215.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 11:30:00 | 216.81 | 216.42 | 215.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 13:15:00 | 214.54 | 215.92 | 215.38 | SL hit (close<static) qty=1.00 sl=214.80 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-27 11:00:00 | 163.11 | 2025-05-27 13:15:00 | 161.08 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-05-29 10:30:00 | 161.69 | 2025-05-29 11:15:00 | 162.60 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-06-05 10:45:00 | 157.30 | 2025-06-20 10:15:00 | 153.05 | STOP_HIT | 1.00 | 2.70% |
| SELL | retest2 | 2025-06-05 11:30:00 | 157.55 | 2025-06-20 10:15:00 | 153.05 | STOP_HIT | 1.00 | 2.86% |
| SELL | retest2 | 2025-06-05 13:30:00 | 157.27 | 2025-06-23 13:15:00 | 152.94 | STOP_HIT | 1.00 | 2.75% |
| SELL | retest2 | 2025-06-06 09:15:00 | 155.67 | 2025-06-23 13:15:00 | 152.94 | STOP_HIT | 1.00 | 1.75% |
| SELL | retest2 | 2025-06-10 11:15:00 | 156.70 | 2025-06-23 13:15:00 | 152.94 | STOP_HIT | 1.00 | 2.40% |
| SELL | retest2 | 2025-06-11 11:00:00 | 156.35 | 2025-06-23 13:15:00 | 152.94 | STOP_HIT | 1.00 | 2.18% |
| SELL | retest2 | 2025-06-11 13:15:00 | 156.78 | 2025-06-23 13:15:00 | 152.94 | STOP_HIT | 1.00 | 2.45% |
| SELL | retest2 | 2025-06-11 14:00:00 | 156.59 | 2025-06-23 13:15:00 | 152.94 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2025-06-19 15:15:00 | 150.51 | 2025-06-23 13:15:00 | 152.94 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-06-20 09:45:00 | 150.90 | 2025-06-23 13:15:00 | 152.94 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-07-10 10:30:00 | 159.66 | 2025-07-10 13:15:00 | 161.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-07-11 11:30:00 | 159.78 | 2025-07-17 14:15:00 | 159.96 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-07-11 12:15:00 | 159.62 | 2025-07-17 14:15:00 | 159.96 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-07-14 12:45:00 | 159.73 | 2025-07-17 14:15:00 | 159.96 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-07-24 12:30:00 | 164.31 | 2025-07-25 09:15:00 | 161.78 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-07-31 14:15:00 | 159.94 | 2025-08-04 15:15:00 | 159.25 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-08-18 12:00:00 | 156.17 | 2025-08-19 13:15:00 | 158.69 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-08-25 15:00:00 | 159.53 | 2025-09-01 15:15:00 | 156.20 | STOP_HIT | 1.00 | 2.09% |
| BUY | retest2 | 2025-09-11 10:00:00 | 169.70 | 2025-09-11 12:15:00 | 168.65 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-09-11 12:00:00 | 169.98 | 2025-09-11 12:15:00 | 168.65 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-09-12 09:15:00 | 170.21 | 2025-09-15 15:15:00 | 169.12 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-09-12 10:00:00 | 169.63 | 2025-09-15 15:15:00 | 169.12 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-09-12 13:30:00 | 169.57 | 2025-09-15 15:15:00 | 169.12 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-09-15 10:30:00 | 169.90 | 2025-09-15 15:15:00 | 169.12 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-09-18 12:45:00 | 171.39 | 2025-09-26 09:15:00 | 170.33 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-09-18 13:15:00 | 171.84 | 2025-09-26 09:15:00 | 170.33 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-19 12:00:00 | 171.47 | 2025-09-26 09:15:00 | 170.33 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-09-19 12:45:00 | 171.53 | 2025-09-26 09:15:00 | 170.33 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-09-23 09:15:00 | 172.84 | 2025-09-26 09:15:00 | 170.33 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-09-23 12:45:00 | 172.50 | 2025-09-26 09:15:00 | 170.33 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-09-25 09:45:00 | 173.60 | 2025-09-26 09:15:00 | 170.33 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-09-30 10:00:00 | 168.44 | 2025-10-03 09:15:00 | 173.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-09-30 14:15:00 | 168.70 | 2025-10-03 09:15:00 | 173.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-09-30 15:00:00 | 168.70 | 2025-10-03 09:15:00 | 173.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-10-01 09:15:00 | 168.76 | 2025-10-03 09:15:00 | 173.00 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-10-01 10:15:00 | 167.66 | 2025-10-03 09:15:00 | 173.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-11-20 12:30:00 | 172.65 | 2025-11-26 13:15:00 | 169.28 | STOP_HIT | 1.00 | 1.95% |
| SELL | retest2 | 2025-11-20 14:15:00 | 172.58 | 2025-11-26 13:15:00 | 169.28 | STOP_HIT | 1.00 | 1.91% |
| SELL | retest2 | 2025-11-20 14:45:00 | 172.40 | 2025-11-26 13:15:00 | 169.28 | STOP_HIT | 1.00 | 1.81% |
| SELL | retest2 | 2025-11-20 15:15:00 | 172.45 | 2025-11-26 13:15:00 | 169.28 | STOP_HIT | 1.00 | 1.84% |
| SELL | retest2 | 2025-12-05 09:30:00 | 165.75 | 2025-12-11 13:15:00 | 165.57 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-12-08 10:00:00 | 165.96 | 2025-12-11 13:15:00 | 165.57 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-12-08 10:45:00 | 166.16 | 2025-12-11 13:15:00 | 165.57 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2025-12-17 09:15:00 | 171.04 | 2025-12-18 09:15:00 | 169.22 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-17 11:00:00 | 170.19 | 2025-12-18 09:15:00 | 169.22 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-12-17 15:00:00 | 170.39 | 2025-12-18 09:15:00 | 169.22 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-01-14 09:15:00 | 184.30 | 2026-01-20 11:15:00 | 185.27 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2026-01-27 09:15:00 | 189.00 | 2026-01-30 14:15:00 | 193.25 | STOP_HIT | 1.00 | 2.25% |
| BUY | retest2 | 2026-02-05 10:30:00 | 194.40 | 2026-02-13 13:15:00 | 204.69 | STOP_HIT | 1.00 | 5.29% |
| SELL | retest2 | 2026-02-17 09:15:00 | 203.79 | 2026-02-18 09:15:00 | 206.97 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-02-20 09:30:00 | 206.15 | 2026-03-02 10:15:00 | 209.96 | STOP_HIT | 1.00 | 1.85% |
| SELL | retest2 | 2026-03-06 12:15:00 | 199.39 | 2026-03-09 09:15:00 | 189.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:45:00 | 199.05 | 2026-03-09 09:15:00 | 189.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:15:00 | 199.39 | 2026-03-10 09:15:00 | 193.12 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2026-03-06 14:45:00 | 199.05 | 2026-03-10 09:15:00 | 193.12 | STOP_HIT | 0.50 | 2.98% |
| BUY | retest2 | 2026-03-20 09:15:00 | 197.67 | 2026-03-23 09:15:00 | 187.50 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest2 | 2026-04-07 14:00:00 | 196.48 | 2026-04-23 12:15:00 | 211.37 | STOP_HIT | 1.00 | 7.58% |
| BUY | retest2 | 2026-04-29 13:45:00 | 216.57 | 2026-04-30 10:15:00 | 211.02 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-05-06 11:00:00 | 212.71 | 2026-05-06 11:15:00 | 213.73 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2026-05-08 11:30:00 | 216.81 | 2026-05-08 13:15:00 | 214.54 | STOP_HIT | 1.00 | -1.05% |
