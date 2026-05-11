# Sammaan Capital Ltd. (SAMMAANCAP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 148.78
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 6 |
| ALERT3 | 83 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 73 |
| PARTIAL | 13 |
| TARGET_HIT | 3 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 88 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 63
- **Target hits / Stop hits / Partials:** 3 / 72 / 13
- **Avg / median % per leg:** -0.43% / -1.44%
- **Sum % (uncompounded):** -37.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 1 | 7.7% | 1 | 12 | 0 | -1.70% | -22.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 1 | 7.7% | 1 | 12 | 0 | -1.70% | -22.1% |
| SELL (all) | 75 | 24 | 32.0% | 2 | 60 | 13 | -0.21% | -15.8% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 1 | 1 | 2 | 5.23% | 20.9% |
| SELL @ 3rd Alert (retest2) | 71 | 20 | 28.2% | 1 | 59 | 11 | -0.52% | -36.7% |
| retest1 (combined) | 4 | 4 | 100.0% | 1 | 1 | 2 | 5.23% | 20.9% |
| retest2 (combined) | 84 | 21 | 25.0% | 2 | 71 | 11 | -0.70% | -58.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 11:15:00 | 175.85 | 169.00 | 169.00 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 159.45 | 169.00 | 169.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 158.02 | 168.89 | 168.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 09:15:00 | 170.78 | 163.64 | 165.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 09:15:00 | 170.78 | 163.64 | 165.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 170.78 | 163.64 | 165.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:00:00 | 170.78 | 163.64 | 165.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 169.11 | 163.69 | 165.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 09:45:00 | 168.35 | 165.99 | 166.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 10:15:00 | 159.93 | 165.50 | 166.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-05 13:15:00 | 165.59 | 164.55 | 165.85 | SL hit (close>ema200) qty=0.50 sl=164.55 alert=retest2 |

### Cycle 3 — BUY (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 12:15:00 | 166.83 | 154.30 | 154.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 13:15:00 | 166.95 | 154.43 | 154.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 157.56 | 158.39 | 156.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 10:30:00 | 157.81 | 158.39 | 156.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 156.32 | 158.37 | 156.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:00:00 | 156.32 | 158.37 | 156.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 154.99 | 158.33 | 156.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 154.99 | 158.33 | 156.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 157.01 | 158.27 | 156.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:30:00 | 158.10 | 158.24 | 156.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 12:15:00 | 155.82 | 158.18 | 156.65 | SL hit (close<static) qty=1.00 sl=156.02 alert=retest2 |

### Cycle 4 — SELL (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 15:15:00 | 150.96 | 155.50 | 155.52 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 160.62 | 155.53 | 155.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 162.21 | 155.60 | 155.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 09:15:00 | 156.14 | 156.73 | 156.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 156.14 | 156.73 | 156.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 156.14 | 156.73 | 156.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 156.14 | 156.73 | 156.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 154.75 | 156.71 | 156.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:00:00 | 154.75 | 156.71 | 156.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 153.75 | 156.68 | 156.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:45:00 | 153.78 | 156.68 | 156.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 156.10 | 156.64 | 156.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:15:00 | 153.37 | 156.64 | 156.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 151.70 | 156.60 | 156.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 151.70 | 156.60 | 156.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 141.42 | 155.66 | 155.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 09:15:00 | 139.87 | 152.57 | 153.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 15:15:00 | 149.00 | 148.94 | 151.64 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 10:15:00 | 146.13 | 148.94 | 151.62 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 151.09 | 148.97 | 151.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:45:00 | 151.30 | 148.97 | 151.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 12:15:00 | 138.82 | 148.20 | 151.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-02-12 09:15:00 | 131.52 | 147.60 | 150.67 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 7 — BUY (started 2025-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 13:15:00 | 127.25 | 122.69 | 122.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 128.06 | 122.83 | 122.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 12:15:00 | 123.83 | 124.63 | 123.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 12:15:00 | 123.83 | 124.63 | 123.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 123.83 | 124.63 | 123.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 123.75 | 124.63 | 123.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 123.40 | 124.62 | 123.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:45:00 | 123.40 | 124.62 | 123.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 123.75 | 124.61 | 123.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:30:00 | 123.65 | 124.61 | 123.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 123.14 | 124.59 | 123.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 123.45 | 124.59 | 123.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 122.73 | 124.57 | 123.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 123.10 | 124.57 | 123.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 121.63 | 124.55 | 123.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 121.63 | 124.55 | 123.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 123.35 | 123.94 | 123.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 124.25 | 123.92 | 123.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-25 11:15:00 | 136.68 | 124.28 | 123.70 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 116.72 | 126.94 | 126.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 11:15:00 | 115.56 | 126.83 | 126.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 125.38 | 124.53 | 125.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 125.38 | 124.53 | 125.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 125.38 | 124.53 | 125.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 125.38 | 124.53 | 125.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 125.66 | 124.54 | 125.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 125.66 | 124.54 | 125.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 124.45 | 124.54 | 125.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:30:00 | 123.80 | 124.52 | 125.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:00:00 | 122.89 | 124.51 | 125.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 117.61 | 123.76 | 125.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 11:15:00 | 116.75 | 123.63 | 124.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 124.28 | 123.41 | 124.83 | SL hit (close>ema200) qty=0.50 sl=123.41 alert=retest2 |

### Cycle 9 — BUY (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 12:15:00 | 138.44 | 126.01 | 125.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 139.38 | 127.20 | 126.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 13:15:00 | 171.00 | 174.36 | 162.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 14:00:00 | 171.00 | 174.36 | 162.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 158.38 | 174.20 | 162.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 158.38 | 174.20 | 162.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 158.48 | 174.05 | 162.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 154.25 | 174.05 | 162.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 164.08 | 173.82 | 162.67 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 12:15:00 | 149.13 | 157.24 | 157.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 147.09 | 156.11 | 156.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 149.30 | 149.12 | 152.25 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 15:15:00 | 148.51 | 149.12 | 152.23 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 151.81 | 149.18 | 152.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:45:00 | 152.35 | 149.18 | 152.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 141.08 | 148.70 | 151.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 147.17 | 144.31 | 147.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 147.17 | 144.31 | 147.96 | SL hit (close>ema200) qty=0.50 sl=144.31 alert=retest1 |

### Cycle 11 — BUY (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 13:15:00 | 154.32 | 148.80 | 148.78 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 141.00 | 148.72 | 148.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 138.82 | 147.75 | 148.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 09:15:00 | 147.25 | 147.07 | 147.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 147.25 | 147.07 | 147.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 147.25 | 147.07 | 147.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 142.40 | 147.03 | 147.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 135.28 | 144.24 | 146.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 153.70 | 143.03 | 145.32 | SL hit (close>ema200) qty=0.50 sl=143.03 alert=retest2 |

### Cycle 13 — BUY (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 14:15:00 | 154.10 | 146.78 | 146.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 156.65 | 146.95 | 146.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 10:15:00 | 148.76 | 148.77 | 147.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 11:00:00 | 148.76 | 148.77 | 147.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 146.14 | 148.76 | 147.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 146.14 | 148.76 | 147.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 147.06 | 148.74 | 147.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 147.51 | 148.74 | 147.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 12:45:00 | 147.57 | 148.71 | 147.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 13:15:00 | 147.60 | 148.71 | 147.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 145.63 | 148.61 | 147.82 | SL hit (close<static) qty=1.00 sl=145.68 alert=retest2 |

### Cycle 14 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 140.90 | 147.13 | 147.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 139.20 | 147.05 | 147.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 148.65 | 146.91 | 147.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 148.65 | 146.91 | 147.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 148.65 | 146.91 | 147.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 148.65 | 146.91 | 147.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 147.41 | 146.91 | 147.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 146.55 | 146.92 | 147.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:45:00 | 146.50 | 146.92 | 147.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 146.00 | 146.93 | 147.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:45:00 | 146.49 | 146.91 | 147.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 146.70 | 146.91 | 147.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:30:00 | 148.20 | 146.91 | 147.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 146.59 | 146.85 | 147.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 13:15:00 | 148.83 | 146.90 | 147.02 | SL hit (close>static) qty=1.00 sl=148.68 alert=retest2 |

### Cycle 15 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 151.74 | 147.18 | 147.16 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-18 13:30:00 | 176.10 | 2024-06-20 11:15:00 | 181.39 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-06-20 10:45:00 | 175.76 | 2024-06-20 11:15:00 | 181.39 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2024-06-24 09:15:00 | 175.00 | 2024-06-27 13:15:00 | 166.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-24 09:15:00 | 175.00 | 2024-06-28 12:15:00 | 169.65 | STOP_HIT | 0.50 | 3.06% |
| SELL | retest2 | 2024-06-27 12:30:00 | 167.69 | 2024-07-01 12:15:00 | 170.90 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-06-28 13:15:00 | 167.32 | 2024-07-01 12:15:00 | 170.90 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-07-02 12:00:00 | 168.02 | 2024-07-03 11:15:00 | 170.65 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-07-02 12:30:00 | 167.97 | 2024-07-03 11:15:00 | 170.65 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-07-08 09:45:00 | 168.96 | 2024-07-09 09:15:00 | 173.60 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-07-08 11:00:00 | 168.80 | 2024-07-09 09:15:00 | 173.60 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-07-08 11:30:00 | 169.00 | 2024-07-09 09:15:00 | 173.60 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2024-07-08 14:45:00 | 168.05 | 2024-07-09 09:15:00 | 173.60 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2024-07-11 12:15:00 | 169.71 | 2024-07-16 10:15:00 | 170.88 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-07-11 13:15:00 | 169.40 | 2024-07-16 10:15:00 | 170.88 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-07-16 12:00:00 | 169.47 | 2024-07-18 09:15:00 | 173.64 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2024-07-18 11:15:00 | 169.50 | 2024-07-23 12:15:00 | 161.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 11:15:00 | 169.50 | 2024-07-29 09:15:00 | 170.32 | STOP_HIT | 0.50 | -0.48% |
| SELL | retest2 | 2024-07-19 09:15:00 | 168.65 | 2024-07-30 09:15:00 | 173.35 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2024-07-19 12:15:00 | 168.62 | 2024-07-30 09:15:00 | 173.35 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2024-07-19 14:00:00 | 168.04 | 2024-07-30 09:15:00 | 173.35 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2024-07-24 09:45:00 | 168.50 | 2024-07-30 09:15:00 | 173.35 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2024-07-29 13:15:00 | 169.27 | 2024-07-30 09:15:00 | 173.35 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2024-08-26 09:45:00 | 168.35 | 2024-09-02 10:15:00 | 159.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-26 09:45:00 | 168.35 | 2024-09-05 13:15:00 | 165.59 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2024-09-13 15:00:00 | 168.30 | 2024-09-19 09:15:00 | 159.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-16 09:45:00 | 168.23 | 2024-09-19 09:15:00 | 159.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 15:00:00 | 168.30 | 2024-09-20 14:15:00 | 163.39 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2024-09-16 09:45:00 | 168.23 | 2024-09-20 14:15:00 | 163.39 | STOP_HIT | 0.50 | 2.88% |
| SELL | retest2 | 2024-09-26 12:30:00 | 168.23 | 2024-10-03 09:15:00 | 159.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 12:30:00 | 168.23 | 2024-10-07 09:15:00 | 151.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-19 15:15:00 | 148.90 | 2024-11-21 10:15:00 | 153.64 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2024-12-18 09:30:00 | 158.10 | 2024-12-18 12:15:00 | 155.82 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest1 | 2025-02-07 10:15:00 | 146.13 | 2025-02-11 12:15:00 | 138.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-07 10:15:00 | 146.13 | 2025-02-12 09:15:00 | 131.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-17 13:45:00 | 119.75 | 2025-04-21 09:15:00 | 122.65 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-04-30 15:15:00 | 120.01 | 2025-05-02 09:15:00 | 122.38 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-05-02 12:00:00 | 119.95 | 2025-05-06 14:15:00 | 113.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 12:45:00 | 119.96 | 2025-05-06 14:15:00 | 113.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 12:00:00 | 119.95 | 2025-05-08 10:15:00 | 118.46 | STOP_HIT | 0.50 | 1.24% |
| SELL | retest2 | 2025-05-02 12:45:00 | 119.96 | 2025-05-08 10:15:00 | 118.46 | STOP_HIT | 0.50 | 1.25% |
| SELL | retest2 | 2025-05-21 11:45:00 | 120.86 | 2025-05-22 09:15:00 | 123.62 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-05-22 11:30:00 | 121.30 | 2025-05-22 12:15:00 | 125.45 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-05-22 12:15:00 | 121.36 | 2025-05-22 12:15:00 | 125.45 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-05-27 09:45:00 | 121.01 | 2025-05-28 10:15:00 | 122.75 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-05-27 13:15:00 | 121.34 | 2025-05-28 10:15:00 | 122.75 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-05-28 10:15:00 | 121.57 | 2025-05-30 11:15:00 | 122.93 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-05-29 10:30:00 | 121.72 | 2025-05-30 11:15:00 | 122.93 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-05-29 11:00:00 | 121.52 | 2025-05-30 11:15:00 | 122.93 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-05-30 09:45:00 | 121.67 | 2025-05-30 13:15:00 | 124.63 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-06-24 09:15:00 | 124.25 | 2025-06-25 11:15:00 | 136.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-04 12:15:00 | 124.62 | 2025-07-04 14:15:00 | 122.86 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-07-04 13:00:00 | 124.00 | 2025-07-04 14:15:00 | 122.86 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-07-04 13:45:00 | 123.99 | 2025-07-04 14:15:00 | 122.86 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-07-10 10:30:00 | 132.55 | 2025-07-14 10:15:00 | 125.33 | STOP_HIT | 1.00 | -5.45% |
| BUY | retest2 | 2025-07-10 12:15:00 | 132.54 | 2025-07-14 10:15:00 | 125.33 | STOP_HIT | 1.00 | -5.44% |
| BUY | retest2 | 2025-07-10 13:45:00 | 131.92 | 2025-07-14 10:15:00 | 125.33 | STOP_HIT | 1.00 | -5.00% |
| BUY | retest2 | 2025-07-17 09:15:00 | 132.40 | 2025-07-25 13:15:00 | 126.44 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest2 | 2025-08-01 11:15:00 | 129.73 | 2025-08-01 14:15:00 | 125.69 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-08-21 13:30:00 | 123.80 | 2025-08-28 09:15:00 | 117.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 15:00:00 | 122.89 | 2025-08-28 11:15:00 | 116.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 13:30:00 | 123.80 | 2025-08-29 10:15:00 | 124.28 | STOP_HIT | 0.50 | -0.39% |
| SELL | retest2 | 2025-08-21 15:00:00 | 122.89 | 2025-08-29 10:15:00 | 124.28 | STOP_HIT | 0.50 | -1.13% |
| SELL | retest2 | 2025-09-01 10:30:00 | 123.61 | 2025-09-01 12:15:00 | 125.91 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest1 | 2026-01-02 15:15:00 | 148.51 | 2026-01-12 11:15:00 | 141.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-02 15:15:00 | 148.51 | 2026-01-29 13:15:00 | 147.17 | STOP_HIT | 0.50 | 0.90% |
| SELL | retest2 | 2026-01-30 09:15:00 | 145.00 | 2026-01-30 10:15:00 | 149.38 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2026-02-01 11:45:00 | 146.13 | 2026-02-03 12:15:00 | 148.61 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-02-04 09:15:00 | 146.52 | 2026-02-04 10:15:00 | 148.85 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-02-04 10:15:00 | 147.10 | 2026-02-04 10:15:00 | 148.85 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-02-05 09:15:00 | 148.75 | 2026-02-11 10:15:00 | 148.50 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2026-02-09 12:15:00 | 148.86 | 2026-02-11 10:15:00 | 148.50 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2026-02-09 14:15:00 | 149.00 | 2026-02-18 09:15:00 | 149.25 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2026-02-09 15:15:00 | 148.50 | 2026-02-18 09:15:00 | 149.25 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-02-10 15:15:00 | 147.00 | 2026-02-19 10:15:00 | 150.63 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-02-11 09:30:00 | 146.41 | 2026-02-19 10:15:00 | 150.63 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2026-02-12 10:15:00 | 146.87 | 2026-02-19 10:15:00 | 150.63 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-02-12 14:30:00 | 146.72 | 2026-02-19 10:15:00 | 150.63 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2026-03-12 09:15:00 | 142.40 | 2026-03-23 09:15:00 | 135.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 09:15:00 | 142.40 | 2026-03-25 09:15:00 | 153.70 | STOP_HIT | 0.50 | -7.94% |
| SELL | retest2 | 2026-03-25 12:45:00 | 143.60 | 2026-03-27 12:15:00 | 150.37 | STOP_HIT | 1.00 | -4.71% |
| SELL | retest2 | 2026-04-02 09:15:00 | 142.82 | 2026-04-08 09:15:00 | 150.03 | STOP_HIT | 1.00 | -5.05% |
| BUY | retest2 | 2026-04-22 11:15:00 | 147.51 | 2026-04-23 10:15:00 | 145.63 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-04-22 12:45:00 | 147.57 | 2026-04-23 10:15:00 | 145.63 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-04-22 13:15:00 | 147.60 | 2026-04-23 10:15:00 | 145.63 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-05-04 13:15:00 | 146.55 | 2026-05-06 13:15:00 | 148.83 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-05-04 14:45:00 | 146.50 | 2026-05-06 13:15:00 | 148.83 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-05-05 09:15:00 | 146.00 | 2026-05-06 13:15:00 | 148.83 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-05-05 11:45:00 | 146.49 | 2026-05-06 13:15:00 | 148.83 | STOP_HIT | 1.00 | -1.60% |
