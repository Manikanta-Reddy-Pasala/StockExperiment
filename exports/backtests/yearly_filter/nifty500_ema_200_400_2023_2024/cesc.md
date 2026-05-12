# CESC Ltd. (CESC)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 185.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 2 |
| ALERT3 | 60 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 43 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 35
- **Target hits / Stop hits / Partials:** 4 / 39 / 5
- **Avg / median % per leg:** 0.00% / -1.29%
- **Sum % (uncompounded):** 0.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 4 | 22.2% | 4 | 14 | 0 | 1.00% | 17.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 4 | 22.2% | 4 | 14 | 0 | 1.00% | 17.9% |
| SELL (all) | 30 | 9 | 30.0% | 0 | 25 | 5 | -0.60% | -17.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 9 | 30.0% | 0 | 25 | 5 | -0.60% | -17.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 48 | 13 | 27.1% | 4 | 39 | 5 | 0.00% | 0.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 71.75 | 70.31 | 70.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 09:15:00 | 72.30 | 70.67 | 70.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 14:15:00 | 71.15 | 71.35 | 70.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-28 15:00:00 | 71.15 | 71.35 | 70.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 70.90 | 71.34 | 70.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 09:15:00 | 71.40 | 71.34 | 70.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-31 14:15:00 | 78.54 | 74.53 | 73.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 11:15:00 | 114.85 | 125.33 | 125.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 14:15:00 | 114.30 | 125.03 | 125.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 11:15:00 | 123.70 | 123.21 | 124.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-27 12:00:00 | 123.70 | 123.21 | 124.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 12:15:00 | 123.30 | 123.21 | 124.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 12:45:00 | 124.15 | 123.21 | 124.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 125.65 | 123.23 | 124.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 13:00:00 | 124.50 | 123.29 | 124.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-01 12:30:00 | 125.00 | 123.35 | 124.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 12:15:00 | 135.45 | 124.33 | 124.62 | SL hit (close>static) qty=1.00 sl=132.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 10:15:00 | 136.60 | 124.93 | 124.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 12:15:00 | 140.80 | 125.91 | 125.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 12:15:00 | 140.40 | 140.48 | 135.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-09 13:00:00 | 140.40 | 140.48 | 135.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 10:15:00 | 137.50 | 140.50 | 135.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:45:00 | 136.10 | 140.50 | 135.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 134.65 | 143.77 | 139.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 134.65 | 143.77 | 139.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 128.90 | 143.62 | 139.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 128.90 | 143.62 | 139.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 133.85 | 143.52 | 139.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:15:00 | 134.75 | 143.52 | 139.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 12:30:00 | 134.45 | 142.75 | 139.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-07 15:15:00 | 147.90 | 142.74 | 139.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 12:15:00 | 174.24 | 185.40 | 185.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 13:15:00 | 173.54 | 185.28 | 185.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 184.80 | 179.74 | 182.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 184.80 | 179.74 | 182.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 184.80 | 179.74 | 182.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:30:00 | 188.40 | 179.74 | 182.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 183.70 | 179.78 | 182.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 11:15:00 | 183.65 | 179.78 | 182.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 11:45:00 | 183.62 | 179.82 | 182.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 11:15:00 | 187.25 | 180.19 | 182.19 | SL hit (close>static) qty=1.00 sl=186.39 alert=retest2 |

### Cycle 5 — BUY (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 11:15:00 | 194.65 | 183.96 | 183.92 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 174.69 | 184.64 | 184.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 10:15:00 | 172.14 | 183.92 | 184.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 15:15:00 | 138.30 | 137.68 | 148.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-05 09:15:00 | 138.52 | 137.68 | 148.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 147.22 | 138.86 | 146.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 11:00:00 | 147.22 | 138.86 | 146.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 11:15:00 | 148.97 | 138.96 | 146.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 12:00:00 | 148.97 | 138.96 | 146.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 145.32 | 139.42 | 146.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:30:00 | 145.75 | 139.42 | 146.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 146.28 | 139.49 | 146.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:30:00 | 146.71 | 139.49 | 146.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 146.74 | 139.56 | 146.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 12:45:00 | 147.10 | 139.56 | 146.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 146.41 | 139.63 | 146.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 13:30:00 | 146.69 | 139.63 | 146.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 146.89 | 139.70 | 146.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 14:30:00 | 146.72 | 139.70 | 146.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 13:15:00 | 147.90 | 146.32 | 148.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 13:30:00 | 148.78 | 146.32 | 148.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 149.45 | 146.35 | 148.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 14:30:00 | 149.52 | 146.35 | 148.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 149.68 | 146.38 | 148.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 152.45 | 146.38 | 148.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 149.03 | 146.74 | 148.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 11:00:00 | 149.03 | 146.74 | 148.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 11:15:00 | 149.82 | 146.77 | 148.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 11:45:00 | 149.84 | 146.77 | 148.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 12:15:00 | 151.05 | 146.82 | 148.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 12:45:00 | 151.14 | 146.82 | 148.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 11:15:00 | 161.08 | 149.60 | 149.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 162.73 | 153.82 | 152.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 163.25 | 163.97 | 159.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:30:00 | 163.20 | 163.97 | 159.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 163.13 | 165.76 | 162.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:45:00 | 162.70 | 165.76 | 162.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 161.84 | 165.50 | 162.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 161.84 | 165.50 | 162.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 162.27 | 165.47 | 162.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 161.46 | 165.47 | 162.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 161.76 | 165.43 | 162.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 161.76 | 165.43 | 162.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 160.93 | 165.39 | 162.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 161.77 | 165.39 | 162.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 162.35 | 165.29 | 162.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:30:00 | 162.63 | 165.29 | 162.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 163.18 | 165.21 | 162.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:15:00 | 163.39 | 165.17 | 162.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-04 09:15:00 | 179.73 | 168.69 | 164.93 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 11:15:00 | 164.07 | 168.82 | 168.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 163.56 | 168.68 | 168.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 09:15:00 | 168.70 | 168.08 | 168.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 168.70 | 168.08 | 168.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 168.70 | 168.08 | 168.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 170.10 | 168.08 | 168.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 167.81 | 168.08 | 168.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 13:45:00 | 166.26 | 168.03 | 168.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 11:15:00 | 157.95 | 167.65 | 168.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 167.60 | 161.76 | 164.38 | SL hit (close>ema200) qty=0.50 sl=161.76 alert=retest2 |

### Cycle 9 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 171.11 | 165.49 | 165.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 09:15:00 | 171.41 | 165.75 | 165.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 09:15:00 | 173.24 | 174.31 | 171.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 10:00:00 | 173.24 | 174.31 | 171.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 171.85 | 174.21 | 171.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 170.64 | 174.21 | 171.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 171.30 | 174.19 | 171.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:45:00 | 171.28 | 174.19 | 171.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 171.34 | 174.10 | 171.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:45:00 | 170.75 | 174.10 | 171.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 170.30 | 174.06 | 171.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 170.80 | 174.06 | 171.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 171.90 | 174.04 | 171.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 10:30:00 | 172.50 | 174.02 | 171.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:15:00 | 173.01 | 173.89 | 171.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:30:00 | 172.44 | 173.87 | 171.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 15:15:00 | 172.80 | 173.83 | 171.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 171.59 | 173.78 | 171.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 171.09 | 173.78 | 171.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 171.36 | 173.80 | 171.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:00:00 | 171.36 | 173.80 | 171.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 171.55 | 173.77 | 171.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:30:00 | 171.30 | 173.77 | 171.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 171.10 | 173.75 | 171.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 173.58 | 173.75 | 171.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 14:15:00 | 170.25 | 173.66 | 171.52 | SL hit (close<static) qty=1.00 sl=171.02 alert=retest2 |

### Cycle 10 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 165.65 | 170.80 | 170.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 163.70 | 169.39 | 169.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 12:15:00 | 154.66 | 154.26 | 159.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 12:30:00 | 154.57 | 154.26 | 159.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 158.86 | 154.33 | 157.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:30:00 | 160.24 | 154.33 | 157.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 158.79 | 154.38 | 157.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 11:30:00 | 157.94 | 155.10 | 158.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:00:00 | 157.39 | 155.18 | 158.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 150.04 | 155.21 | 158.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 155.34 | 155.21 | 158.02 | SL hit (close>static) qty=0.50 sl=155.21 alert=retest2 |

### Cycle 11 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 171.49 | 157.03 | 156.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 173.40 | 157.34 | 157.14 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-30 09:15:00 | 71.40 | 2023-07-31 14:15:00 | 78.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-28 13:00:00 | 124.50 | 2024-04-04 12:15:00 | 135.45 | STOP_HIT | 1.00 | -8.80% |
| SELL | retest2 | 2024-04-01 12:30:00 | 125.00 | 2024-04-04 12:15:00 | 135.45 | STOP_HIT | 1.00 | -8.36% |
| BUY | retest2 | 2024-06-04 13:15:00 | 134.75 | 2024-06-07 15:15:00 | 147.90 | TARGET_HIT | 1.00 | 9.76% |
| BUY | retest2 | 2024-06-05 12:30:00 | 134.45 | 2024-06-10 09:15:00 | 148.23 | TARGET_HIT | 1.00 | 10.25% |
| SELL | retest2 | 2024-12-03 11:15:00 | 183.65 | 2024-12-04 11:15:00 | 187.25 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-12-03 11:45:00 | 183.62 | 2024-12-04 11:15:00 | 187.25 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-06-23 12:15:00 | 163.39 | 2025-07-04 09:15:00 | 179.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 10:45:00 | 163.62 | 2025-08-11 11:15:00 | 160.96 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-08-04 13:45:00 | 163.50 | 2025-08-11 11:15:00 | 160.96 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-08-04 14:30:00 | 163.50 | 2025-08-11 11:15:00 | 160.96 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-08-26 13:45:00 | 166.26 | 2025-08-28 11:15:00 | 157.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 13:45:00 | 166.26 | 2025-09-15 09:15:00 | 167.60 | STOP_HIT | 0.50 | -0.81% |
| SELL | retest2 | 2025-09-15 10:15:00 | 166.91 | 2025-09-19 15:15:00 | 171.29 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-09-15 12:00:00 | 166.40 | 2025-09-19 15:15:00 | 171.29 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-09-17 12:15:00 | 166.88 | 2025-09-19 15:15:00 | 171.29 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-09-30 11:45:00 | 165.34 | 2025-10-08 09:15:00 | 167.39 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-09-30 12:30:00 | 165.50 | 2025-10-08 09:15:00 | 167.39 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-10-06 10:15:00 | 165.40 | 2025-10-08 09:15:00 | 167.39 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-10-07 09:45:00 | 165.56 | 2025-10-08 09:15:00 | 167.39 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-10-07 12:30:00 | 164.16 | 2025-10-08 09:15:00 | 167.39 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-11-12 10:30:00 | 172.50 | 2025-11-20 14:15:00 | 170.25 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-11-13 10:15:00 | 173.01 | 2025-11-24 09:15:00 | 169.19 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-11-13 11:30:00 | 172.44 | 2025-11-24 09:15:00 | 169.19 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-11-13 15:15:00 | 172.80 | 2025-11-24 09:15:00 | 169.19 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-11-20 09:15:00 | 173.58 | 2025-11-24 09:15:00 | 169.19 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-11-21 09:30:00 | 172.22 | 2025-11-24 09:15:00 | 169.19 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-11-26 09:15:00 | 172.39 | 2025-11-27 09:15:00 | 170.16 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-11-27 09:45:00 | 171.59 | 2025-11-27 11:15:00 | 170.99 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-12-01 09:15:00 | 172.02 | 2025-12-08 11:15:00 | 169.61 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-12-01 10:00:00 | 171.95 | 2025-12-08 11:15:00 | 169.61 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-12-16 10:15:00 | 171.63 | 2025-12-16 11:15:00 | 169.67 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-02-27 11:30:00 | 157.94 | 2026-03-02 09:15:00 | 150.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 11:30:00 | 157.94 | 2026-03-02 09:15:00 | 155.34 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2026-02-27 15:00:00 | 157.39 | 2026-03-02 09:15:00 | 149.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:00:00 | 157.39 | 2026-03-02 09:15:00 | 155.34 | STOP_HIT | 0.50 | 1.30% |
| SELL | retest2 | 2026-03-05 11:00:00 | 158.10 | 2026-03-09 09:15:00 | 150.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 11:00:00 | 158.10 | 2026-03-10 15:15:00 | 154.80 | STOP_HIT | 0.50 | 2.09% |
| SELL | retest2 | 2026-03-11 09:45:00 | 158.07 | 2026-03-12 09:15:00 | 160.57 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-03-18 12:15:00 | 156.18 | 2026-03-18 13:15:00 | 157.55 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-03-18 15:15:00 | 156.10 | 2026-03-20 09:15:00 | 157.43 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-03-19 09:30:00 | 156.21 | 2026-03-20 09:15:00 | 157.43 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-03-20 10:45:00 | 156.13 | 2026-03-23 10:15:00 | 148.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 10:45:00 | 156.13 | 2026-04-06 14:15:00 | 154.00 | STOP_HIT | 0.50 | 1.36% |
| SELL | retest2 | 2026-04-08 10:15:00 | 156.29 | 2026-04-13 09:15:00 | 159.29 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-04-08 11:45:00 | 155.70 | 2026-04-13 09:15:00 | 159.29 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2026-04-09 09:45:00 | 156.02 | 2026-04-13 09:15:00 | 159.29 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-04-09 12:30:00 | 156.00 | 2026-04-13 09:15:00 | 159.29 | STOP_HIT | 1.00 | -2.11% |
