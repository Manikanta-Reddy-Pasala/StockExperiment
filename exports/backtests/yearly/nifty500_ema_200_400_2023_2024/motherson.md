# Samvardhana Motherson International Ltd. (MOTHERSON)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 131.57
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 4 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 4 |
| TARGET_HIT | 8 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 14 / 2
- **Target hits / Stop hits / Partials:** 8 / 4 / 4
- **Avg / median % per leg:** 6.21% / 7.35%
- **Sum % (uncompounded):** 99.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 6 | 2 | 0 | 7.16% | 57.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 6 | 75.0% | 6 | 2 | 0 | 7.16% | 57.3% |
| SELL (all) | 8 | 8 | 100.0% | 2 | 2 | 4 | 5.27% | 42.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 8 | 100.0% | 2 | 2 | 4 | 5.27% | 42.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 14 | 87.5% | 8 | 4 | 4 | 6.21% | 99.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 11:15:00 | 61.33 | 62.81 | 62.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 11:15:00 | 60.87 | 62.71 | 62.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 09:15:00 | 61.70 | 60.70 | 61.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 09:15:00 | 61.70 | 60.70 | 61.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 61.70 | 60.70 | 61.51 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-12-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 12:15:00 | 65.03 | 61.96 | 61.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 14:15:00 | 65.83 | 62.04 | 62.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 62.43 | 62.74 | 62.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 62.43 | 62.74 | 62.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 62.43 | 62.74 | 62.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 79.40 | 77.98 | 76.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-26 09:15:00 | 87.34 | 80.72 | 78.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 118.77 | 129.17 | 129.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 117.81 | 128.95 | 129.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 10:15:00 | 116.10 | 115.68 | 120.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-06 11:00:00 | 116.10 | 115.68 | 120.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 88.58 | 85.86 | 90.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 10:30:00 | 88.51 | 85.88 | 90.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 13:00:00 | 88.14 | 85.93 | 90.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 85.73 | 86.25 | 90.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 09:15:00 | 84.08 | 86.26 | 90.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-27 09:15:00 | 87.33 | 86.26 | 90.63 | SL hit (close>static) qty=0.50 sl=86.26 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 95.57 | 89.44 | 89.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 11:15:00 | 95.81 | 89.90 | 89.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 99.73 | 99.84 | 96.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 13:45:00 | 99.72 | 99.84 | 96.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 99.50 | 101.64 | 99.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:45:00 | 98.99 | 101.64 | 99.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 98.82 | 101.61 | 99.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:00:00 | 98.82 | 101.61 | 99.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 97.90 | 101.58 | 99.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 97.90 | 101.58 | 99.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 99.87 | 101.37 | 99.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 99.87 | 101.37 | 99.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 99.02 | 101.34 | 99.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:45:00 | 98.77 | 101.34 | 99.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 99.05 | 101.32 | 99.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:15:00 | 98.85 | 101.32 | 99.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 12:15:00 | 89.94 | 98.77 | 98.78 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 107.30 | 97.72 | 97.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 107.91 | 97.82 | 97.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 103.15 | 104.02 | 101.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:45:00 | 103.31 | 104.02 | 101.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 101.59 | 103.94 | 101.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 101.59 | 103.94 | 101.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 101.55 | 103.92 | 101.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 102.38 | 103.92 | 101.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 101.59 | 103.90 | 101.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:00:00 | 102.92 | 103.89 | 101.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 13:30:00 | 102.89 | 103.85 | 101.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:00:00 | 103.05 | 103.85 | 101.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 09:45:00 | 103.05 | 105.18 | 103.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 103.58 | 105.17 | 103.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 13:00:00 | 103.83 | 105.14 | 103.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 14:30:00 | 103.88 | 105.11 | 103.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 102.53 | 104.98 | 103.62 | SL hit (close<static) qty=1.00 sl=102.90 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 12:15:00 | 110.78 | 119.61 | 119.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 107.29 | 118.30 | 118.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 117.41 | 115.14 | 117.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 117.41 | 115.14 | 117.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 117.41 | 115.14 | 117.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 117.41 | 115.14 | 117.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 117.79 | 115.17 | 117.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 117.79 | 115.17 | 117.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 118.73 | 115.20 | 117.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:45:00 | 118.46 | 115.20 | 117.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 117.50 | 115.35 | 117.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:30:00 | 117.32 | 115.35 | 117.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 11:15:00 | 118.05 | 115.37 | 117.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 11:45:00 | 118.19 | 115.37 | 117.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 117.93 | 115.40 | 117.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 12:30:00 | 118.08 | 115.40 | 117.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 119.61 | 115.48 | 117.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:00:00 | 119.61 | 115.48 | 117.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 120.06 | 115.53 | 117.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:30:00 | 120.75 | 115.53 | 117.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 119.32 | 115.87 | 117.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:45:00 | 119.77 | 115.87 | 117.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 127.31 | 118.48 | 118.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 129.13 | 118.58 | 118.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 120.05 | 121.28 | 120.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 120.05 | 121.28 | 120.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 120.05 | 121.28 | 120.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 120.05 | 121.28 | 120.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 119.50 | 121.26 | 120.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 119.50 | 121.26 | 120.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 119.83 | 121.25 | 120.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:30:00 | 119.62 | 121.25 | 120.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 120.66 | 121.25 | 120.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:30:00 | 120.15 | 121.25 | 120.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 120.16 | 121.24 | 120.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:45:00 | 120.30 | 121.24 | 120.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 120.30 | 121.23 | 120.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:15:00 | 119.42 | 121.23 | 120.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 120.22 | 121.22 | 120.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 121.73 | 121.17 | 120.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 79.40 | 2024-04-26 09:15:00 | 87.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-25 10:30:00 | 88.51 | 2025-03-27 09:15:00 | 84.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 10:30:00 | 88.51 | 2025-03-27 09:15:00 | 87.33 | STOP_HIT | 0.50 | 1.33% |
| SELL | retest2 | 2025-03-25 13:00:00 | 88.14 | 2025-03-27 09:15:00 | 83.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 13:00:00 | 88.14 | 2025-03-27 09:15:00 | 87.33 | STOP_HIT | 0.50 | 0.92% |
| SELL | retest2 | 2025-03-27 09:15:00 | 85.73 | 2025-04-04 09:15:00 | 81.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 12:15:00 | 88.25 | 2025-04-04 09:15:00 | 83.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-27 09:15:00 | 85.73 | 2025-04-04 13:15:00 | 79.42 | TARGET_HIT | 0.50 | 7.35% |
| SELL | retest2 | 2025-03-28 12:15:00 | 88.25 | 2025-04-07 09:15:00 | 77.16 | TARGET_HIT | 0.50 | 12.57% |
| BUY | retest2 | 2025-10-09 11:00:00 | 102.92 | 2025-11-10 14:15:00 | 102.53 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-10-09 13:30:00 | 102.89 | 2025-11-10 14:15:00 | 102.53 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-10-13 12:00:00 | 103.05 | 2025-11-14 09:15:00 | 113.21 | TARGET_HIT | 1.00 | 9.86% |
| BUY | retest2 | 2025-11-07 09:45:00 | 103.05 | 2025-11-14 09:15:00 | 113.18 | TARGET_HIT | 1.00 | 9.83% |
| BUY | retest2 | 2025-11-07 13:00:00 | 103.83 | 2025-11-14 09:15:00 | 113.36 | TARGET_HIT | 1.00 | 9.17% |
| BUY | retest2 | 2025-11-07 14:30:00 | 103.88 | 2025-11-14 09:15:00 | 113.36 | TARGET_HIT | 1.00 | 9.12% |
| BUY | retest2 | 2025-11-11 13:30:00 | 104.45 | 2025-11-27 10:15:00 | 114.90 | TARGET_HIT | 1.00 | 10.00% |
