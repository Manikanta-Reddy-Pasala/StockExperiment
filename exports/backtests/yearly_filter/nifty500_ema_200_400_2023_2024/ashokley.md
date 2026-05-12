# Ashok Leyland Ltd. (ASHOKLEY)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 168.77
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
| ALERT2 | 11 |
| ALERT2_SKIP | 6 |
| ALERT3 | 79 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 65 |
| PARTIAL | 12 |
| TARGET_HIT | 9 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 51
- **Target hits / Stop hits / Partials:** 9 / 60 / 12
- **Avg / median % per leg:** 1.28% / -0.82%
- **Sum % (uncompounded):** 103.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 14 | 36.8% | 6 | 28 | 4 | 1.86% | 70.8% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 4.99% | 39.9% |
| BUY @ 3rd Alert (retest2) | 30 | 6 | 20.0% | 6 | 24 | 0 | 1.03% | 30.9% |
| SELL (all) | 43 | 16 | 37.2% | 3 | 32 | 8 | 0.77% | 33.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 43 | 16 | 37.2% | 3 | 32 | 8 | 0.77% | 33.1% |
| retest1 (combined) | 8 | 8 | 100.0% | 0 | 4 | 4 | 4.99% | 39.9% |
| retest2 (combined) | 73 | 22 | 30.1% | 9 | 56 | 8 | 0.88% | 64.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 10:15:00 | 85.13 | 88.38 | 88.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 11:15:00 | 84.65 | 88.34 | 88.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 13:15:00 | 86.60 | 86.17 | 87.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-08 14:00:00 | 86.60 | 86.17 | 87.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 86.90 | 86.19 | 87.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 14:45:00 | 85.30 | 86.21 | 87.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 10:45:00 | 86.18 | 86.19 | 87.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-12 18:15:00 | 88.05 | 86.24 | 87.03 | SL hit (close>static) qty=1.00 sl=87.25 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 11:15:00 | 90.40 | 87.46 | 87.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 13:15:00 | 90.68 | 87.52 | 87.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 14:15:00 | 87.30 | 87.76 | 87.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 14:15:00 | 87.30 | 87.76 | 87.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 14:15:00 | 87.30 | 87.76 | 87.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 15:00:00 | 87.30 | 87.76 | 87.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 15:15:00 | 87.25 | 87.76 | 87.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 09:15:00 | 87.50 | 87.76 | 87.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-05 11:15:00 | 86.78 | 87.75 | 87.61 | SL hit (close<static) qty=1.00 sl=87.10 alert=retest2 |

### Cycle 3 — SELL (started 2023-12-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 11:15:00 | 84.95 | 87.54 | 87.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 14:15:00 | 84.65 | 87.47 | 87.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-26 09:15:00 | 87.35 | 87.35 | 87.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-26 09:15:00 | 87.35 | 87.35 | 87.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 87.35 | 87.35 | 87.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 10:00:00 | 87.35 | 87.35 | 87.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 10:15:00 | 87.00 | 87.34 | 87.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 11:15:00 | 86.75 | 87.34 | 87.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 15:00:00 | 86.90 | 87.32 | 87.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-27 09:15:00 | 87.75 | 87.32 | 87.43 | SL hit (close>static) qty=1.00 sl=87.45 alert=retest2 |

### Cycle 4 — BUY (started 2023-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 15:15:00 | 90.58 | 87.53 | 87.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 09:15:00 | 92.85 | 87.59 | 87.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 87.78 | 88.55 | 88.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 09:15:00 | 87.78 | 88.55 | 88.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 87.78 | 88.55 | 88.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:00:00 | 87.78 | 88.55 | 88.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 87.73 | 88.54 | 88.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 11:00:00 | 87.73 | 88.54 | 88.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 88.18 | 88.49 | 88.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 10:30:00 | 88.03 | 88.49 | 88.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 15:15:00 | 88.25 | 88.48 | 88.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-12 09:15:00 | 88.10 | 88.48 | 88.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 87.88 | 88.48 | 88.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-12 09:30:00 | 87.95 | 88.48 | 88.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 10:15:00 | 88.03 | 88.47 | 88.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 13:00:00 | 88.15 | 88.46 | 88.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 12:00:00 | 88.13 | 88.44 | 88.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-16 12:15:00 | 87.65 | 88.44 | 88.12 | SL hit (close<static) qty=1.00 sl=87.80 alert=retest2 |

### Cycle 5 — SELL (started 2024-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 14:15:00 | 85.05 | 87.83 | 87.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 11:15:00 | 84.20 | 87.72 | 87.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 14:15:00 | 88.00 | 87.30 | 87.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 14:15:00 | 88.00 | 87.30 | 87.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 88.00 | 87.30 | 87.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 15:00:00 | 88.00 | 87.30 | 87.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 15:15:00 | 87.78 | 87.30 | 87.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-01 09:15:00 | 87.80 | 87.30 | 87.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 87.53 | 87.31 | 87.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 11:45:00 | 87.30 | 87.32 | 87.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 12:15:00 | 87.08 | 87.32 | 87.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 13:00:00 | 87.40 | 87.32 | 87.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 15:00:00 | 87.23 | 87.32 | 87.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 87.15 | 87.31 | 87.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 13:15:00 | 86.73 | 87.31 | 87.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 10:45:00 | 86.65 | 87.30 | 87.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-05 11:15:00 | 88.55 | 87.31 | 87.53 | SL hit (close>static) qty=1.00 sl=87.98 alert=retest2 |

### Cycle 6 — BUY (started 2024-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 13:15:00 | 87.65 | 86.15 | 86.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 10:15:00 | 87.95 | 86.21 | 86.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 13:15:00 | 86.25 | 86.32 | 86.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 13:15:00 | 86.25 | 86.32 | 86.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 13:15:00 | 86.25 | 86.32 | 86.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 14:00:00 | 86.25 | 86.32 | 86.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 14:15:00 | 84.65 | 86.30 | 86.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 15:00:00 | 84.65 | 86.30 | 86.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 15:15:00 | 85.18 | 86.29 | 86.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:15:00 | 84.33 | 86.29 | 86.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 12:15:00 | 86.28 | 86.17 | 86.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-22 12:45:00 | 86.15 | 86.17 | 86.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 14:15:00 | 86.28 | 86.17 | 86.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-22 15:00:00 | 86.28 | 86.17 | 86.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 15:15:00 | 86.53 | 86.17 | 86.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 09:15:00 | 86.85 | 86.17 | 86.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 10:45:00 | 86.93 | 86.19 | 86.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 12:15:00 | 86.73 | 86.20 | 86.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 13:45:00 | 86.83 | 86.21 | 86.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-30 09:15:00 | 95.53 | 87.43 | 86.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 10:15:00 | 111.50 | 119.83 | 119.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 14:15:00 | 111.24 | 119.49 | 119.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 14:15:00 | 111.00 | 110.03 | 113.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 15:00:00 | 111.00 | 110.03 | 113.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 114.35 | 110.09 | 113.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:30:00 | 114.68 | 110.09 | 113.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 114.99 | 110.13 | 113.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 114.99 | 110.13 | 113.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 112.28 | 110.10 | 112.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:00:00 | 112.28 | 110.10 | 112.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 114.70 | 110.20 | 112.60 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 12:15:00 | 116.00 | 114.03 | 114.03 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 11:15:00 | 111.49 | 114.02 | 114.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 12:15:00 | 111.25 | 113.99 | 114.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 09:15:00 | 114.54 | 112.22 | 112.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 114.54 | 112.22 | 112.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 114.54 | 112.22 | 112.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 114.54 | 112.22 | 112.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 115.48 | 112.25 | 112.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:00:00 | 114.44 | 112.84 | 113.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:45:00 | 114.03 | 112.85 | 113.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 13:00:00 | 114.15 | 112.87 | 113.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:15:00 | 108.72 | 112.65 | 113.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 12:15:00 | 108.33 | 112.61 | 113.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 12:15:00 | 108.44 | 112.61 | 113.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 12:15:00 | 103.00 | 111.62 | 112.53 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 10 — BUY (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 12:15:00 | 115.31 | 106.40 | 106.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 115.60 | 109.60 | 108.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 116.44 | 117.38 | 114.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 10:15:00 | 116.51 | 117.38 | 114.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 10:15:00 | 116.94 | 117.36 | 114.49 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 12:00:00 | 116.56 | 117.36 | 114.81 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:15:00 | 116.79 | 117.32 | 114.83 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 14:15:00 | 122.34 | 117.84 | 115.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 14:15:00 | 122.39 | 117.84 | 115.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 09:15:00 | 122.79 | 117.94 | 115.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 09:15:00 | 122.63 | 117.94 | 115.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 122.50 | 122.62 | 119.55 | SL hit (close<ema200) qty=0.50 sl=122.62 alert=retest1 |

### Cycle 11 — SELL (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 10:15:00 | 172.24 | 185.63 | 185.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-25 14:15:00 | 171.07 | 185.08 | 185.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 175.71 | 173.61 | 178.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-10 10:00:00 | 175.71 | 173.61 | 178.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 177.43 | 173.76 | 178.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:00:00 | 175.38 | 173.91 | 178.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:45:00 | 175.01 | 173.95 | 178.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 09:30:00 | 175.30 | 174.11 | 178.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 11:00:00 | 175.20 | 174.14 | 178.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 178.36 | 174.27 | 178.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 178.36 | 174.27 | 178.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 178.45 | 174.32 | 178.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:00:00 | 178.45 | 174.32 | 178.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 176.78 | 174.34 | 178.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 15:00:00 | 176.61 | 174.41 | 178.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 11:15:00 | 179.88 | 174.54 | 178.05 | SL hit (close>static) qty=1.00 sl=178.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-25 13:15:00 | 72.45 | 2023-06-16 09:15:00 | 79.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-11-09 14:45:00 | 85.30 | 2023-11-12 18:15:00 | 88.05 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2023-11-10 10:45:00 | 86.18 | 2023-11-12 18:15:00 | 88.05 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2023-11-20 09:30:00 | 85.88 | 2023-11-21 12:15:00 | 87.75 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2023-12-05 09:15:00 | 87.50 | 2023-12-05 11:15:00 | 86.78 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-12-05 13:15:00 | 87.43 | 2023-12-12 09:15:00 | 87.43 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2023-12-06 09:15:00 | 87.83 | 2023-12-12 12:15:00 | 86.83 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-12-08 15:00:00 | 87.55 | 2023-12-12 12:15:00 | 86.83 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-12-12 09:15:00 | 88.00 | 2023-12-12 12:15:00 | 86.83 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2023-12-14 10:30:00 | 88.05 | 2023-12-15 11:15:00 | 87.13 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-12-14 13:30:00 | 88.03 | 2023-12-15 11:15:00 | 87.13 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2023-12-14 15:00:00 | 88.08 | 2023-12-15 11:15:00 | 87.13 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-12-19 09:15:00 | 88.08 | 2023-12-19 09:15:00 | 87.30 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-12-20 09:15:00 | 88.30 | 2023-12-20 13:15:00 | 86.00 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2023-12-26 11:15:00 | 86.75 | 2023-12-27 09:15:00 | 87.75 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2023-12-26 15:00:00 | 86.90 | 2023-12-27 09:15:00 | 87.75 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-12-27 12:15:00 | 86.90 | 2023-12-28 09:15:00 | 87.55 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2023-12-27 13:00:00 | 86.88 | 2023-12-28 09:15:00 | 87.55 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-01-12 13:00:00 | 88.15 | 2024-01-16 12:15:00 | 87.65 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-01-15 12:00:00 | 88.13 | 2024-01-16 12:15:00 | 87.65 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-02-01 11:45:00 | 87.30 | 2024-02-05 11:15:00 | 88.55 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-02-01 12:15:00 | 87.08 | 2024-02-05 11:15:00 | 88.55 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-02-01 13:00:00 | 87.40 | 2024-02-05 11:15:00 | 88.55 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-02-01 15:00:00 | 87.23 | 2024-02-05 11:15:00 | 88.55 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-02-02 13:15:00 | 86.73 | 2024-02-05 11:15:00 | 88.55 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-02-05 10:45:00 | 86.65 | 2024-02-05 11:15:00 | 88.55 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-02-09 09:30:00 | 86.40 | 2024-02-16 09:15:00 | 88.03 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-02-09 13:30:00 | 86.53 | 2024-02-16 09:15:00 | 88.03 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-02-19 09:15:00 | 86.70 | 2024-02-26 13:15:00 | 87.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-02-20 13:15:00 | 86.65 | 2024-02-26 13:15:00 | 87.80 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-02-21 10:15:00 | 86.50 | 2024-02-26 13:15:00 | 87.80 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-02-21 13:15:00 | 86.55 | 2024-02-26 13:15:00 | 87.80 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-02-27 11:45:00 | 86.73 | 2024-03-13 09:15:00 | 82.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-27 13:30:00 | 86.48 | 2024-03-13 10:15:00 | 82.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-28 09:45:00 | 86.40 | 2024-03-13 10:15:00 | 82.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-04 10:45:00 | 86.43 | 2024-03-13 10:15:00 | 82.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-27 11:45:00 | 86.73 | 2024-03-26 11:15:00 | 84.75 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2024-02-27 13:30:00 | 86.48 | 2024-03-26 11:15:00 | 84.75 | STOP_HIT | 0.50 | 2.00% |
| SELL | retest2 | 2024-02-28 09:45:00 | 86.40 | 2024-03-26 11:15:00 | 84.75 | STOP_HIT | 0.50 | 1.91% |
| SELL | retest2 | 2024-03-04 10:45:00 | 86.43 | 2024-03-26 11:15:00 | 84.75 | STOP_HIT | 0.50 | 1.94% |
| SELL | retest2 | 2024-03-27 13:15:00 | 85.05 | 2024-03-28 12:15:00 | 85.83 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-03-27 14:00:00 | 84.93 | 2024-03-28 12:15:00 | 85.83 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-03-28 09:45:00 | 85.10 | 2024-03-28 12:15:00 | 85.83 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-04-23 09:15:00 | 86.85 | 2024-04-30 09:15:00 | 95.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-23 10:45:00 | 86.93 | 2024-04-30 09:15:00 | 95.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-23 12:15:00 | 86.73 | 2024-04-30 09:15:00 | 95.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-23 13:45:00 | 86.83 | 2024-04-30 09:15:00 | 95.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-17 11:15:00 | 119.55 | 2024-09-18 09:15:00 | 118.73 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-09-17 12:30:00 | 119.58 | 2024-09-18 09:15:00 | 118.73 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-09-26 10:00:00 | 119.85 | 2024-09-30 14:15:00 | 117.93 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-09-26 11:30:00 | 119.83 | 2024-09-30 14:15:00 | 117.93 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-01-06 11:00:00 | 114.44 | 2025-01-09 11:15:00 | 108.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 11:45:00 | 114.03 | 2025-01-09 12:15:00 | 108.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 13:00:00 | 114.15 | 2025-01-09 12:15:00 | 108.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 11:00:00 | 114.44 | 2025-01-13 12:15:00 | 103.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-06 11:45:00 | 114.03 | 2025-01-13 12:15:00 | 102.74 | TARGET_HIT | 0.50 | 9.91% |
| SELL | retest2 | 2025-01-06 13:00:00 | 114.15 | 2025-01-13 13:15:00 | 102.63 | TARGET_HIT | 0.50 | 10.09% |
| BUY | retest1 | 2025-06-13 10:15:00 | 116.51 | 2025-06-25 14:15:00 | 122.34 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-16 10:15:00 | 116.94 | 2025-06-25 14:15:00 | 122.39 | PARTIAL | 0.50 | 4.66% |
| BUY | retest1 | 2025-06-19 12:00:00 | 116.56 | 2025-06-26 09:15:00 | 122.79 | PARTIAL | 0.50 | 5.34% |
| BUY | retest1 | 2025-06-20 09:15:00 | 116.79 | 2025-06-26 09:15:00 | 122.63 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-13 10:15:00 | 116.51 | 2025-07-18 10:15:00 | 122.50 | STOP_HIT | 0.50 | 5.14% |
| BUY | retest1 | 2025-06-16 10:15:00 | 116.94 | 2025-07-18 10:15:00 | 122.50 | STOP_HIT | 0.50 | 4.75% |
| BUY | retest1 | 2025-06-19 12:00:00 | 116.56 | 2025-07-18 10:15:00 | 122.50 | STOP_HIT | 0.50 | 5.10% |
| BUY | retest1 | 2025-06-20 09:15:00 | 116.79 | 2025-07-18 10:15:00 | 122.50 | STOP_HIT | 0.50 | 4.89% |
| BUY | retest2 | 2025-07-31 11:30:00 | 121.30 | 2025-08-01 09:15:00 | 118.80 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-07-31 12:15:00 | 121.35 | 2025-08-01 09:15:00 | 118.80 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-07-31 13:15:00 | 121.30 | 2025-08-01 09:15:00 | 118.80 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-07-31 14:00:00 | 121.50 | 2025-08-01 09:15:00 | 118.80 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-08-04 09:15:00 | 122.25 | 2025-08-07 11:15:00 | 119.70 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-08-07 09:45:00 | 120.93 | 2025-08-07 11:15:00 | 119.70 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-08-13 10:00:00 | 120.59 | 2025-08-13 13:15:00 | 119.99 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-08-13 11:00:00 | 120.69 | 2025-08-13 13:15:00 | 119.99 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-08-18 09:15:00 | 129.60 | 2025-09-19 10:15:00 | 142.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-15 15:00:00 | 175.38 | 2026-04-22 11:15:00 | 179.88 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-04-16 09:45:00 | 175.01 | 2026-04-22 11:15:00 | 179.88 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2026-04-17 09:30:00 | 175.30 | 2026-04-22 11:15:00 | 179.88 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-04-20 11:00:00 | 175.20 | 2026-04-22 11:15:00 | 179.88 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-04-21 15:00:00 | 176.61 | 2026-04-22 11:15:00 | 179.88 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-04-23 09:15:00 | 174.40 | 2026-04-29 11:15:00 | 165.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 09:15:00 | 174.40 | 2026-05-07 12:15:00 | 172.80 | STOP_HIT | 0.50 | 0.92% |
