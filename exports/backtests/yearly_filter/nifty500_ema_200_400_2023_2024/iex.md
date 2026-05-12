# Indian Energy Exchange Ltd. (IEX)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 134.07
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 64 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 62 |
| PARTIAL | 21 |
| TARGET_HIT | 24 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 83 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 35
- **Target hits / Stop hits / Partials:** 24 / 38 / 21
- **Avg / median % per leg:** 2.83% / 5.00%
- **Sum % (uncompounded):** 235.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 12 | 60.0% | 12 | 8 | 0 | 4.76% | 95.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 12 | 60.0% | 12 | 8 | 0 | 4.76% | 95.1% |
| SELL (all) | 63 | 36 | 57.1% | 12 | 30 | 21 | 2.22% | 140.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 63 | 36 | 57.1% | 12 | 30 | 21 | 2.22% | 140.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 83 | 48 | 57.8% | 24 | 38 | 21 | 2.83% | 235.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 09:15:00 | 134.45 | 132.52 | 132.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 10:15:00 | 135.00 | 132.55 | 132.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 11:15:00 | 132.20 | 133.05 | 132.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 11:15:00 | 132.20 | 133.05 | 132.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 11:15:00 | 132.20 | 133.05 | 132.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 12:00:00 | 132.20 | 133.05 | 132.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 12:15:00 | 131.25 | 133.03 | 132.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 13:00:00 | 131.25 | 133.03 | 132.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2023-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 10:15:00 | 125.80 | 132.52 | 132.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 12:15:00 | 123.75 | 132.37 | 132.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 132.80 | 129.65 | 130.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 132.80 | 129.65 | 130.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 132.80 | 129.65 | 130.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 10:00:00 | 132.80 | 129.65 | 130.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 10:15:00 | 133.35 | 129.69 | 130.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 11:00:00 | 133.35 | 129.69 | 130.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 131.65 | 130.27 | 131.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:45:00 | 131.55 | 130.27 | 131.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 15:15:00 | 131.30 | 130.32 | 131.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:15:00 | 130.90 | 130.32 | 131.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 130.50 | 130.32 | 131.09 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 15:15:00 | 137.90 | 131.69 | 131.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 09:15:00 | 139.10 | 131.77 | 131.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 143.95 | 145.68 | 140.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-20 15:00:00 | 143.95 | 145.68 | 140.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 147.00 | 145.67 | 140.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 12:30:00 | 147.70 | 145.71 | 140.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 09:45:00 | 147.80 | 145.77 | 141.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-28 13:15:00 | 162.47 | 147.77 | 142.64 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 11:15:00 | 144.75 | 147.17 | 147.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 10:15:00 | 144.25 | 146.83 | 146.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 09:15:00 | 147.00 | 146.72 | 146.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 09:15:00 | 147.00 | 146.72 | 146.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 147.00 | 146.72 | 146.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:30:00 | 148.20 | 146.72 | 146.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 10:15:00 | 146.80 | 146.72 | 146.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 09:45:00 | 146.30 | 146.75 | 146.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 11:30:00 | 146.35 | 146.75 | 146.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-27 09:15:00 | 148.05 | 146.77 | 146.95 | SL hit (close>static) qty=1.00 sl=147.35 alert=retest2 |

### Cycle 5 — BUY (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 11:15:00 | 149.85 | 146.98 | 146.98 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 10:15:00 | 143.10 | 146.95 | 146.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 142.55 | 146.79 | 146.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 13:15:00 | 140.50 | 140.38 | 142.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-01 14:00:00 | 140.50 | 140.38 | 142.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 14:15:00 | 142.30 | 140.46 | 142.91 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 13:15:00 | 148.90 | 144.44 | 144.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 14:15:00 | 149.00 | 144.49 | 144.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 10:15:00 | 149.50 | 149.88 | 147.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 10:45:00 | 150.25 | 149.88 | 147.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 148.95 | 149.87 | 147.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 09:45:00 | 150.20 | 149.83 | 147.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 11:15:00 | 150.00 | 149.83 | 147.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-09 09:15:00 | 145.70 | 149.74 | 147.67 | SL hit (close<static) qty=1.00 sl=147.30 alert=retest2 |

### Cycle 8 — SELL (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 09:15:00 | 180.13 | 197.20 | 197.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 178.10 | 195.99 | 196.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 11:15:00 | 175.40 | 175.29 | 182.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-28 11:45:00 | 175.76 | 175.29 | 182.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 178.32 | 175.71 | 182.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 10:15:00 | 177.61 | 175.71 | 182.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 11:45:00 | 178.17 | 175.75 | 182.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 12:30:00 | 178.08 | 175.78 | 182.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 14:45:00 | 178.17 | 175.83 | 182.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 12:15:00 | 183.01 | 176.22 | 181.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 12:45:00 | 183.26 | 176.22 | 181.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 13:15:00 | 184.74 | 176.31 | 181.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-06 13:15:00 | 184.74 | 176.31 | 181.91 | SL hit (close>static) qty=1.00 sl=183.35 alert=retest2 |

### Cycle 9 — BUY (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 12:15:00 | 172.00 | 170.73 | 170.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-07 14:15:00 | 173.80 | 170.77 | 170.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 193.49 | 197.77 | 191.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 193.49 | 197.77 | 191.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 193.30 | 197.73 | 191.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 191.99 | 197.73 | 191.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 192.94 | 197.60 | 191.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 189.96 | 197.60 | 191.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 190.04 | 197.36 | 191.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 190.04 | 197.36 | 191.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 190.70 | 197.30 | 191.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:15:00 | 190.50 | 197.30 | 191.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 190.10 | 196.01 | 190.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 189.22 | 196.01 | 190.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 189.95 | 191.86 | 189.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 189.95 | 191.86 | 189.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 190.00 | 191.84 | 189.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:45:00 | 189.90 | 191.84 | 189.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 189.94 | 191.82 | 189.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:00:00 | 189.94 | 191.82 | 189.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 189.46 | 191.80 | 189.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:45:00 | 189.37 | 191.80 | 189.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 188.66 | 191.77 | 189.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 186.92 | 191.77 | 189.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 190.46 | 191.50 | 189.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:15:00 | 191.14 | 191.50 | 189.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 11:30:00 | 191.25 | 191.48 | 189.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 12:15:00 | 191.52 | 191.48 | 189.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:30:00 | 191.01 | 191.40 | 189.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-09 12:15:00 | 210.25 | 193.84 | 191.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 14:15:00 | 143.63 | 191.33 | 191.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 140.38 | 190.37 | 191.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 147.28 | 146.60 | 156.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 147.28 | 146.60 | 156.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 148.15 | 140.97 | 146.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:00:00 | 148.15 | 140.97 | 146.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 147.42 | 141.03 | 146.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:15:00 | 147.08 | 141.03 | 146.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:45:00 | 146.59 | 141.32 | 146.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:45:00 | 147.31 | 141.44 | 146.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:00:00 | 147.39 | 141.88 | 146.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 148.07 | 142.05 | 147.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 148.60 | 142.05 | 147.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 148.67 | 142.11 | 147.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 148.67 | 142.11 | 147.00 | SL hit (close>static) qty=1.00 sl=148.49 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-12-21 12:30:00 | 147.70 | 2023-12-28 13:15:00 | 162.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-22 09:45:00 | 147.80 | 2023-12-28 13:15:00 | 162.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-17 15:00:00 | 147.60 | 2024-01-18 09:15:00 | 137.45 | STOP_HIT | 1.00 | -6.88% |
| BUY | retest2 | 2024-01-31 09:30:00 | 147.85 | 2024-02-05 14:15:00 | 146.60 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-02-05 09:30:00 | 147.85 | 2024-02-07 10:15:00 | 144.45 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-02-06 09:15:00 | 150.75 | 2024-02-08 10:15:00 | 145.85 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2024-02-08 09:30:00 | 147.85 | 2024-02-12 10:15:00 | 140.55 | STOP_HIT | 1.00 | -4.94% |
| SELL | retest2 | 2024-02-26 09:45:00 | 146.30 | 2024-02-27 09:15:00 | 148.05 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-02-26 11:30:00 | 146.35 | 2024-02-27 09:15:00 | 148.05 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-02-27 13:15:00 | 146.30 | 2024-03-02 09:15:00 | 150.85 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-03-06 11:30:00 | 145.95 | 2024-03-06 14:15:00 | 147.70 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-05-08 09:45:00 | 150.20 | 2024-05-09 09:15:00 | 145.70 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2024-05-08 11:15:00 | 150.00 | 2024-05-09 09:15:00 | 145.70 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2024-05-17 14:15:00 | 150.50 | 2024-05-21 09:15:00 | 165.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 10:00:00 | 152.10 | 2024-06-06 10:15:00 | 167.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-22 10:15:00 | 172.19 | 2024-07-29 12:15:00 | 189.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-22 13:00:00 | 171.40 | 2024-07-29 12:15:00 | 188.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-22 14:15:00 | 171.39 | 2024-07-29 12:15:00 | 188.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-24 09:45:00 | 173.00 | 2024-07-29 12:15:00 | 190.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-14 11:45:00 | 201.46 | 2024-10-14 12:15:00 | 199.80 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-12-04 10:15:00 | 177.61 | 2024-12-06 13:15:00 | 184.74 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2024-12-04 11:45:00 | 178.17 | 2024-12-06 13:15:00 | 184.74 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2024-12-04 12:30:00 | 178.08 | 2024-12-06 13:15:00 | 184.74 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2024-12-04 14:45:00 | 178.17 | 2024-12-06 13:15:00 | 184.74 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2024-12-19 09:15:00 | 180.60 | 2024-12-19 11:15:00 | 183.50 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-12-19 10:00:00 | 181.89 | 2024-12-19 11:15:00 | 183.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-12-20 10:00:00 | 182.57 | 2024-12-20 10:15:00 | 184.70 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-12-20 12:45:00 | 181.77 | 2025-01-07 09:15:00 | 172.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-30 13:15:00 | 179.45 | 2025-01-07 09:15:00 | 172.05 | PARTIAL | 0.50 | 4.12% |
| SELL | retest2 | 2025-01-01 09:15:00 | 180.65 | 2025-01-07 09:15:00 | 171.81 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-01-01 10:45:00 | 181.11 | 2025-01-07 09:15:00 | 171.90 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2025-01-01 11:15:00 | 180.85 | 2025-01-07 09:15:00 | 171.71 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-01-03 10:15:00 | 180.95 | 2025-01-07 10:15:00 | 171.62 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2025-01-03 12:00:00 | 180.75 | 2025-01-07 11:15:00 | 170.48 | PARTIAL | 0.50 | 5.68% |
| SELL | retest2 | 2024-12-20 12:45:00 | 181.77 | 2025-01-13 12:15:00 | 163.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-30 13:15:00 | 179.45 | 2025-01-13 12:15:00 | 163.00 | TARGET_HIT | 0.50 | 9.17% |
| SELL | retest2 | 2025-01-01 09:15:00 | 180.65 | 2025-01-13 12:15:00 | 162.85 | TARGET_HIT | 0.50 | 9.85% |
| SELL | retest2 | 2025-01-01 10:45:00 | 181.11 | 2025-01-13 13:15:00 | 161.50 | TARGET_HIT | 0.50 | 10.82% |
| SELL | retest2 | 2025-01-01 11:15:00 | 180.85 | 2025-01-13 13:15:00 | 162.59 | TARGET_HIT | 0.50 | 10.10% |
| SELL | retest2 | 2025-01-03 10:15:00 | 180.95 | 2025-01-13 13:15:00 | 162.76 | TARGET_HIT | 0.50 | 10.05% |
| SELL | retest2 | 2025-01-03 12:00:00 | 180.75 | 2025-01-13 13:15:00 | 162.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 10:15:00 | 180.78 | 2025-02-07 10:15:00 | 183.92 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-02-10 09:15:00 | 179.10 | 2025-02-11 12:15:00 | 170.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 10:30:00 | 173.50 | 2025-02-14 10:15:00 | 164.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 11:00:00 | 173.00 | 2025-02-14 11:15:00 | 164.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 179.10 | 2025-02-17 09:15:00 | 161.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-11 10:30:00 | 173.50 | 2025-02-27 11:15:00 | 156.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-11 11:00:00 | 173.00 | 2025-02-27 11:15:00 | 155.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-07 09:45:00 | 172.77 | 2025-04-07 12:15:00 | 172.00 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-04-07 12:00:00 | 172.33 | 2025-04-07 12:15:00 | 172.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-06-27 10:15:00 | 191.14 | 2025-07-09 12:15:00 | 210.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-27 11:30:00 | 191.25 | 2025-07-09 12:15:00 | 210.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-27 12:15:00 | 191.52 | 2025-07-09 12:15:00 | 210.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-30 09:30:00 | 191.01 | 2025-07-09 12:15:00 | 210.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-24 13:15:00 | 147.08 | 2025-10-29 10:15:00 | 148.67 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-10-27 10:45:00 | 146.59 | 2025-10-29 10:15:00 | 148.67 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-10-27 12:45:00 | 147.31 | 2025-10-29 10:15:00 | 148.67 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-10-28 14:00:00 | 147.39 | 2025-10-29 10:15:00 | 148.67 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-10-30 11:30:00 | 146.15 | 2025-10-31 14:15:00 | 138.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 11:30:00 | 146.15 | 2025-11-20 09:15:00 | 140.75 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2025-12-01 11:00:00 | 146.20 | 2025-12-02 10:15:00 | 147.66 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-12-01 14:30:00 | 146.16 | 2025-12-02 10:15:00 | 147.66 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-12-02 09:15:00 | 145.65 | 2025-12-02 10:15:00 | 147.66 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-12-08 12:15:00 | 142.32 | 2025-12-26 09:15:00 | 135.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 10:15:00 | 142.21 | 2025-12-26 09:15:00 | 135.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 10:45:00 | 142.05 | 2025-12-26 09:15:00 | 134.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 11:45:00 | 142.70 | 2025-12-26 09:15:00 | 135.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-15 15:00:00 | 142.22 | 2025-12-26 09:15:00 | 135.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 11:15:00 | 142.03 | 2025-12-26 09:15:00 | 134.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 14:15:00 | 142.17 | 2025-12-26 09:15:00 | 135.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 09:45:00 | 142.10 | 2025-12-26 09:15:00 | 134.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 12:15:00 | 142.32 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.72% |
| SELL | retest2 | 2025-12-12 10:15:00 | 142.21 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.80% |
| SELL | retest2 | 2025-12-12 10:45:00 | 142.05 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.92% |
| SELL | retest2 | 2025-12-12 11:45:00 | 142.70 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.43% |
| SELL | retest2 | 2025-12-15 15:00:00 | 142.22 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.79% |
| SELL | retest2 | 2025-12-23 11:15:00 | 142.03 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.94% |
| SELL | retest2 | 2025-12-23 14:15:00 | 142.17 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.83% |
| SELL | retest2 | 2025-12-24 09:45:00 | 142.10 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.88% |
| SELL | retest2 | 2026-01-19 13:00:00 | 137.42 | 2026-01-20 13:15:00 | 130.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 14:45:00 | 137.77 | 2026-01-20 13:15:00 | 130.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 13:00:00 | 137.42 | 2026-02-01 12:15:00 | 123.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 14:45:00 | 137.77 | 2026-02-01 12:15:00 | 123.99 | TARGET_HIT | 0.50 | 10.00% |
