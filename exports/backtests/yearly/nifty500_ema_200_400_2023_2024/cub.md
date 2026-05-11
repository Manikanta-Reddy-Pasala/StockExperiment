# City Union Bank Ltd. (CUB)

## Backtest Summary

- **Window:** 2022-04-07 14:15:00 → 2026-05-08 15:15:00 (7049 bars)
- **Last close:** 258.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT2_SKIP | 6 |
| ALERT3 | 53 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 46 |
| PARTIAL | 5 |
| TARGET_HIT | 16 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 47 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 32
- **Target hits / Stop hits / Partials:** 13 / 33 / 1
- **Avg / median % per leg:** 0.82% / -1.76%
- **Sum % (uncompounded):** 38.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 11 | 44.0% | 11 | 14 | 0 | 2.72% | 68.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 11 | 44.0% | 11 | 14 | 0 | 2.72% | 68.1% |
| SELL (all) | 22 | 4 | 18.2% | 2 | 19 | 1 | -1.34% | -29.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 4 | 18.2% | 2 | 19 | 1 | -1.34% | -29.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 47 | 15 | 31.9% | 13 | 33 | 1 | 0.82% | 38.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 10:15:00 | 133.55 | 131.14 | 131.14 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 122.70 | 131.09 | 131.11 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 13:15:00 | 133.15 | 129.34 | 129.34 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 15:15:00 | 125.95 | 129.35 | 129.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 124.55 | 128.84 | 129.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 15:15:00 | 129.60 | 128.40 | 128.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 15:15:00 | 129.60 | 128.40 | 128.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 15:15:00 | 129.60 | 128.40 | 128.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 09:15:00 | 126.25 | 128.40 | 128.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-10 10:15:00 | 131.50 | 128.27 | 128.76 | SL hit (close>static) qty=1.00 sl=129.80 alert=retest2 |

### Cycle 5 — BUY (started 2023-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 15:15:00 | 140.95 | 129.22 | 129.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-18 09:15:00 | 143.00 | 131.73 | 130.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 11:15:00 | 132.65 | 132.96 | 131.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-23 11:45:00 | 132.55 | 132.96 | 131.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 14:15:00 | 130.45 | 132.92 | 131.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 15:00:00 | 130.45 | 132.92 | 131.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 15:15:00 | 129.80 | 132.89 | 131.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-25 09:15:00 | 131.00 | 132.89 | 131.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-25 13:15:00 | 131.10 | 132.86 | 131.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-26 09:15:00 | 129.15 | 132.78 | 131.35 | SL hit (close<static) qty=1.00 sl=129.80 alert=retest2 |

### Cycle 6 — SELL (started 2024-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 10:15:00 | 137.70 | 145.92 | 145.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 11:15:00 | 137.20 | 145.84 | 145.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-16 12:15:00 | 140.90 | 140.40 | 142.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-16 12:45:00 | 140.60 | 140.40 | 142.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 10:15:00 | 137.00 | 134.14 | 137.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-22 11:00:00 | 137.00 | 134.14 | 137.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 11:15:00 | 136.80 | 134.16 | 137.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 13:15:00 | 136.40 | 134.19 | 137.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 14:30:00 | 136.20 | 134.33 | 137.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 09:45:00 | 136.50 | 134.45 | 137.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 11:45:00 | 136.50 | 134.48 | 137.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 138.90 | 134.57 | 137.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-01 09:15:00 | 138.90 | 134.57 | 137.18 | SL hit (close>static) qty=1.00 sl=137.60 alert=retest2 |

### Cycle 7 — BUY (started 2024-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 11:15:00 | 153.40 | 139.26 | 139.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 14:15:00 | 154.30 | 139.69 | 139.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 152.30 | 152.81 | 148.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-09 10:00:00 | 152.30 | 152.81 | 148.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 148.85 | 152.69 | 148.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 14:00:00 | 148.85 | 152.69 | 148.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 14:15:00 | 145.95 | 152.62 | 148.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 15:00:00 | 145.95 | 152.62 | 148.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 15:15:00 | 146.95 | 152.56 | 148.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 10:30:00 | 149.95 | 152.49 | 148.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 13:30:00 | 149.15 | 152.38 | 148.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 13:00:00 | 149.20 | 152.17 | 148.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 13:45:00 | 149.60 | 152.14 | 148.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 145.90 | 152.36 | 148.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:30:00 | 145.45 | 152.36 | 148.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 146.50 | 152.30 | 148.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-21 13:15:00 | 144.70 | 152.10 | 148.90 | SL hit (close<static) qty=1.00 sl=144.75 alert=retest2 |

### Cycle 8 — SELL (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 11:15:00 | 154.50 | 163.97 | 164.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 154.22 | 163.78 | 163.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 09:15:00 | 168.11 | 159.80 | 161.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 09:15:00 | 168.11 | 159.80 | 161.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 168.11 | 159.80 | 161.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:45:00 | 169.00 | 159.80 | 161.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 172.17 | 159.92 | 161.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 11:00:00 | 172.17 | 159.92 | 161.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 174.94 | 163.19 | 163.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 176.15 | 163.32 | 163.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 12:15:00 | 169.74 | 170.23 | 167.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-13 13:00:00 | 169.74 | 170.23 | 167.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 168.06 | 170.53 | 167.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 167.78 | 170.53 | 167.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 169.59 | 170.52 | 167.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:30:00 | 167.40 | 170.52 | 167.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 174.51 | 179.37 | 175.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 174.51 | 179.37 | 175.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 177.00 | 179.35 | 175.13 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 165.88 | 173.23 | 173.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 14:15:00 | 163.72 | 171.84 | 172.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 171.60 | 171.28 | 172.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 10:00:00 | 171.60 | 171.28 | 172.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 172.70 | 171.29 | 172.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:45:00 | 173.21 | 171.29 | 172.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 172.55 | 171.31 | 172.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:30:00 | 172.73 | 171.31 | 172.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 172.96 | 171.33 | 172.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:00:00 | 172.96 | 171.33 | 172.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 172.98 | 171.34 | 172.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:45:00 | 173.10 | 171.34 | 172.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 172.30 | 171.39 | 172.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:30:00 | 172.65 | 171.39 | 172.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 172.65 | 171.40 | 172.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:00:00 | 172.65 | 171.40 | 172.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 172.13 | 171.41 | 172.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 170.75 | 171.41 | 172.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 15:00:00 | 171.77 | 171.40 | 172.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 15:15:00 | 175.50 | 171.44 | 172.18 | SL hit (close>static) qty=1.00 sl=172.69 alert=retest2 |

### Cycle 11 — BUY (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 10:15:00 | 173.63 | 172.76 | 172.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 12:15:00 | 174.05 | 172.77 | 172.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 172.58 | 172.81 | 172.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 172.58 | 172.81 | 172.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 172.58 | 172.81 | 172.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 172.58 | 172.81 | 172.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 171.74 | 172.80 | 172.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 171.74 | 172.80 | 172.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 170.64 | 172.77 | 172.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:00:00 | 170.64 | 172.77 | 172.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 170.31 | 172.73 | 172.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 168.27 | 172.67 | 172.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 15:15:00 | 156.50 | 156.33 | 161.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 162.10 | 156.41 | 161.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 162.10 | 156.41 | 161.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 09:30:00 | 162.00 | 156.41 | 161.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 162.00 | 156.47 | 161.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 13:15:00 | 160.26 | 156.58 | 161.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-27 14:15:00 | 144.23 | 157.24 | 161.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 10:15:00 | 182.40 | 162.75 | 162.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 11:15:00 | 184.25 | 162.97 | 162.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 192.86 | 193.07 | 184.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:00:00 | 192.86 | 193.07 | 184.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 206.15 | 208.99 | 201.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 210.60 | 208.91 | 202.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 15:00:00 | 209.27 | 211.08 | 205.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 12:30:00 | 209.65 | 210.97 | 205.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 13:00:00 | 209.34 | 210.97 | 205.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 206.99 | 211.76 | 206.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 206.99 | 211.76 | 206.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 207.50 | 211.72 | 206.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-28 11:15:00 | 200.09 | 210.50 | 206.61 | SL hit (close<static) qty=1.00 sl=200.30 alert=retest2 |

### Cycle 14 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 246.70 | 275.28 | 275.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 246.30 | 275.00 | 275.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 257.50 | 252.87 | 260.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 14:15:00 | 255.00 | 252.98 | 260.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 255.00 | 252.98 | 260.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 14:45:00 | 256.30 | 252.98 | 260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 259.51 | 252.97 | 259.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 263.48 | 252.97 | 259.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 265.98 | 253.10 | 259.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:30:00 | 266.00 | 253.10 | 259.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 269.89 | 253.27 | 259.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 11:00:00 | 269.89 | 253.27 | 259.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 261.03 | 256.56 | 260.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 260.46 | 256.60 | 260.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:00:00 | 260.75 | 256.60 | 260.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 265.62 | 256.81 | 260.32 | SL hit (close>static) qty=1.00 sl=262.68 alert=retest2 |

### Cycle 15 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 269.10 | 263.15 | 263.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 13:15:00 | 269.85 | 263.28 | 263.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 262.00 | 263.44 | 263.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 10:15:00 | 262.00 | 263.44 | 263.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 262.00 | 263.44 | 263.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 262.00 | 263.44 | 263.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 261.80 | 263.43 | 263.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 261.75 | 263.43 | 263.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-25 12:45:00 | 136.35 | 2023-05-26 12:15:00 | 139.50 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2023-05-29 09:15:00 | 128.10 | 2023-05-29 09:15:00 | 130.44 | PARTIAL | 0.50 | -1.82% |
| SELL | retest2 | 2023-05-29 09:15:00 | 128.10 | 2023-06-01 09:15:00 | 123.89 | TARGET_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2023-10-09 09:15:00 | 126.25 | 2023-10-10 10:15:00 | 131.50 | STOP_HIT | 1.00 | -4.16% |
| BUY | retest2 | 2023-10-25 09:15:00 | 131.00 | 2023-10-26 09:15:00 | 129.15 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2023-10-25 13:15:00 | 131.10 | 2023-10-26 09:15:00 | 129.15 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2023-10-27 09:15:00 | 131.90 | 2023-11-09 13:15:00 | 145.09 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-22 13:15:00 | 136.40 | 2024-04-01 09:15:00 | 138.90 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-03-26 14:30:00 | 136.20 | 2024-04-01 09:15:00 | 138.90 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-03-28 09:45:00 | 136.50 | 2024-04-01 09:15:00 | 138.90 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-03-28 11:45:00 | 136.50 | 2024-04-01 09:15:00 | 138.90 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-05-10 10:30:00 | 149.95 | 2024-05-21 13:15:00 | 144.70 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2024-05-10 13:30:00 | 149.15 | 2024-05-21 13:15:00 | 144.70 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2024-05-13 13:00:00 | 149.20 | 2024-05-21 13:15:00 | 144.70 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2024-05-13 13:45:00 | 149.60 | 2024-05-21 13:15:00 | 144.70 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2024-06-07 10:45:00 | 147.40 | 2024-06-20 09:15:00 | 162.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-07 11:30:00 | 148.30 | 2024-06-20 09:15:00 | 163.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-07 13:30:00 | 147.50 | 2024-06-20 09:15:00 | 162.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-10 09:15:00 | 148.15 | 2024-06-20 09:15:00 | 162.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-14 14:15:00 | 151.60 | 2024-06-20 11:15:00 | 166.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-14 15:15:00 | 151.70 | 2024-06-20 11:15:00 | 166.87 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-30 13:15:00 | 170.75 | 2025-01-30 15:15:00 | 175.50 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-01-30 15:00:00 | 171.77 | 2025-01-30 15:15:00 | 175.50 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-02-01 15:15:00 | 170.22 | 2025-02-03 09:15:00 | 173.61 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-02-03 12:30:00 | 171.90 | 2025-02-03 14:15:00 | 172.90 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-02-06 11:15:00 | 175.53 | 2025-02-07 10:15:00 | 173.63 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2025-02-06 11:45:00 | 175.31 | 2025-02-07 10:15:00 | 173.63 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2025-03-21 13:15:00 | 160.26 | 2025-03-27 14:15:00 | 144.23 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-02 14:00:00 | 161.40 | 2025-04-03 11:15:00 | 162.84 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-04-02 14:45:00 | 161.69 | 2025-04-03 11:15:00 | 162.84 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-04-07 09:15:00 | 156.88 | 2025-04-11 10:15:00 | 162.82 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2025-04-08 10:30:00 | 156.95 | 2025-04-11 15:15:00 | 167.40 | STOP_HIT | 1.00 | -6.66% |
| SELL | retest2 | 2025-04-09 10:00:00 | 158.12 | 2025-04-11 15:15:00 | 167.40 | STOP_HIT | 1.00 | -5.87% |
| BUY | retest2 | 2025-07-31 11:45:00 | 210.60 | 2025-08-28 11:15:00 | 200.09 | STOP_HIT | 1.00 | -4.99% |
| BUY | retest2 | 2025-08-11 15:00:00 | 209.27 | 2025-08-28 11:15:00 | 200.09 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2025-08-12 12:30:00 | 209.65 | 2025-08-28 11:15:00 | 200.09 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest2 | 2025-08-12 13:00:00 | 209.34 | 2025-08-28 11:15:00 | 200.09 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2025-09-19 12:30:00 | 209.00 | 2025-09-22 14:15:00 | 206.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-09-22 12:15:00 | 209.43 | 2025-09-22 14:15:00 | 206.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-09-24 09:30:00 | 208.88 | 2025-10-20 12:15:00 | 229.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-26 10:15:00 | 208.64 | 2025-10-20 12:15:00 | 229.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-13 09:15:00 | 269.75 | 2026-01-28 15:15:00 | 294.03 | TARGET_HIT | 1.00 | 9.00% |
| BUY | retest2 | 2026-01-20 10:00:00 | 267.30 | 2026-01-30 09:15:00 | 296.73 | TARGET_HIT | 1.00 | 11.01% |
| BUY | retest2 | 2026-03-05 09:15:00 | 267.75 | 2026-03-06 14:15:00 | 260.70 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-03-05 10:00:00 | 266.50 | 2026-03-06 14:15:00 | 260.70 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-04-24 13:30:00 | 260.46 | 2026-04-27 09:15:00 | 265.62 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2026-04-24 14:00:00 | 260.75 | 2026-04-27 09:15:00 | 265.62 | STOP_HIT | 1.00 | -1.87% |
