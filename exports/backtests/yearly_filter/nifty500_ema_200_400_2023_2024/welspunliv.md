# Welspun Living Ltd. (WELSPUNLIV)

## Backtest Summary

- **Window:** 2023-09-21 09:15:00 → 2026-05-11 15:15:00 (4154 bars)
- **Last close:** 135.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 14 |
| ALERT2 | 13 |
| ALERT2_SKIP | 8 |
| ALERT3 | 79 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 51 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 41
- **Target hits / Stop hits / Partials:** 4 / 47 / 7
- **Avg / median % per leg:** -1.20% / -1.78%
- **Sum % (uncompounded):** -69.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 0 | 0.0% | 0 | 20 | 0 | -1.86% | -37.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 0 | 0.0% | 0 | 20 | 0 | -1.86% | -37.2% |
| SELL (all) | 38 | 17 | 44.7% | 4 | 27 | 7 | -0.86% | -32.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 38 | 17 | 44.7% | 4 | 27 | 7 | -0.86% | -32.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 58 | 17 | 29.3% | 4 | 47 | 7 | -1.20% | -69.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 14:15:00 | 141.05 | 149.48 | 149.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 14:15:00 | 140.50 | 148.46 | 148.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 148.75 | 147.22 | 148.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 09:15:00 | 148.75 | 147.22 | 148.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 148.75 | 147.22 | 148.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 10:00:00 | 148.75 | 147.22 | 148.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 10:15:00 | 149.20 | 147.24 | 148.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 10:30:00 | 150.20 | 147.24 | 148.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 09:15:00 | 154.80 | 149.19 | 149.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 11:15:00 | 158.75 | 149.73 | 149.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 12:15:00 | 151.40 | 151.60 | 150.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-30 13:00:00 | 151.40 | 151.60 | 150.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 149.50 | 151.57 | 150.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 149.50 | 151.57 | 150.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 149.70 | 151.56 | 150.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:15:00 | 148.50 | 151.56 | 150.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 15:15:00 | 150.20 | 151.38 | 150.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 10:00:00 | 149.50 | 151.36 | 150.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 149.15 | 151.34 | 150.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 10:45:00 | 148.70 | 151.34 | 150.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 149.85 | 151.16 | 150.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 11:45:00 | 151.15 | 151.16 | 150.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 13:15:00 | 150.95 | 151.15 | 150.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 147.10 | 151.08 | 150.45 | SL hit (close<static) qty=1.00 sl=148.75 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 10:15:00 | 139.55 | 149.87 | 149.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 12:15:00 | 138.80 | 149.67 | 149.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 147.95 | 146.93 | 148.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 11:15:00 | 147.95 | 146.93 | 148.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 147.95 | 146.93 | 148.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:00:00 | 147.95 | 146.93 | 148.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 146.75 | 146.93 | 148.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 13:30:00 | 145.95 | 146.92 | 148.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 145.80 | 146.91 | 148.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-18 09:30:00 | 145.45 | 146.77 | 148.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 13:15:00 | 138.65 | 145.02 | 146.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 14:15:00 | 138.51 | 144.96 | 146.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 14:15:00 | 138.18 | 144.96 | 146.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 10:15:00 | 131.35 | 143.69 | 145.96 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 12:15:00 | 151.20 | 145.34 | 145.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 10:15:00 | 153.37 | 145.65 | 145.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 10:15:00 | 172.62 | 173.85 | 165.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-14 11:00:00 | 172.62 | 173.85 | 165.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 180.75 | 185.65 | 177.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 13:45:00 | 181.09 | 185.65 | 177.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 178.85 | 185.58 | 177.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:45:00 | 177.99 | 185.58 | 177.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 177.67 | 185.05 | 177.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:30:00 | 177.40 | 185.05 | 177.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 177.25 | 184.97 | 177.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:00:00 | 177.25 | 184.97 | 177.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 179.97 | 184.92 | 177.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:30:00 | 178.04 | 184.92 | 177.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 178.74 | 184.54 | 177.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:30:00 | 178.15 | 184.46 | 177.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 175.60 | 184.37 | 177.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:15:00 | 175.45 | 184.37 | 177.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 177.76 | 184.07 | 177.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:15:00 | 178.20 | 184.07 | 177.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 178.20 | 184.02 | 177.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 181.68 | 184.02 | 177.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 13:45:00 | 178.96 | 183.90 | 177.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 15:15:00 | 179.50 | 183.84 | 177.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 176.31 | 183.72 | 177.79 | SL hit (close<static) qty=1.00 sl=177.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 14:15:00 | 163.59 | 174.33 | 174.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 09:15:00 | 161.39 | 172.16 | 173.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 15:15:00 | 160.80 | 160.77 | 165.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-07 09:15:00 | 159.77 | 160.77 | 165.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 159.79 | 155.08 | 159.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 14:30:00 | 160.18 | 155.08 | 159.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 160.20 | 155.13 | 159.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:15:00 | 160.85 | 155.13 | 159.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 160.90 | 155.18 | 159.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:30:00 | 160.98 | 155.18 | 159.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 160.75 | 155.24 | 159.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:30:00 | 160.60 | 155.24 | 159.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 159.86 | 155.29 | 159.87 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 12:15:00 | 174.73 | 163.05 | 163.02 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 15:15:00 | 156.53 | 163.17 | 163.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 154.27 | 162.80 | 163.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 09:15:00 | 162.76 | 162.12 | 162.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 162.76 | 162.12 | 162.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 162.76 | 162.12 | 162.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:30:00 | 162.94 | 162.12 | 162.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 162.05 | 162.12 | 162.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 11:15:00 | 161.33 | 162.12 | 162.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 11:15:00 | 163.42 | 162.13 | 162.63 | SL hit (close>static) qty=1.00 sl=162.79 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 14:15:00 | 144.11 | 132.19 | 132.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 146.37 | 133.24 | 132.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 138.80 | 141.11 | 137.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 138.80 | 141.11 | 137.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 138.80 | 141.11 | 137.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 137.57 | 141.11 | 137.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 137.70 | 141.08 | 137.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 137.70 | 141.08 | 137.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 138.01 | 141.05 | 137.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 140.70 | 139.34 | 137.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 11:15:00 | 136.98 | 139.33 | 137.15 | SL hit (close<static) qty=1.00 sl=137.48 alert=retest2 |

### Cycle 9 — SELL (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 14:15:00 | 127.36 | 135.94 | 135.97 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 12:15:00 | 142.87 | 136.01 | 136.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 15:15:00 | 143.70 | 136.23 | 136.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 09:15:00 | 137.79 | 139.60 | 138.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 137.79 | 139.60 | 138.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 137.79 | 139.60 | 138.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 137.94 | 139.60 | 138.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 137.51 | 139.58 | 138.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 137.51 | 139.58 | 138.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 138.50 | 139.55 | 138.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:30:00 | 138.00 | 139.55 | 138.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 138.15 | 139.54 | 138.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 138.15 | 139.54 | 138.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 138.32 | 139.52 | 138.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 138.60 | 139.52 | 138.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 13:15:00 | 137.30 | 139.82 | 138.61 | SL hit (close<static) qty=1.00 sl=137.88 alert=retest2 |

### Cycle 11 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 124.11 | 137.66 | 137.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 123.95 | 137.52 | 137.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 124.55 | 118.86 | 124.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 124.55 | 118.86 | 124.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 124.55 | 118.86 | 124.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 125.59 | 118.86 | 124.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 124.12 | 118.92 | 124.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:30:00 | 123.19 | 119.24 | 124.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 122.60 | 119.33 | 124.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:15:00 | 123.19 | 119.37 | 124.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 121.71 | 119.48 | 124.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 124.18 | 119.72 | 124.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 124.80 | 119.72 | 124.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 124.15 | 119.76 | 124.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 124.30 | 119.76 | 124.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 124.09 | 119.80 | 124.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 124.09 | 119.80 | 124.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 124.44 | 119.89 | 124.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:00:00 | 124.44 | 119.89 | 124.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 124.56 | 119.94 | 124.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 15:15:00 | 123.99 | 119.94 | 124.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 128.72 | 120.06 | 124.33 | SL hit (close>static) qty=1.00 sl=125.91 alert=retest2 |

### Cycle 12 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 132.44 | 123.62 | 123.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 11:15:00 | 132.75 | 123.97 | 123.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 131.50 | 131.57 | 128.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 131.50 | 131.57 | 128.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 129.54 | 133.94 | 130.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 128.47 | 133.94 | 130.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 131.35 | 133.91 | 130.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 11:15:00 | 131.51 | 133.91 | 130.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 11:45:00 | 131.68 | 133.89 | 130.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 15:00:00 | 132.00 | 133.83 | 130.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 14:15:00 | 131.51 | 134.65 | 132.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 133.29 | 134.40 | 132.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 133.29 | 134.40 | 132.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 131.66 | 134.36 | 132.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 131.39 | 134.36 | 132.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 131.84 | 134.34 | 132.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:15:00 | 131.54 | 134.34 | 132.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 132.15 | 134.10 | 132.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:00:00 | 132.15 | 134.10 | 132.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 132.01 | 134.04 | 132.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 132.00 | 134.02 | 132.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 132.14 | 134.00 | 132.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:30:00 | 132.10 | 134.00 | 132.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 131.82 | 133.85 | 132.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 131.82 | 133.85 | 132.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 131.49 | 133.82 | 132.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 129.95 | 133.82 | 132.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 130.28 | 133.75 | 132.37 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 13:15:00 | 129.22 | 133.63 | 132.33 | SL hit (close<static) qty=1.00 sl=129.54 alert=retest2 |

### Cycle 13 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 122.00 | 131.23 | 131.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 13:15:00 | 120.56 | 131.04 | 131.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 10:15:00 | 127.78 | 126.73 | 128.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 10:15:00 | 127.78 | 126.73 | 128.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 127.78 | 126.73 | 128.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:30:00 | 128.09 | 126.73 | 128.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 128.19 | 126.75 | 128.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:00:00 | 128.19 | 126.75 | 128.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 126.16 | 126.72 | 128.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 125.15 | 126.71 | 128.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 10:15:00 | 125.69 | 126.46 | 128.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 13:30:00 | 124.96 | 126.46 | 128.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:45:00 | 125.77 | 126.40 | 128.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 143.59 | 126.15 | 128.01 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 143.59 | 126.15 | 128.01 | SL hit (close>static) qty=1.00 sl=129.38 alert=retest2 |

### Cycle 14 — BUY (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 11:15:00 | 139.50 | 129.78 | 129.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 140.15 | 129.98 | 129.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 11:15:00 | 132.69 | 135.26 | 133.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 11:15:00 | 132.69 | 135.26 | 133.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 132.69 | 135.26 | 133.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 132.69 | 135.26 | 133.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 132.60 | 135.24 | 133.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:00:00 | 132.60 | 135.24 | 133.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 135.30 | 135.20 | 133.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:30:00 | 132.58 | 135.15 | 133.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 130.81 | 135.11 | 133.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 131.65 | 135.11 | 133.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 127.70 | 134.99 | 133.06 | SL hit (close<static) qty=1.00 sl=129.23 alert=retest2 |

### Cycle 15 — SELL (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 13:15:00 | 116.76 | 131.42 | 131.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 116.47 | 130.36 | 130.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 11:15:00 | 118.73 | 118.50 | 123.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 12:00:00 | 118.73 | 118.50 | 123.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 123.74 | 118.53 | 122.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 122.50 | 118.83 | 122.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:15:00 | 122.36 | 118.96 | 122.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 120.75 | 119.35 | 122.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 122.35 | 119.51 | 122.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 122.35 | 119.54 | 122.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:15:00 | 124.24 | 119.54 | 122.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 124.90 | 119.59 | 122.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 124.90 | 119.59 | 122.79 | SL hit (close>static) qty=1.00 sl=124.89 alert=retest2 |

### Cycle 16 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 132.32 | 124.89 | 124.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 133.79 | 125.06 | 124.95 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-06 11:45:00 | 151.15 | 2024-05-07 09:15:00 | 147.10 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2024-05-06 13:15:00 | 150.95 | 2024-05-07 09:15:00 | 147.10 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-05-16 13:30:00 | 145.95 | 2024-05-30 13:15:00 | 138.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-16 15:15:00 | 145.80 | 2024-05-30 14:15:00 | 138.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-18 09:30:00 | 145.45 | 2024-05-30 14:15:00 | 138.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-16 13:30:00 | 145.95 | 2024-06-04 10:15:00 | 131.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-16 15:15:00 | 145.80 | 2024-06-04 10:15:00 | 131.22 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-18 09:30:00 | 145.45 | 2024-06-04 10:15:00 | 130.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-19 10:00:00 | 146.06 | 2024-06-26 10:15:00 | 149.06 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-09-18 09:15:00 | 181.68 | 2024-09-19 09:15:00 | 176.31 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2024-09-18 13:45:00 | 178.96 | 2024-09-19 09:15:00 | 176.31 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-09-18 15:15:00 | 179.50 | 2024-09-19 09:15:00 | 176.31 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-01-02 11:15:00 | 161.33 | 2025-01-02 11:15:00 | 163.42 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-01-06 09:15:00 | 160.15 | 2025-01-09 12:15:00 | 152.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 09:15:00 | 160.15 | 2025-01-13 11:15:00 | 144.14 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-05 09:15:00 | 140.70 | 2025-06-06 11:15:00 | 136.98 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-06-10 09:15:00 | 138.78 | 2025-06-12 11:15:00 | 137.13 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-10 11:00:00 | 138.75 | 2025-06-12 11:15:00 | 137.13 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-06-10 12:45:00 | 138.44 | 2025-06-12 11:15:00 | 137.13 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-07-14 15:15:00 | 138.60 | 2025-07-23 13:15:00 | 137.30 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-07-24 09:15:00 | 139.65 | 2025-07-25 12:15:00 | 137.23 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-07-24 10:30:00 | 138.65 | 2025-07-25 12:15:00 | 137.23 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-07-24 11:30:00 | 138.57 | 2025-07-25 12:15:00 | 137.23 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-07-24 15:00:00 | 140.87 | 2025-07-25 12:15:00 | 137.23 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-09-11 09:30:00 | 123.19 | 2025-09-17 09:15:00 | 128.72 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-09-11 11:30:00 | 122.60 | 2025-09-17 09:15:00 | 128.72 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2025-09-11 13:15:00 | 123.19 | 2025-09-17 09:15:00 | 128.72 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-09-12 09:15:00 | 121.71 | 2025-09-17 09:15:00 | 128.72 | STOP_HIT | 1.00 | -5.76% |
| SELL | retest2 | 2025-09-16 15:15:00 | 123.99 | 2025-09-17 09:15:00 | 128.72 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2025-09-22 09:15:00 | 123.90 | 2025-09-26 09:15:00 | 117.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:00:00 | 123.81 | 2025-09-26 09:15:00 | 117.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:30:00 | 123.60 | 2025-09-26 09:15:00 | 117.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 123.90 | 2025-10-09 09:15:00 | 119.07 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2025-09-22 12:00:00 | 123.81 | 2025-10-09 09:15:00 | 119.07 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2025-09-22 13:30:00 | 123.60 | 2025-10-09 09:15:00 | 119.07 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2025-10-15 09:30:00 | 121.41 | 2025-10-15 14:15:00 | 125.18 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-10-17 13:00:00 | 121.46 | 2025-10-23 09:15:00 | 129.99 | STOP_HIT | 1.00 | -7.02% |
| SELL | retest2 | 2025-10-17 15:15:00 | 121.45 | 2025-10-23 09:15:00 | 129.99 | STOP_HIT | 1.00 | -7.03% |
| BUY | retest2 | 2025-12-09 11:15:00 | 131.51 | 2026-01-06 13:15:00 | 129.22 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-12-09 11:45:00 | 131.68 | 2026-01-06 13:15:00 | 129.22 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-12-09 15:00:00 | 132.00 | 2026-01-06 13:15:00 | 129.22 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-12-29 14:15:00 | 131.51 | 2026-01-06 13:15:00 | 129.22 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-01-08 09:30:00 | 131.49 | 2026-01-08 10:15:00 | 128.78 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-01-29 09:15:00 | 125.15 | 2026-02-03 09:15:00 | 143.59 | STOP_HIT | 1.00 | -14.73% |
| SELL | retest2 | 2026-01-30 10:15:00 | 125.69 | 2026-02-03 09:15:00 | 143.59 | STOP_HIT | 1.00 | -14.24% |
| SELL | retest2 | 2026-01-30 13:30:00 | 124.96 | 2026-02-03 09:15:00 | 143.59 | STOP_HIT | 1.00 | -14.91% |
| SELL | retest2 | 2026-02-01 11:45:00 | 125.77 | 2026-02-03 09:15:00 | 143.59 | STOP_HIT | 1.00 | -14.17% |
| SELL | retest2 | 2026-02-04 09:15:00 | 138.80 | 2026-02-04 09:15:00 | 148.88 | STOP_HIT | 1.00 | -7.26% |
| SELL | retest2 | 2026-02-05 09:30:00 | 140.86 | 2026-02-06 11:15:00 | 139.50 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2026-02-05 10:00:00 | 142.02 | 2026-02-06 11:15:00 | 139.50 | STOP_HIT | 1.00 | 1.77% |
| SELL | retest2 | 2026-02-05 10:45:00 | 141.12 | 2026-02-06 11:15:00 | 139.50 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2026-02-25 11:15:00 | 131.65 | 2026-02-25 12:15:00 | 127.70 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2026-04-09 09:30:00 | 122.50 | 2026-04-15 09:15:00 | 124.90 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-04-09 13:15:00 | 122.36 | 2026-04-15 09:15:00 | 124.90 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-04-13 09:15:00 | 120.75 | 2026-04-15 09:15:00 | 124.90 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2026-04-13 15:15:00 | 122.35 | 2026-04-15 09:15:00 | 124.90 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-04-16 11:45:00 | 124.04 | 2026-04-17 09:15:00 | 126.42 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-04-16 14:45:00 | 123.92 | 2026-04-17 09:15:00 | 126.42 | STOP_HIT | 1.00 | -2.02% |
