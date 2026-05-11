# Wipro Ltd. (WIPRO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 197.88
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 213 |
| ALERT1 | 147 |
| ALERT2 | 145 |
| ALERT2_SKIP | 70 |
| ALERT3 | 354 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 176 |
| PARTIAL | 9 |
| TARGET_HIT | 2 |
| STOP_HIT | 175 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 186 (incl. partial bookings)
- **Trades open at end:** 6
- **Winners / losers:** 48 / 138
- **Target hits / Stop hits / Partials:** 2 / 175 / 9
- **Avg / median % per leg:** -0.11% / -0.69%
- **Sum % (uncompounded):** -20.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 88 | 19 | 21.6% | 0 | 88 | 0 | -0.29% | -25.7% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.38% | -6.9% |
| BUY @ 3rd Alert (retest2) | 83 | 19 | 22.9% | 0 | 83 | 0 | -0.23% | -18.9% |
| SELL (all) | 98 | 29 | 29.6% | 2 | 87 | 9 | 0.05% | 4.8% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.22% | 9.6% |
| SELL @ 3rd Alert (retest2) | 95 | 26 | 27.4% | 2 | 85 | 8 | -0.05% | -4.9% |
| retest1 (combined) | 8 | 3 | 37.5% | 0 | 7 | 1 | 0.35% | 2.8% |
| retest2 (combined) | 178 | 45 | 25.3% | 2 | 168 | 8 | -0.13% | -23.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 11:15:00 | 191.15 | 192.47 | 192.61 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 10:15:00 | 193.28 | 192.07 | 192.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 11:15:00 | 193.40 | 192.33 | 192.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 15:15:00 | 198.60 | 198.65 | 196.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-24 09:15:00 | 197.80 | 198.65 | 196.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 198.70 | 198.66 | 197.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 13:30:00 | 199.60 | 199.10 | 197.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 10:15:00 | 199.55 | 198.25 | 198.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 11:30:00 | 199.78 | 198.82 | 198.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-05 15:15:00 | 202.28 | 202.48 | 202.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 15:15:00 | 202.28 | 202.48 | 202.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 09:15:00 | 199.90 | 201.96 | 202.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 09:15:00 | 200.75 | 200.34 | 201.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 09:15:00 | 200.75 | 200.34 | 201.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 200.75 | 200.34 | 201.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-08 09:30:00 | 200.00 | 200.91 | 201.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-16 14:15:00 | 190.00 | 192.65 | 194.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-06-20 09:15:00 | 191.58 | 190.56 | 191.98 | SL hit (close>ema200) qty=0.50 sl=190.56 alert=retest2 |

### Cycle 4 — BUY (started 2023-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 12:15:00 | 192.85 | 192.10 | 192.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 13:15:00 | 193.03 | 192.28 | 192.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 09:15:00 | 191.30 | 192.26 | 192.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 191.30 | 192.26 | 192.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 191.30 | 192.26 | 192.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:30:00 | 191.35 | 192.26 | 192.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-06-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 10:15:00 | 191.65 | 192.14 | 192.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 14:15:00 | 191.03 | 191.72 | 191.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 11:15:00 | 190.68 | 190.48 | 190.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 11:45:00 | 190.55 | 190.48 | 190.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 191.48 | 190.63 | 190.82 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 11:15:00 | 191.60 | 190.94 | 190.94 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 13:15:00 | 190.70 | 190.98 | 191.00 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 09:15:00 | 193.58 | 191.52 | 191.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 14:15:00 | 194.43 | 192.98 | 192.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-05 09:15:00 | 196.25 | 196.85 | 195.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-05 10:00:00 | 196.25 | 196.85 | 195.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 198.40 | 197.75 | 197.09 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 11:15:00 | 195.78 | 196.86 | 196.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 14:15:00 | 194.95 | 196.26 | 196.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 14:15:00 | 195.95 | 195.39 | 195.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 14:15:00 | 195.95 | 195.39 | 195.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 195.95 | 195.39 | 195.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 15:00:00 | 195.95 | 195.39 | 195.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 196.00 | 195.51 | 195.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 09:15:00 | 196.13 | 195.51 | 195.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 196.53 | 195.71 | 195.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 12:15:00 | 195.63 | 195.77 | 195.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-12 13:15:00 | 196.78 | 196.07 | 196.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2023-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 13:15:00 | 196.78 | 196.07 | 196.06 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 14:15:00 | 195.80 | 196.02 | 196.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-12 15:15:00 | 195.50 | 195.92 | 195.98 | Break + close below crossover candle low |

### Cycle 12 — BUY (started 2023-07-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 09:15:00 | 198.00 | 196.33 | 196.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 11:15:00 | 200.30 | 197.96 | 197.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-19 13:15:00 | 208.38 | 208.50 | 206.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-19 14:00:00 | 208.38 | 208.50 | 206.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 207.43 | 208.31 | 207.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 11:00:00 | 207.43 | 208.31 | 207.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 207.98 | 208.25 | 207.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 14:15:00 | 208.50 | 208.19 | 207.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 15:00:00 | 208.90 | 208.33 | 207.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-21 09:15:00 | 204.23 | 207.56 | 207.44 | SL hit (close<static) qty=1.00 sl=207.33 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 10:15:00 | 203.30 | 206.71 | 207.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 14:15:00 | 202.50 | 204.63 | 205.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 201.65 | 201.31 | 202.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 202.40 | 201.56 | 202.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 202.40 | 201.56 | 202.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 09:30:00 | 202.80 | 201.56 | 202.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 201.93 | 201.64 | 202.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 11:15:00 | 201.28 | 201.64 | 202.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 11:45:00 | 201.60 | 201.70 | 202.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 14:00:00 | 201.40 | 201.71 | 201.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-31 11:30:00 | 201.53 | 200.55 | 200.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-31 13:15:00 | 201.93 | 200.96 | 200.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2023-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 13:15:00 | 201.93 | 200.96 | 200.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 202.50 | 201.27 | 201.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 202.45 | 202.65 | 202.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 202.45 | 202.65 | 202.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 202.45 | 202.65 | 202.10 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 199.60 | 201.62 | 201.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 11:15:00 | 199.48 | 200.74 | 201.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 204.50 | 200.91 | 201.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 204.50 | 200.91 | 201.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 204.50 | 200.91 | 201.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 10:00:00 | 204.50 | 200.91 | 201.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 10:15:00 | 203.85 | 201.50 | 201.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 10:15:00 | 205.05 | 203.79 | 202.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 09:15:00 | 206.95 | 207.18 | 205.86 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 10:45:00 | 208.35 | 207.43 | 206.09 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 207.80 | 208.97 | 208.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-11 09:15:00 | 207.80 | 208.97 | 208.22 | SL hit (close<ema400) qty=1.00 sl=208.22 alert=retest1 |

### Cycle 17 — SELL (started 2023-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 13:15:00 | 206.80 | 207.82 | 207.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 15:15:00 | 206.53 | 207.39 | 207.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 13:15:00 | 207.25 | 206.46 | 206.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 13:15:00 | 207.25 | 206.46 | 206.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 13:15:00 | 207.25 | 206.46 | 206.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 14:00:00 | 207.25 | 206.46 | 206.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 207.65 | 206.70 | 207.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 15:00:00 | 207.65 | 206.70 | 207.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 15:15:00 | 207.00 | 206.76 | 207.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:15:00 | 208.38 | 206.76 | 207.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2023-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 09:15:00 | 209.55 | 207.32 | 207.26 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 09:15:00 | 204.50 | 207.67 | 207.91 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 13:15:00 | 207.90 | 207.26 | 207.20 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 10:15:00 | 207.15 | 207.18 | 207.18 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 11:15:00 | 207.53 | 207.25 | 207.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 14:15:00 | 208.38 | 207.63 | 207.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 10:15:00 | 207.80 | 207.81 | 207.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 10:15:00 | 207.80 | 207.81 | 207.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 10:15:00 | 207.80 | 207.81 | 207.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 11:00:00 | 207.80 | 207.81 | 207.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 207.63 | 207.90 | 207.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 15:00:00 | 207.63 | 207.90 | 207.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 207.70 | 207.86 | 207.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 09:15:00 | 209.38 | 207.86 | 207.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 11:45:00 | 207.73 | 208.00 | 207.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 12:30:00 | 207.98 | 207.88 | 207.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-24 14:15:00 | 206.40 | 207.54 | 207.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-08-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 14:15:00 | 206.40 | 207.54 | 207.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 09:15:00 | 205.48 | 206.97 | 207.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 09:15:00 | 205.00 | 204.61 | 205.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 205.00 | 204.61 | 205.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 205.00 | 204.61 | 205.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 09:30:00 | 205.68 | 204.61 | 205.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 204.68 | 204.43 | 204.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 14:45:00 | 203.95 | 204.28 | 204.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 12:00:00 | 203.95 | 204.06 | 204.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 14:30:00 | 203.88 | 204.12 | 204.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-01 09:15:00 | 204.15 | 204.30 | 204.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 09:15:00 | 204.38 | 204.31 | 204.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 09:45:00 | 204.45 | 204.31 | 204.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-09-01 10:15:00 | 205.65 | 204.58 | 204.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 10:15:00 | 205.65 | 204.58 | 204.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 12:15:00 | 206.00 | 204.95 | 204.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 15:15:00 | 215.75 | 215.82 | 213.19 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 09:15:00 | 216.80 | 215.82 | 213.19 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 214.18 | 215.28 | 213.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 12:00:00 | 214.18 | 215.28 | 213.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 213.50 | 214.92 | 213.58 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-06 12:15:00 | 213.50 | 214.92 | 213.58 | SL hit (close<ema400) qty=1.00 sl=213.58 alert=retest1 |

### Cycle 25 — SELL (started 2023-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 14:15:00 | 217.85 | 218.89 | 218.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 09:15:00 | 216.53 | 218.27 | 218.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 13:15:00 | 214.78 | 214.74 | 216.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-21 14:00:00 | 214.78 | 214.74 | 216.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 14:15:00 | 207.90 | 207.08 | 207.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 14:45:00 | 208.00 | 207.08 | 207.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 15:15:00 | 207.75 | 207.21 | 207.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-28 09:15:00 | 207.18 | 207.21 | 207.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 207.45 | 207.26 | 207.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 11:00:00 | 206.15 | 207.04 | 207.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-05 15:15:00 | 203.50 | 203.19 | 203.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2023-10-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 15:15:00 | 203.50 | 203.19 | 203.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 10:15:00 | 204.70 | 203.57 | 203.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 13:15:00 | 203.68 | 203.81 | 203.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-06 13:30:00 | 203.68 | 203.81 | 203.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 203.88 | 203.83 | 203.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 15:15:00 | 203.75 | 203.83 | 203.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 15:15:00 | 203.75 | 203.81 | 203.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 09:15:00 | 203.33 | 203.81 | 203.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 204.78 | 204.00 | 203.69 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 14:15:00 | 202.88 | 203.58 | 203.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 15:15:00 | 202.50 | 203.37 | 203.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 203.68 | 203.43 | 203.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 203.68 | 203.43 | 203.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 203.68 | 203.43 | 203.52 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 14:15:00 | 203.88 | 203.59 | 203.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 208.70 | 204.65 | 204.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 12:15:00 | 209.65 | 209.69 | 207.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-12 13:00:00 | 209.65 | 209.69 | 207.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 14:15:00 | 208.75 | 209.31 | 208.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 14:45:00 | 208.38 | 209.31 | 208.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 15:15:00 | 208.35 | 209.12 | 208.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 09:15:00 | 206.40 | 209.12 | 208.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 205.45 | 208.38 | 207.86 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 11:15:00 | 204.98 | 207.14 | 207.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 13:15:00 | 204.45 | 205.26 | 205.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 190.85 | 190.03 | 191.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 190.85 | 190.03 | 191.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 190.85 | 190.03 | 191.59 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 10:15:00 | 191.95 | 191.28 | 191.20 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-11-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 14:15:00 | 190.53 | 191.10 | 191.14 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-11-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 11:15:00 | 191.63 | 191.24 | 191.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 15:15:00 | 192.00 | 191.61 | 191.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 13:15:00 | 191.80 | 191.83 | 191.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 13:15:00 | 191.80 | 191.83 | 191.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 13:15:00 | 191.80 | 191.83 | 191.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 13:30:00 | 191.63 | 191.83 | 191.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 14:15:00 | 191.83 | 191.83 | 191.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 14:45:00 | 191.80 | 191.83 | 191.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 191.88 | 191.84 | 191.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:15:00 | 193.05 | 191.84 | 191.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 12:30:00 | 192.38 | 192.31 | 191.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 13:45:00 | 192.05 | 192.37 | 192.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 15:15:00 | 192.00 | 192.23 | 192.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 10:15:00 | 191.83 | 192.17 | 192.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 11:00:00 | 191.83 | 192.17 | 192.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-11-08 11:15:00 | 191.70 | 192.08 | 192.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 11:15:00 | 191.70 | 192.08 | 192.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-08 14:15:00 | 191.35 | 191.78 | 191.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 15:15:00 | 189.88 | 189.80 | 190.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-12 18:15:00 | 191.13 | 189.80 | 190.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 191.50 | 190.14 | 190.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 09:15:00 | 190.45 | 190.14 | 190.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 10:15:00 | 190.48 | 190.26 | 190.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-15 09:15:00 | 194.50 | 191.35 | 190.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 194.50 | 191.35 | 190.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 13:15:00 | 195.88 | 193.54 | 192.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 14:15:00 | 197.70 | 198.05 | 196.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-17 15:00:00 | 197.70 | 198.05 | 196.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 199.50 | 198.25 | 197.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-20 10:15:00 | 200.35 | 198.25 | 197.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 09:15:00 | 201.03 | 199.49 | 198.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 11:15:00 | 200.38 | 200.29 | 199.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 14:15:00 | 200.23 | 200.14 | 199.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 199.60 | 200.68 | 200.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 10:00:00 | 199.60 | 200.68 | 200.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 199.30 | 200.40 | 200.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 11:00:00 | 199.30 | 200.40 | 200.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-11-24 11:15:00 | 199.00 | 200.12 | 200.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 11:15:00 | 199.00 | 200.12 | 200.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 12:15:00 | 198.83 | 199.86 | 200.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 14:15:00 | 198.45 | 198.06 | 198.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-28 15:00:00 | 198.45 | 198.06 | 198.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 202.23 | 198.98 | 199.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:30:00 | 202.30 | 198.98 | 199.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 10:15:00 | 202.88 | 199.76 | 199.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 10:15:00 | 204.68 | 202.87 | 201.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 10:15:00 | 204.20 | 204.30 | 202.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 11:00:00 | 204.20 | 204.30 | 202.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 12:15:00 | 203.70 | 204.01 | 203.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 12:45:00 | 203.45 | 204.01 | 203.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 13:15:00 | 203.50 | 203.90 | 203.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 14:00:00 | 203.50 | 203.90 | 203.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 14:15:00 | 203.65 | 203.85 | 203.54 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 11:15:00 | 201.95 | 203.19 | 203.31 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 09:15:00 | 205.85 | 203.18 | 203.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 13:15:00 | 207.73 | 205.41 | 204.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 12:15:00 | 209.25 | 209.34 | 208.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-08 13:00:00 | 209.25 | 209.34 | 208.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 209.23 | 209.32 | 208.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:30:00 | 207.75 | 209.32 | 208.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 210.18 | 210.20 | 208.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 09:45:00 | 211.85 | 210.60 | 209.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 15:00:00 | 211.25 | 211.72 | 210.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 09:15:00 | 211.55 | 211.58 | 210.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-13 11:15:00 | 208.18 | 210.15 | 210.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2023-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 11:15:00 | 208.18 | 210.15 | 210.24 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 214.95 | 210.82 | 210.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 14:15:00 | 217.25 | 214.33 | 212.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 14:15:00 | 223.00 | 223.24 | 220.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-18 15:15:00 | 222.15 | 223.24 | 220.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 218.93 | 222.21 | 220.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:00:00 | 218.93 | 222.21 | 220.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 218.75 | 221.52 | 220.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:30:00 | 218.88 | 221.52 | 220.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 15:15:00 | 219.10 | 219.96 | 219.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 09:15:00 | 222.00 | 219.96 | 219.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 218.40 | 220.95 | 220.63 | SL hit (close<static) qty=1.00 sl=218.65 alert=retest2 |

### Cycle 41 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 216.00 | 219.96 | 220.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 215.25 | 219.02 | 219.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 219.95 | 218.09 | 218.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 219.95 | 218.09 | 218.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 219.95 | 218.09 | 218.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:00:00 | 219.95 | 218.09 | 218.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 219.43 | 218.36 | 218.74 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 13:15:00 | 226.23 | 220.39 | 219.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 14:15:00 | 231.05 | 222.52 | 220.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 11:15:00 | 232.58 | 233.39 | 229.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 12:00:00 | 232.58 | 233.39 | 229.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 235.88 | 235.25 | 233.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 12:45:00 | 239.33 | 236.40 | 235.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:15:00 | 238.40 | 237.65 | 236.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 12:15:00 | 233.43 | 235.49 | 235.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2024-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 12:15:00 | 233.43 | 235.49 | 235.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 09:15:00 | 228.80 | 233.90 | 234.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-05 09:15:00 | 230.30 | 228.09 | 229.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 09:15:00 | 230.30 | 228.09 | 229.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 230.30 | 228.09 | 229.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 09:30:00 | 231.25 | 228.09 | 229.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 229.05 | 228.28 | 229.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-05 13:30:00 | 228.20 | 228.49 | 229.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-05 15:00:00 | 228.35 | 228.46 | 229.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-08 09:30:00 | 227.80 | 228.46 | 229.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-08 10:15:00 | 228.40 | 228.46 | 229.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 228.80 | 227.10 | 227.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-09 12:45:00 | 227.43 | 227.50 | 227.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-09 13:45:00 | 226.88 | 227.29 | 227.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-12 09:15:00 | 233.00 | 226.82 | 226.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 09:15:00 | 233.00 | 226.82 | 226.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 09:15:00 | 249.45 | 234.97 | 231.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 09:15:00 | 242.88 | 244.23 | 238.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 242.75 | 242.84 | 240.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 242.75 | 242.84 | 240.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 11:15:00 | 244.28 | 242.75 | 240.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 11:45:00 | 243.65 | 240.94 | 240.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 09:15:00 | 244.38 | 240.96 | 240.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 10:15:00 | 244.08 | 241.45 | 241.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 240.05 | 241.94 | 241.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-20 11:15:00 | 240.45 | 241.47 | 241.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 11:15:00 | 240.45 | 241.47 | 241.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 12:15:00 | 240.20 | 241.21 | 241.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-23 09:15:00 | 241.23 | 240.42 | 240.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 241.23 | 240.42 | 240.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 241.23 | 240.42 | 240.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:30:00 | 242.83 | 240.42 | 240.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 240.55 | 240.45 | 240.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 11:30:00 | 239.40 | 239.96 | 240.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-30 09:15:00 | 239.95 | 237.36 | 237.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 09:15:00 | 239.95 | 237.36 | 237.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 11:15:00 | 240.88 | 238.46 | 237.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 236.33 | 238.39 | 238.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 14:15:00 | 236.33 | 238.39 | 238.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 236.33 | 238.39 | 238.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 15:00:00 | 236.33 | 238.39 | 238.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 235.95 | 237.91 | 237.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 09:15:00 | 235.80 | 237.91 | 237.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 09:15:00 | 236.18 | 237.56 | 237.66 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 14:15:00 | 239.03 | 237.76 | 237.67 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 11:15:00 | 236.40 | 237.41 | 237.55 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 10:15:00 | 240.75 | 237.74 | 237.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-02 11:15:00 | 241.63 | 238.52 | 237.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 14:15:00 | 241.43 | 241.80 | 240.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 14:15:00 | 241.43 | 241.80 | 240.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 241.43 | 241.80 | 240.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:45:00 | 240.60 | 241.80 | 240.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 14:15:00 | 247.48 | 248.21 | 246.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 14:45:00 | 246.68 | 248.21 | 246.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 248.03 | 248.07 | 246.63 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-02-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 15:15:00 | 244.50 | 245.90 | 246.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 243.28 | 245.37 | 245.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 13:15:00 | 245.80 | 244.70 | 245.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 13:15:00 | 245.80 | 244.70 | 245.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 13:15:00 | 245.80 | 244.70 | 245.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 14:00:00 | 245.80 | 244.70 | 245.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 14:15:00 | 245.25 | 244.81 | 245.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 14:30:00 | 245.25 | 244.81 | 245.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 15:15:00 | 245.50 | 244.95 | 245.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 09:15:00 | 250.25 | 244.95 | 245.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 09:15:00 | 250.00 | 245.96 | 245.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-12 10:15:00 | 253.50 | 247.47 | 246.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-13 09:15:00 | 248.60 | 250.10 | 248.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 09:15:00 | 248.60 | 250.10 | 248.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 248.60 | 250.10 | 248.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 09:30:00 | 247.43 | 250.10 | 248.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 249.60 | 250.00 | 248.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-13 13:15:00 | 250.80 | 249.65 | 248.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 09:45:00 | 251.18 | 252.14 | 250.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 13:15:00 | 262.80 | 264.68 | 264.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2024-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 13:15:00 | 262.80 | 264.68 | 264.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 14:15:00 | 261.15 | 263.98 | 264.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 12:15:00 | 262.85 | 262.84 | 263.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-22 13:00:00 | 262.85 | 262.84 | 263.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 262.80 | 262.83 | 263.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:00:00 | 262.80 | 262.83 | 263.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 265.08 | 263.28 | 263.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 265.08 | 263.28 | 263.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 265.10 | 263.64 | 263.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 267.50 | 263.64 | 263.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 09:15:00 | 267.35 | 264.39 | 264.16 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 09:15:00 | 264.58 | 265.47 | 265.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 10:15:00 | 261.80 | 264.73 | 265.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 259.73 | 259.65 | 261.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 15:00:00 | 259.73 | 259.65 | 261.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 260.50 | 259.82 | 261.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 260.93 | 259.82 | 261.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 260.15 | 259.89 | 261.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 10:30:00 | 259.68 | 259.96 | 261.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 11:15:00 | 259.70 | 259.96 | 261.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 14:15:00 | 259.83 | 259.85 | 260.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-04 09:15:00 | 262.83 | 261.03 | 260.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2024-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 09:15:00 | 262.83 | 261.03 | 260.95 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 12:15:00 | 260.38 | 260.88 | 260.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 09:15:00 | 258.58 | 260.15 | 260.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 254.20 | 254.07 | 256.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 13:45:00 | 254.05 | 254.07 | 256.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 256.35 | 254.53 | 256.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 15:00:00 | 256.35 | 254.53 | 256.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 256.85 | 254.99 | 256.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:15:00 | 256.88 | 254.99 | 256.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 256.60 | 255.31 | 256.40 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 12:15:00 | 258.63 | 257.05 | 257.02 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 10:15:00 | 256.10 | 257.07 | 257.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 11:15:00 | 254.90 | 256.64 | 256.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 09:15:00 | 256.52 | 255.95 | 256.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 09:15:00 | 256.52 | 255.95 | 256.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 256.52 | 255.95 | 256.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 11:15:00 | 253.93 | 255.73 | 256.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-14 14:15:00 | 259.13 | 255.56 | 255.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-03-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 14:15:00 | 259.13 | 255.56 | 255.29 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 15:15:00 | 255.00 | 255.84 | 255.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 251.20 | 254.91 | 255.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 251.75 | 248.55 | 250.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 251.75 | 248.55 | 250.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 251.75 | 248.55 | 250.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 09:30:00 | 251.03 | 248.55 | 250.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 10:15:00 | 251.80 | 249.20 | 250.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 11:00:00 | 251.80 | 249.20 | 250.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 12:15:00 | 250.30 | 249.59 | 250.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 12:30:00 | 250.68 | 249.59 | 250.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 13:15:00 | 250.43 | 249.76 | 250.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 13:30:00 | 251.08 | 249.76 | 250.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 14:15:00 | 250.50 | 249.91 | 250.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 14:30:00 | 250.58 | 249.91 | 250.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 15:15:00 | 250.50 | 250.02 | 250.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 09:15:00 | 240.83 | 250.02 | 250.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-01 12:15:00 | 241.73 | 240.77 | 240.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 12:15:00 | 241.73 | 240.77 | 240.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 14:15:00 | 242.53 | 241.32 | 241.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 09:15:00 | 240.15 | 241.28 | 241.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 09:15:00 | 240.15 | 241.28 | 241.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 240.15 | 241.28 | 241.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 09:45:00 | 239.88 | 241.28 | 241.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 11:15:00 | 240.58 | 241.06 | 241.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 11:45:00 | 240.43 | 241.06 | 241.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 12:15:00 | 240.40 | 240.93 | 240.95 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 10:15:00 | 242.85 | 241.24 | 241.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-04 14:15:00 | 243.40 | 242.29 | 241.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 11:15:00 | 242.75 | 242.83 | 242.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 11:15:00 | 242.75 | 242.83 | 242.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 11:15:00 | 242.75 | 242.83 | 242.29 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 10:15:00 | 240.25 | 241.83 | 242.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 12:15:00 | 238.58 | 240.29 | 240.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 13:15:00 | 239.15 | 238.59 | 239.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 13:15:00 | 239.15 | 238.59 | 239.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 13:15:00 | 239.15 | 238.59 | 239.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 13:30:00 | 239.68 | 238.59 | 239.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 14:15:00 | 238.43 | 238.56 | 239.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 14:30:00 | 239.25 | 238.56 | 239.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 237.80 | 238.46 | 239.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:30:00 | 238.03 | 238.46 | 239.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 237.93 | 238.35 | 239.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 10:30:00 | 239.88 | 238.35 | 239.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 13:15:00 | 226.05 | 223.72 | 225.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:00:00 | 226.05 | 223.72 | 225.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 226.58 | 224.29 | 225.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 15:00:00 | 226.58 | 224.29 | 225.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 15:15:00 | 226.05 | 224.65 | 225.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:15:00 | 228.60 | 224.65 | 225.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 231.43 | 226.00 | 226.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:45:00 | 231.15 | 226.00 | 226.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 10:15:00 | 232.00 | 227.20 | 226.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 09:15:00 | 235.43 | 231.40 | 230.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 13:15:00 | 233.50 | 233.90 | 232.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 14:00:00 | 233.50 | 233.90 | 232.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 232.40 | 233.60 | 232.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 14:30:00 | 232.93 | 233.60 | 232.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 232.58 | 233.40 | 232.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:45:00 | 231.53 | 232.91 | 232.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 231.98 | 232.72 | 232.22 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 14:15:00 | 231.40 | 231.95 | 231.96 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 12:15:00 | 232.28 | 231.95 | 231.95 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 14:15:00 | 231.35 | 231.89 | 231.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 15:15:00 | 231.20 | 231.75 | 231.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 09:15:00 | 229.73 | 229.47 | 230.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 09:15:00 | 229.73 | 229.47 | 230.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 229.73 | 229.47 | 230.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 10:45:00 | 229.23 | 229.44 | 230.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 11:15:00 | 228.50 | 229.44 | 230.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 13:15:00 | 229.25 | 229.36 | 229.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 10:15:00 | 228.68 | 229.42 | 229.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 227.85 | 229.11 | 229.38 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-07 13:15:00 | 232.00 | 229.39 | 229.42 | SL hit (close>static) qty=1.00 sl=231.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 14:15:00 | 231.55 | 229.83 | 229.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-07 15:15:00 | 232.33 | 230.33 | 229.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 13:15:00 | 231.25 | 231.45 | 230.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-08 13:30:00 | 231.38 | 231.45 | 230.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 229.43 | 231.05 | 230.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 10:00:00 | 229.43 | 231.05 | 230.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 230.00 | 230.84 | 230.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 11:15:00 | 230.30 | 230.84 | 230.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-09 13:15:00 | 229.08 | 230.47 | 230.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 13:15:00 | 229.08 | 230.47 | 230.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 14:15:00 | 227.63 | 229.90 | 230.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 14:15:00 | 225.85 | 225.48 | 226.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-13 15:00:00 | 225.85 | 225.48 | 226.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 226.83 | 225.89 | 226.72 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 229.33 | 227.29 | 227.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 231.78 | 229.40 | 228.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 229.75 | 229.91 | 229.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:00:00 | 229.75 | 229.91 | 229.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 228.68 | 229.66 | 228.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 229.83 | 229.66 | 228.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 232.25 | 230.18 | 229.28 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 13:15:00 | 229.95 | 230.50 | 230.52 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 15:15:00 | 230.68 | 230.55 | 230.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 09:15:00 | 233.03 | 231.04 | 230.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 13:15:00 | 233.00 | 233.30 | 232.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 14:00:00 | 233.00 | 233.30 | 232.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 231.65 | 232.97 | 232.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 231.65 | 232.97 | 232.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 231.08 | 232.59 | 232.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 228.73 | 232.59 | 232.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 227.93 | 231.66 | 231.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 14:15:00 | 225.95 | 228.38 | 230.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 13:15:00 | 228.13 | 227.79 | 228.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-28 14:00:00 | 228.13 | 227.79 | 228.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 221.80 | 220.33 | 221.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 217.38 | 221.67 | 221.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:30:00 | 220.20 | 220.80 | 221.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 209.19 | 219.73 | 220.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-04 14:15:00 | 219.88 | 219.64 | 220.62 | SL hit (close>ema200) qty=0.50 sl=219.64 alert=retest2 |

### Cycle 76 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 225.60 | 221.33 | 221.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 228.05 | 225.54 | 223.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 237.75 | 238.81 | 234.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:00:00 | 237.75 | 238.81 | 234.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 238.30 | 239.12 | 238.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:00:00 | 238.30 | 239.12 | 238.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 238.50 | 238.99 | 238.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 242.33 | 238.99 | 238.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 15:15:00 | 238.58 | 239.40 | 239.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-06-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 15:15:00 | 238.58 | 239.40 | 239.41 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 09:15:00 | 244.60 | 240.44 | 239.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 11:15:00 | 246.63 | 244.85 | 243.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 09:15:00 | 246.00 | 246.20 | 244.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-20 10:00:00 | 246.00 | 246.20 | 244.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 245.20 | 245.93 | 245.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:45:00 | 244.68 | 245.93 | 245.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 245.80 | 245.90 | 245.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 248.68 | 245.90 | 245.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 12:45:00 | 246.40 | 247.12 | 246.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 14:15:00 | 244.98 | 246.59 | 245.99 | SL hit (close<static) qty=1.00 sl=245.03 alert=retest2 |

### Cycle 79 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 245.28 | 245.77 | 245.81 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 11:15:00 | 246.60 | 245.97 | 245.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 14:15:00 | 248.85 | 246.67 | 246.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 15:15:00 | 247.25 | 247.53 | 247.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 09:15:00 | 245.43 | 247.53 | 247.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 247.28 | 247.48 | 247.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:30:00 | 245.60 | 247.48 | 247.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 248.25 | 247.63 | 247.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:45:00 | 247.33 | 247.63 | 247.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 267.25 | 269.21 | 267.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:00:00 | 267.25 | 269.21 | 267.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 265.63 | 268.49 | 267.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:00:00 | 265.63 | 268.49 | 267.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 265.75 | 267.94 | 267.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:00:00 | 265.75 | 267.94 | 267.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 10:15:00 | 264.90 | 266.47 | 266.63 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 12:15:00 | 267.33 | 266.73 | 266.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 09:15:00 | 268.40 | 267.37 | 267.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 10:15:00 | 269.65 | 269.68 | 268.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 10:30:00 | 269.30 | 269.68 | 268.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 268.90 | 270.05 | 269.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 268.90 | 270.05 | 269.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 267.88 | 269.62 | 269.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 267.18 | 269.62 | 269.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 267.85 | 269.26 | 269.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 11:30:00 | 267.93 | 269.26 | 269.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 12:15:00 | 267.80 | 268.97 | 269.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 13:15:00 | 266.20 | 268.42 | 268.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 14:15:00 | 267.08 | 266.80 | 267.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 15:00:00 | 267.08 | 266.80 | 267.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 267.95 | 267.03 | 267.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 272.00 | 267.03 | 267.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 276.85 | 269.00 | 268.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 10:15:00 | 277.10 | 270.62 | 269.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 13:15:00 | 278.10 | 278.87 | 275.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 14:00:00 | 278.10 | 278.87 | 275.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 280.80 | 283.02 | 281.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:00:00 | 280.80 | 283.02 | 281.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 281.33 | 282.69 | 281.15 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 09:15:00 | 255.38 | 275.69 | 278.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 12:15:00 | 251.68 | 264.84 | 272.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 10:15:00 | 252.38 | 251.94 | 257.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-24 11:00:00 | 252.38 | 251.94 | 257.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 257.93 | 253.48 | 254.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:45:00 | 258.35 | 253.48 | 254.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 259.27 | 254.64 | 254.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 11:15:00 | 260.90 | 255.89 | 255.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 261.05 | 261.66 | 259.97 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 11:00:00 | 262.88 | 261.90 | 260.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 12:00:00 | 262.40 | 262.00 | 260.43 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 260.75 | 261.60 | 260.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 13:45:00 | 260.27 | 261.60 | 260.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 260.50 | 261.38 | 260.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-30 14:15:00 | 260.50 | 261.38 | 260.51 | SL hit (close<ema400) qty=1.00 sl=260.51 alert=retest1 |

### Cycle 87 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 257.58 | 260.34 | 260.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 12:15:00 | 253.98 | 257.98 | 259.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 248.68 | 245.91 | 250.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 248.68 | 245.91 | 250.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 248.13 | 246.10 | 248.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 12:45:00 | 247.65 | 246.97 | 248.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 246.60 | 247.75 | 248.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 10:15:00 | 246.18 | 245.70 | 245.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 10:15:00 | 246.18 | 245.70 | 245.68 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 245.55 | 245.67 | 245.67 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 12:15:00 | 246.78 | 245.89 | 245.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 14:15:00 | 247.40 | 246.33 | 246.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 11:15:00 | 261.30 | 261.49 | 259.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 12:00:00 | 261.30 | 261.49 | 259.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 261.10 | 262.17 | 260.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:00:00 | 261.10 | 262.17 | 260.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 261.15 | 261.96 | 260.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:15:00 | 259.88 | 261.96 | 260.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 259.65 | 261.50 | 260.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:30:00 | 259.63 | 261.50 | 260.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 259.58 | 261.12 | 260.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 258.10 | 261.12 | 260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 09:15:00 | 257.18 | 260.33 | 260.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 256.40 | 258.28 | 259.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 261.68 | 258.42 | 259.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 261.68 | 258.42 | 259.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 261.68 | 258.42 | 259.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:45:00 | 261.95 | 258.42 | 259.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 261.20 | 258.98 | 259.26 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 12:15:00 | 261.30 | 259.74 | 259.58 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 11:15:00 | 258.98 | 259.55 | 259.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 13:15:00 | 258.48 | 259.24 | 259.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 09:15:00 | 260.48 | 259.34 | 259.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 260.48 | 259.34 | 259.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 260.48 | 259.34 | 259.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:00:00 | 260.48 | 259.34 | 259.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 10:15:00 | 265.58 | 260.59 | 259.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 12:15:00 | 268.15 | 263.11 | 261.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 13:15:00 | 268.88 | 268.98 | 267.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-30 14:00:00 | 268.88 | 268.98 | 267.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 267.85 | 268.93 | 267.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:00:00 | 267.85 | 268.93 | 267.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 266.93 | 268.53 | 267.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:00:00 | 266.93 | 268.53 | 267.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 268.35 | 268.49 | 267.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 11:45:00 | 269.50 | 267.82 | 267.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 14:15:00 | 269.25 | 268.14 | 267.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 262.48 | 266.86 | 267.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 09:15:00 | 262.48 | 266.86 | 267.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 10:15:00 | 259.38 | 265.37 | 266.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 12:15:00 | 262.00 | 261.49 | 263.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-05 13:00:00 | 262.00 | 261.49 | 263.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 261.75 | 261.94 | 262.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:30:00 | 263.13 | 261.94 | 262.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 263.88 | 262.17 | 262.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 12:00:00 | 263.88 | 262.17 | 262.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 261.33 | 262.00 | 262.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 13:45:00 | 260.75 | 261.65 | 262.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 14:15:00 | 263.02 | 260.72 | 260.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 263.02 | 260.72 | 260.63 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 256.85 | 260.12 | 260.51 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 264.00 | 261.04 | 260.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 265.20 | 261.87 | 261.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 15:15:00 | 275.52 | 275.55 | 272.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 09:15:00 | 275.15 | 275.55 | 272.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 268.23 | 274.34 | 273.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:00:00 | 268.23 | 274.34 | 273.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 268.65 | 273.20 | 273.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:30:00 | 267.58 | 273.20 | 273.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 268.98 | 272.36 | 272.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 267.35 | 270.62 | 271.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 11:15:00 | 267.45 | 267.31 | 268.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 11:45:00 | 267.10 | 267.31 | 268.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 268.15 | 267.48 | 268.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:45:00 | 268.23 | 267.48 | 268.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 266.70 | 267.32 | 268.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 09:30:00 | 265.20 | 266.86 | 267.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 14:15:00 | 269.55 | 268.00 | 267.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 14:15:00 | 269.55 | 268.00 | 267.92 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 11:15:00 | 266.05 | 267.60 | 267.78 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 09:15:00 | 270.55 | 268.10 | 267.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 274.55 | 270.72 | 269.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 12:15:00 | 270.88 | 271.84 | 270.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 13:00:00 | 270.88 | 271.84 | 270.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 271.80 | 271.83 | 270.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:30:00 | 272.65 | 271.46 | 270.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 268.77 | 272.08 | 271.74 | SL hit (close<static) qty=1.00 sl=270.35 alert=retest2 |

### Cycle 103 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 267.35 | 271.14 | 271.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 265.43 | 269.99 | 270.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 267.40 | 267.02 | 268.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 09:45:00 | 268.52 | 267.02 | 268.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 270.23 | 267.66 | 268.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:00:00 | 270.23 | 267.66 | 268.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 269.80 | 268.09 | 268.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:30:00 | 268.25 | 268.14 | 268.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:00:00 | 268.00 | 267.72 | 268.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 13:15:00 | 266.50 | 265.82 | 265.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 266.50 | 265.82 | 265.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 267.45 | 266.15 | 265.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 265.10 | 266.18 | 266.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 12:15:00 | 265.10 | 266.18 | 266.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 265.10 | 266.18 | 266.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:45:00 | 263.83 | 266.18 | 266.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 13:15:00 | 263.00 | 265.54 | 265.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 262.48 | 264.93 | 265.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 09:15:00 | 264.68 | 264.58 | 265.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 09:15:00 | 264.68 | 264.58 | 265.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 264.68 | 264.58 | 265.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 265.05 | 264.58 | 265.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 264.23 | 263.98 | 264.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 269.85 | 263.98 | 264.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 269.35 | 265.05 | 264.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 12:15:00 | 274.10 | 268.41 | 266.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 10:15:00 | 270.43 | 271.43 | 269.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 10:30:00 | 270.25 | 271.43 | 269.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 268.10 | 270.77 | 269.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:00:00 | 268.10 | 270.77 | 269.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 266.75 | 269.96 | 268.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 266.75 | 269.96 | 268.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 267.20 | 268.15 | 268.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 265.00 | 267.52 | 267.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 10:15:00 | 267.33 | 266.86 | 267.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 10:15:00 | 267.33 | 266.86 | 267.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 267.33 | 266.86 | 267.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:45:00 | 267.77 | 266.86 | 267.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 264.95 | 266.48 | 267.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 266.20 | 266.48 | 267.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 264.08 | 265.89 | 266.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:30:00 | 267.77 | 265.89 | 266.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 09:15:00 | 275.10 | 267.49 | 267.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 09:15:00 | 279.27 | 273.81 | 271.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 12:15:00 | 274.65 | 274.65 | 272.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 13:00:00 | 274.65 | 274.65 | 272.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 12:15:00 | 274.13 | 274.40 | 273.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 12:30:00 | 273.50 | 274.40 | 273.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 13:15:00 | 274.25 | 274.37 | 273.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 10:00:00 | 274.75 | 273.94 | 273.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-24 09:30:00 | 274.73 | 274.28 | 273.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 274.70 | 273.82 | 273.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 273.20 | 273.70 | 273.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 273.20 | 273.70 | 273.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 271.58 | 273.28 | 273.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 15:15:00 | 272.58 | 272.23 | 272.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 09:15:00 | 271.77 | 272.23 | 272.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 273.85 | 272.56 | 272.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:00:00 | 273.85 | 272.56 | 272.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2024-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 10:15:00 | 277.27 | 273.50 | 273.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 11:15:00 | 278.43 | 274.49 | 273.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 12:15:00 | 277.75 | 277.83 | 276.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-29 13:00:00 | 277.75 | 277.83 | 276.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 277.02 | 282.20 | 280.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:00:00 | 277.02 | 282.20 | 280.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 274.45 | 280.65 | 280.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 274.45 | 280.65 | 280.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 11:15:00 | 274.93 | 279.51 | 279.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 270.25 | 275.68 | 277.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 10:15:00 | 272.10 | 271.45 | 273.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 11:00:00 | 272.10 | 271.45 | 273.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 278.48 | 272.98 | 273.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 278.48 | 272.98 | 273.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 280.00 | 274.39 | 274.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 14:15:00 | 281.95 | 278.33 | 276.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 14:15:00 | 284.58 | 284.73 | 282.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 15:00:00 | 284.58 | 284.73 | 282.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 286.45 | 287.25 | 285.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:30:00 | 285.75 | 287.25 | 285.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 285.15 | 286.83 | 285.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 285.15 | 286.83 | 285.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 284.52 | 286.37 | 285.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:15:00 | 285.93 | 286.37 | 285.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 285.58 | 285.71 | 285.55 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 11:15:00 | 283.90 | 285.35 | 285.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 11:15:00 | 283.60 | 284.67 | 284.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 281.65 | 278.38 | 280.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 281.65 | 278.38 | 280.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 281.65 | 278.38 | 280.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 281.65 | 278.38 | 280.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 281.75 | 279.05 | 280.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 281.77 | 279.05 | 280.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 280.50 | 280.95 | 281.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 283.20 | 280.95 | 281.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 280.30 | 280.82 | 280.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 281.98 | 280.82 | 280.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 280.23 | 280.70 | 280.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:30:00 | 281.27 | 280.70 | 280.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 278.23 | 280.21 | 280.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:30:00 | 280.20 | 280.21 | 280.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 283.65 | 280.11 | 280.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 283.65 | 280.11 | 280.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 284.13 | 280.92 | 280.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 289.70 | 285.04 | 283.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 11:15:00 | 292.27 | 292.79 | 290.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 11:45:00 | 292.38 | 292.79 | 290.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 291.93 | 292.36 | 291.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:15:00 | 289.93 | 292.36 | 291.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 290.60 | 292.01 | 291.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:30:00 | 289.43 | 292.01 | 291.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 287.00 | 291.01 | 290.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 286.85 | 291.01 | 290.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 11:15:00 | 286.55 | 290.11 | 290.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 14:15:00 | 285.58 | 288.26 | 289.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 289.18 | 288.26 | 289.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 10:15:00 | 289.18 | 288.26 | 289.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 289.18 | 288.26 | 289.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:30:00 | 289.02 | 288.26 | 289.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 288.98 | 288.40 | 289.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:00:00 | 288.98 | 288.40 | 289.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 289.05 | 288.53 | 289.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:00:00 | 289.05 | 288.53 | 289.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 288.65 | 288.56 | 288.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:30:00 | 289.40 | 288.56 | 288.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 288.77 | 288.60 | 288.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:30:00 | 289.85 | 288.60 | 288.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 289.50 | 288.78 | 289.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 290.38 | 288.78 | 289.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 289.73 | 288.97 | 289.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:45:00 | 290.68 | 288.97 | 289.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 290.08 | 289.19 | 289.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 12:15:00 | 291.35 | 289.81 | 289.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 12:15:00 | 289.70 | 291.12 | 290.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 12:15:00 | 289.70 | 291.12 | 290.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 289.70 | 291.12 | 290.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 13:00:00 | 289.70 | 291.12 | 290.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 13:15:00 | 290.75 | 291.05 | 290.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 14:45:00 | 291.55 | 291.16 | 290.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 12:15:00 | 306.80 | 310.35 | 310.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 306.80 | 310.35 | 310.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 304.85 | 308.71 | 309.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 308.80 | 308.05 | 309.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 11:00:00 | 308.80 | 308.05 | 309.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 308.45 | 308.13 | 309.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:30:00 | 310.00 | 308.13 | 309.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 305.60 | 307.62 | 308.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 12:15:00 | 305.10 | 307.13 | 308.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 12:45:00 | 305.25 | 306.61 | 307.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:15:00 | 304.80 | 305.75 | 306.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 11:30:00 | 304.95 | 305.59 | 306.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 307.85 | 305.73 | 306.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:15:00 | 309.05 | 305.73 | 306.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-27 10:15:00 | 310.50 | 306.68 | 306.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 310.50 | 306.68 | 306.62 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 304.80 | 306.94 | 307.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 304.10 | 306.37 | 306.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 302.90 | 301.84 | 303.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 14:00:00 | 302.90 | 301.84 | 303.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 299.60 | 301.46 | 303.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 298.95 | 300.91 | 302.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 14:15:00 | 303.95 | 301.82 | 301.98 | SL hit (close>static) qty=1.00 sl=303.35 alert=retest2 |

### Cycle 120 — BUY (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 10:15:00 | 300.75 | 295.32 | 294.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-10 11:15:00 | 302.20 | 296.70 | 295.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 09:15:00 | 294.10 | 298.01 | 296.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-13 09:15:00 | 294.10 | 298.01 | 296.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 294.10 | 298.01 | 296.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:00:00 | 294.10 | 298.01 | 296.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 293.80 | 297.17 | 296.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 11:15:00 | 292.80 | 297.17 | 296.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 11:15:00 | 293.20 | 296.37 | 296.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 11:30:00 | 293.75 | 296.37 | 296.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2025-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 12:15:00 | 292.45 | 295.59 | 295.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 291.70 | 294.81 | 295.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 294.10 | 293.44 | 294.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 10:15:00 | 294.10 | 293.44 | 294.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 294.10 | 293.44 | 294.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 294.10 | 293.44 | 294.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 293.10 | 293.11 | 293.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:30:00 | 291.20 | 292.89 | 293.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 11:00:00 | 291.40 | 292.60 | 293.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 09:15:00 | 292.25 | 292.36 | 293.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 302.00 | 287.42 | 288.11 | SL hit (close>static) qty=1.00 sl=294.10 alert=retest2 |

### Cycle 122 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 300.95 | 290.12 | 289.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 09:15:00 | 302.95 | 298.07 | 294.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 14:15:00 | 298.55 | 299.03 | 296.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 15:00:00 | 298.55 | 299.03 | 296.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 312.15 | 317.38 | 314.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:45:00 | 311.30 | 317.38 | 314.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 11:15:00 | 311.05 | 316.12 | 314.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 11:30:00 | 311.15 | 316.12 | 314.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2025-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 13:15:00 | 307.75 | 313.33 | 313.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 302.70 | 309.71 | 311.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 308.70 | 306.40 | 308.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 308.70 | 306.40 | 308.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 308.70 | 306.40 | 308.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:30:00 | 309.65 | 306.40 | 308.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 309.20 | 306.96 | 308.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 308.15 | 306.96 | 308.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 309.45 | 307.46 | 308.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:45:00 | 309.35 | 307.46 | 308.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 310.00 | 307.97 | 308.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:45:00 | 310.60 | 307.97 | 308.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 312.70 | 309.85 | 309.56 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 306.55 | 309.46 | 309.65 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 312.20 | 310.10 | 309.90 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 11:15:00 | 307.20 | 309.82 | 310.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 12:15:00 | 304.25 | 308.71 | 309.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 09:15:00 | 307.85 | 307.42 | 308.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 307.85 | 307.42 | 308.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 307.85 | 307.42 | 308.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:30:00 | 308.40 | 307.42 | 308.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 307.40 | 307.42 | 308.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:45:00 | 309.05 | 307.42 | 308.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 309.60 | 307.85 | 308.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:00:00 | 309.60 | 307.85 | 308.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 312.10 | 308.70 | 308.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:00:00 | 312.10 | 308.70 | 308.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 13:15:00 | 313.45 | 309.65 | 309.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 09:15:00 | 317.50 | 312.29 | 310.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 11:15:00 | 315.20 | 315.32 | 313.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 11:30:00 | 315.70 | 315.32 | 313.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 316.95 | 316.13 | 314.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 12:30:00 | 318.80 | 316.71 | 315.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:30:00 | 318.55 | 317.38 | 316.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:00:00 | 318.20 | 317.38 | 316.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:30:00 | 318.70 | 317.39 | 316.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 318.50 | 317.61 | 316.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 319.45 | 317.49 | 316.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 09:45:00 | 318.60 | 317.63 | 316.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 10:15:00 | 318.85 | 317.63 | 316.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 14:30:00 | 319.40 | 318.65 | 317.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 316.35 | 318.27 | 317.74 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 316.35 | 318.27 | 317.74 | SL hit (close<static) qty=1.00 sl=316.50 alert=retest2 |

### Cycle 129 — SELL (started 2025-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 10:15:00 | 313.85 | 317.39 | 317.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 11:15:00 | 312.95 | 316.50 | 316.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 313.00 | 311.77 | 313.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 313.00 | 311.77 | 313.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 313.00 | 311.77 | 313.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:15:00 | 309.30 | 311.11 | 312.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:30:00 | 308.70 | 310.29 | 312.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:30:00 | 309.25 | 309.85 | 311.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 308.85 | 309.85 | 311.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 305.80 | 307.29 | 308.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:45:00 | 303.35 | 306.39 | 308.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 13:15:00 | 311.15 | 307.69 | 307.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 13:15:00 | 311.15 | 307.69 | 307.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 312.55 | 308.66 | 308.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 10:15:00 | 311.90 | 312.82 | 311.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 10:45:00 | 312.35 | 312.82 | 311.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 313.45 | 313.12 | 312.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 14:30:00 | 312.50 | 313.12 | 312.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 308.10 | 312.11 | 311.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 308.10 | 312.11 | 311.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 308.25 | 311.34 | 311.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 11:15:00 | 306.20 | 310.31 | 310.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 293.40 | 293.12 | 297.08 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 13:15:00 | 291.45 | 293.08 | 296.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 292.35 | 293.19 | 295.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 285.80 | 293.19 | 295.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 09:15:00 | 276.88 | 283.07 | 285.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 283.25 | 281.61 | 283.32 | SL hit (close>ema200) qty=0.50 sl=281.61 alert=retest1 |

### Cycle 132 — BUY (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 15:15:00 | 284.60 | 283.96 | 283.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 10:15:00 | 285.40 | 284.43 | 284.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 285.20 | 285.21 | 284.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 285.50 | 285.21 | 284.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 287.00 | 285.56 | 284.91 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 283.25 | 284.52 | 284.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 280.90 | 283.01 | 283.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 278.70 | 278.04 | 280.06 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 09:15:00 | 269.70 | 278.04 | 280.06 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 260.75 | 260.81 | 262.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 11:30:00 | 260.00 | 260.65 | 262.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 12:00:00 | 260.00 | 260.65 | 262.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 13:15:00 | 264.75 | 261.69 | 262.43 | SL hit (close>ema400) qty=1.00 sl=262.43 alert=retest1 |

### Cycle 134 — BUY (started 2025-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 15:15:00 | 265.80 | 263.18 | 263.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 09:15:00 | 268.45 | 264.23 | 263.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 09:15:00 | 266.55 | 267.19 | 265.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 266.55 | 267.19 | 265.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 266.55 | 267.19 | 265.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 12:15:00 | 269.15 | 267.35 | 266.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 13:00:00 | 269.90 | 267.48 | 266.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 12:15:00 | 268.70 | 270.01 | 269.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 13:15:00 | 269.00 | 269.73 | 269.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 269.05 | 269.60 | 269.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:45:00 | 268.45 | 269.60 | 269.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 267.55 | 269.19 | 269.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 267.55 | 269.19 | 269.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-26 15:15:00 | 267.50 | 268.85 | 268.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 267.50 | 268.85 | 268.94 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 09:15:00 | 270.80 | 269.24 | 269.11 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 10:15:00 | 267.20 | 269.23 | 269.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 12:15:00 | 264.70 | 267.93 | 268.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 15:15:00 | 262.85 | 262.83 | 264.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 09:15:00 | 262.30 | 262.83 | 264.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 265.00 | 263.32 | 264.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 265.00 | 263.32 | 264.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 263.95 | 263.44 | 264.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 13:30:00 | 263.00 | 263.30 | 264.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 259.70 | 263.48 | 264.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:15:00 | 249.85 | 256.23 | 259.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 12:15:00 | 246.71 | 252.01 | 256.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 236.70 | 246.26 | 252.24 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 138 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 243.55 | 241.83 | 241.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 15:15:00 | 244.30 | 242.81 | 242.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 233.55 | 243.44 | 243.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 233.55 | 243.44 | 243.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 233.55 | 243.44 | 243.35 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 10:15:00 | 233.90 | 241.54 | 242.49 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 12:15:00 | 243.60 | 239.43 | 238.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 13:15:00 | 244.70 | 240.48 | 239.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 13:15:00 | 241.65 | 242.55 | 241.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 14:00:00 | 241.65 | 242.55 | 241.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 240.10 | 242.15 | 241.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 240.10 | 242.15 | 241.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 239.80 | 241.68 | 241.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:00:00 | 241.10 | 241.56 | 241.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:45:00 | 241.50 | 241.55 | 241.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 10:00:00 | 241.25 | 241.27 | 241.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-28 10:15:00 | 239.95 | 241.01 | 241.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 10:15:00 | 239.95 | 241.01 | 241.10 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 12:15:00 | 242.30 | 241.20 | 241.08 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 13:15:00 | 239.90 | 241.11 | 241.20 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 14:15:00 | 241.85 | 241.26 | 241.26 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 240.80 | 241.17 | 241.21 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 244.94 | 241.92 | 241.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 245.59 | 242.88 | 242.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 11:15:00 | 243.04 | 243.13 | 242.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-05 12:00:00 | 243.04 | 243.13 | 242.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 244.44 | 243.63 | 242.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 244.44 | 243.63 | 242.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 242.33 | 243.46 | 243.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 242.33 | 243.46 | 243.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 242.64 | 243.30 | 243.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:30:00 | 242.45 | 243.30 | 243.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 241.16 | 242.87 | 242.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:00:00 | 241.16 | 242.87 | 242.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 240.70 | 242.44 | 242.67 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 244.03 | 242.91 | 242.77 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 241.65 | 242.58 | 242.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 240.50 | 242.10 | 242.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 10:15:00 | 241.82 | 241.76 | 242.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 10:15:00 | 241.82 | 241.76 | 242.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 241.82 | 241.76 | 242.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 11:00:00 | 241.82 | 241.76 | 242.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 11:15:00 | 241.55 | 241.72 | 242.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 11:30:00 | 241.78 | 241.72 | 242.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 242.10 | 241.80 | 242.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:45:00 | 242.42 | 241.80 | 242.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 241.71 | 241.78 | 242.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 247.15 | 241.78 | 242.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 250.81 | 243.59 | 242.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 252.44 | 246.48 | 244.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 252.35 | 253.01 | 249.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 252.35 | 253.01 | 249.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 251.74 | 252.09 | 250.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 251.29 | 252.09 | 250.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 251.41 | 252.25 | 251.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:00:00 | 251.41 | 252.25 | 251.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 251.23 | 252.05 | 251.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 254.00 | 252.44 | 251.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 251.76 | 253.01 | 253.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 251.76 | 253.01 | 253.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 250.40 | 251.80 | 252.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 251.53 | 251.15 | 251.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 251.53 | 251.15 | 251.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 251.53 | 251.15 | 251.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 251.85 | 251.15 | 251.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 250.35 | 250.99 | 251.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 248.06 | 250.98 | 251.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 249.57 | 248.24 | 249.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:00:00 | 249.85 | 248.78 | 248.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 12:15:00 | 250.46 | 249.12 | 249.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 250.46 | 249.12 | 249.09 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 14:15:00 | 248.20 | 249.19 | 249.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 10:15:00 | 247.65 | 248.71 | 249.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 249.43 | 248.25 | 248.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 249.43 | 248.25 | 248.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 249.43 | 248.25 | 248.56 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 250.32 | 248.77 | 248.75 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 247.13 | 249.31 | 249.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 246.14 | 247.27 | 248.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 247.30 | 246.84 | 247.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 247.30 | 246.84 | 247.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 247.30 | 246.84 | 247.49 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 249.15 | 247.56 | 247.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 250.54 | 248.76 | 248.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 257.34 | 257.34 | 255.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 10:00:00 | 257.34 | 257.34 | 255.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 259.50 | 259.26 | 257.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 260.92 | 259.26 | 257.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:00:00 | 260.39 | 260.10 | 258.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 260.30 | 259.95 | 258.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 263.49 | 263.97 | 264.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 13:15:00 | 263.49 | 263.97 | 264.00 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 268.20 | 264.68 | 264.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 12:15:00 | 269.54 | 267.18 | 266.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 12:15:00 | 268.11 | 268.64 | 267.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 13:00:00 | 268.11 | 268.64 | 267.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 268.00 | 268.65 | 267.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 268.94 | 268.65 | 267.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 11:15:00 | 266.95 | 268.12 | 267.81 | SL hit (close<static) qty=1.00 sl=267.19 alert=retest2 |

### Cycle 159 — SELL (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 12:15:00 | 265.50 | 267.60 | 267.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 265.06 | 266.79 | 267.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 13:15:00 | 265.55 | 265.18 | 266.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 14:00:00 | 265.55 | 265.18 | 266.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 266.11 | 265.36 | 266.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 266.11 | 265.36 | 266.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 266.01 | 265.49 | 266.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 267.35 | 265.49 | 266.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 265.55 | 265.50 | 266.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 266.85 | 265.50 | 266.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 265.35 | 265.47 | 265.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 11:30:00 | 264.95 | 265.26 | 265.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 267.30 | 265.30 | 265.57 | SL hit (close>static) qty=1.00 sl=266.30 alert=retest2 |

### Cycle 160 — BUY (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 14:15:00 | 267.00 | 265.88 | 265.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 09:15:00 | 269.80 | 266.80 | 266.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 267.10 | 267.94 | 267.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 14:15:00 | 267.10 | 267.94 | 267.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 267.10 | 267.94 | 267.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 267.10 | 267.94 | 267.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 267.20 | 267.79 | 267.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 269.50 | 267.79 | 267.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 13:00:00 | 267.80 | 268.78 | 268.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 15:00:00 | 267.75 | 268.43 | 268.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 267.75 | 268.22 | 268.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 267.75 | 268.22 | 268.23 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 269.85 | 268.51 | 268.35 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 11:15:00 | 267.05 | 268.14 | 268.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 265.80 | 267.67 | 268.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 14:15:00 | 267.95 | 267.67 | 267.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 14:15:00 | 267.95 | 267.67 | 267.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 267.95 | 267.67 | 267.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 15:00:00 | 267.95 | 267.67 | 267.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 267.30 | 267.60 | 267.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:45:00 | 265.70 | 267.11 | 267.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 11:15:00 | 252.41 | 257.09 | 260.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 258.80 | 255.66 | 258.03 | SL hit (close>ema200) qty=0.50 sl=255.66 alert=retest2 |

### Cycle 164 — BUY (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 11:15:00 | 260.40 | 258.44 | 258.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 13:15:00 | 262.45 | 259.77 | 259.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 261.00 | 261.18 | 260.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 13:00:00 | 261.00 | 261.18 | 260.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 261.10 | 261.24 | 260.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 15:00:00 | 261.10 | 261.24 | 260.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 258.75 | 260.74 | 260.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 268.80 | 260.74 | 260.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:30:00 | 261.40 | 264.70 | 263.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:15:00 | 262.50 | 264.70 | 263.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:45:00 | 261.50 | 263.73 | 263.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 13:15:00 | 260.25 | 262.68 | 262.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 260.25 | 262.68 | 262.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 259.50 | 261.44 | 262.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 12:15:00 | 260.00 | 259.93 | 260.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 12:15:00 | 260.00 | 259.93 | 260.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 260.00 | 259.93 | 260.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:45:00 | 260.45 | 259.93 | 260.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 260.80 | 260.10 | 260.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:30:00 | 260.70 | 260.10 | 260.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 261.60 | 260.40 | 260.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 261.60 | 260.40 | 260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 261.00 | 260.52 | 260.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 261.65 | 260.52 | 260.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 261.50 | 260.72 | 260.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:45:00 | 262.20 | 260.72 | 260.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 12:15:00 | 261.15 | 260.94 | 260.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 14:15:00 | 261.85 | 261.21 | 261.06 | Break + close above crossover candle high |

### Cycle 167 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 257.80 | 260.69 | 260.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 256.70 | 259.89 | 260.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 14:15:00 | 259.90 | 258.90 | 259.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 14:15:00 | 259.90 | 258.90 | 259.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 259.90 | 258.90 | 259.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:00:00 | 259.90 | 258.90 | 259.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 259.85 | 259.09 | 259.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 252.00 | 259.09 | 259.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 14:15:00 | 239.40 | 242.33 | 243.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 242.30 | 240.82 | 242.11 | SL hit (close>ema200) qty=0.50 sl=240.82 alert=retest2 |

### Cycle 168 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 245.01 | 241.75 | 241.43 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 11:15:00 | 240.58 | 241.45 | 241.56 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 242.47 | 241.73 | 241.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 245.22 | 242.38 | 241.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 244.98 | 246.10 | 244.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 11:15:00 | 244.98 | 246.10 | 244.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 244.98 | 246.10 | 244.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 244.98 | 246.10 | 244.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 245.60 | 245.91 | 245.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 246.08 | 245.91 | 245.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:45:00 | 245.65 | 245.93 | 245.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 10:15:00 | 250.19 | 251.72 | 251.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 250.19 | 251.72 | 251.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 249.33 | 250.50 | 251.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 251.05 | 250.19 | 250.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 251.05 | 250.19 | 250.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 251.05 | 250.19 | 250.66 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 251.64 | 250.94 | 250.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 252.14 | 251.18 | 250.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 250.70 | 251.26 | 251.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 250.70 | 251.26 | 251.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 250.70 | 251.26 | 251.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 250.70 | 251.26 | 251.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 251.00 | 251.21 | 251.06 | EMA400 retest candle locked (from upside) |

### Cycle 173 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 249.70 | 250.84 | 250.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 12:15:00 | 249.15 | 250.39 | 250.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 248.19 | 244.25 | 245.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 248.19 | 244.25 | 245.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 248.19 | 244.25 | 245.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 248.29 | 244.25 | 245.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 248.39 | 245.90 | 245.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 255.20 | 249.28 | 247.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 253.15 | 254.02 | 251.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 251.60 | 253.23 | 252.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 251.60 | 253.23 | 252.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 251.35 | 253.23 | 252.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 252.04 | 252.99 | 252.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:30:00 | 251.44 | 252.99 | 252.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 251.65 | 252.54 | 252.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:45:00 | 251.57 | 252.54 | 252.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 251.93 | 252.34 | 252.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 251.91 | 252.34 | 252.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 252.00 | 252.27 | 252.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 250.60 | 252.27 | 252.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 250.50 | 251.92 | 251.95 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 252.75 | 251.67 | 251.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 254.20 | 252.33 | 251.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 253.71 | 253.93 | 253.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 253.71 | 253.93 | 253.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 255.14 | 255.80 | 255.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:45:00 | 254.86 | 255.80 | 255.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 255.34 | 255.71 | 255.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:30:00 | 255.24 | 255.71 | 255.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 255.76 | 255.72 | 255.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:45:00 | 255.00 | 255.72 | 255.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 250.73 | 254.81 | 254.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 246.29 | 249.48 | 251.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 239.20 | 238.15 | 241.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:45:00 | 238.87 | 238.15 | 241.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 240.32 | 239.07 | 240.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 240.32 | 239.07 | 240.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 239.74 | 239.20 | 240.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:00:00 | 238.78 | 239.53 | 240.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 240.80 | 239.88 | 240.00 | SL hit (close>static) qty=1.00 sl=240.70 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 241.80 | 240.27 | 240.17 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 10:15:00 | 239.66 | 240.04 | 240.08 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 240.90 | 240.23 | 240.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 10:15:00 | 241.80 | 240.71 | 240.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 245.15 | 247.79 | 246.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 245.15 | 247.79 | 246.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 245.15 | 247.79 | 246.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:45:00 | 245.13 | 247.79 | 246.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 244.86 | 247.20 | 246.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 244.86 | 247.20 | 246.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 244.74 | 246.07 | 246.11 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 10:15:00 | 248.91 | 246.40 | 246.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 09:15:00 | 250.60 | 248.25 | 247.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 249.38 | 249.45 | 248.50 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 11:15:00 | 251.72 | 249.65 | 248.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 243.00 | 250.38 | 249.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 243.00 | 250.38 | 249.80 | SL hit (close<ema400) qty=1.00 sl=249.80 alert=retest1 |

### Cycle 183 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 242.33 | 248.77 | 249.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 11:15:00 | 241.45 | 247.31 | 248.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 245.95 | 242.45 | 243.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 245.95 | 242.45 | 243.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 245.95 | 242.45 | 243.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 245.95 | 242.45 | 243.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 247.12 | 243.39 | 243.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 246.34 | 243.39 | 243.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 247.22 | 244.15 | 244.12 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 242.45 | 244.09 | 244.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 241.89 | 243.18 | 243.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 15:15:00 | 242.70 | 242.68 | 243.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 09:15:00 | 243.12 | 242.68 | 243.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 242.05 | 242.55 | 243.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:45:00 | 241.42 | 242.26 | 242.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:30:00 | 241.44 | 242.17 | 242.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:15:00 | 241.61 | 242.17 | 242.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:45:00 | 241.55 | 241.85 | 242.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 238.84 | 238.71 | 239.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:15:00 | 238.50 | 238.71 | 239.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 12:15:00 | 240.25 | 239.09 | 239.60 | SL hit (close>static) qty=1.00 sl=240.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 240.00 | 238.82 | 238.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 241.11 | 239.80 | 239.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 242.72 | 244.73 | 243.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 242.72 | 244.73 | 243.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 242.72 | 244.73 | 243.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:15:00 | 245.15 | 243.98 | 243.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 241.91 | 243.47 | 243.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 241.91 | 243.47 | 243.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 240.92 | 242.22 | 242.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 243.12 | 242.21 | 242.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 243.12 | 242.21 | 242.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 243.12 | 242.21 | 242.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 243.12 | 242.21 | 242.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 245.00 | 242.76 | 242.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 244.55 | 242.76 | 242.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 245.93 | 243.40 | 243.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 12:15:00 | 246.29 | 243.98 | 243.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 244.14 | 245.63 | 245.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 244.14 | 245.63 | 245.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 244.14 | 245.63 | 245.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 244.00 | 245.63 | 245.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 246.01 | 245.71 | 245.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:15:00 | 247.19 | 245.68 | 245.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 248.30 | 245.23 | 245.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 11:15:00 | 246.55 | 246.57 | 246.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 14:00:00 | 246.47 | 246.50 | 246.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 245.56 | 246.31 | 246.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 245.56 | 246.31 | 246.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 245.59 | 246.17 | 246.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:30:00 | 248.30 | 246.52 | 246.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 15:15:00 | 257.39 | 258.54 | 258.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 15:15:00 | 257.39 | 258.54 | 258.61 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 259.34 | 258.70 | 258.68 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 258.34 | 258.63 | 258.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 257.72 | 258.35 | 258.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 258.96 | 258.47 | 258.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 258.96 | 258.47 | 258.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 258.96 | 258.47 | 258.55 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 259.06 | 258.66 | 258.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 260.04 | 259.11 | 258.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 260.38 | 260.87 | 260.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 260.38 | 260.87 | 260.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 260.38 | 260.87 | 260.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 260.38 | 260.87 | 260.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 260.18 | 260.73 | 260.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:45:00 | 260.10 | 260.73 | 260.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 260.26 | 260.64 | 260.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 260.22 | 260.64 | 260.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 260.00 | 260.51 | 260.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:45:00 | 260.10 | 260.51 | 260.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 259.42 | 260.29 | 260.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 259.42 | 260.29 | 260.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 259.07 | 260.05 | 260.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 259.07 | 260.05 | 260.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 259.30 | 259.90 | 259.96 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 260.88 | 260.09 | 260.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 09:15:00 | 264.02 | 261.35 | 260.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 13:15:00 | 262.08 | 262.27 | 261.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-18 14:00:00 | 262.08 | 262.27 | 261.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 269.41 | 270.89 | 269.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:30:00 | 268.91 | 270.89 | 269.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 268.70 | 270.45 | 269.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 268.70 | 270.45 | 269.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 268.10 | 269.98 | 269.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:45:00 | 268.14 | 269.98 | 269.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 268.14 | 268.83 | 268.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 267.83 | 268.60 | 268.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 15:15:00 | 263.80 | 263.10 | 264.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 15:15:00 | 263.80 | 263.10 | 264.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 263.80 | 263.10 | 264.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 266.45 | 263.10 | 264.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 265.35 | 263.55 | 264.15 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 11:15:00 | 266.55 | 264.61 | 264.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 268.95 | 266.77 | 265.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 263.20 | 267.83 | 267.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 263.20 | 267.83 | 267.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 263.20 | 267.83 | 267.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:45:00 | 261.80 | 267.83 | 267.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 263.80 | 267.02 | 266.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 262.70 | 267.02 | 266.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 11:15:00 | 264.40 | 266.50 | 266.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 15:15:00 | 263.00 | 264.70 | 265.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 265.00 | 264.76 | 265.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 265.00 | 264.76 | 265.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 265.00 | 264.76 | 265.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 265.00 | 264.76 | 265.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 265.70 | 264.84 | 265.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 265.70 | 264.84 | 265.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 265.10 | 264.89 | 265.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 267.60 | 264.89 | 265.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 271.00 | 266.11 | 265.81 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 263.70 | 266.71 | 266.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 263.00 | 265.97 | 266.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 263.70 | 262.66 | 263.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 263.70 | 262.66 | 263.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 263.70 | 262.66 | 263.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 264.00 | 262.66 | 263.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 263.55 | 262.83 | 263.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:30:00 | 264.30 | 262.83 | 263.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 263.05 | 262.88 | 263.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 262.35 | 262.96 | 263.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 262.25 | 262.96 | 263.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 262.10 | 262.86 | 263.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 264.25 | 263.14 | 263.37 | SL hit (close>static) qty=1.00 sl=263.75 alert=retest2 |

### Cycle 200 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 268.95 | 262.85 | 262.82 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 250.00 | 263.09 | 263.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 239.85 | 246.21 | 252.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 241.50 | 240.43 | 243.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:30:00 | 241.20 | 240.43 | 243.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 242.80 | 240.97 | 242.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:30:00 | 241.30 | 240.92 | 242.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 240.30 | 237.75 | 237.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 240.30 | 237.75 | 237.55 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 236.85 | 237.62 | 237.64 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 12:15:00 | 243.38 | 238.58 | 237.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 247.35 | 242.69 | 240.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 14:15:00 | 242.59 | 243.38 | 242.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-03 15:00:00 | 242.59 | 243.38 | 242.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 205 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 233.70 | 241.31 | 241.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 220.50 | 228.44 | 230.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 213.24 | 213.17 | 216.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 213.24 | 213.17 | 216.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 217.47 | 214.03 | 216.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 217.47 | 214.03 | 216.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 215.91 | 214.41 | 216.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:15:00 | 217.30 | 214.41 | 216.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 217.68 | 215.06 | 216.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:30:00 | 217.00 | 215.06 | 216.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 218.00 | 215.65 | 216.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:45:00 | 218.64 | 215.65 | 216.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 211.08 | 214.80 | 215.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 208.62 | 212.10 | 213.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 12:45:00 | 210.42 | 211.14 | 212.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 199.90 | 205.66 | 208.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 204.30 | 202.30 | 204.74 | SL hit (close>ema200) qty=0.50 sl=202.30 alert=retest2 |

### Cycle 206 — BUY (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 14:15:00 | 198.93 | 196.71 | 196.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 10:15:00 | 199.11 | 197.65 | 197.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 200.61 | 201.54 | 200.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 200.61 | 201.54 | 200.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 200.61 | 201.54 | 200.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:30:00 | 202.06 | 201.76 | 200.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 202.63 | 202.24 | 201.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 199.08 | 201.04 | 200.92 | SL hit (close<static) qty=1.00 sl=199.70 alert=retest2 |

### Cycle 207 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 197.39 | 200.31 | 200.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 197.00 | 199.65 | 200.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 196.46 | 192.90 | 194.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 196.46 | 192.90 | 194.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 196.46 | 192.90 | 194.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 196.08 | 192.90 | 194.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 197.09 | 193.73 | 194.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 197.09 | 193.73 | 194.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 190.63 | 190.25 | 191.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 189.49 | 190.26 | 191.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 189.49 | 190.00 | 191.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 188.69 | 190.31 | 191.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:45:00 | 189.39 | 190.20 | 191.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 188.65 | 188.64 | 189.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 188.06 | 188.59 | 189.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 10:15:00 | 190.59 | 189.78 | 189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 10:15:00 | 190.59 | 189.78 | 189.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 11:15:00 | 190.95 | 190.02 | 189.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 190.69 | 190.73 | 190.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 190.69 | 190.73 | 190.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 190.69 | 190.73 | 190.30 | EMA400 retest candle locked (from upside) |

### Cycle 209 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 187.79 | 189.90 | 190.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 186.90 | 188.97 | 189.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 193.86 | 189.95 | 189.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 193.86 | 189.95 | 189.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 193.86 | 189.95 | 189.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 193.60 | 189.95 | 189.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 193.02 | 190.57 | 190.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 195.13 | 192.44 | 191.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 13:15:00 | 202.98 | 203.09 | 200.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 14:00:00 | 202.98 | 203.09 | 200.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 202.75 | 203.13 | 202.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 202.48 | 203.13 | 202.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 201.04 | 202.78 | 202.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 201.04 | 202.78 | 202.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 201.95 | 202.62 | 202.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:15:00 | 202.00 | 202.62 | 202.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 202.39 | 203.05 | 202.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 204.10 | 206.45 | 206.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 12:15:00 | 204.10 | 206.45 | 206.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 09:15:00 | 203.00 | 204.93 | 205.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 203.78 | 203.43 | 204.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:30:00 | 203.73 | 203.43 | 204.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 205.20 | 203.78 | 204.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 205.25 | 203.78 | 204.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 204.96 | 204.02 | 204.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:00:00 | 204.96 | 204.02 | 204.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 205.12 | 204.32 | 204.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 205.12 | 204.32 | 204.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 204.98 | 204.45 | 204.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:00:00 | 204.98 | 204.45 | 204.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 203.71 | 203.88 | 204.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:45:00 | 204.16 | 203.88 | 204.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 204.05 | 203.91 | 204.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:45:00 | 204.20 | 203.91 | 204.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 203.83 | 203.91 | 204.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:15:00 | 202.92 | 203.85 | 204.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 15:15:00 | 202.87 | 203.77 | 204.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 202.94 | 201.29 | 201.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:45:00 | 202.97 | 201.63 | 202.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 204.79 | 202.26 | 202.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 204.79 | 202.26 | 202.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 14:15:00 | 205.18 | 203.20 | 202.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 12:15:00 | 203.39 | 203.99 | 203.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 12:15:00 | 203.39 | 203.99 | 203.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 203.39 | 203.99 | 203.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:45:00 | 203.24 | 203.99 | 203.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 202.17 | 203.63 | 203.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 202.17 | 203.63 | 203.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 201.60 | 203.22 | 203.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 201.60 | 203.22 | 203.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 201.70 | 202.92 | 202.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 200.98 | 202.20 | 202.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 11:15:00 | 201.37 | 201.32 | 201.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 12:00:00 | 201.37 | 201.32 | 201.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 201.53 | 201.36 | 201.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:45:00 | 201.80 | 201.36 | 201.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 201.90 | 201.47 | 201.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:00:00 | 201.90 | 201.47 | 201.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 200.40 | 201.26 | 201.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 201.46 | 201.26 | 201.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 201.24 | 201.22 | 201.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:15:00 | 200.92 | 201.22 | 201.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 200.64 | 201.15 | 201.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:30:00 | 200.58 | 200.94 | 201.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:00:00 | 200.82 | 200.94 | 201.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 200.22 | 200.26 | 200.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:45:00 | 199.89 | 200.17 | 200.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:00:00 | 199.87 | 200.11 | 200.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-15 10:45:00 | 192.73 | 2023-05-17 10:15:00 | 191.48 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-05-24 13:30:00 | 199.60 | 2023-06-05 15:15:00 | 202.28 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2023-05-26 10:15:00 | 199.55 | 2023-06-05 15:15:00 | 202.28 | STOP_HIT | 1.00 | 1.37% |
| BUY | retest2 | 2023-05-26 11:30:00 | 199.78 | 2023-06-05 15:15:00 | 202.28 | STOP_HIT | 1.00 | 1.25% |
| SELL | retest2 | 2023-06-08 09:30:00 | 200.00 | 2023-06-16 14:15:00 | 190.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-08 09:30:00 | 200.00 | 2023-06-20 09:15:00 | 191.58 | STOP_HIT | 0.50 | 4.21% |
| SELL | retest2 | 2023-07-12 12:15:00 | 195.63 | 2023-07-12 13:15:00 | 196.78 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2023-07-20 14:15:00 | 208.50 | 2023-07-21 09:15:00 | 204.23 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2023-07-20 15:00:00 | 208.90 | 2023-07-21 09:15:00 | 204.23 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2023-07-27 11:15:00 | 201.28 | 2023-07-31 13:15:00 | 201.93 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2023-07-27 11:45:00 | 201.60 | 2023-07-31 13:15:00 | 201.93 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2023-07-27 14:00:00 | 201.40 | 2023-07-31 13:15:00 | 201.93 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2023-07-31 11:30:00 | 201.53 | 2023-07-31 13:15:00 | 201.93 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-08-09 10:45:00 | 208.35 | 2023-08-11 09:15:00 | 207.80 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2023-08-24 09:15:00 | 209.38 | 2023-08-24 14:15:00 | 206.40 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2023-08-24 11:45:00 | 207.73 | 2023-08-24 14:15:00 | 206.40 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2023-08-24 12:30:00 | 207.98 | 2023-08-24 14:15:00 | 206.40 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-08-30 14:45:00 | 203.95 | 2023-09-01 10:15:00 | 205.65 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2023-08-31 12:00:00 | 203.95 | 2023-09-01 10:15:00 | 205.65 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2023-08-31 14:30:00 | 203.88 | 2023-09-01 10:15:00 | 205.65 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2023-09-01 09:15:00 | 204.15 | 2023-09-01 10:15:00 | 205.65 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2023-09-06 09:15:00 | 216.80 | 2023-09-06 12:15:00 | 213.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2023-09-06 14:45:00 | 213.73 | 2023-09-18 14:15:00 | 217.85 | STOP_HIT | 1.00 | 1.93% |
| SELL | retest2 | 2023-09-28 11:00:00 | 206.15 | 2023-10-05 15:15:00 | 203.50 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2023-11-06 09:15:00 | 193.05 | 2023-11-08 11:15:00 | 191.70 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2023-11-06 12:30:00 | 192.38 | 2023-11-08 11:15:00 | 191.70 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2023-11-07 13:45:00 | 192.05 | 2023-11-08 11:15:00 | 191.70 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2023-11-07 15:15:00 | 192.00 | 2023-11-08 11:15:00 | 191.70 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2023-11-13 09:15:00 | 190.45 | 2023-11-15 09:15:00 | 194.50 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2023-11-13 10:15:00 | 190.48 | 2023-11-15 09:15:00 | 194.50 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2023-11-20 10:15:00 | 200.35 | 2023-11-24 11:15:00 | 199.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-11-21 09:15:00 | 201.03 | 2023-11-24 11:15:00 | 199.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2023-11-22 11:15:00 | 200.38 | 2023-11-24 11:15:00 | 199.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-11-22 14:15:00 | 200.23 | 2023-11-24 11:15:00 | 199.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-12-12 09:45:00 | 211.85 | 2023-12-13 11:15:00 | 208.18 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2023-12-12 15:00:00 | 211.25 | 2023-12-13 11:15:00 | 208.18 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2023-12-13 09:15:00 | 211.55 | 2023-12-13 11:15:00 | 208.18 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2023-12-20 09:15:00 | 222.00 | 2023-12-20 13:15:00 | 218.40 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-01-01 12:45:00 | 239.33 | 2024-01-02 12:15:00 | 233.43 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-01-02 09:15:00 | 238.40 | 2024-01-02 12:15:00 | 233.43 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-01-05 13:30:00 | 228.20 | 2024-01-12 09:15:00 | 233.00 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-01-05 15:00:00 | 228.35 | 2024-01-12 09:15:00 | 233.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-01-08 09:30:00 | 227.80 | 2024-01-12 09:15:00 | 233.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-01-08 10:15:00 | 228.40 | 2024-01-12 09:15:00 | 233.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-01-09 12:45:00 | 227.43 | 2024-01-12 09:15:00 | 233.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-01-09 13:45:00 | 226.88 | 2024-01-12 09:15:00 | 233.00 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-01-17 11:15:00 | 244.28 | 2024-01-20 11:15:00 | 240.45 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-01-18 11:45:00 | 243.65 | 2024-01-20 11:15:00 | 240.45 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-01-19 09:15:00 | 244.38 | 2024-01-20 11:15:00 | 240.45 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-01-19 10:15:00 | 244.08 | 2024-01-20 11:15:00 | 240.45 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-01-23 11:30:00 | 239.40 | 2024-01-30 09:15:00 | 239.95 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-02-13 13:15:00 | 250.80 | 2024-02-21 13:15:00 | 262.80 | STOP_HIT | 1.00 | 4.78% |
| BUY | retest2 | 2024-02-14 09:45:00 | 251.18 | 2024-02-21 13:15:00 | 262.80 | STOP_HIT | 1.00 | 4.63% |
| SELL | retest2 | 2024-03-01 10:30:00 | 259.68 | 2024-03-04 09:15:00 | 262.83 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-03-01 11:15:00 | 259.70 | 2024-03-04 09:15:00 | 262.83 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-03-01 14:15:00 | 259.83 | 2024-03-04 09:15:00 | 262.83 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-03-13 11:15:00 | 253.93 | 2024-03-14 14:15:00 | 259.13 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-03-22 09:15:00 | 240.83 | 2024-04-01 12:15:00 | 241.73 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-05-03 10:45:00 | 229.23 | 2024-05-07 13:15:00 | 232.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-05-03 11:15:00 | 228.50 | 2024-05-07 13:15:00 | 232.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-05-06 13:15:00 | 229.25 | 2024-05-07 13:15:00 | 232.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-05-07 10:15:00 | 228.68 | 2024-05-07 13:15:00 | 232.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-05-09 11:15:00 | 230.30 | 2024-05-09 13:15:00 | 229.08 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-06-04 09:15:00 | 217.38 | 2024-06-04 12:15:00 | 209.19 | PARTIAL | 0.50 | 3.77% |
| SELL | retest2 | 2024-06-04 09:15:00 | 217.38 | 2024-06-04 14:15:00 | 219.88 | STOP_HIT | 0.50 | -1.15% |
| SELL | retest2 | 2024-06-04 10:30:00 | 220.20 | 2024-06-05 10:15:00 | 225.60 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-06-13 09:15:00 | 242.33 | 2024-06-14 15:15:00 | 238.58 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-06-21 09:15:00 | 248.68 | 2024-06-21 14:15:00 | 244.98 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-06-21 12:45:00 | 246.40 | 2024-06-21 14:15:00 | 244.98 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-06-21 14:45:00 | 246.80 | 2024-06-24 14:15:00 | 245.28 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-06-24 09:15:00 | 246.03 | 2024-06-24 14:15:00 | 245.28 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-30 11:00:00 | 262.88 | 2024-07-30 14:15:00 | 260.50 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest1 | 2024-07-30 12:00:00 | 262.40 | 2024-07-30 14:15:00 | 260.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-08-01 09:15:00 | 262.15 | 2024-08-01 13:15:00 | 259.93 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-08-07 12:45:00 | 247.65 | 2024-08-14 10:15:00 | 246.18 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2024-08-08 09:15:00 | 246.60 | 2024-08-14 10:15:00 | 246.18 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2024-09-03 11:45:00 | 269.50 | 2024-09-04 09:15:00 | 262.48 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2024-09-03 14:15:00 | 269.25 | 2024-09-04 09:15:00 | 262.48 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-09-06 13:45:00 | 260.75 | 2024-09-10 14:15:00 | 263.02 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-09-24 09:30:00 | 265.20 | 2024-09-24 14:15:00 | 269.55 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-10-01 09:30:00 | 272.65 | 2024-10-03 09:15:00 | 268.77 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-10-04 12:30:00 | 268.25 | 2024-10-09 13:15:00 | 266.50 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2024-10-07 10:00:00 | 268.00 | 2024-10-09 13:15:00 | 266.50 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2024-10-23 10:00:00 | 274.75 | 2024-10-25 09:15:00 | 273.20 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-10-24 09:30:00 | 274.73 | 2024-10-25 09:15:00 | 273.20 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-10-25 09:15:00 | 274.70 | 2024-10-25 09:15:00 | 273.20 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-12-03 14:45:00 | 291.55 | 2024-12-20 12:15:00 | 306.80 | STOP_HIT | 1.00 | 5.23% |
| SELL | retest2 | 2024-12-24 12:15:00 | 305.10 | 2024-12-27 10:15:00 | 310.50 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2024-12-24 12:45:00 | 305.25 | 2024-12-27 10:15:00 | 310.50 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-12-26 10:15:00 | 304.80 | 2024-12-27 10:15:00 | 310.50 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-12-26 11:30:00 | 304.95 | 2024-12-27 10:15:00 | 310.50 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-01-02 09:15:00 | 298.95 | 2025-01-02 14:15:00 | 303.95 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-01-03 10:00:00 | 298.95 | 2025-01-10 10:15:00 | 300.75 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-01-10 09:30:00 | 296.90 | 2025-01-10 10:15:00 | 300.75 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-01-15 09:30:00 | 291.20 | 2025-01-20 09:15:00 | 302.00 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-01-15 11:00:00 | 291.40 | 2025-01-20 09:15:00 | 302.00 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2025-01-16 09:15:00 | 292.25 | 2025-01-20 09:15:00 | 302.00 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2025-02-06 12:30:00 | 318.80 | 2025-02-11 09:15:00 | 316.35 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-02-07 12:30:00 | 318.55 | 2025-02-11 09:15:00 | 316.35 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-02-07 13:00:00 | 318.20 | 2025-02-11 09:15:00 | 316.35 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-02-07 13:30:00 | 318.70 | 2025-02-11 09:15:00 | 316.35 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-02-10 09:15:00 | 319.45 | 2025-02-11 10:15:00 | 313.85 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-02-10 09:45:00 | 318.60 | 2025-02-11 10:15:00 | 313.85 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-02-10 10:15:00 | 318.85 | 2025-02-11 10:15:00 | 313.85 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-02-10 14:30:00 | 319.40 | 2025-02-11 10:15:00 | 313.85 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-02-13 12:15:00 | 309.30 | 2025-02-18 13:15:00 | 311.15 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-02-13 13:30:00 | 308.70 | 2025-02-18 13:15:00 | 311.15 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-02-14 09:30:00 | 309.25 | 2025-02-18 13:15:00 | 311.15 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-02-14 10:15:00 | 308.85 | 2025-02-18 13:15:00 | 311.15 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-02-17 10:45:00 | 303.35 | 2025-02-18 13:15:00 | 311.15 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest1 | 2025-02-27 13:15:00 | 291.45 | 2025-03-04 09:15:00 | 276.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-27 13:15:00 | 291.45 | 2025-03-05 09:15:00 | 283.25 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2025-02-28 09:15:00 | 285.80 | 2025-03-05 15:15:00 | 284.60 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest1 | 2025-03-12 09:15:00 | 269.70 | 2025-03-19 13:15:00 | 264.75 | STOP_HIT | 1.00 | 1.84% |
| SELL | retest2 | 2025-03-19 11:30:00 | 260.00 | 2025-03-19 13:15:00 | 264.75 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-03-19 12:00:00 | 260.00 | 2025-03-19 13:15:00 | 264.75 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-03-21 12:15:00 | 269.15 | 2025-03-26 15:15:00 | 267.50 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-03-24 13:00:00 | 269.90 | 2025-03-26 15:15:00 | 267.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-03-26 12:15:00 | 268.70 | 2025-03-26 15:15:00 | 267.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-03-26 13:15:00 | 269.00 | 2025-03-26 15:15:00 | 267.50 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-04-02 13:30:00 | 263.00 | 2025-04-04 09:15:00 | 249.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 09:15:00 | 259.70 | 2025-04-04 12:15:00 | 246.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-02 13:30:00 | 263.00 | 2025-04-07 09:15:00 | 236.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-03 09:15:00 | 259.70 | 2025-04-07 09:15:00 | 233.73 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-25 12:00:00 | 241.10 | 2025-04-28 10:15:00 | 239.95 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-04-25 12:45:00 | 241.50 | 2025-04-28 10:15:00 | 239.95 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-04-28 10:00:00 | 241.25 | 2025-04-28 10:15:00 | 239.95 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-05-15 13:00:00 | 254.00 | 2025-05-19 13:15:00 | 251.76 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-05-22 09:15:00 | 248.06 | 2025-05-26 12:15:00 | 250.46 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-05-23 11:15:00 | 249.57 | 2025-05-26 12:15:00 | 250.46 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-05-26 12:00:00 | 249.85 | 2025-05-26 12:15:00 | 250.46 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-06-13 10:15:00 | 260.92 | 2025-06-23 13:15:00 | 263.49 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2025-06-13 15:00:00 | 260.39 | 2025-06-23 13:15:00 | 263.49 | STOP_HIT | 1.00 | 1.19% |
| BUY | retest2 | 2025-06-16 10:15:00 | 260.30 | 2025-06-23 13:15:00 | 263.49 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2025-06-27 09:15:00 | 268.94 | 2025-06-27 11:15:00 | 266.95 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-07-01 11:30:00 | 264.95 | 2025-07-02 09:15:00 | 267.30 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-07-02 13:15:00 | 264.85 | 2025-07-02 14:15:00 | 267.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-07-04 09:15:00 | 269.50 | 2025-07-08 09:15:00 | 267.75 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-07-07 13:00:00 | 267.80 | 2025-07-08 09:15:00 | 267.75 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-07-07 15:00:00 | 267.75 | 2025-07-08 09:15:00 | 267.75 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-07-10 09:45:00 | 265.70 | 2025-07-14 11:15:00 | 252.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-10 09:45:00 | 265.70 | 2025-07-15 10:15:00 | 258.80 | STOP_HIT | 0.50 | 2.60% |
| BUY | retest2 | 2025-07-18 09:15:00 | 268.80 | 2025-07-21 13:15:00 | 260.25 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2025-07-21 09:30:00 | 261.40 | 2025-07-21 13:15:00 | 260.25 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-07-21 10:15:00 | 262.50 | 2025-07-21 13:15:00 | 260.25 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-07-21 11:45:00 | 261.50 | 2025-07-21 13:15:00 | 260.25 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-07-28 09:15:00 | 252.00 | 2025-08-06 14:15:00 | 239.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 09:15:00 | 252.00 | 2025-08-07 14:15:00 | 242.30 | STOP_HIT | 0.50 | 3.85% |
| BUY | retest2 | 2025-08-19 09:15:00 | 246.08 | 2025-08-28 10:15:00 | 250.19 | STOP_HIT | 1.00 | 1.67% |
| BUY | retest2 | 2025-08-19 09:45:00 | 245.65 | 2025-08-28 10:15:00 | 250.19 | STOP_HIT | 1.00 | 1.85% |
| SELL | retest2 | 2025-10-01 10:00:00 | 238.78 | 2025-10-01 14:15:00 | 240.80 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest1 | 2025-10-16 11:15:00 | 251.72 | 2025-10-17 09:15:00 | 243.00 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2025-10-30 09:45:00 | 241.42 | 2025-11-06 12:15:00 | 240.25 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2025-10-30 10:30:00 | 241.44 | 2025-11-10 10:15:00 | 240.46 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-10-30 11:15:00 | 241.61 | 2025-11-10 12:15:00 | 240.00 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-10-31 10:45:00 | 241.55 | 2025-11-10 12:15:00 | 240.00 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-11-06 10:15:00 | 238.50 | 2025-11-10 12:15:00 | 240.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-11-07 09:15:00 | 237.00 | 2025-11-10 12:15:00 | 240.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-11-14 15:15:00 | 245.15 | 2025-11-18 09:15:00 | 241.91 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-11-21 12:15:00 | 247.19 | 2025-12-09 15:15:00 | 257.39 | STOP_HIT | 1.00 | 4.13% |
| BUY | retest2 | 2025-11-24 09:15:00 | 248.30 | 2025-12-09 15:15:00 | 257.39 | STOP_HIT | 1.00 | 3.66% |
| BUY | retest2 | 2025-11-25 11:15:00 | 246.55 | 2025-12-09 15:15:00 | 257.39 | STOP_HIT | 1.00 | 4.40% |
| BUY | retest2 | 2025-11-25 14:00:00 | 246.47 | 2025-12-09 15:15:00 | 257.39 | STOP_HIT | 1.00 | 4.43% |
| BUY | retest2 | 2025-11-26 09:30:00 | 248.30 | 2025-12-09 15:15:00 | 257.39 | STOP_HIT | 1.00 | 3.66% |
| SELL | retest2 | 2026-01-13 09:30:00 | 262.35 | 2026-01-13 14:15:00 | 264.25 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-01-13 12:45:00 | 262.25 | 2026-01-13 14:15:00 | 264.25 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-01-13 14:15:00 | 262.10 | 2026-01-13 14:15:00 | 264.25 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-01-14 09:15:00 | 261.45 | 2026-01-16 09:15:00 | 268.95 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-01-23 10:30:00 | 241.30 | 2026-01-29 13:15:00 | 240.30 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2026-02-20 09:15:00 | 208.62 | 2026-02-24 09:15:00 | 199.90 | PARTIAL | 0.50 | 4.18% |
| SELL | retest2 | 2026-02-20 09:15:00 | 208.62 | 2026-02-25 09:15:00 | 204.30 | STOP_HIT | 0.50 | 2.07% |
| SELL | retest2 | 2026-02-20 12:45:00 | 210.42 | 2026-03-02 09:15:00 | 198.19 | PARTIAL | 0.50 | 5.81% |
| SELL | retest2 | 2026-02-20 12:45:00 | 210.42 | 2026-03-05 14:15:00 | 195.41 | STOP_HIT | 0.50 | 7.13% |
| BUY | retest2 | 2026-03-12 10:30:00 | 202.06 | 2026-03-13 11:15:00 | 199.08 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-03-12 15:00:00 | 202.63 | 2026-03-13 11:15:00 | 199.08 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-03-20 12:15:00 | 189.49 | 2026-03-27 10:15:00 | 190.59 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-03-20 13:30:00 | 189.49 | 2026-03-27 10:15:00 | 190.59 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-03-23 09:15:00 | 188.69 | 2026-03-27 10:15:00 | 190.59 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-03-23 09:45:00 | 189.39 | 2026-03-27 10:15:00 | 190.59 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-03-24 10:30:00 | 188.06 | 2026-03-27 10:15:00 | 190.59 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-04-10 12:15:00 | 202.00 | 2026-04-17 12:15:00 | 204.10 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2026-04-13 11:00:00 | 202.39 | 2026-04-17 12:15:00 | 204.10 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2026-04-23 14:15:00 | 202.92 | 2026-04-27 12:15:00 | 204.79 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-04-23 15:15:00 | 202.87 | 2026-04-27 12:15:00 | 204.79 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-04-27 11:15:00 | 202.94 | 2026-04-27 12:15:00 | 204.79 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-04-27 11:45:00 | 202.97 | 2026-04-27 12:15:00 | 204.79 | STOP_HIT | 1.00 | -0.90% |
