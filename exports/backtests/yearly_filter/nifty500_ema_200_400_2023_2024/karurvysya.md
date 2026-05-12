# Karur Vysya Bank Ltd. (KARURVYSYA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 304.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 5 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 26
- **Target hits / Stop hits / Partials:** 2 / 26 / 0
- **Avg / median % per leg:** -1.43% / -2.00%
- **Sum % (uncompounded):** -40.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 2 | 14.3% | 2 | 12 | 0 | -0.17% | -2.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 2 | 14.3% | 2 | 12 | 0 | -0.17% | -2.4% |
| SELL (all) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.69% | -37.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.69% | -37.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 2 | 7.1% | 2 | 26 | 0 | -1.43% | -40.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 165.63 | 177.64 | 177.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 10:15:00 | 163.75 | 177.26 | 177.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 14:15:00 | 179.83 | 174.20 | 175.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 14:15:00 | 179.83 | 174.20 | 175.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 179.83 | 174.20 | 175.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 15:00:00 | 179.83 | 174.20 | 175.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 178.75 | 174.24 | 175.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 09:15:00 | 175.04 | 174.24 | 175.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 10:30:00 | 177.60 | 174.31 | 175.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 11:00:00 | 177.66 | 174.31 | 175.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 12:15:00 | 181.94 | 174.44 | 175.85 | SL hit (close>static) qty=1.00 sl=181.67 alert=retest2 |

### Cycle 2 — BUY (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 11:15:00 | 186.29 | 177.01 | 177.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 189.48 | 177.43 | 177.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 178.76 | 182.73 | 180.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 09:15:00 | 178.76 | 182.73 | 180.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 178.76 | 182.73 | 180.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:00:00 | 178.76 | 182.73 | 180.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 178.98 | 182.70 | 180.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 09:15:00 | 181.75 | 181.69 | 180.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 09:15:00 | 176.98 | 181.63 | 180.06 | SL hit (close<static) qty=1.00 sl=178.02 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 11:15:00 | 176.54 | 185.48 | 185.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 14:15:00 | 174.61 | 185.20 | 185.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 14:15:00 | 183.53 | 183.38 | 184.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 14:15:00 | 183.53 | 183.38 | 184.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 183.53 | 183.38 | 184.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 14:45:00 | 183.62 | 183.38 | 184.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 184.58 | 183.39 | 184.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:15:00 | 182.43 | 183.39 | 184.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 14:15:00 | 182.63 | 183.35 | 184.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 10:15:00 | 186.07 | 183.37 | 184.31 | SL hit (close>static) qty=1.00 sl=185.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 11:15:00 | 189.71 | 185.06 | 185.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 10:15:00 | 191.50 | 185.35 | 185.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 11:15:00 | 189.64 | 189.66 | 187.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-11 12:00:00 | 189.64 | 189.66 | 187.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 13:15:00 | 187.91 | 189.63 | 187.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 14:00:00 | 187.91 | 189.63 | 187.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 189.07 | 189.62 | 187.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 14:30:00 | 186.35 | 189.62 | 187.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 185.76 | 189.60 | 187.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:00:00 | 185.76 | 189.60 | 187.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 187.10 | 189.57 | 187.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 11:30:00 | 188.23 | 189.56 | 187.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 12:30:00 | 188.18 | 189.55 | 187.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 189.96 | 189.48 | 187.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 188.26 | 189.44 | 187.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 187.09 | 189.42 | 187.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:45:00 | 187.41 | 189.42 | 187.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 185.10 | 189.37 | 187.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-14 10:15:00 | 185.10 | 189.37 | 187.81 | SL hit (close<static) qty=1.00 sl=185.21 alert=retest2 |

### Cycle 5 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 173.66 | 186.55 | 186.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 169.75 | 184.36 | 185.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 10:15:00 | 173.24 | 172.73 | 177.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 11:00:00 | 173.24 | 172.73 | 177.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 177.63 | 172.98 | 177.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 13:15:00 | 176.36 | 173.10 | 177.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 15:00:00 | 176.17 | 173.17 | 177.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 15:00:00 | 174.65 | 173.42 | 177.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:15:00 | 176.37 | 173.55 | 177.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 175.21 | 173.57 | 177.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-03 13:15:00 | 179.25 | 174.25 | 177.33 | SL hit (close>static) qty=1.00 sl=179.16 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 15:15:00 | 181.68 | 178.65 | 178.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 183.36 | 178.70 | 178.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 13:15:00 | 178.93 | 179.00 | 178.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 13:15:00 | 178.93 | 179.00 | 178.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 178.93 | 179.00 | 178.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:30:00 | 178.33 | 179.00 | 178.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 180.04 | 179.01 | 178.82 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 175.00 | 178.66 | 178.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 173.52 | 178.37 | 178.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-13 09:15:00 | 178.70 | 177.94 | 178.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 178.70 | 177.94 | 178.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 178.70 | 177.94 | 178.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-13 09:30:00 | 178.54 | 177.94 | 178.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 177.84 | 177.94 | 178.27 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 10:15:00 | 183.18 | 178.61 | 178.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 185.48 | 178.92 | 178.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 12:15:00 | 183.10 | 183.27 | 181.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:00:00 | 183.10 | 183.27 | 181.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 213.20 | 217.29 | 213.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 213.20 | 217.29 | 213.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 212.50 | 217.24 | 213.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:00:00 | 212.50 | 217.24 | 213.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 213.05 | 217.20 | 213.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 11:15:00 | 214.35 | 217.02 | 213.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 213.22 | 216.86 | 213.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 210.18 | 216.75 | 213.09 | SL hit (close<static) qty=1.00 sl=212.25 alert=retest2 |

### Cycle 9 — SELL (started 2026-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 14:15:00 | 281.45 | 287.62 | 287.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-09 13:15:00 | 277.25 | 287.26 | 287.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 10:15:00 | 284.55 | 284.43 | 285.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-21 11:00:00 | 284.55 | 284.43 | 285.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 287.55 | 284.46 | 285.87 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 299.20 | 287.13 | 287.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 304.90 | 289.45 | 288.36 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-10-18 09:15:00 | 175.04 | 2024-10-18 12:15:00 | 181.94 | STOP_HIT | 1.00 | -3.94% |
| SELL | retest2 | 2024-10-18 10:30:00 | 177.60 | 2024-10-18 12:15:00 | 181.94 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2024-10-18 11:00:00 | 177.66 | 2024-10-18 12:15:00 | 181.94 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2024-10-23 09:15:00 | 175.96 | 2024-10-23 10:15:00 | 181.81 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2024-11-19 09:15:00 | 181.75 | 2024-11-21 09:15:00 | 176.98 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-11-22 13:30:00 | 180.42 | 2024-11-22 14:15:00 | 176.82 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-11-25 09:15:00 | 183.19 | 2024-11-29 09:15:00 | 201.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-24 11:45:00 | 180.42 | 2024-12-30 15:15:00 | 175.38 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-01-17 09:15:00 | 182.43 | 2025-01-20 10:15:00 | 186.07 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-01-17 14:15:00 | 182.63 | 2025-01-20 10:15:00 | 186.07 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-01-20 12:00:00 | 183.28 | 2025-01-20 13:15:00 | 190.18 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2025-01-28 09:15:00 | 181.97 | 2025-01-28 11:15:00 | 185.93 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-02-12 11:30:00 | 188.23 | 2025-02-14 10:15:00 | 185.10 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-02-12 12:30:00 | 188.18 | 2025-02-14 10:15:00 | 185.10 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-02-13 09:15:00 | 189.96 | 2025-02-14 10:15:00 | 185.10 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-02-14 09:15:00 | 188.26 | 2025-02-14 10:15:00 | 185.10 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-03-25 13:15:00 | 176.36 | 2025-04-03 13:15:00 | 179.25 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-03-25 15:00:00 | 176.17 | 2025-04-03 13:15:00 | 179.25 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-03-27 15:00:00 | 174.65 | 2025-04-03 13:15:00 | 179.25 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-04-01 10:15:00 | 176.37 | 2025-04-03 13:15:00 | 179.25 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-04-07 09:15:00 | 172.50 | 2025-04-16 09:15:00 | 179.58 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2025-04-09 09:15:00 | 172.81 | 2025-04-16 09:15:00 | 179.58 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2025-08-29 11:15:00 | 214.35 | 2025-09-01 10:15:00 | 210.18 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-09-01 09:15:00 | 213.22 | 2025-09-01 10:15:00 | 210.18 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-09-17 09:45:00 | 213.89 | 2025-09-24 14:15:00 | 211.76 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-09-23 09:15:00 | 213.43 | 2025-09-24 14:15:00 | 211.76 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-09-29 15:15:00 | 213.98 | 2025-09-30 09:15:00 | 209.05 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-10-06 10:00:00 | 213.86 | 2025-10-20 10:15:00 | 235.25 | TARGET_HIT | 1.00 | 10.00% |
