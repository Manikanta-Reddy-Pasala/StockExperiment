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
| CROSSOVER | 201 |
| ALERT1 | 138 |
| ALERT2 | 136 |
| ALERT2_SKIP | 58 |
| ALERT3 | 379 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 176 |
| PARTIAL | 15 |
| TARGET_HIT | 7 |
| STOP_HIT | 172 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 194 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 65 / 129
- **Target hits / Stop hits / Partials:** 7 / 172 / 15
- **Avg / median % per leg:** 0.01% / -0.69%
- **Sum % (uncompounded):** 2.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 88 | 20 | 22.7% | 5 | 83 | 0 | -0.57% | -50.5% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.46% | -7.4% |
| BUY @ 3rd Alert (retest2) | 85 | 20 | 23.5% | 5 | 80 | 0 | -0.51% | -43.2% |
| SELL (all) | 106 | 45 | 42.5% | 2 | 89 | 15 | 0.50% | 52.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 106 | 45 | 42.5% | 2 | 89 | 15 | 0.50% | 52.7% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.46% | -7.4% |
| retest2 (combined) | 191 | 65 | 34.0% | 7 | 169 | 15 | 0.05% | 9.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 11:15:00 | 159.25 | 158.63 | 158.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 12:15:00 | 160.05 | 158.91 | 158.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-17 13:15:00 | 158.90 | 158.91 | 158.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 13:15:00 | 158.90 | 158.91 | 158.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 13:15:00 | 158.90 | 158.91 | 158.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 14:00:00 | 158.90 | 158.91 | 158.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 14:15:00 | 159.20 | 158.97 | 158.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-18 09:15:00 | 160.35 | 159.03 | 158.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-19 10:15:00 | 158.70 | 159.20 | 159.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 10:15:00 | 158.70 | 159.20 | 159.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-22 09:15:00 | 154.95 | 158.07 | 158.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 09:15:00 | 156.55 | 156.21 | 157.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-23 10:15:00 | 157.00 | 156.21 | 157.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 11:15:00 | 157.35 | 156.42 | 157.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 12:00:00 | 157.35 | 156.42 | 157.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 12:15:00 | 157.20 | 156.58 | 157.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 13:15:00 | 157.40 | 156.58 | 157.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 13:15:00 | 157.00 | 156.66 | 157.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 13:30:00 | 157.50 | 156.66 | 157.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 14:15:00 | 156.70 | 156.67 | 157.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 14:30:00 | 156.50 | 156.67 | 157.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 15:15:00 | 156.40 | 156.62 | 156.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 09:15:00 | 157.10 | 156.62 | 156.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 156.30 | 156.55 | 156.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 09:30:00 | 157.30 | 156.55 | 156.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 12:15:00 | 157.10 | 156.67 | 156.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 14:00:00 | 156.70 | 156.67 | 156.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-26 09:45:00 | 156.55 | 155.75 | 156.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-29 10:15:00 | 156.65 | 155.74 | 155.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-29 10:15:00 | 156.65 | 155.92 | 155.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 10:15:00 | 156.65 | 155.92 | 155.91 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 13:15:00 | 155.50 | 155.89 | 155.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 14:15:00 | 155.05 | 155.72 | 155.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 09:15:00 | 154.40 | 154.18 | 154.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 09:15:00 | 154.40 | 154.18 | 154.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 154.40 | 154.18 | 154.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-31 12:30:00 | 153.60 | 154.03 | 154.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-31 14:00:00 | 153.40 | 153.90 | 154.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-01 09:45:00 | 153.65 | 153.62 | 154.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-01 10:30:00 | 153.60 | 153.62 | 154.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 13:15:00 | 154.55 | 153.74 | 154.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 14:00:00 | 154.55 | 153.74 | 154.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 14:15:00 | 153.45 | 153.68 | 153.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 10:15:00 | 153.15 | 153.66 | 153.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 11:15:00 | 153.00 | 153.59 | 153.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-05 13:00:00 | 153.20 | 153.21 | 153.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-05 14:00:00 | 153.25 | 153.22 | 153.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 14:15:00 | 153.20 | 153.21 | 153.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-05 15:15:00 | 153.15 | 153.21 | 153.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 09:15:00 | 145.92 | 148.85 | 150.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 09:15:00 | 145.73 | 148.85 | 150.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 09:15:00 | 145.97 | 148.85 | 150.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 09:15:00 | 145.92 | 148.85 | 150.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 09:15:00 | 145.49 | 148.85 | 150.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 09:15:00 | 145.54 | 148.85 | 150.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 09:15:00 | 145.59 | 148.85 | 150.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 09:15:00 | 145.49 | 148.85 | 150.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-06-07 11:15:00 | 148.95 | 148.86 | 150.22 | SL hit (close>ema200) qty=0.50 sl=148.86 alert=retest2 |

### Cycle 5 — BUY (started 2023-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 15:15:00 | 126.05 | 125.41 | 125.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 09:15:00 | 126.45 | 125.62 | 125.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 10:15:00 | 129.25 | 130.08 | 128.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-21 11:00:00 | 129.25 | 130.08 | 128.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 11:15:00 | 130.15 | 130.09 | 129.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 12:30:00 | 130.40 | 129.96 | 129.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 14:15:00 | 128.50 | 129.53 | 129.04 | SL hit (close<static) qty=1.00 sl=128.80 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 10:15:00 | 127.55 | 128.68 | 128.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 12:15:00 | 127.10 | 128.15 | 128.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 10:15:00 | 127.35 | 127.20 | 127.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-23 11:00:00 | 127.35 | 127.20 | 127.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 11:15:00 | 128.45 | 127.45 | 127.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 12:00:00 | 128.45 | 127.45 | 127.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 12:15:00 | 128.45 | 127.65 | 127.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 13:00:00 | 128.45 | 127.65 | 127.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 13:15:00 | 127.65 | 127.65 | 127.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-23 14:45:00 | 127.45 | 127.58 | 127.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 09:45:00 | 127.40 | 127.48 | 127.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-26 10:15:00 | 128.70 | 127.73 | 127.84 | SL hit (close>static) qty=1.00 sl=128.60 alert=retest2 |

### Cycle 7 — BUY (started 2023-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 11:15:00 | 129.55 | 128.09 | 127.99 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 10:15:00 | 127.85 | 128.32 | 128.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-28 12:15:00 | 127.60 | 128.11 | 128.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 09:15:00 | 127.70 | 127.52 | 127.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 09:15:00 | 127.70 | 127.52 | 127.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 127.70 | 127.52 | 127.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 09:45:00 | 127.95 | 127.52 | 127.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 10:15:00 | 127.55 | 127.53 | 127.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 11:15:00 | 127.45 | 127.53 | 127.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 14:00:00 | 127.30 | 127.43 | 127.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-06 12:15:00 | 126.75 | 126.16 | 126.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-07-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 12:15:00 | 126.75 | 126.16 | 126.15 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 11:15:00 | 124.95 | 125.99 | 126.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 09:15:00 | 124.05 | 125.28 | 125.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 124.80 | 124.20 | 124.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 124.80 | 124.20 | 124.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 124.80 | 124.20 | 124.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:00:00 | 124.80 | 124.20 | 124.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 124.95 | 124.35 | 124.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:45:00 | 124.90 | 124.35 | 124.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 123.90 | 124.10 | 124.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 10:15:00 | 123.80 | 124.10 | 124.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 11:00:00 | 123.85 | 124.05 | 124.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 11:30:00 | 123.55 | 123.92 | 124.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 09:30:00 | 123.70 | 123.56 | 123.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 122.85 | 122.49 | 123.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 10:30:00 | 122.80 | 122.49 | 123.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 11:15:00 | 122.50 | 122.49 | 123.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 11:30:00 | 122.80 | 122.49 | 123.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 12:15:00 | 123.15 | 122.63 | 123.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 13:00:00 | 123.15 | 122.63 | 123.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 13:15:00 | 123.35 | 122.77 | 123.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 14:00:00 | 123.35 | 122.77 | 123.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 123.70 | 122.96 | 123.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 15:00:00 | 123.70 | 122.96 | 123.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-17 09:15:00 | 124.65 | 123.45 | 123.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 124.65 | 123.45 | 123.33 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 13:15:00 | 123.20 | 123.59 | 123.63 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 10:15:00 | 124.75 | 123.81 | 123.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 11:15:00 | 124.90 | 124.03 | 123.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 12:15:00 | 124.70 | 124.73 | 124.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-20 13:00:00 | 124.70 | 124.73 | 124.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 13:15:00 | 124.15 | 124.61 | 124.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 14:00:00 | 124.15 | 124.61 | 124.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 14:15:00 | 124.00 | 124.49 | 124.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 15:15:00 | 124.10 | 124.49 | 124.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 124.10 | 124.41 | 124.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 09:15:00 | 124.50 | 124.41 | 124.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 10:15:00 | 123.85 | 124.24 | 124.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 15:15:00 | 123.45 | 123.84 | 124.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 12:15:00 | 124.15 | 123.67 | 123.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 12:15:00 | 124.15 | 123.67 | 123.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 12:15:00 | 124.15 | 123.67 | 123.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 12:45:00 | 123.85 | 123.67 | 123.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 13:15:00 | 124.20 | 123.78 | 123.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 13:45:00 | 124.30 | 123.78 | 123.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 123.55 | 123.64 | 123.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:30:00 | 123.90 | 123.64 | 123.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 123.00 | 123.51 | 123.72 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 09:15:00 | 124.95 | 123.75 | 123.72 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 12:15:00 | 123.45 | 124.00 | 124.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 14:15:00 | 119.70 | 123.01 | 123.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 09:15:00 | 123.55 | 122.74 | 123.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 09:15:00 | 123.55 | 122.74 | 123.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 123.55 | 122.74 | 123.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 09:30:00 | 124.00 | 122.74 | 123.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 10:15:00 | 123.35 | 122.86 | 123.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 10:45:00 | 124.05 | 122.86 | 123.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 11:15:00 | 123.00 | 122.89 | 123.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 11:30:00 | 123.50 | 122.89 | 123.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 14:15:00 | 122.75 | 122.75 | 123.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 15:00:00 | 122.75 | 122.75 | 123.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 123.60 | 122.89 | 123.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:45:00 | 123.60 | 122.89 | 123.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 123.70 | 123.05 | 123.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:30:00 | 123.50 | 123.05 | 123.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 12:15:00 | 122.95 | 123.06 | 123.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-31 13:15:00 | 122.70 | 123.06 | 123.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-01 09:15:00 | 126.15 | 123.53 | 123.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 09:15:00 | 126.15 | 123.53 | 123.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 10:15:00 | 128.00 | 124.42 | 123.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 12:15:00 | 126.40 | 127.15 | 126.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 13:00:00 | 126.40 | 127.15 | 126.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 125.75 | 126.87 | 126.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:30:00 | 125.40 | 126.87 | 126.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 126.95 | 126.88 | 126.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 09:15:00 | 127.50 | 126.90 | 126.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-03 09:15:00 | 125.30 | 126.58 | 126.09 | SL hit (close<static) qty=1.00 sl=125.60 alert=retest2 |

### Cycle 18 — SELL (started 2023-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 15:15:00 | 128.35 | 129.10 | 129.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 127.10 | 128.70 | 128.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 125.75 | 125.72 | 126.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 13:15:00 | 127.90 | 126.16 | 126.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 13:15:00 | 127.90 | 126.16 | 126.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 14:00:00 | 127.90 | 126.16 | 126.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 14:15:00 | 126.30 | 126.19 | 126.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 09:15:00 | 125.95 | 126.21 | 126.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-23 11:15:00 | 125.35 | 124.78 | 124.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 11:15:00 | 125.35 | 124.78 | 124.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 12:15:00 | 125.70 | 124.97 | 124.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 14:15:00 | 125.00 | 125.09 | 124.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 14:15:00 | 125.00 | 125.09 | 124.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 125.00 | 125.09 | 124.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 15:00:00 | 125.00 | 125.09 | 124.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 125.40 | 125.15 | 124.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 09:15:00 | 126.00 | 125.15 | 124.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-25 09:15:00 | 124.50 | 125.24 | 125.20 | SL hit (close<static) qty=1.00 sl=124.90 alert=retest2 |

### Cycle 20 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 124.35 | 125.06 | 125.12 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 125.85 | 124.99 | 124.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 126.40 | 125.58 | 125.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 126.15 | 126.18 | 125.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-30 15:00:00 | 126.15 | 126.18 | 125.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 125.75 | 126.09 | 125.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:00:00 | 125.75 | 126.09 | 125.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 126.25 | 126.12 | 125.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:30:00 | 125.50 | 126.12 | 125.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 14:15:00 | 126.25 | 126.28 | 126.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 14:45:00 | 125.85 | 126.28 | 126.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 126.40 | 126.30 | 126.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 09:15:00 | 127.25 | 126.30 | 126.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-05 09:15:00 | 139.98 | 135.37 | 132.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 135.65 | 138.51 | 138.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 132.45 | 135.48 | 137.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 134.05 | 134.00 | 135.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 12:30:00 | 134.30 | 134.00 | 135.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 134.40 | 134.23 | 135.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:30:00 | 135.45 | 134.23 | 135.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 135.00 | 134.38 | 135.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:30:00 | 135.15 | 134.38 | 135.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 134.90 | 134.49 | 135.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 12:30:00 | 134.35 | 134.51 | 135.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-15 09:15:00 | 136.00 | 135.00 | 135.19 | SL hit (close>static) qty=1.00 sl=135.25 alert=retest2 |

### Cycle 23 — BUY (started 2023-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 15:15:00 | 132.05 | 131.80 | 131.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 09:15:00 | 134.25 | 132.29 | 132.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 12:15:00 | 132.50 | 132.69 | 132.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-28 13:00:00 | 132.50 | 132.69 | 132.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 131.55 | 132.47 | 132.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:00:00 | 131.55 | 132.47 | 132.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 130.75 | 132.12 | 132.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 130.75 | 132.12 | 132.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 15:15:00 | 131.30 | 131.96 | 132.02 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 12:15:00 | 132.75 | 132.16 | 132.09 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 09:15:00 | 130.95 | 131.96 | 132.03 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 12:15:00 | 133.25 | 132.29 | 132.17 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 10:15:00 | 131.05 | 132.02 | 132.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 129.30 | 131.47 | 131.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 132.75 | 130.78 | 131.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 132.75 | 130.78 | 131.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 132.75 | 130.78 | 131.24 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-10-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 12:15:00 | 132.90 | 131.61 | 131.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 14:15:00 | 133.20 | 132.16 | 131.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 131.50 | 133.06 | 132.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 131.50 | 133.06 | 132.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 131.50 | 133.06 | 132.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 10:15:00 | 132.60 | 133.06 | 132.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 11:30:00 | 132.55 | 132.86 | 132.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 13:15:00 | 132.25 | 132.62 | 132.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 13:15:00 | 131.80 | 132.46 | 132.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2023-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 13:15:00 | 131.80 | 132.46 | 132.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 14:15:00 | 131.50 | 132.27 | 132.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 132.90 | 132.22 | 132.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 132.90 | 132.22 | 132.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 132.90 | 132.22 | 132.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:45:00 | 132.60 | 132.22 | 132.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 10:15:00 | 133.35 | 132.45 | 132.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 11:15:00 | 133.90 | 132.74 | 132.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 12:15:00 | 133.30 | 133.52 | 133.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-11 12:45:00 | 133.25 | 133.52 | 133.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 13:15:00 | 133.30 | 133.48 | 133.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 13:45:00 | 133.15 | 133.48 | 133.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 14:15:00 | 133.20 | 133.42 | 133.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 15:00:00 | 133.20 | 133.42 | 133.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 15:15:00 | 133.10 | 133.36 | 133.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 09:15:00 | 133.45 | 133.36 | 133.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 11:15:00 | 133.80 | 135.15 | 135.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 133.80 | 135.15 | 135.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 11:15:00 | 132.20 | 133.45 | 134.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 124.50 | 123.99 | 125.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-27 10:00:00 | 124.50 | 123.99 | 125.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 09:15:00 | 125.20 | 124.81 | 125.30 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 13:15:00 | 125.65 | 125.32 | 125.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 14:15:00 | 125.95 | 125.45 | 125.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 11:15:00 | 125.45 | 125.62 | 125.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 11:15:00 | 125.45 | 125.62 | 125.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 11:15:00 | 125.45 | 125.62 | 125.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 12:00:00 | 125.45 | 125.62 | 125.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 124.75 | 125.45 | 125.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 13:00:00 | 124.75 | 125.45 | 125.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2023-11-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 13:15:00 | 124.80 | 125.32 | 125.38 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-11-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 10:15:00 | 125.85 | 125.37 | 125.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 12:15:00 | 126.60 | 125.70 | 125.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 11:15:00 | 132.10 | 132.42 | 130.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 11:45:00 | 132.30 | 132.42 | 130.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 131.60 | 132.30 | 131.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:45:00 | 131.75 | 132.30 | 131.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 131.40 | 132.12 | 131.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:30:00 | 131.10 | 132.12 | 131.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 131.30 | 131.88 | 131.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 13:00:00 | 131.30 | 131.88 | 131.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2023-11-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 15:15:00 | 131.30 | 131.63 | 131.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 09:15:00 | 130.50 | 131.40 | 131.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 11:15:00 | 131.50 | 131.37 | 131.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-10 12:00:00 | 131.50 | 131.37 | 131.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 12:15:00 | 131.20 | 131.34 | 131.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 12:30:00 | 131.40 | 131.34 | 131.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 131.25 | 131.32 | 131.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 13:30:00 | 131.45 | 131.32 | 131.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 131.45 | 131.35 | 131.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 15:00:00 | 131.45 | 131.35 | 131.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 15:15:00 | 131.50 | 131.38 | 131.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 18:15:00 | 132.35 | 131.38 | 131.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 132.10 | 131.52 | 131.53 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 09:15:00 | 132.45 | 131.71 | 131.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 134.10 | 132.76 | 132.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 14:15:00 | 135.90 | 136.04 | 134.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 15:00:00 | 135.90 | 136.04 | 134.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 10:15:00 | 136.80 | 137.57 | 137.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-21 10:45:00 | 137.00 | 137.57 | 137.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 11:15:00 | 137.85 | 137.63 | 137.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 13:30:00 | 139.20 | 138.24 | 137.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 10:15:00 | 143.30 | 143.73 | 143.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 10:15:00 | 143.30 | 143.73 | 143.75 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-11-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 11:15:00 | 144.15 | 143.81 | 143.79 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 12:15:00 | 143.30 | 143.71 | 143.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-30 13:15:00 | 142.80 | 143.53 | 143.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 15:15:00 | 144.15 | 143.64 | 143.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 15:15:00 | 144.15 | 143.64 | 143.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 15:15:00 | 144.15 | 143.64 | 143.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-01 09:15:00 | 144.10 | 143.64 | 143.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 143.30 | 143.57 | 143.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-01 12:15:00 | 143.05 | 143.67 | 143.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-01 13:15:00 | 143.15 | 143.61 | 143.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-01 14:30:00 | 143.05 | 143.41 | 143.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-04 09:15:00 | 146.60 | 143.90 | 143.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2023-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 09:15:00 | 146.60 | 143.90 | 143.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 10:15:00 | 148.65 | 145.35 | 144.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 10:15:00 | 153.90 | 154.50 | 151.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-08 11:00:00 | 153.90 | 154.50 | 151.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 152.05 | 153.94 | 151.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:00:00 | 152.05 | 153.94 | 151.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 151.55 | 153.46 | 151.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:45:00 | 151.20 | 153.46 | 151.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 14:15:00 | 152.40 | 153.25 | 151.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 15:15:00 | 153.00 | 153.25 | 151.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 09:45:00 | 153.40 | 153.38 | 152.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-12 14:15:00 | 151.35 | 152.40 | 152.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 14:15:00 | 151.35 | 152.40 | 152.49 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 13:15:00 | 153.05 | 152.57 | 152.50 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-12-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 15:15:00 | 152.00 | 152.57 | 152.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-15 10:15:00 | 150.95 | 152.24 | 152.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-18 09:15:00 | 153.85 | 151.95 | 152.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 09:15:00 | 153.85 | 151.95 | 152.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 153.85 | 151.95 | 152.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 10:00:00 | 153.85 | 151.95 | 152.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2023-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 10:15:00 | 154.35 | 152.43 | 152.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 11:15:00 | 154.85 | 152.91 | 152.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 14:15:00 | 153.05 | 153.14 | 152.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-18 14:45:00 | 153.20 | 153.14 | 152.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 152.05 | 152.89 | 152.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:00:00 | 152.05 | 152.89 | 152.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 153.70 | 153.05 | 152.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 11:15:00 | 154.15 | 153.05 | 152.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 12:45:00 | 154.05 | 153.38 | 153.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 15:00:00 | 154.20 | 153.68 | 153.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 12:15:00 | 150.75 | 153.11 | 153.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 12:15:00 | 150.75 | 153.11 | 153.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 146.60 | 151.80 | 152.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 15:15:00 | 147.50 | 147.35 | 149.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 09:15:00 | 147.30 | 147.35 | 149.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 149.05 | 147.83 | 148.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:45:00 | 149.15 | 147.83 | 148.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 11:15:00 | 149.05 | 148.08 | 148.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:45:00 | 149.30 | 148.08 | 148.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 12:15:00 | 149.00 | 148.26 | 148.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 13:00:00 | 149.00 | 148.26 | 148.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 13:15:00 | 149.55 | 148.52 | 149.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 13:30:00 | 149.10 | 148.52 | 149.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2023-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 14:15:00 | 152.80 | 149.38 | 149.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 156.15 | 151.25 | 150.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 11:15:00 | 154.70 | 154.86 | 153.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 12:00:00 | 154.70 | 154.86 | 153.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 164.10 | 167.37 | 165.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 09:45:00 | 163.70 | 167.37 | 165.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 164.00 | 166.69 | 165.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 163.45 | 166.69 | 165.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 164.70 | 166.29 | 165.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 11:30:00 | 164.00 | 166.29 | 165.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 12:15:00 | 165.50 | 166.14 | 165.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 13:30:00 | 166.10 | 166.28 | 165.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 14:00:00 | 166.85 | 166.28 | 165.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 15:15:00 | 166.10 | 166.17 | 165.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-03 09:15:00 | 163.65 | 165.66 | 165.30 | SL hit (close<static) qty=1.00 sl=164.70 alert=retest2 |

### Cycle 48 — SELL (started 2024-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 11:15:00 | 164.25 | 165.04 | 165.06 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-01-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 12:15:00 | 165.80 | 165.01 | 164.91 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 15:15:00 | 163.65 | 164.83 | 164.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 09:15:00 | 161.60 | 164.18 | 164.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-08 13:15:00 | 163.25 | 162.95 | 163.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 13:15:00 | 163.25 | 162.95 | 163.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 13:15:00 | 163.25 | 162.95 | 163.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-08 14:00:00 | 163.25 | 162.95 | 163.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 160.65 | 162.49 | 163.53 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 09:15:00 | 165.95 | 163.46 | 163.28 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 09:15:00 | 161.90 | 164.47 | 164.60 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 10:15:00 | 166.05 | 164.69 | 164.53 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 163.20 | 164.43 | 164.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 09:15:00 | 155.60 | 162.59 | 163.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 141.25 | 140.98 | 147.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-19 10:00:00 | 141.25 | 140.98 | 147.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 139.20 | 141.37 | 142.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 11:30:00 | 138.00 | 140.48 | 142.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 12:30:00 | 138.20 | 140.13 | 141.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 13:45:00 | 138.20 | 139.75 | 141.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 11:15:00 | 140.75 | 138.22 | 138.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 11:15:00 | 140.75 | 138.22 | 138.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 09:15:00 | 145.55 | 140.73 | 139.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 12:15:00 | 146.00 | 146.62 | 145.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 12:15:00 | 146.00 | 146.62 | 145.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 146.00 | 146.62 | 145.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:30:00 | 145.65 | 146.62 | 145.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 12:15:00 | 146.45 | 146.66 | 145.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 12:30:00 | 146.25 | 146.66 | 145.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 13:15:00 | 146.50 | 146.63 | 145.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 13:30:00 | 146.00 | 146.63 | 145.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 146.25 | 146.47 | 146.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-05 09:15:00 | 147.05 | 146.47 | 146.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-05 15:00:00 | 146.60 | 147.62 | 146.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 10:00:00 | 147.00 | 150.30 | 149.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-07 10:15:00 | 144.45 | 149.13 | 148.76 | SL hit (close<static) qty=1.00 sl=146.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 11:15:00 | 144.05 | 148.12 | 148.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 10:15:00 | 141.15 | 144.34 | 145.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 09:15:00 | 142.75 | 141.57 | 142.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 09:15:00 | 142.75 | 141.57 | 142.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 142.75 | 141.57 | 142.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 10:00:00 | 142.75 | 141.57 | 142.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 143.50 | 141.95 | 142.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 11:15:00 | 143.05 | 141.95 | 142.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 11:15:00 | 142.00 | 141.96 | 142.80 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 15:15:00 | 144.15 | 143.27 | 143.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 09:15:00 | 145.50 | 144.61 | 144.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 11:15:00 | 144.75 | 144.76 | 144.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 11:30:00 | 144.90 | 144.76 | 144.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 147.35 | 146.05 | 145.45 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 144.90 | 145.66 | 145.71 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 146.80 | 145.52 | 145.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 12:15:00 | 147.60 | 146.14 | 145.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 11:15:00 | 146.65 | 146.76 | 146.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 11:15:00 | 146.65 | 146.76 | 146.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 11:15:00 | 146.65 | 146.76 | 146.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 11:45:00 | 146.50 | 146.76 | 146.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 12:15:00 | 146.90 | 146.78 | 146.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 13:30:00 | 147.40 | 146.86 | 146.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 14:15:00 | 147.30 | 146.86 | 146.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 09:15:00 | 149.40 | 146.92 | 146.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-27 13:15:00 | 145.50 | 147.00 | 146.78 | SL hit (close<static) qty=1.00 sl=146.20 alert=retest2 |

### Cycle 60 — SELL (started 2024-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 14:15:00 | 145.10 | 146.62 | 146.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 15:15:00 | 144.95 | 146.28 | 146.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 12:15:00 | 141.90 | 141.67 | 143.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 13:00:00 | 141.90 | 141.67 | 143.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 142.80 | 141.88 | 142.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 143.55 | 141.88 | 142.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 143.35 | 142.17 | 142.93 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-03-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 13:15:00 | 144.70 | 143.47 | 143.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 09:15:00 | 150.85 | 145.21 | 144.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 09:15:00 | 151.70 | 152.33 | 149.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 09:45:00 | 151.40 | 152.33 | 149.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 149.30 | 151.37 | 149.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:00:00 | 149.30 | 151.37 | 149.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 150.70 | 151.24 | 149.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 13:30:00 | 151.00 | 151.00 | 149.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 09:15:00 | 147.45 | 149.96 | 149.65 | SL hit (close<static) qty=1.00 sl=149.10 alert=retest2 |

### Cycle 62 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 146.65 | 149.30 | 149.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 09:15:00 | 143.20 | 147.34 | 148.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 137.10 | 136.22 | 139.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:45:00 | 137.20 | 136.22 | 139.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 135.55 | 136.46 | 137.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:15:00 | 134.50 | 136.46 | 137.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 10:15:00 | 135.00 | 136.19 | 136.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 14:00:00 | 135.15 | 135.48 | 136.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 15:00:00 | 134.35 | 135.25 | 135.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 132.20 | 134.54 | 135.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-21 13:15:00 | 135.65 | 135.02 | 135.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 13:15:00 | 135.65 | 135.02 | 135.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 15:15:00 | 136.10 | 135.36 | 135.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 135.70 | 136.01 | 135.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 15:00:00 | 135.70 | 136.01 | 135.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 15:15:00 | 135.95 | 136.00 | 135.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:15:00 | 136.30 | 136.00 | 135.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 135.75 | 135.95 | 135.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:30:00 | 135.30 | 135.95 | 135.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 135.45 | 135.85 | 135.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 10:45:00 | 135.40 | 135.85 | 135.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 11:15:00 | 135.35 | 135.75 | 135.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 12:00:00 | 135.35 | 135.75 | 135.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 12:15:00 | 135.40 | 135.68 | 135.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 13:00:00 | 135.40 | 135.68 | 135.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 13:15:00 | 135.75 | 135.69 | 135.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 13:30:00 | 135.75 | 135.69 | 135.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2024-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 14:15:00 | 134.05 | 135.37 | 135.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 14:15:00 | 133.65 | 134.66 | 135.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 09:15:00 | 134.90 | 134.50 | 134.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 09:15:00 | 134.90 | 134.50 | 134.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 134.90 | 134.50 | 134.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:30:00 | 135.10 | 134.50 | 134.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 134.70 | 134.54 | 134.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 12:15:00 | 134.40 | 134.53 | 134.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 13:45:00 | 134.40 | 134.53 | 134.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 15:00:00 | 134.35 | 134.49 | 134.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-01 09:15:00 | 138.70 | 135.31 | 135.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 138.70 | 135.31 | 135.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 11:15:00 | 139.60 | 136.76 | 135.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 14:15:00 | 144.65 | 144.68 | 143.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-04 15:00:00 | 144.65 | 144.68 | 143.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 143.50 | 144.52 | 143.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:30:00 | 142.75 | 144.52 | 143.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 144.20 | 144.46 | 143.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 11:00:00 | 144.20 | 144.46 | 143.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 13:15:00 | 145.15 | 145.56 | 144.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 14:00:00 | 145.15 | 145.56 | 144.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 149.10 | 147.52 | 146.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 10:15:00 | 150.70 | 147.52 | 146.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 12:15:00 | 151.60 | 148.42 | 147.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 151.65 | 149.65 | 148.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:45:00 | 150.90 | 150.16 | 148.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 147.90 | 151.09 | 150.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:30:00 | 146.55 | 151.09 | 150.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-04-15 13:15:00 | 148.55 | 149.54 | 149.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 13:15:00 | 148.55 | 149.54 | 149.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 14:15:00 | 146.40 | 148.91 | 149.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 10:15:00 | 148.80 | 148.53 | 148.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 10:45:00 | 148.65 | 148.53 | 148.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 147.95 | 148.41 | 148.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 12:15:00 | 147.25 | 148.41 | 148.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 10:15:00 | 147.55 | 148.15 | 148.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 147.55 | 148.02 | 148.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 15:15:00 | 147.40 | 147.19 | 147.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 15:15:00 | 147.40 | 147.23 | 147.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:15:00 | 148.25 | 147.23 | 147.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 147.75 | 147.33 | 147.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-22 11:15:00 | 147.90 | 147.59 | 147.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2024-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 11:15:00 | 147.90 | 147.59 | 147.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 13:15:00 | 148.90 | 147.94 | 147.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 09:15:00 | 157.15 | 157.20 | 155.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 10:00:00 | 157.15 | 157.20 | 155.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 156.65 | 157.62 | 156.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 10:45:00 | 156.40 | 157.62 | 156.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 11:15:00 | 156.35 | 157.37 | 156.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 12:00:00 | 156.35 | 157.37 | 156.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 12:15:00 | 156.00 | 157.09 | 156.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 12:45:00 | 156.00 | 157.09 | 156.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 14:15:00 | 155.15 | 156.62 | 156.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 15:00:00 | 155.15 | 156.62 | 156.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 155.35 | 156.36 | 156.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 09:15:00 | 156.25 | 156.36 | 156.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 11:15:00 | 155.45 | 156.24 | 156.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 11:15:00 | 155.45 | 156.24 | 156.28 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 13:15:00 | 157.05 | 156.34 | 156.31 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 14:15:00 | 156.10 | 156.29 | 156.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 15:15:00 | 156.00 | 156.23 | 156.27 | Break + close below crossover candle low |

### Cycle 71 — BUY (started 2024-05-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 09:15:00 | 157.85 | 156.56 | 156.41 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-05-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 11:15:00 | 155.15 | 156.29 | 156.32 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-05-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 12:15:00 | 157.70 | 156.57 | 156.44 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 12:15:00 | 154.65 | 156.21 | 156.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 13:15:00 | 153.40 | 155.65 | 156.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 150.60 | 150.23 | 152.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 150.60 | 150.23 | 152.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 144.95 | 144.67 | 145.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:30:00 | 145.60 | 144.67 | 145.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 145.50 | 144.84 | 145.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 145.60 | 144.84 | 145.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 144.20 | 144.71 | 145.32 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 11:15:00 | 146.35 | 145.42 | 145.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 10:15:00 | 149.45 | 146.74 | 146.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 147.00 | 147.39 | 146.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 14:00:00 | 147.00 | 147.39 | 146.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 146.70 | 147.46 | 146.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:30:00 | 146.90 | 147.46 | 146.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 147.45 | 147.46 | 146.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 11:45:00 | 147.90 | 147.61 | 147.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-21 09:15:00 | 162.69 | 155.32 | 151.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 153.15 | 155.02 | 155.03 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 14:15:00 | 155.50 | 155.05 | 155.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 156.15 | 155.38 | 155.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 09:15:00 | 159.25 | 160.21 | 158.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 09:15:00 | 159.25 | 160.21 | 158.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 159.25 | 160.21 | 158.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:00:00 | 159.25 | 160.21 | 158.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 157.70 | 159.64 | 158.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:00:00 | 157.70 | 159.64 | 158.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 157.55 | 159.23 | 158.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 157.70 | 159.23 | 158.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 157.10 | 158.24 | 158.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 12:15:00 | 156.70 | 157.45 | 157.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 10:15:00 | 157.10 | 156.74 | 157.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 10:15:00 | 157.10 | 156.74 | 157.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 157.10 | 156.74 | 157.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:00:00 | 157.10 | 156.74 | 157.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 155.95 | 156.58 | 157.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 13:15:00 | 155.50 | 156.47 | 157.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 159.55 | 155.00 | 155.43 | SL hit (close>static) qty=1.00 sl=157.35 alert=retest2 |

### Cycle 79 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 158.70 | 155.74 | 155.72 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 143.85 | 153.98 | 155.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 142.15 | 151.61 | 154.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 152.10 | 148.82 | 151.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 152.10 | 148.82 | 151.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 152.10 | 148.82 | 151.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 152.10 | 148.82 | 151.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 154.50 | 149.95 | 151.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 154.50 | 149.95 | 151.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 155.85 | 151.13 | 152.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 155.85 | 151.13 | 152.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 157.10 | 153.17 | 152.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 165.15 | 156.62 | 154.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 183.00 | 183.68 | 180.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 09:30:00 | 182.46 | 183.68 | 180.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 183.07 | 183.56 | 181.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 181.68 | 183.56 | 181.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 181.15 | 182.88 | 181.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 15:00:00 | 181.15 | 182.88 | 181.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 181.49 | 182.60 | 181.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 181.90 | 182.60 | 181.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 182.27 | 182.54 | 181.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 11:15:00 | 183.45 | 181.64 | 181.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 12:00:00 | 183.49 | 182.01 | 181.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 14:00:00 | 183.79 | 182.33 | 181.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 12:00:00 | 183.95 | 182.59 | 182.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 182.75 | 182.85 | 182.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 09:15:00 | 184.30 | 182.85 | 182.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 09:15:00 | 181.58 | 182.59 | 182.37 | SL hit (close<static) qty=1.00 sl=182.44 alert=retest2 |

### Cycle 82 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 10:15:00 | 180.45 | 182.17 | 182.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 11:15:00 | 179.62 | 181.66 | 181.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 10:15:00 | 179.85 | 179.66 | 180.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 10:15:00 | 179.85 | 179.66 | 180.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 179.85 | 179.66 | 180.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:45:00 | 180.28 | 179.66 | 180.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 181.10 | 179.39 | 180.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:45:00 | 182.16 | 179.39 | 180.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 179.87 | 179.49 | 179.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:30:00 | 179.91 | 179.49 | 179.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 180.40 | 179.67 | 180.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:45:00 | 180.41 | 179.67 | 180.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 179.06 | 179.55 | 179.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 13:45:00 | 178.67 | 179.31 | 179.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 182.26 | 180.22 | 180.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 182.26 | 180.22 | 180.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 186.05 | 181.70 | 180.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 10:15:00 | 186.78 | 187.00 | 184.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 10:30:00 | 186.90 | 187.00 | 184.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 184.39 | 186.48 | 184.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 184.39 | 186.48 | 184.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 183.59 | 185.90 | 184.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 183.59 | 185.90 | 184.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 183.87 | 185.50 | 184.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:15:00 | 184.85 | 185.50 | 184.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 11:15:00 | 184.58 | 184.84 | 184.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 11:15:00 | 184.58 | 184.84 | 184.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 10:15:00 | 181.65 | 183.62 | 184.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 183.55 | 182.49 | 183.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 183.55 | 182.49 | 183.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 183.55 | 182.49 | 183.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:45:00 | 184.95 | 182.49 | 183.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 184.30 | 182.85 | 183.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:00:00 | 184.30 | 182.85 | 183.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 184.22 | 183.12 | 183.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:30:00 | 184.10 | 183.12 | 183.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 184.20 | 183.38 | 183.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 184.20 | 183.38 | 183.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 183.48 | 183.40 | 183.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 15:15:00 | 182.81 | 183.40 | 183.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 09:15:00 | 173.67 | 181.36 | 182.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-11 09:15:00 | 177.10 | 176.96 | 179.10 | SL hit (close>ema200) qty=0.50 sl=176.96 alert=retest2 |

### Cycle 85 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 180.55 | 178.22 | 178.00 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 15:15:00 | 177.15 | 178.10 | 178.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 174.71 | 177.42 | 177.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 171.87 | 171.12 | 172.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 10:15:00 | 171.87 | 171.12 | 172.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 171.87 | 171.12 | 172.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 172.62 | 171.12 | 172.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 172.00 | 171.33 | 172.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:15:00 | 170.88 | 171.33 | 172.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 162.34 | 169.24 | 170.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 173.30 | 169.60 | 170.47 | SL hit (close>ema200) qty=0.50 sl=169.60 alert=retest2 |

### Cycle 87 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 173.37 | 171.45 | 171.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 174.24 | 172.01 | 171.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 190.68 | 190.88 | 188.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 09:30:00 | 190.70 | 190.88 | 188.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 190.60 | 190.82 | 188.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 188.45 | 190.82 | 188.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 190.43 | 190.52 | 189.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 190.12 | 190.52 | 189.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 192.10 | 190.83 | 189.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:15:00 | 195.25 | 190.83 | 189.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 09:30:00 | 195.00 | 194.85 | 192.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 13:15:00 | 189.08 | 191.42 | 191.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 13:15:00 | 189.08 | 191.42 | 191.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 15:15:00 | 187.99 | 190.27 | 190.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 193.11 | 190.84 | 191.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 193.11 | 190.84 | 191.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 193.11 | 190.84 | 191.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 193.11 | 190.84 | 191.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 192.42 | 191.15 | 191.28 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 195.01 | 191.92 | 191.61 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 15:15:00 | 190.60 | 191.37 | 191.47 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 10:15:00 | 193.18 | 191.81 | 191.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 12:15:00 | 195.00 | 192.60 | 192.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 196.46 | 196.57 | 194.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 13:00:00 | 196.46 | 196.57 | 194.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 193.93 | 195.92 | 194.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 193.93 | 195.92 | 194.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 193.80 | 195.49 | 194.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 195.46 | 195.49 | 194.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 12:15:00 | 193.00 | 194.78 | 194.61 | SL hit (close<static) qty=1.00 sl=193.08 alert=retest2 |

### Cycle 92 — SELL (started 2024-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 14:15:00 | 192.81 | 194.20 | 194.36 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 14:15:00 | 194.69 | 194.25 | 194.23 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 15:15:00 | 193.75 | 194.15 | 194.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 192.34 | 193.79 | 194.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 191.37 | 187.79 | 189.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 191.37 | 187.79 | 189.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 191.37 | 187.79 | 189.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:30:00 | 190.58 | 187.79 | 189.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 191.06 | 188.45 | 189.43 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 193.71 | 190.19 | 190.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 194.64 | 191.08 | 190.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 195.20 | 195.35 | 193.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:00:00 | 195.20 | 195.35 | 193.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 194.19 | 194.91 | 193.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 193.80 | 194.91 | 193.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 196.07 | 195.39 | 194.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 197.63 | 195.78 | 195.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 13:30:00 | 198.00 | 197.40 | 196.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 09:15:00 | 191.59 | 195.68 | 195.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 09:15:00 | 191.59 | 195.68 | 195.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 12:15:00 | 189.60 | 193.41 | 194.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 15:15:00 | 190.05 | 189.09 | 190.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-27 09:15:00 | 190.92 | 189.09 | 190.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 191.53 | 189.58 | 190.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:30:00 | 191.99 | 189.58 | 190.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 192.63 | 190.19 | 191.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:45:00 | 192.64 | 190.19 | 191.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 12:15:00 | 194.83 | 191.98 | 191.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 14:15:00 | 195.81 | 193.15 | 192.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 10:15:00 | 200.20 | 201.31 | 198.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 10:45:00 | 200.49 | 201.31 | 198.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 202.05 | 203.68 | 202.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:00:00 | 202.05 | 203.68 | 202.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 202.45 | 203.43 | 202.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 206.00 | 203.21 | 202.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-18 13:15:00 | 226.60 | 223.41 | 221.64 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 12:15:00 | 211.54 | 227.74 | 229.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 201.11 | 216.40 | 222.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 09:15:00 | 208.77 | 206.84 | 213.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 09:30:00 | 207.35 | 206.84 | 213.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 210.80 | 208.62 | 211.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 10:30:00 | 209.35 | 208.40 | 211.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 208.70 | 207.16 | 207.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 11:15:00 | 208.70 | 207.16 | 207.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-03 12:15:00 | 210.75 | 207.87 | 207.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 208.40 | 208.99 | 208.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 13:15:00 | 208.40 | 208.99 | 208.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 208.40 | 208.99 | 208.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:30:00 | 209.14 | 208.99 | 208.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 207.93 | 208.77 | 208.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 209.73 | 208.57 | 208.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 204.81 | 207.82 | 207.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 204.81 | 207.82 | 207.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 201.87 | 206.63 | 207.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 202.28 | 201.83 | 204.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 202.10 | 201.83 | 204.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 203.38 | 202.14 | 204.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:30:00 | 203.93 | 202.14 | 204.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 203.95 | 202.65 | 204.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 203.95 | 202.65 | 204.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 204.65 | 203.05 | 204.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 204.65 | 203.05 | 204.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 204.55 | 203.35 | 204.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:15:00 | 205.40 | 203.35 | 204.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 205.40 | 203.76 | 204.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 208.19 | 203.76 | 204.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 207.19 | 204.93 | 204.71 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 203.00 | 204.69 | 204.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 10:15:00 | 202.58 | 204.08 | 204.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 09:15:00 | 203.80 | 203.20 | 203.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 09:15:00 | 203.80 | 203.20 | 203.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 203.80 | 203.20 | 203.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 204.00 | 203.20 | 203.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 203.45 | 203.25 | 203.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:30:00 | 203.78 | 203.25 | 203.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 203.00 | 203.20 | 203.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 09:30:00 | 202.43 | 203.10 | 203.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 13:15:00 | 192.31 | 200.73 | 202.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-16 10:15:00 | 194.53 | 193.13 | 195.84 | SL hit (close>ema200) qty=0.50 sl=193.13 alert=retest2 |

### Cycle 103 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 177.32 | 174.58 | 174.49 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 14:15:00 | 174.00 | 175.00 | 175.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 173.36 | 174.51 | 174.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 172.57 | 172.08 | 173.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 172.57 | 172.08 | 173.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 164.76 | 162.42 | 163.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 164.76 | 162.42 | 163.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 165.00 | 162.94 | 163.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 165.17 | 162.94 | 163.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 161.70 | 161.69 | 162.33 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 165.61 | 163.05 | 162.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 167.67 | 165.89 | 164.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 14:15:00 | 166.00 | 166.46 | 165.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 15:00:00 | 166.00 | 166.46 | 165.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 177.90 | 178.39 | 177.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:00:00 | 177.90 | 178.39 | 177.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 177.19 | 178.05 | 177.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 177.15 | 178.05 | 177.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 179.41 | 178.32 | 177.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:15:00 | 180.59 | 178.68 | 178.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 13:15:00 | 186.89 | 188.25 | 188.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 186.89 | 188.25 | 188.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 14:15:00 | 185.79 | 187.76 | 188.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 12:15:00 | 184.73 | 183.89 | 185.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 13:00:00 | 184.73 | 183.89 | 185.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 185.11 | 184.17 | 184.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:00:00 | 185.11 | 184.17 | 184.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 185.00 | 184.33 | 184.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 185.20 | 184.33 | 184.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 182.57 | 183.98 | 184.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:45:00 | 181.77 | 183.49 | 184.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 181.58 | 179.76 | 179.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 181.58 | 179.76 | 179.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 10:15:00 | 182.72 | 181.09 | 180.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 178.63 | 180.92 | 180.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 13:15:00 | 178.63 | 180.92 | 180.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 178.63 | 180.92 | 180.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 178.63 | 180.92 | 180.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 178.35 | 180.40 | 180.40 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 15:15:00 | 179.20 | 180.16 | 180.29 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 182.50 | 180.68 | 180.46 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 15:15:00 | 180.38 | 180.58 | 180.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 09:15:00 | 178.25 | 180.11 | 180.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 12:15:00 | 179.59 | 179.40 | 179.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-02 13:00:00 | 179.59 | 179.40 | 179.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 180.02 | 179.53 | 179.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:45:00 | 180.16 | 179.53 | 179.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 181.35 | 179.89 | 180.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 15:00:00 | 181.35 | 179.89 | 180.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 15:15:00 | 181.20 | 180.15 | 180.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:15:00 | 182.10 | 180.15 | 180.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 181.21 | 180.36 | 180.27 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 178.42 | 180.05 | 180.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 177.25 | 179.49 | 179.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 174.40 | 173.28 | 175.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 13:45:00 | 173.30 | 173.28 | 175.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 173.95 | 173.20 | 174.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 173.95 | 173.20 | 174.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 178.46 | 174.35 | 174.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 178.46 | 174.35 | 174.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 10:15:00 | 176.03 | 174.68 | 174.65 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 12:15:00 | 173.80 | 174.49 | 174.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 14:15:00 | 173.30 | 174.10 | 174.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 164.47 | 164.23 | 167.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 09:45:00 | 164.88 | 164.23 | 167.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 167.50 | 164.89 | 167.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 167.50 | 164.89 | 167.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 166.35 | 165.18 | 167.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 165.82 | 165.18 | 167.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:15:00 | 165.59 | 166.07 | 167.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 10:15:00 | 168.80 | 166.75 | 167.22 | SL hit (close>static) qty=1.00 sl=167.65 alert=retest2 |

### Cycle 115 — BUY (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 14:15:00 | 167.73 | 167.52 | 167.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 171.09 | 168.28 | 167.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 171.40 | 171.95 | 170.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 171.40 | 171.95 | 170.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 171.40 | 171.95 | 170.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:45:00 | 170.94 | 171.95 | 170.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 171.11 | 172.43 | 171.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 171.11 | 172.43 | 171.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 171.21 | 172.18 | 171.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:45:00 | 170.13 | 172.18 | 171.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 171.89 | 172.00 | 171.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 171.33 | 172.00 | 171.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 171.56 | 171.91 | 171.69 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 170.18 | 171.29 | 171.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 167.54 | 170.54 | 171.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 168.25 | 167.90 | 169.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 168.25 | 167.90 | 169.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 170.84 | 168.34 | 168.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:30:00 | 171.63 | 168.34 | 168.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 172.30 | 169.13 | 169.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 12:15:00 | 173.77 | 170.64 | 169.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 13:15:00 | 170.46 | 170.61 | 169.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 14:00:00 | 170.46 | 170.61 | 169.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 168.92 | 170.27 | 169.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 15:00:00 | 168.92 | 170.27 | 169.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 169.02 | 170.02 | 169.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 164.22 | 170.02 | 169.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 163.80 | 168.78 | 169.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 162.98 | 167.62 | 168.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 164.70 | 163.99 | 165.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 164.70 | 163.99 | 165.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 168.54 | 164.90 | 166.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 168.54 | 164.90 | 166.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 168.70 | 165.66 | 166.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:45:00 | 167.30 | 166.00 | 166.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 174.85 | 168.11 | 167.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 174.85 | 168.11 | 167.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 10:15:00 | 177.51 | 169.99 | 168.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 174.18 | 175.51 | 172.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 12:00:00 | 174.18 | 175.51 | 172.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 174.04 | 175.16 | 173.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 174.04 | 175.16 | 173.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 175.84 | 175.33 | 173.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:15:00 | 176.30 | 175.33 | 173.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 11:15:00 | 172.41 | 174.25 | 174.13 | SL hit (close<static) qty=1.00 sl=173.20 alert=retest2 |

### Cycle 120 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 168.18 | 173.03 | 173.59 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 15:15:00 | 174.50 | 171.90 | 171.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 178.80 | 173.28 | 172.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 09:15:00 | 181.64 | 181.76 | 179.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-07 10:00:00 | 181.64 | 181.76 | 179.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 178.82 | 181.69 | 180.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:45:00 | 179.63 | 181.69 | 180.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 178.21 | 180.99 | 180.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 178.21 | 180.99 | 180.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 176.84 | 179.58 | 179.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 14:15:00 | 176.47 | 178.60 | 179.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 173.25 | 171.85 | 174.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 173.25 | 171.85 | 174.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 171.94 | 171.80 | 173.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:15:00 | 169.15 | 170.79 | 172.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 160.69 | 165.21 | 167.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 12:15:00 | 165.42 | 164.98 | 167.05 | SL hit (close>ema200) qty=0.50 sl=164.98 alert=retest2 |

### Cycle 123 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 168.95 | 166.22 | 166.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 15:15:00 | 170.10 | 168.98 | 167.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 168.33 | 168.85 | 167.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 168.33 | 168.85 | 167.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 167.39 | 168.56 | 167.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 167.39 | 168.56 | 167.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 167.91 | 168.43 | 167.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 15:00:00 | 169.32 | 168.60 | 168.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 165.40 | 167.95 | 167.91 | SL hit (close<static) qty=1.00 sl=167.24 alert=retest2 |

### Cycle 124 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 164.50 | 167.26 | 167.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 12:15:00 | 163.90 | 166.11 | 166.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 12:15:00 | 165.94 | 164.84 | 165.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 12:15:00 | 165.94 | 164.84 | 165.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 165.94 | 164.84 | 165.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 13:00:00 | 165.94 | 164.84 | 165.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 167.37 | 165.35 | 165.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 13:30:00 | 167.35 | 165.35 | 165.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 166.45 | 165.57 | 165.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 15:15:00 | 165.51 | 165.57 | 165.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 11:15:00 | 157.23 | 161.50 | 163.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 160.95 | 159.05 | 161.44 | SL hit (close>ema200) qty=0.50 sl=159.05 alert=retest2 |

### Cycle 125 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 157.62 | 155.83 | 155.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 162.57 | 157.89 | 156.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 162.68 | 162.69 | 160.78 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 14:30:00 | 164.25 | 163.00 | 161.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 160.85 | 162.63 | 161.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-10 09:15:00 | 160.85 | 162.63 | 161.40 | SL hit (close<ema400) qty=1.00 sl=161.40 alert=retest1 |

### Cycle 126 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 159.52 | 160.90 | 160.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 158.15 | 160.35 | 160.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 158.45 | 158.42 | 159.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 157.23 | 158.42 | 159.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 156.02 | 157.94 | 159.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:15:00 | 155.74 | 157.64 | 158.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 154.61 | 156.62 | 157.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:45:00 | 155.08 | 156.27 | 157.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 11:15:00 | 161.84 | 157.47 | 157.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 161.84 | 157.47 | 157.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 162.15 | 159.11 | 158.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 14:15:00 | 176.00 | 176.78 | 174.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 15:00:00 | 176.00 | 176.78 | 174.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 176.31 | 176.48 | 174.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 11:45:00 | 177.04 | 176.78 | 175.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 13:15:00 | 175.18 | 176.95 | 176.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 175.18 | 176.95 | 176.97 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 177.90 | 176.65 | 176.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 180.85 | 177.67 | 177.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 176.66 | 179.64 | 178.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 176.66 | 179.64 | 178.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 176.66 | 179.64 | 178.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 176.66 | 179.64 | 178.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 178.79 | 179.47 | 178.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 11:30:00 | 179.06 | 179.26 | 178.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 12:30:00 | 178.98 | 179.05 | 178.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 15:15:00 | 179.10 | 178.80 | 178.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 172.05 | 177.50 | 178.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 172.05 | 177.50 | 178.01 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 178.43 | 176.74 | 176.61 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 174.16 | 176.25 | 176.42 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 179.40 | 176.90 | 176.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 180.40 | 178.18 | 177.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 12:15:00 | 190.38 | 190.43 | 188.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 12:45:00 | 189.93 | 190.43 | 188.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 187.68 | 189.65 | 188.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 187.68 | 189.65 | 188.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 188.53 | 189.43 | 188.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 187.78 | 189.43 | 188.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 189.17 | 189.38 | 188.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:30:00 | 189.20 | 189.38 | 188.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 189.89 | 189.48 | 189.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:30:00 | 189.09 | 189.48 | 189.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 191.09 | 191.33 | 190.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:45:00 | 190.98 | 191.33 | 190.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 191.25 | 191.31 | 190.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 09:15:00 | 199.41 | 191.31 | 190.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 14:45:00 | 191.61 | 192.70 | 191.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 15:15:00 | 190.00 | 192.16 | 191.73 | SL hit (close<static) qty=1.00 sl=190.06 alert=retest2 |

### Cycle 134 — SELL (started 2025-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 12:15:00 | 191.48 | 193.04 | 193.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 190.19 | 192.47 | 192.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 193.26 | 192.13 | 192.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 193.26 | 192.13 | 192.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 193.26 | 192.13 | 192.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 193.26 | 192.13 | 192.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 192.28 | 192.16 | 192.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 191.32 | 192.16 | 192.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:00:00 | 190.80 | 191.89 | 192.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 190.70 | 191.65 | 192.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 193.92 | 191.85 | 192.14 | SL hit (close>static) qty=1.00 sl=193.40 alert=retest2 |

### Cycle 135 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 195.67 | 192.99 | 192.62 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 191.51 | 193.25 | 193.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 191.12 | 192.82 | 193.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 09:15:00 | 193.23 | 192.55 | 192.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 193.23 | 192.55 | 192.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 193.23 | 192.55 | 192.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:30:00 | 191.70 | 192.55 | 192.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 196.00 | 193.24 | 193.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 11:15:00 | 196.55 | 193.90 | 193.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 11:15:00 | 195.50 | 195.67 | 194.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 12:00:00 | 195.50 | 195.67 | 194.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 194.13 | 195.36 | 194.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:00:00 | 194.13 | 195.36 | 194.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 191.90 | 194.67 | 194.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 191.90 | 194.67 | 194.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 190.24 | 193.78 | 194.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 188.49 | 192.72 | 193.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 194.41 | 190.40 | 191.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 194.41 | 190.40 | 191.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 194.41 | 190.40 | 191.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:15:00 | 195.90 | 190.40 | 191.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 195.15 | 192.14 | 192.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 195.82 | 193.41 | 192.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 194.12 | 194.43 | 193.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 13:15:00 | 194.12 | 194.43 | 193.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 194.12 | 194.43 | 193.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:30:00 | 194.24 | 194.43 | 193.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 194.71 | 194.49 | 193.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:30:00 | 193.83 | 194.49 | 193.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 195.12 | 194.66 | 193.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 10:30:00 | 196.33 | 195.10 | 194.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:15:00 | 196.36 | 196.66 | 196.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 196.60 | 197.80 | 197.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 196.60 | 197.80 | 197.86 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 199.25 | 198.07 | 197.96 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 194.29 | 197.26 | 197.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 193.66 | 195.98 | 196.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 195.71 | 195.39 | 196.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 12:00:00 | 195.71 | 195.39 | 196.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 195.35 | 195.26 | 195.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 10:45:00 | 194.90 | 195.19 | 195.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 198.20 | 195.92 | 195.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 198.20 | 195.92 | 195.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 12:15:00 | 198.84 | 196.51 | 195.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 11:15:00 | 198.04 | 198.42 | 197.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 11:45:00 | 198.00 | 198.42 | 197.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 197.90 | 198.31 | 197.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 197.50 | 198.31 | 197.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 197.99 | 198.25 | 197.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:30:00 | 197.80 | 198.25 | 197.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 197.77 | 198.15 | 197.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:30:00 | 197.53 | 198.15 | 197.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 197.80 | 198.08 | 197.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 198.68 | 198.08 | 197.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:45:00 | 198.06 | 198.09 | 197.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 10:45:00 | 198.87 | 198.13 | 197.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:45:00 | 197.99 | 200.29 | 200.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 200.80 | 200.39 | 200.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:30:00 | 203.41 | 200.84 | 200.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 10:15:00 | 198.25 | 200.76 | 200.67 | SL hit (close<static) qty=1.00 sl=198.32 alert=retest2 |

### Cycle 144 — SELL (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 11:15:00 | 197.10 | 200.03 | 200.35 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 202.93 | 200.39 | 200.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 215.02 | 203.92 | 201.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 208.26 | 208.76 | 206.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:45:00 | 208.50 | 208.76 | 206.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 193.49 | 207.28 | 207.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:00:00 | 193.49 | 207.28 | 207.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 193.30 | 204.49 | 205.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 10:15:00 | 187.87 | 191.39 | 196.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 187.65 | 187.61 | 190.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:30:00 | 187.82 | 187.61 | 190.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 189.89 | 188.33 | 190.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 190.31 | 188.33 | 190.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 188.60 | 188.67 | 190.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 189.86 | 188.67 | 190.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 182.58 | 181.53 | 182.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 182.58 | 181.53 | 182.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 182.80 | 181.79 | 182.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 180.95 | 181.79 | 182.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 186.81 | 183.07 | 182.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 186.81 | 183.07 | 182.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 187.71 | 184.00 | 183.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 15:15:00 | 188.66 | 189.08 | 187.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 09:15:00 | 186.92 | 189.08 | 187.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 187.29 | 188.72 | 187.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:45:00 | 189.93 | 188.33 | 187.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-09 12:15:00 | 208.92 | 203.29 | 200.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 206.79 | 207.36 | 207.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 204.12 | 206.02 | 206.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 204.14 | 204.09 | 205.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:30:00 | 203.91 | 204.09 | 205.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 203.59 | 204.11 | 204.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:15:00 | 201.76 | 204.11 | 204.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 15:15:00 | 191.67 | 195.94 | 199.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-24 09:15:00 | 181.58 | 183.65 | 190.64 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 149 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 135.88 | 133.01 | 132.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 10:15:00 | 137.81 | 133.97 | 133.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 11:15:00 | 136.46 | 136.98 | 135.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-11 12:00:00 | 136.46 | 136.98 | 135.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 137.65 | 137.29 | 136.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:30:00 | 136.21 | 137.29 | 136.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 140.54 | 140.59 | 139.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:45:00 | 140.40 | 140.59 | 139.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 140.72 | 140.47 | 139.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 139.08 | 140.47 | 139.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 140.97 | 140.55 | 140.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:30:00 | 140.22 | 140.55 | 140.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 140.09 | 140.65 | 140.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 138.81 | 140.65 | 140.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 141.30 | 140.78 | 140.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 12:30:00 | 141.39 | 140.93 | 140.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:15:00 | 141.61 | 140.97 | 140.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:45:00 | 142.16 | 141.54 | 140.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 142.86 | 144.28 | 144.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 142.86 | 144.28 | 144.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 141.29 | 142.73 | 143.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 140.91 | 140.68 | 141.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:30:00 | 140.99 | 140.68 | 141.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 141.34 | 140.41 | 141.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 141.78 | 140.41 | 141.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 141.90 | 140.71 | 141.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 141.90 | 140.71 | 141.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 142.10 | 140.99 | 141.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:30:00 | 142.20 | 140.99 | 141.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 142.01 | 141.50 | 141.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 143.29 | 141.94 | 141.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 142.05 | 142.26 | 141.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 13:00:00 | 142.05 | 142.26 | 141.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 141.19 | 142.04 | 141.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:45:00 | 140.95 | 142.04 | 141.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 141.19 | 141.87 | 141.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:00:00 | 141.80 | 141.74 | 141.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:00:00 | 141.60 | 141.71 | 141.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 140.32 | 141.80 | 141.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 140.32 | 141.80 | 141.98 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 143.07 | 142.10 | 141.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 144.79 | 143.24 | 142.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 143.82 | 144.23 | 143.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 143.82 | 144.23 | 143.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 143.82 | 144.23 | 143.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 143.82 | 144.23 | 143.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 143.72 | 144.13 | 143.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 13:30:00 | 144.13 | 143.98 | 143.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 14:00:00 | 144.06 | 143.98 | 143.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 14:15:00 | 143.41 | 143.87 | 143.67 | SL hit (close<static) qty=1.00 sl=143.45 alert=retest2 |

### Cycle 154 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 146.00 | 148.03 | 148.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 145.07 | 147.17 | 147.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 10:15:00 | 138.67 | 138.57 | 139.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 11:00:00 | 138.67 | 138.57 | 139.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 140.04 | 138.86 | 139.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:00:00 | 140.04 | 138.86 | 139.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 139.98 | 139.09 | 139.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:15:00 | 140.73 | 139.09 | 139.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 140.55 | 139.38 | 140.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 139.59 | 139.42 | 139.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 139.12 | 138.95 | 139.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 141.57 | 139.76 | 139.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 141.57 | 139.76 | 139.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 141.77 | 140.55 | 140.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 142.41 | 142.63 | 141.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:00:00 | 142.41 | 142.63 | 141.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 141.97 | 142.27 | 141.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 141.97 | 142.27 | 141.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 141.48 | 142.11 | 141.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:15:00 | 141.56 | 142.11 | 141.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 141.82 | 142.05 | 141.90 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 140.45 | 141.63 | 141.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 139.85 | 141.27 | 141.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 11:15:00 | 140.05 | 139.93 | 140.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 12:00:00 | 140.05 | 139.93 | 140.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 140.50 | 140.05 | 140.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:45:00 | 140.47 | 140.05 | 140.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 140.25 | 140.09 | 140.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 14:00:00 | 140.06 | 140.31 | 140.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 15:00:00 | 139.72 | 140.19 | 140.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 137.84 | 135.35 | 135.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 137.84 | 135.35 | 135.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 14:15:00 | 138.00 | 136.23 | 135.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 145.65 | 146.63 | 145.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 145.65 | 146.63 | 145.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 146.32 | 146.56 | 145.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 145.01 | 146.56 | 145.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 146.32 | 147.85 | 147.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 146.32 | 147.85 | 147.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 146.72 | 147.62 | 147.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:15:00 | 146.41 | 147.62 | 147.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 145.77 | 147.25 | 147.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:00:00 | 145.77 | 147.25 | 147.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 143.31 | 146.46 | 146.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 139.44 | 142.06 | 143.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 15:15:00 | 140.15 | 139.96 | 141.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:15:00 | 141.74 | 139.96 | 141.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 140.56 | 140.08 | 141.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:30:00 | 140.14 | 140.11 | 141.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 139.52 | 140.11 | 141.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 13:15:00 | 140.02 | 139.07 | 139.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 139.71 | 139.28 | 139.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 139.71 | 139.28 | 139.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 140.16 | 139.53 | 139.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 14:15:00 | 139.44 | 139.62 | 139.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 14:15:00 | 139.44 | 139.62 | 139.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 139.44 | 139.62 | 139.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 139.44 | 139.62 | 139.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 139.50 | 139.60 | 139.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 138.12 | 139.60 | 139.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 137.77 | 139.23 | 139.29 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 139.60 | 139.21 | 139.19 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 13:15:00 | 139.13 | 139.18 | 139.19 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 139.47 | 139.24 | 139.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 140.14 | 139.47 | 139.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 138.90 | 139.46 | 139.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 13:15:00 | 138.90 | 139.46 | 139.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 138.90 | 139.46 | 139.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 138.90 | 139.46 | 139.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 138.41 | 139.25 | 139.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 137.78 | 138.83 | 139.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 139.39 | 138.25 | 138.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 139.39 | 138.25 | 138.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 139.39 | 138.25 | 138.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 139.39 | 138.25 | 138.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 138.25 | 138.25 | 138.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 138.20 | 138.25 | 138.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 13:00:00 | 137.87 | 138.19 | 138.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 140.75 | 137.52 | 137.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 140.75 | 137.52 | 137.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 14:15:00 | 143.18 | 139.86 | 138.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 12:15:00 | 141.15 | 141.35 | 140.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 13:00:00 | 141.15 | 141.35 | 140.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 141.70 | 141.28 | 140.39 | EMA400 retest candle locked (from upside) |

### Cycle 166 — SELL (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 15:15:00 | 140.27 | 140.58 | 140.59 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 142.70 | 141.00 | 140.78 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 140.75 | 141.09 | 141.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 140.05 | 140.88 | 141.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 11:15:00 | 141.30 | 140.94 | 141.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 11:15:00 | 141.30 | 140.94 | 141.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 141.30 | 140.94 | 141.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:00:00 | 141.30 | 140.94 | 141.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 140.63 | 140.88 | 141.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:45:00 | 141.14 | 140.88 | 141.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 145.94 | 141.22 | 141.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 13:15:00 | 146.90 | 144.30 | 142.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 146.06 | 147.38 | 145.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 10:00:00 | 146.06 | 147.38 | 145.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 147.32 | 147.37 | 145.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 11:45:00 | 148.35 | 147.57 | 146.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 13:45:00 | 148.20 | 147.89 | 146.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 13:15:00 | 147.96 | 148.52 | 147.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 14:00:00 | 148.01 | 148.42 | 147.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 148.15 | 148.37 | 147.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:45:00 | 147.75 | 148.37 | 147.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 145.90 | 147.76 | 147.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 145.90 | 147.76 | 147.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 146.51 | 147.51 | 147.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 145.84 | 147.51 | 147.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 145.47 | 147.10 | 147.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 145.47 | 147.10 | 147.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 12:15:00 | 145.20 | 146.72 | 147.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 141.63 | 141.60 | 142.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 141.63 | 141.60 | 142.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 141.63 | 141.60 | 142.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 142.25 | 141.60 | 142.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 142.04 | 140.57 | 141.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 142.04 | 140.57 | 141.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 142.42 | 140.94 | 141.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 142.27 | 140.94 | 141.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 142.93 | 141.92 | 141.82 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 141.20 | 142.31 | 142.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 140.66 | 141.80 | 142.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 11:15:00 | 141.27 | 140.99 | 141.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 11:15:00 | 141.27 | 140.99 | 141.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 141.27 | 140.99 | 141.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:30:00 | 141.33 | 140.99 | 141.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 140.95 | 140.99 | 141.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 13:30:00 | 140.66 | 140.96 | 141.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:30:00 | 140.15 | 140.80 | 141.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:30:00 | 140.50 | 140.48 | 140.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 141.46 | 140.41 | 140.50 | SL hit (close>static) qty=1.00 sl=141.38 alert=retest2 |

### Cycle 173 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 141.20 | 140.57 | 140.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 142.03 | 140.86 | 140.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 13:15:00 | 140.87 | 141.11 | 140.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 13:15:00 | 140.87 | 141.11 | 140.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 13:15:00 | 140.87 | 141.11 | 140.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:00:00 | 140.87 | 141.11 | 140.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 140.97 | 141.08 | 140.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:30:00 | 140.75 | 141.08 | 140.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 140.96 | 141.06 | 140.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 141.40 | 141.06 | 140.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 140.10 | 141.39 | 141.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 140.10 | 141.39 | 141.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 139.19 | 140.95 | 141.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 134.71 | 133.07 | 134.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 134.71 | 133.07 | 134.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 134.71 | 133.07 | 134.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 135.33 | 133.07 | 134.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 134.91 | 133.43 | 134.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 135.15 | 133.43 | 134.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 134.82 | 134.06 | 134.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 134.82 | 134.06 | 134.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 134.17 | 134.08 | 134.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 133.67 | 133.93 | 134.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:45:00 | 133.64 | 133.85 | 134.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:15:00 | 134.00 | 133.83 | 133.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:45:00 | 133.86 | 133.87 | 133.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 134.30 | 133.95 | 133.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:45:00 | 134.48 | 133.95 | 133.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 134.49 | 134.06 | 134.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 134.49 | 134.06 | 134.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 10:15:00 | 134.65 | 134.20 | 134.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 133.38 | 134.14 | 134.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 133.38 | 134.14 | 134.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 133.38 | 134.14 | 134.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 133.38 | 134.14 | 134.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 134.57 | 134.23 | 134.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 133.38 | 134.23 | 134.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 134.04 | 134.45 | 134.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:45:00 | 144.12 | 137.80 | 135.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-09 09:15:00 | 158.53 | 152.41 | 148.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 139.70 | 146.33 | 146.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 137.42 | 139.43 | 140.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 131.90 | 130.36 | 132.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 131.90 | 130.36 | 132.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 131.90 | 130.36 | 132.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 132.06 | 130.36 | 132.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 131.20 | 130.72 | 131.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:30:00 | 131.84 | 130.72 | 131.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 128.18 | 127.64 | 128.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:45:00 | 128.50 | 127.64 | 128.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 128.68 | 127.85 | 128.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 14:45:00 | 128.61 | 127.85 | 128.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 128.70 | 128.02 | 128.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:15:00 | 128.33 | 128.02 | 128.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 127.11 | 127.32 | 127.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 125.93 | 127.36 | 127.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 124.20 | 126.95 | 127.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 10:15:00 | 119.63 | 123.79 | 125.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 14:15:00 | 122.77 | 122.06 | 123.95 | SL hit (close>ema200) qty=0.50 sl=122.06 alert=retest2 |

### Cycle 177 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 126.15 | 124.72 | 124.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 126.70 | 125.51 | 125.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 125.34 | 126.30 | 125.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 125.34 | 126.30 | 125.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 125.34 | 126.30 | 125.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 125.34 | 126.30 | 125.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 125.52 | 126.15 | 125.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 125.52 | 126.15 | 125.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 125.76 | 126.07 | 125.68 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 124.60 | 125.38 | 125.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 120.60 | 124.42 | 125.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 122.70 | 121.83 | 123.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 09:45:00 | 122.58 | 121.83 | 123.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 123.62 | 122.19 | 123.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 123.62 | 122.19 | 123.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 123.92 | 122.53 | 123.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 123.92 | 122.53 | 123.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 124.25 | 122.88 | 123.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 124.25 | 122.88 | 123.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 125.41 | 123.68 | 123.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 10:15:00 | 126.65 | 124.89 | 124.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 125.55 | 125.71 | 125.02 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 14:00:00 | 127.14 | 126.08 | 125.41 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 15:00:00 | 126.95 | 126.25 | 125.55 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 126.13 | 126.32 | 125.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 126.10 | 126.32 | 125.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 126.00 | 126.14 | 125.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:45:00 | 126.06 | 126.14 | 125.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 125.90 | 126.09 | 125.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 123.84 | 126.09 | 125.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 123.68 | 125.61 | 125.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 180 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 123.68 | 125.61 | 125.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 10:15:00 | 121.83 | 124.86 | 125.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 124.21 | 123.85 | 124.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 12:15:00 | 124.21 | 123.85 | 124.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 124.21 | 123.85 | 124.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 124.21 | 123.85 | 124.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 124.42 | 123.97 | 124.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:45:00 | 124.28 | 123.97 | 124.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 125.00 | 124.17 | 124.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 125.00 | 124.17 | 124.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 124.98 | 124.34 | 124.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 125.30 | 124.34 | 124.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 125.20 | 124.68 | 124.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 126.31 | 125.30 | 124.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 125.54 | 125.87 | 125.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 125.54 | 125.87 | 125.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 125.54 | 125.87 | 125.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:45:00 | 125.60 | 125.87 | 125.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 125.41 | 125.78 | 125.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 125.49 | 125.78 | 125.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 124.75 | 125.57 | 125.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 124.75 | 125.57 | 125.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 124.96 | 125.45 | 125.41 | EMA400 retest candle locked (from upside) |

### Cycle 182 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 124.75 | 125.31 | 125.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 123.77 | 125.00 | 125.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 124.75 | 124.65 | 124.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 12:00:00 | 124.75 | 124.65 | 124.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 124.41 | 124.60 | 124.89 | EMA400 retest candle locked (from downside) |

### Cycle 183 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 125.80 | 125.16 | 125.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 126.94 | 125.72 | 125.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 12:15:00 | 127.14 | 127.48 | 126.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 13:00:00 | 127.14 | 127.48 | 126.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 127.12 | 127.41 | 126.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:45:00 | 127.45 | 127.42 | 126.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 126.36 | 127.12 | 126.94 | SL hit (close<static) qty=1.00 sl=126.81 alert=retest2 |

### Cycle 184 — SELL (started 2026-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 13:15:00 | 125.92 | 126.67 | 126.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 15:15:00 | 125.16 | 126.22 | 126.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 119.97 | 119.54 | 121.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 119.97 | 119.54 | 121.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 121.30 | 120.01 | 120.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 121.30 | 120.01 | 120.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 121.56 | 120.32 | 120.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 122.54 | 120.32 | 120.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 121.77 | 120.99 | 121.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:15:00 | 122.58 | 120.99 | 121.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 122.17 | 121.45 | 121.37 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 118.99 | 121.11 | 121.27 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 123.90 | 121.33 | 120.98 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 120.23 | 121.94 | 122.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 118.70 | 120.72 | 121.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 120.14 | 119.18 | 120.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 120.14 | 119.18 | 120.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 120.14 | 119.18 | 120.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 120.01 | 119.18 | 120.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 120.10 | 119.36 | 120.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 119.45 | 119.36 | 120.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 121.04 | 120.10 | 120.15 | SL hit (close>static) qty=1.00 sl=120.75 alert=retest2 |

### Cycle 189 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 122.02 | 120.49 | 120.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 123.14 | 121.42 | 120.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 119.90 | 121.55 | 121.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 119.90 | 121.55 | 121.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 119.90 | 121.55 | 121.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 119.90 | 121.55 | 121.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 119.80 | 121.20 | 120.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 119.80 | 121.20 | 120.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 119.57 | 120.65 | 120.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 118.48 | 120.22 | 120.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 120.96 | 119.88 | 120.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 120.96 | 119.88 | 120.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 120.96 | 119.88 | 120.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 121.22 | 119.88 | 120.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 121.23 | 120.15 | 120.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 121.33 | 120.15 | 120.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 121.42 | 120.53 | 120.49 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 118.07 | 120.12 | 120.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 117.79 | 119.66 | 120.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 118.30 | 117.17 | 118.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 118.30 | 117.17 | 118.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 118.30 | 117.17 | 118.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 118.56 | 117.17 | 118.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 119.69 | 117.68 | 118.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 119.69 | 117.68 | 118.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 119.61 | 118.06 | 118.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 119.61 | 118.06 | 118.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 123.14 | 119.35 | 118.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 123.61 | 120.20 | 119.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 10:15:00 | 121.23 | 121.55 | 120.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 10:15:00 | 121.23 | 121.55 | 120.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 121.23 | 121.55 | 120.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 120.63 | 121.55 | 120.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 120.88 | 121.33 | 120.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:45:00 | 120.58 | 121.33 | 120.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 120.15 | 121.10 | 120.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 120.15 | 121.10 | 120.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 118.91 | 120.66 | 120.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 118.91 | 120.66 | 120.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 118.82 | 120.29 | 120.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 116.79 | 119.59 | 120.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 119.50 | 117.11 | 118.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 119.50 | 117.11 | 118.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 119.50 | 117.11 | 118.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 119.50 | 117.11 | 118.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 118.87 | 117.46 | 118.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:15:00 | 119.84 | 117.46 | 118.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 120.36 | 118.98 | 118.84 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 117.73 | 118.63 | 118.74 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 119.53 | 118.91 | 118.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 120.31 | 119.23 | 118.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 127.72 | 128.54 | 126.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 127.05 | 128.54 | 126.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 128.50 | 129.64 | 128.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 129.10 | 129.64 | 128.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 10:15:00 | 125.61 | 132.03 | 132.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 10:15:00 | 125.61 | 132.03 | 132.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 14:15:00 | 125.39 | 128.37 | 130.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 126.89 | 126.21 | 127.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 126.89 | 126.21 | 127.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 126.89 | 126.21 | 127.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:45:00 | 127.15 | 126.21 | 127.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 127.20 | 126.41 | 127.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:45:00 | 127.49 | 126.41 | 127.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 127.37 | 126.60 | 127.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:00:00 | 127.37 | 126.60 | 127.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 127.60 | 126.80 | 127.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 13:00:00 | 127.60 | 126.80 | 127.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 127.51 | 126.94 | 127.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 13:45:00 | 127.74 | 126.94 | 127.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 127.17 | 126.99 | 127.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 127.49 | 126.99 | 127.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 127.25 | 127.04 | 127.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:45:00 | 125.95 | 126.78 | 127.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:15:00 | 125.80 | 125.13 | 125.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 127.14 | 125.92 | 125.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 127.14 | 125.92 | 125.85 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 124.05 | 125.73 | 125.89 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 126.81 | 125.85 | 125.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 127.00 | 126.49 | 126.15 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-16 15:00:00 | 159.15 | 2023-05-17 11:15:00 | 159.25 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2023-05-17 09:15:00 | 158.75 | 2023-05-17 11:15:00 | 159.25 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2023-05-18 09:15:00 | 160.35 | 2023-05-19 10:15:00 | 158.70 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2023-05-24 14:00:00 | 156.70 | 2023-05-29 10:15:00 | 156.65 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2023-05-26 09:45:00 | 156.55 | 2023-05-29 10:15:00 | 156.65 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2023-05-29 10:15:00 | 156.65 | 2023-05-29 10:15:00 | 156.65 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2023-05-31 12:30:00 | 153.60 | 2023-06-07 09:15:00 | 145.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-31 14:00:00 | 153.40 | 2023-06-07 09:15:00 | 145.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-01 09:45:00 | 153.65 | 2023-06-07 09:15:00 | 145.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-01 10:30:00 | 153.60 | 2023-06-07 09:15:00 | 145.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-02 10:15:00 | 153.15 | 2023-06-07 09:15:00 | 145.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-02 11:15:00 | 153.00 | 2023-06-07 09:15:00 | 145.54 | PARTIAL | 0.50 | 4.88% |
| SELL | retest2 | 2023-06-05 13:00:00 | 153.20 | 2023-06-07 09:15:00 | 145.59 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2023-06-05 14:00:00 | 153.25 | 2023-06-07 09:15:00 | 145.49 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2023-05-31 12:30:00 | 153.60 | 2023-06-07 11:15:00 | 148.95 | STOP_HIT | 0.50 | 3.03% |
| SELL | retest2 | 2023-05-31 14:00:00 | 153.40 | 2023-06-07 11:15:00 | 148.95 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2023-06-01 09:45:00 | 153.65 | 2023-06-07 11:15:00 | 148.95 | STOP_HIT | 0.50 | 3.06% |
| SELL | retest2 | 2023-06-01 10:30:00 | 153.60 | 2023-06-07 11:15:00 | 148.95 | STOP_HIT | 0.50 | 3.03% |
| SELL | retest2 | 2023-06-02 10:15:00 | 153.15 | 2023-06-07 11:15:00 | 148.95 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2023-06-02 11:15:00 | 153.00 | 2023-06-07 11:15:00 | 148.95 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2023-06-05 13:00:00 | 153.20 | 2023-06-07 11:15:00 | 148.95 | STOP_HIT | 0.50 | 2.77% |
| SELL | retest2 | 2023-06-05 14:00:00 | 153.25 | 2023-06-07 11:15:00 | 148.95 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2023-06-05 15:15:00 | 153.15 | 2023-06-08 14:15:00 | 137.70 | TARGET_HIT | 1.00 | 10.09% |
| BUY | retest2 | 2023-06-21 12:30:00 | 130.40 | 2023-06-21 14:15:00 | 128.50 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2023-06-23 14:45:00 | 127.45 | 2023-06-26 10:15:00 | 128.70 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-06-26 09:45:00 | 127.40 | 2023-06-26 10:15:00 | 128.70 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2023-06-26 10:30:00 | 127.50 | 2023-06-26 11:15:00 | 129.55 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2023-06-30 11:15:00 | 127.45 | 2023-07-06 12:15:00 | 126.75 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2023-06-30 14:00:00 | 127.30 | 2023-07-06 12:15:00 | 126.75 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2023-07-12 10:15:00 | 123.80 | 2023-07-17 09:15:00 | 124.65 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2023-07-12 11:00:00 | 123.85 | 2023-07-17 09:15:00 | 124.65 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-07-12 11:30:00 | 123.55 | 2023-07-17 09:15:00 | 124.65 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2023-07-13 09:30:00 | 123.70 | 2023-07-17 09:15:00 | 124.65 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-07-31 13:15:00 | 122.70 | 2023-08-01 09:15:00 | 126.15 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2023-08-03 09:15:00 | 127.50 | 2023-08-03 09:15:00 | 125.30 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2023-08-04 09:15:00 | 127.05 | 2023-08-11 15:15:00 | 128.35 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2023-08-04 11:00:00 | 127.00 | 2023-08-11 15:15:00 | 128.35 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2023-08-04 15:15:00 | 127.00 | 2023-08-11 15:15:00 | 128.35 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2023-08-08 09:30:00 | 127.65 | 2023-08-11 15:15:00 | 128.35 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2023-08-18 09:15:00 | 125.95 | 2023-08-23 11:15:00 | 125.35 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2023-08-24 09:15:00 | 126.00 | 2023-08-25 09:15:00 | 124.50 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2023-09-01 09:15:00 | 127.25 | 2023-09-05 09:15:00 | 139.98 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-14 12:30:00 | 134.35 | 2023-09-15 09:15:00 | 136.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2023-09-15 15:00:00 | 134.20 | 2023-09-27 15:15:00 | 132.05 | STOP_HIT | 1.00 | 1.60% |
| SELL | retest2 | 2023-09-18 09:15:00 | 134.50 | 2023-09-27 15:15:00 | 132.05 | STOP_HIT | 1.00 | 1.82% |
| SELL | retest2 | 2023-09-18 10:15:00 | 134.25 | 2023-09-27 15:15:00 | 132.05 | STOP_HIT | 1.00 | 1.64% |
| SELL | retest2 | 2023-09-21 12:00:00 | 132.85 | 2023-09-27 15:15:00 | 132.05 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2023-09-26 10:15:00 | 132.80 | 2023-09-27 15:15:00 | 132.05 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2023-10-09 10:15:00 | 132.60 | 2023-10-09 13:15:00 | 131.80 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-10-09 11:30:00 | 132.55 | 2023-10-09 13:15:00 | 131.80 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2023-10-09 13:15:00 | 132.25 | 2023-10-09 13:15:00 | 131.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2023-10-12 09:15:00 | 133.45 | 2023-10-18 11:15:00 | 133.80 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2023-11-21 13:30:00 | 139.20 | 2023-11-30 10:15:00 | 143.30 | STOP_HIT | 1.00 | 2.95% |
| SELL | retest2 | 2023-12-01 12:15:00 | 143.05 | 2023-12-04 09:15:00 | 146.60 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2023-12-01 13:15:00 | 143.15 | 2023-12-04 09:15:00 | 146.60 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2023-12-01 14:30:00 | 143.05 | 2023-12-04 09:15:00 | 146.60 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2023-12-08 15:15:00 | 153.00 | 2023-12-12 14:15:00 | 151.35 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-12-12 09:45:00 | 153.40 | 2023-12-12 14:15:00 | 151.35 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2023-12-19 11:15:00 | 154.15 | 2023-12-20 12:15:00 | 150.75 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2023-12-19 12:45:00 | 154.05 | 2023-12-20 12:15:00 | 150.75 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2023-12-19 15:00:00 | 154.20 | 2023-12-20 12:15:00 | 150.75 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-01-02 13:30:00 | 166.10 | 2024-01-03 09:15:00 | 163.65 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-01-02 14:00:00 | 166.85 | 2024-01-03 09:15:00 | 163.65 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-01-02 15:15:00 | 166.10 | 2024-01-03 09:15:00 | 163.65 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-01-23 11:30:00 | 138.00 | 2024-01-29 11:15:00 | 140.75 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-01-23 12:30:00 | 138.20 | 2024-01-29 11:15:00 | 140.75 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-01-23 13:45:00 | 138.20 | 2024-01-29 11:15:00 | 140.75 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-02-05 09:15:00 | 147.05 | 2024-02-07 10:15:00 | 144.45 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-02-05 15:00:00 | 146.60 | 2024-02-07 10:15:00 | 144.45 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-02-07 10:00:00 | 147.00 | 2024-02-07 10:15:00 | 144.45 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-02-26 13:30:00 | 147.40 | 2024-02-27 13:15:00 | 145.50 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-02-26 14:15:00 | 147.30 | 2024-02-27 13:15:00 | 145.50 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-02-27 09:15:00 | 149.40 | 2024-02-27 13:15:00 | 145.50 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2024-03-05 13:30:00 | 151.00 | 2024-03-06 09:15:00 | 147.45 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-03-15 10:15:00 | 134.50 | 2024-03-21 13:15:00 | 135.65 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-03-19 10:15:00 | 135.00 | 2024-03-21 13:15:00 | 135.65 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-03-19 14:00:00 | 135.15 | 2024-03-21 13:15:00 | 135.65 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-03-19 15:00:00 | 134.35 | 2024-03-21 13:15:00 | 135.65 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-03-28 12:15:00 | 134.40 | 2024-04-01 09:15:00 | 138.70 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2024-03-28 13:45:00 | 134.40 | 2024-04-01 09:15:00 | 138.70 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2024-03-28 15:00:00 | 134.35 | 2024-04-01 09:15:00 | 138.70 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2024-04-10 10:15:00 | 150.70 | 2024-04-15 13:15:00 | 148.55 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-04-10 12:15:00 | 151.60 | 2024-04-15 13:15:00 | 148.55 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-04-12 09:15:00 | 151.65 | 2024-04-15 13:15:00 | 148.55 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-04-12 09:45:00 | 150.90 | 2024-04-15 13:15:00 | 148.55 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-04-16 12:15:00 | 147.25 | 2024-04-22 11:15:00 | 147.90 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-04-18 10:15:00 | 147.55 | 2024-04-22 11:15:00 | 147.90 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-04-18 13:15:00 | 147.55 | 2024-04-22 11:15:00 | 147.90 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-04-19 15:15:00 | 147.40 | 2024-04-22 11:15:00 | 147.90 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-04-30 09:15:00 | 156.25 | 2024-05-02 11:15:00 | 155.45 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-05-17 11:45:00 | 147.90 | 2024-05-21 09:15:00 | 162.69 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-30 13:15:00 | 155.50 | 2024-06-03 09:15:00 | 159.55 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2024-06-21 11:15:00 | 183.45 | 2024-06-25 09:15:00 | 181.58 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-06-21 12:00:00 | 183.49 | 2024-06-25 10:15:00 | 180.45 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-06-21 14:00:00 | 183.79 | 2024-06-25 10:15:00 | 180.45 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-06-24 12:00:00 | 183.95 | 2024-06-25 10:15:00 | 180.45 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-06-25 09:15:00 | 184.30 | 2024-06-25 10:15:00 | 180.45 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-06-27 13:45:00 | 178.67 | 2024-06-28 09:15:00 | 182.26 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-07-02 14:15:00 | 184.85 | 2024-07-04 11:15:00 | 184.58 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-07-09 15:15:00 | 182.81 | 2024-07-10 09:15:00 | 173.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-09 15:15:00 | 182.81 | 2024-07-11 09:15:00 | 177.10 | STOP_HIT | 0.50 | 3.12% |
| SELL | retest2 | 2024-07-23 09:15:00 | 170.88 | 2024-07-23 12:15:00 | 162.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-23 09:15:00 | 170.88 | 2024-07-24 09:15:00 | 173.30 | STOP_HIT | 0.50 | -1.42% |
| BUY | retest2 | 2024-08-02 10:15:00 | 195.25 | 2024-08-05 13:15:00 | 189.08 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2024-08-05 09:30:00 | 195.00 | 2024-08-05 13:15:00 | 189.08 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2024-08-09 09:15:00 | 195.46 | 2024-08-09 12:15:00 | 193.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-08-22 09:15:00 | 197.63 | 2024-08-23 09:15:00 | 191.59 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2024-08-22 13:30:00 | 198.00 | 2024-08-23 09:15:00 | 191.59 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2024-09-03 09:15:00 | 206.00 | 2024-09-18 13:15:00 | 226.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-27 10:30:00 | 209.35 | 2024-10-03 11:15:00 | 208.70 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2024-10-07 09:15:00 | 209.73 | 2024-10-07 09:15:00 | 204.81 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-10-14 09:30:00 | 202.43 | 2024-10-14 13:15:00 | 192.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 09:30:00 | 202.43 | 2024-10-16 10:15:00 | 194.53 | STOP_HIT | 0.50 | 3.90% |
| BUY | retest2 | 2024-12-06 10:15:00 | 180.59 | 2024-12-17 13:15:00 | 186.89 | STOP_HIT | 1.00 | 3.49% |
| SELL | retest2 | 2024-12-20 12:45:00 | 181.77 | 2024-12-27 09:15:00 | 181.58 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-01-14 12:15:00 | 165.82 | 2025-01-15 10:15:00 | 168.80 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-01-15 09:15:00 | 165.59 | 2025-01-15 10:15:00 | 168.80 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-01-28 14:45:00 | 167.30 | 2025-01-29 09:15:00 | 174.85 | STOP_HIT | 1.00 | -4.51% |
| BUY | retest2 | 2025-01-31 10:15:00 | 176.30 | 2025-02-01 11:15:00 | 172.41 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-02-13 15:15:00 | 169.15 | 2025-02-17 09:15:00 | 160.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:15:00 | 169.15 | 2025-02-17 12:15:00 | 165.42 | STOP_HIT | 0.50 | 2.21% |
| BUY | retest2 | 2025-02-21 15:00:00 | 169.32 | 2025-02-24 09:15:00 | 165.40 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-02-25 15:15:00 | 165.51 | 2025-02-27 11:15:00 | 157.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 15:15:00 | 165.51 | 2025-02-28 09:15:00 | 160.95 | STOP_HIT | 0.50 | 2.76% |
| BUY | retest1 | 2025-03-07 14:30:00 | 164.25 | 2025-03-10 09:15:00 | 160.85 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-03-10 13:15:00 | 160.58 | 2025-03-10 14:15:00 | 159.52 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-03-12 11:15:00 | 155.74 | 2025-03-17 11:15:00 | 161.84 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2025-03-13 09:15:00 | 154.61 | 2025-03-17 11:15:00 | 161.84 | STOP_HIT | 1.00 | -4.68% |
| SELL | retest2 | 2025-03-13 10:45:00 | 155.08 | 2025-03-17 11:15:00 | 161.84 | STOP_HIT | 1.00 | -4.36% |
| BUY | retest2 | 2025-03-26 11:45:00 | 177.04 | 2025-03-28 13:15:00 | 175.18 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-04-04 11:30:00 | 179.06 | 2025-04-07 09:15:00 | 172.05 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2025-04-04 12:30:00 | 178.98 | 2025-04-07 09:15:00 | 172.05 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2025-04-04 15:15:00 | 179.10 | 2025-04-07 09:15:00 | 172.05 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2025-04-25 09:15:00 | 199.41 | 2025-04-25 15:15:00 | 190.00 | STOP_HIT | 1.00 | -4.72% |
| BUY | retest2 | 2025-04-25 14:45:00 | 191.61 | 2025-04-25 15:15:00 | 190.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-04-28 09:15:00 | 192.04 | 2025-04-30 12:15:00 | 191.48 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-05-02 11:15:00 | 191.32 | 2025-05-05 09:15:00 | 193.92 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-05-02 12:00:00 | 190.80 | 2025-05-05 09:15:00 | 193.92 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-05-02 14:15:00 | 190.70 | 2025-05-05 09:15:00 | 193.92 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-05-14 10:30:00 | 196.33 | 2025-05-21 11:15:00 | 196.60 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-05-16 11:15:00 | 196.36 | 2025-05-21 11:15:00 | 196.60 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-05-26 10:45:00 | 194.90 | 2025-05-27 11:15:00 | 198.20 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-05-29 09:15:00 | 198.68 | 2025-06-05 10:15:00 | 198.25 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-05-29 09:45:00 | 198.06 | 2025-06-05 11:15:00 | 197.10 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-05-29 10:45:00 | 198.87 | 2025-06-05 11:15:00 | 197.10 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-06-04 09:45:00 | 197.99 | 2025-06-05 11:15:00 | 197.10 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-06-04 11:30:00 | 203.41 | 2025-06-05 11:15:00 | 197.10 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-06-23 09:15:00 | 180.95 | 2025-06-24 09:15:00 | 186.81 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-06-26 14:45:00 | 189.93 | 2025-07-09 12:15:00 | 208.92 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-21 13:15:00 | 201.76 | 2025-07-22 15:15:00 | 191.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 13:15:00 | 201.76 | 2025-07-24 09:15:00 | 181.58 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-19 12:30:00 | 141.39 | 2025-08-26 09:15:00 | 142.86 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2025-08-19 14:15:00 | 141.61 | 2025-08-26 09:15:00 | 142.86 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2025-08-20 09:45:00 | 142.16 | 2025-08-26 09:15:00 | 142.86 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2025-09-03 10:00:00 | 141.80 | 2025-09-05 11:15:00 | 140.32 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-09-03 11:00:00 | 141.60 | 2025-09-05 11:15:00 | 140.32 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-11 13:30:00 | 144.13 | 2025-09-11 14:15:00 | 143.41 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-09-11 14:00:00 | 144.06 | 2025-09-11 14:15:00 | 143.41 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-09-12 10:15:00 | 144.06 | 2025-09-22 14:15:00 | 146.00 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2025-09-12 11:00:00 | 144.24 | 2025-09-22 14:15:00 | 146.00 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2025-09-12 15:15:00 | 145.70 | 2025-09-22 14:15:00 | 146.00 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-09-29 15:00:00 | 139.59 | 2025-10-01 12:15:00 | 141.57 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-10-01 09:45:00 | 139.12 | 2025-10-01 12:15:00 | 141.57 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-10-10 14:00:00 | 140.06 | 2025-10-20 12:15:00 | 137.84 | STOP_HIT | 1.00 | 1.59% |
| SELL | retest2 | 2025-10-10 15:00:00 | 139.72 | 2025-10-20 12:15:00 | 137.84 | STOP_HIT | 1.00 | 1.35% |
| SELL | retest2 | 2025-11-04 10:30:00 | 140.14 | 2025-11-10 10:15:00 | 139.71 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-11-04 11:15:00 | 139.52 | 2025-11-10 10:15:00 | 139.71 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-11-07 13:15:00 | 140.02 | 2025-11-10 10:15:00 | 139.71 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-11-17 11:15:00 | 138.20 | 2025-11-20 09:15:00 | 140.75 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-11-17 13:00:00 | 137.87 | 2025-11-20 09:15:00 | 140.75 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-12-03 11:45:00 | 148.35 | 2025-12-05 11:15:00 | 145.47 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-12-03 13:45:00 | 148.20 | 2025-12-05 11:15:00 | 145.47 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-12-04 13:15:00 | 147.96 | 2025-12-05 11:15:00 | 145.47 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-12-04 14:00:00 | 148.01 | 2025-12-05 11:15:00 | 145.47 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-12-17 13:30:00 | 140.66 | 2025-12-19 14:15:00 | 141.46 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-17 14:30:00 | 140.15 | 2025-12-19 14:15:00 | 141.46 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-18 13:30:00 | 140.50 | 2025-12-19 14:15:00 | 141.46 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-12-23 09:15:00 | 141.40 | 2025-12-24 13:15:00 | 140.10 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-01-01 09:30:00 | 133.67 | 2026-01-02 15:15:00 | 134.49 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-01-01 11:45:00 | 133.64 | 2026-01-02 15:15:00 | 134.49 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-01-02 13:15:00 | 134.00 | 2026-01-02 15:15:00 | 134.49 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2026-01-02 13:45:00 | 133.86 | 2026-01-02 15:15:00 | 134.49 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-01-06 14:45:00 | 144.12 | 2026-01-09 09:15:00 | 158.53 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-30 09:15:00 | 125.93 | 2026-02-02 10:15:00 | 119.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-30 09:15:00 | 125.93 | 2026-02-02 14:15:00 | 122.77 | STOP_HIT | 0.50 | 2.51% |
| SELL | retest2 | 2026-02-01 12:15:00 | 124.20 | 2026-02-03 14:15:00 | 126.15 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-02-03 12:00:00 | 125.70 | 2026-02-03 14:15:00 | 126.15 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-11 14:00:00 | 127.14 | 2026-02-13 09:15:00 | 123.68 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest1 | 2026-02-11 15:00:00 | 126.95 | 2026-02-13 09:15:00 | 123.68 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2026-02-26 14:45:00 | 127.45 | 2026-02-27 11:15:00 | 126.36 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-03-17 11:15:00 | 119.45 | 2026-03-18 09:15:00 | 121.04 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-04-13 10:15:00 | 129.10 | 2026-04-20 10:15:00 | 125.61 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-04-24 10:45:00 | 125.95 | 2026-04-28 09:15:00 | 127.14 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-04-27 14:15:00 | 125.80 | 2026-04-28 09:15:00 | 127.14 | STOP_HIT | 1.00 | -1.07% |
