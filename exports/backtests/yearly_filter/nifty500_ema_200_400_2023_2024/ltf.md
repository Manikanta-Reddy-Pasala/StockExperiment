# L&T Finance Ltd. (LTF)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 302.85
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
| ALERT2 | 8 |
| ALERT2_SKIP | 2 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 26
- **Target hits / Stop hits / Partials:** 2 / 28 / 2
- **Avg / median % per leg:** -1.17% / -2.10%
- **Sum % (uncompounded):** -37.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 2 | 6 | 0 | -0.07% | -0.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 2 | 6 | 0 | -0.07% | -0.6% |
| SELL (all) | 24 | 4 | 16.7% | 0 | 22 | 2 | -1.54% | -36.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 4 | 16.7% | 0 | 22 | 2 | -1.54% | -36.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 6 | 18.8% | 2 | 28 | 2 | -1.17% | -37.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 12:15:00 | 148.00 | 162.25 | 162.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 14:15:00 | 146.80 | 161.96 | 162.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 09:15:00 | 160.30 | 159.87 | 161.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-27 10:00:00 | 160.30 | 159.87 | 161.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 159.35 | 159.85 | 160.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 10:30:00 | 158.50 | 159.85 | 160.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 11:15:00 | 158.70 | 159.85 | 160.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 14:45:00 | 158.65 | 159.81 | 160.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-01 09:15:00 | 161.70 | 159.83 | 160.90 | SL hit (close>static) qty=1.00 sl=161.10 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 09:15:00 | 169.60 | 161.80 | 161.80 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 15:15:00 | 157.25 | 163.04 | 163.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 09:15:00 | 155.70 | 162.96 | 163.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 12:15:00 | 159.65 | 159.46 | 160.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-03 12:30:00 | 159.65 | 159.46 | 160.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 161.50 | 158.53 | 160.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 161.40 | 158.53 | 160.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 160.20 | 158.54 | 160.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 12:45:00 | 159.15 | 158.55 | 160.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:45:00 | 159.30 | 158.55 | 160.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 09:15:00 | 162.65 | 158.61 | 160.24 | SL hit (close>static) qty=1.00 sl=161.60 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 15:15:00 | 175.70 | 161.64 | 161.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 09:15:00 | 182.44 | 161.84 | 161.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 10:15:00 | 179.42 | 179.47 | 173.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 11:00:00 | 179.42 | 179.47 | 173.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 175.35 | 179.32 | 174.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 10:30:00 | 175.50 | 179.27 | 174.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 09:15:00 | 175.55 | 178.60 | 173.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 11:45:00 | 175.47 | 178.53 | 174.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 175.45 | 178.04 | 173.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 173.85 | 178.51 | 174.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:30:00 | 172.85 | 178.51 | 174.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 176.22 | 178.49 | 174.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 14:00:00 | 177.63 | 178.44 | 174.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 171.40 | 178.34 | 174.96 | SL hit (close<static) qty=1.00 sl=173.51 alert=retest2 |

### Cycle 5 — SELL (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 15:15:00 | 163.90 | 172.57 | 172.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 163.77 | 170.47 | 171.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 13:15:00 | 171.83 | 170.11 | 171.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 13:15:00 | 171.83 | 170.11 | 171.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 171.83 | 170.11 | 171.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 14:00:00 | 171.83 | 170.11 | 171.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 171.80 | 170.13 | 171.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:15:00 | 170.90 | 170.17 | 171.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 09:30:00 | 170.70 | 170.07 | 170.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 11:15:00 | 170.30 | 170.09 | 170.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 14:15:00 | 174.40 | 170.19 | 170.97 | SL hit (close>static) qty=1.00 sl=173.24 alert=retest2 |

### Cycle 6 — BUY (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 14:15:00 | 176.85 | 171.69 | 171.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 14:15:00 | 178.25 | 171.96 | 171.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 176.59 | 178.13 | 175.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 13:00:00 | 176.59 | 178.13 | 175.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 174.98 | 178.10 | 175.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:00:00 | 174.98 | 178.10 | 175.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 174.75 | 178.07 | 175.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 10:30:00 | 175.98 | 177.00 | 175.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 11:15:00 | 173.17 | 176.97 | 175.10 | SL hit (close<static) qty=1.00 sl=173.91 alert=retest2 |

### Cycle 7 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 164.71 | 173.54 | 173.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 164.28 | 173.29 | 173.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 147.40 | 146.20 | 153.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 10:00:00 | 147.40 | 146.20 | 153.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 144.60 | 140.77 | 144.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 15:00:00 | 144.60 | 140.77 | 144.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 144.40 | 140.81 | 144.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 144.90 | 140.81 | 144.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 147.13 | 140.87 | 144.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 147.13 | 140.87 | 144.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 146.45 | 140.93 | 144.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 140.79 | 141.19 | 144.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 11:15:00 | 147.34 | 141.38 | 144.21 | SL hit (close>static) qty=1.00 sl=147.27 alert=retest2 |

### Cycle 8 — BUY (started 2025-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 14:15:00 | 158.80 | 142.93 | 142.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 09:15:00 | 161.75 | 149.94 | 147.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 12:15:00 | 160.74 | 161.56 | 155.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-09 13:00:00 | 160.74 | 161.56 | 155.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 194.71 | 201.63 | 195.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:45:00 | 194.95 | 201.63 | 195.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 195.50 | 201.57 | 195.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 14:00:00 | 196.23 | 201.52 | 195.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:45:00 | 197.54 | 201.41 | 195.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-19 10:15:00 | 215.85 | 202.17 | 196.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 12:15:00 | 273.00 | 290.92 | 290.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 271.40 | 290.73 | 290.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 274.88 | 263.91 | 273.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 274.88 | 263.91 | 273.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 274.88 | 263.91 | 273.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:15:00 | 276.80 | 263.91 | 273.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 277.09 | 264.04 | 273.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:30:00 | 275.16 | 264.04 | 273.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 275.55 | 264.84 | 273.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 11:00:00 | 275.55 | 264.84 | 273.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 275.23 | 265.36 | 273.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:30:00 | 273.59 | 266.13 | 273.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 14:45:00 | 273.34 | 266.59 | 273.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 282.45 | 266.82 | 273.81 | SL hit (close>static) qty=1.00 sl=277.19 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 280.07 | 278.49 | 278.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 286.70 | 278.60 | 278.54 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-03-28 10:30:00 | 158.50 | 2024-04-01 09:15:00 | 161.70 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-03-28 11:15:00 | 158.70 | 2024-04-01 09:15:00 | 161.70 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-03-28 14:45:00 | 158.65 | 2024-04-01 09:15:00 | 161.70 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-06-06 12:45:00 | 159.15 | 2024-06-07 09:15:00 | 162.65 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-06-06 13:45:00 | 159.30 | 2024-06-07 09:15:00 | 162.65 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-07-22 10:30:00 | 175.50 | 2024-08-05 09:15:00 | 171.40 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-07-24 09:15:00 | 175.55 | 2024-08-05 10:15:00 | 168.70 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2024-07-24 11:45:00 | 175.47 | 2024-08-05 10:15:00 | 168.70 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2024-07-26 09:15:00 | 175.45 | 2024-08-05 10:15:00 | 168.70 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2024-08-02 14:00:00 | 177.63 | 2024-08-05 10:15:00 | 168.70 | STOP_HIT | 1.00 | -5.03% |
| SELL | retest2 | 2024-09-11 11:15:00 | 170.90 | 2024-09-12 14:15:00 | 174.40 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-09-12 09:30:00 | 170.70 | 2024-09-12 14:15:00 | 174.40 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-09-12 11:15:00 | 170.30 | 2024-09-12 14:15:00 | 174.40 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-10-09 10:30:00 | 175.98 | 2024-10-09 11:15:00 | 173.17 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-01-22 09:15:00 | 140.79 | 2025-01-30 11:15:00 | 147.34 | STOP_HIT | 1.00 | -4.65% |
| SELL | retest2 | 2025-01-30 14:45:00 | 144.90 | 2025-02-03 10:15:00 | 148.00 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-01-31 11:45:00 | 145.21 | 2025-02-03 10:15:00 | 148.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-01-31 13:45:00 | 145.23 | 2025-02-03 10:15:00 | 148.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-02-11 10:00:00 | 143.29 | 2025-02-12 10:15:00 | 136.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 10:30:00 | 143.01 | 2025-02-12 10:15:00 | 135.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 10:00:00 | 143.29 | 2025-02-27 09:15:00 | 140.49 | STOP_HIT | 0.50 | 1.95% |
| SELL | retest2 | 2025-02-11 10:30:00 | 143.01 | 2025-02-27 09:15:00 | 140.49 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2025-03-10 12:30:00 | 143.22 | 2025-03-19 09:15:00 | 143.49 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-03-10 13:15:00 | 142.95 | 2025-03-19 09:15:00 | 143.49 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-03-12 10:15:00 | 139.63 | 2025-03-19 09:15:00 | 143.49 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-03-12 11:00:00 | 139.20 | 2025-03-19 09:15:00 | 143.49 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-03-13 09:15:00 | 139.69 | 2025-03-19 13:15:00 | 146.56 | STOP_HIT | 1.00 | -4.92% |
| SELL | retest2 | 2025-03-13 10:45:00 | 139.10 | 2025-03-19 13:15:00 | 146.56 | STOP_HIT | 1.00 | -5.36% |
| BUY | retest2 | 2025-08-11 14:00:00 | 196.23 | 2025-08-19 10:15:00 | 215.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-12 09:45:00 | 197.54 | 2025-08-19 14:15:00 | 217.29 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-13 09:30:00 | 273.59 | 2026-04-15 09:15:00 | 282.45 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2026-04-13 14:45:00 | 273.34 | 2026-04-15 09:15:00 | 282.45 | STOP_HIT | 1.00 | -3.33% |
