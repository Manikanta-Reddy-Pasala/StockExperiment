# NHPC Ltd. (NHPC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 80.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 44 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 38 |
| PARTIAL | 14 |
| TARGET_HIT | 3 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 22
- **Target hits / Stop hits / Partials:** 3 / 35 / 14
- **Avg / median % per leg:** 1.20% / 0.81%
- **Sum % (uncompounded):** 62.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 2 | 22.2% | 1 | 8 | 0 | -0.25% | -2.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 2 | 22.2% | 1 | 8 | 0 | -0.25% | -2.3% |
| SELL (all) | 43 | 28 | 65.1% | 2 | 27 | 14 | 1.51% | 64.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 43 | 28 | 65.1% | 2 | 27 | 14 | 1.51% | 64.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 52 | 30 | 57.7% | 3 | 35 | 14 | 1.20% | 62.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 14:15:00 | 95.97 | 100.77 | 100.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 13:15:00 | 95.36 | 99.39 | 100.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 12:15:00 | 98.61 | 98.50 | 99.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-03 13:00:00 | 98.61 | 98.50 | 99.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 85.99 | 82.55 | 85.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 85.99 | 82.55 | 85.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 86.27 | 82.59 | 85.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:00:00 | 86.27 | 82.59 | 85.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 85.35 | 82.61 | 85.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:30:00 | 85.90 | 82.61 | 85.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 87.56 | 82.76 | 85.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:30:00 | 87.28 | 82.76 | 85.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 86.00 | 83.11 | 85.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 11:30:00 | 86.22 | 83.11 | 85.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 85.22 | 83.48 | 85.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:30:00 | 85.60 | 83.48 | 85.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 83.76 | 83.55 | 85.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 10:15:00 | 83.43 | 83.55 | 85.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 13:30:00 | 83.63 | 83.85 | 85.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:45:00 | 83.51 | 83.84 | 85.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 15:15:00 | 83.67 | 83.84 | 85.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 12:15:00 | 79.26 | 83.16 | 84.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 12:15:00 | 79.45 | 83.16 | 84.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 12:15:00 | 79.33 | 83.16 | 84.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 12:15:00 | 79.49 | 83.16 | 84.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 82.83 | 82.72 | 84.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-03 09:15:00 | 82.83 | 82.72 | 84.42 | SL hit (close>ema200) qty=0.50 sl=82.72 alert=retest2 |

### Cycle 2 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 85.30 | 78.77 | 78.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 09:15:00 | 86.50 | 81.74 | 80.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 83.03 | 84.36 | 82.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 10:00:00 | 83.03 | 84.36 | 82.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 82.39 | 84.32 | 82.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:30:00 | 82.30 | 84.32 | 82.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 82.35 | 84.30 | 82.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 82.35 | 84.30 | 82.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 81.68 | 84.28 | 82.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:00:00 | 81.68 | 84.28 | 82.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 82.86 | 84.19 | 82.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:15:00 | 83.21 | 84.19 | 82.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 13:15:00 | 81.09 | 84.07 | 82.47 | SL hit (close<static) qty=1.00 sl=81.64 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 84.26 | 85.46 | 85.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 83.29 | 85.39 | 85.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 85.34 | 84.83 | 85.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 85.34 | 84.83 | 85.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 85.34 | 84.83 | 85.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:30:00 | 83.85 | 84.80 | 85.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 11:15:00 | 79.66 | 83.80 | 84.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 82.05 | 81.00 | 82.58 | SL hit (close>ema200) qty=0.50 sl=81.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 87.02 | 83.67 | 83.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 87.78 | 83.85 | 83.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 83.75 | 84.04 | 83.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 13:15:00 | 83.75 | 84.04 | 83.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 83.75 | 84.04 | 83.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 83.75 | 84.04 | 83.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 83.56 | 84.04 | 83.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 85.11 | 84.04 | 83.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 81.96 | 85.50 | 85.06 | SL hit (close<static) qty=1.00 sl=83.51 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 13:15:00 | 81.79 | 84.67 | 84.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 80.82 | 84.20 | 84.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 78.25 | 78.01 | 79.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:45:00 | 78.33 | 78.01 | 79.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 79.58 | 78.04 | 79.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 79.72 | 78.04 | 79.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 79.26 | 78.06 | 79.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 79.37 | 78.06 | 79.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 80.03 | 78.11 | 79.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 80.06 | 78.11 | 79.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 79.91 | 78.13 | 79.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:30:00 | 80.01 | 78.13 | 79.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 79.67 | 78.14 | 79.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:00:00 | 79.54 | 78.16 | 79.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 80.01 | 78.22 | 79.66 | SL hit (close>static) qty=1.00 sl=79.93 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 84.26 | 77.09 | 77.07 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-13 10:15:00 | 83.43 | 2024-12-30 12:15:00 | 79.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 13:30:00 | 83.63 | 2024-12-30 12:15:00 | 79.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 14:45:00 | 83.51 | 2024-12-30 12:15:00 | 79.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 15:15:00 | 83.67 | 2024-12-30 12:15:00 | 79.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 10:15:00 | 83.43 | 2025-01-03 09:15:00 | 82.83 | STOP_HIT | 0.50 | 0.72% |
| SELL | retest2 | 2024-12-18 13:30:00 | 83.63 | 2025-01-03 09:15:00 | 82.83 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest2 | 2024-12-18 14:45:00 | 83.51 | 2025-01-03 09:15:00 | 82.83 | STOP_HIT | 0.50 | 0.81% |
| SELL | retest2 | 2024-12-18 15:15:00 | 83.67 | 2025-01-03 09:15:00 | 82.83 | STOP_HIT | 0.50 | 1.00% |
| SELL | retest2 | 2025-01-03 10:15:00 | 82.64 | 2025-01-08 09:15:00 | 78.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 15:15:00 | 82.75 | 2025-01-08 09:15:00 | 78.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 10:15:00 | 82.64 | 2025-01-13 09:15:00 | 74.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 15:15:00 | 82.75 | 2025-01-13 09:15:00 | 74.48 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-21 15:00:00 | 82.41 | 2025-03-28 09:15:00 | 85.92 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2025-03-25 10:00:00 | 82.54 | 2025-03-28 09:15:00 | 85.92 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest2 | 2025-05-07 11:15:00 | 83.21 | 2025-05-08 13:15:00 | 81.09 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-05-12 14:15:00 | 83.25 | 2025-06-09 09:15:00 | 91.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-19 09:30:00 | 83.03 | 2025-06-19 11:15:00 | 81.46 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-06-23 10:00:00 | 82.93 | 2025-07-02 09:15:00 | 84.79 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2025-07-01 12:15:00 | 86.07 | 2025-07-02 09:15:00 | 84.79 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-07-02 09:15:00 | 86.00 | 2025-07-25 10:15:00 | 84.88 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-08 09:15:00 | 86.25 | 2025-07-25 10:15:00 | 84.88 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-07-24 14:00:00 | 85.96 | 2025-08-06 11:15:00 | 84.26 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-08-14 09:30:00 | 83.85 | 2025-08-25 11:15:00 | 79.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-14 09:30:00 | 83.85 | 2025-09-11 09:15:00 | 82.05 | STOP_HIT | 0.50 | 2.15% |
| BUY | retest2 | 2025-09-29 09:15:00 | 85.11 | 2025-11-06 09:15:00 | 81.96 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2026-01-01 13:00:00 | 79.54 | 2026-01-02 09:15:00 | 80.01 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-01-19 11:15:00 | 79.20 | 2026-01-23 14:15:00 | 75.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 12:45:00 | 79.52 | 2026-01-23 14:15:00 | 75.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 13:45:00 | 79.52 | 2026-01-23 14:15:00 | 75.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 11:15:00 | 79.20 | 2026-01-29 13:15:00 | 79.18 | STOP_HIT | 0.50 | 0.03% |
| SELL | retest2 | 2026-01-19 12:45:00 | 79.52 | 2026-01-29 13:15:00 | 79.18 | STOP_HIT | 0.50 | 0.43% |
| SELL | retest2 | 2026-01-19 13:45:00 | 79.52 | 2026-01-29 13:15:00 | 79.18 | STOP_HIT | 0.50 | 0.43% |
| SELL | retest2 | 2026-01-29 10:15:00 | 78.44 | 2026-02-04 10:15:00 | 80.33 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-01-29 11:30:00 | 78.66 | 2026-02-04 10:15:00 | 80.33 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-01-30 09:15:00 | 78.20 | 2026-02-04 10:15:00 | 80.33 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-01-30 10:15:00 | 78.24 | 2026-02-04 10:15:00 | 80.33 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-02-10 09:15:00 | 77.30 | 2026-03-02 09:15:00 | 73.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 10:45:00 | 77.25 | 2026-03-02 09:15:00 | 73.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 11:30:00 | 77.32 | 2026-03-02 09:15:00 | 73.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 12:15:00 | 77.46 | 2026-03-02 09:15:00 | 73.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 09:15:00 | 77.30 | 2026-03-12 11:15:00 | 76.05 | STOP_HIT | 0.50 | 1.62% |
| SELL | retest2 | 2026-02-10 10:45:00 | 77.25 | 2026-03-12 11:15:00 | 76.05 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2026-02-10 11:30:00 | 77.32 | 2026-03-12 11:15:00 | 76.05 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2026-02-10 12:15:00 | 77.46 | 2026-03-12 11:15:00 | 76.05 | STOP_HIT | 0.50 | 1.82% |
| SELL | retest2 | 2026-03-19 09:30:00 | 76.42 | 2026-03-19 11:15:00 | 77.66 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-03-19 13:00:00 | 76.43 | 2026-03-20 09:15:00 | 77.95 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-03-23 09:45:00 | 76.25 | 2026-03-25 09:15:00 | 77.70 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-03-24 15:00:00 | 76.52 | 2026-03-25 09:15:00 | 77.70 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-03-27 14:45:00 | 76.64 | 2026-04-15 09:15:00 | 78.98 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2026-04-08 09:45:00 | 76.67 | 2026-04-15 09:15:00 | 78.98 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2026-04-08 12:30:00 | 76.49 | 2026-04-15 09:15:00 | 78.98 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2026-04-10 14:15:00 | 76.67 | 2026-04-15 09:15:00 | 78.98 | STOP_HIT | 1.00 | -3.01% |
