# Billionbrains Garage Ventures Ltd. (GROWW)

## Backtest Summary

- **Window:** 2025-11-12 09:15:00 → 2026-05-08 15:15:00 (840 bars)
- **Last close:** 204.45
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 39 |
| ALERT1 | 25 |
| ALERT2 | 24 |
| ALERT2_SKIP | 13 |
| ALERT3 | 61 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 38 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 27
- **Target hits / Stop hits / Partials:** 3 / 35 / 8
- **Avg / median % per leg:** 0.43% / -0.92%
- **Sum % (uncompounded):** 19.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 3 | 16.7% | 0 | 18 | 0 | -2.08% | -37.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 3 | 16.7% | 0 | 18 | 0 | -2.08% | -37.5% |
| SELL (all) | 28 | 16 | 57.1% | 3 | 17 | 8 | 2.05% | 57.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 16 | 57.1% | 3 | 17 | 8 | 2.05% | 57.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 46 | 19 | 41.3% | 3 | 35 | 8 | 0.43% | 19.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 156.34 | 166.83 | 167.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 154.47 | 162.77 | 165.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 09:15:00 | 166.10 | 160.93 | 163.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 166.10 | 160.93 | 163.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 166.10 | 160.93 | 163.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 167.29 | 160.93 | 163.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 165.52 | 161.84 | 163.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 165.52 | 161.84 | 163.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 165.20 | 162.52 | 163.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:00:00 | 165.20 | 162.52 | 163.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 159.99 | 159.85 | 161.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:15:00 | 156.59 | 159.64 | 161.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:15:00 | 157.15 | 155.64 | 158.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 13:15:00 | 157.21 | 156.43 | 158.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 163.42 | 159.23 | 158.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 163.42 | 159.23 | 158.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 163.42 | 159.23 | 158.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 163.42 | 159.23 | 158.94 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 10:15:00 | 158.45 | 160.05 | 160.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 157.91 | 159.10 | 159.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 158.03 | 157.79 | 158.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 15:00:00 | 158.03 | 157.79 | 158.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 157.90 | 157.81 | 158.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 156.85 | 157.81 | 158.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 155.64 | 157.38 | 158.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 155.23 | 156.78 | 157.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 13:15:00 | 147.47 | 150.97 | 153.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 152.01 | 149.48 | 151.30 | SL hit (close>ema200) qty=0.50 sl=149.48 alert=retest2 |

### Cycle 4 — BUY (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 11:15:00 | 156.00 | 152.53 | 152.22 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 145.66 | 151.62 | 152.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 10:15:00 | 142.77 | 147.23 | 149.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 14:15:00 | 145.85 | 145.56 | 147.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 15:00:00 | 145.85 | 145.56 | 147.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 147.18 | 145.90 | 147.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 15:00:00 | 145.59 | 146.56 | 147.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 09:30:00 | 144.70 | 145.90 | 146.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 144.60 | 145.47 | 145.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:00:00 | 145.14 | 145.40 | 145.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 144.39 | 144.92 | 145.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 144.39 | 144.92 | 145.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 144.92 | 143.69 | 144.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:30:00 | 144.80 | 143.69 | 144.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 144.37 | 143.82 | 144.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:30:00 | 144.22 | 143.81 | 144.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:00:00 | 143.78 | 143.81 | 144.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:30:00 | 144.23 | 143.74 | 143.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 146.13 | 144.22 | 144.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 146.13 | 144.22 | 144.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 146.13 | 144.22 | 144.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 146.13 | 144.22 | 144.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 146.13 | 144.22 | 144.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 146.13 | 144.22 | 144.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 146.13 | 144.22 | 144.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 146.13 | 144.22 | 144.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 11:15:00 | 157.50 | 146.87 | 145.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 163.40 | 164.47 | 158.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 163.89 | 164.47 | 158.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 173.82 | 164.47 | 161.31 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 160.76 | 162.88 | 163.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 159.37 | 161.27 | 162.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 156.09 | 155.66 | 156.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 156.09 | 155.66 | 156.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 156.09 | 155.66 | 156.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 156.09 | 155.66 | 156.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 154.67 | 154.54 | 155.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:30:00 | 154.77 | 154.54 | 155.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 155.61 | 154.75 | 155.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:45:00 | 156.11 | 154.75 | 155.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 155.15 | 154.83 | 155.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 157.19 | 154.83 | 155.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 156.06 | 155.08 | 155.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:15:00 | 155.40 | 155.26 | 155.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:00:00 | 155.18 | 155.42 | 155.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 158.27 | 155.89 | 155.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 158.27 | 155.89 | 155.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 158.27 | 155.89 | 155.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 10:15:00 | 162.93 | 157.30 | 156.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 13:15:00 | 160.95 | 161.75 | 160.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 14:00:00 | 160.95 | 161.75 | 160.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 160.45 | 161.49 | 160.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:30:00 | 160.60 | 161.49 | 160.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 159.50 | 161.09 | 160.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 160.98 | 161.09 | 160.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 161.71 | 160.94 | 160.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 10:15:00 | 161.20 | 160.94 | 160.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 12:15:00 | 158.38 | 160.00 | 159.85 | SL hit (close<static) qty=1.00 sl=159.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 12:15:00 | 158.38 | 160.00 | 159.85 | SL hit (close<static) qty=1.00 sl=159.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 12:15:00 | 158.38 | 160.00 | 159.85 | SL hit (close<static) qty=1.00 sl=159.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 157.74 | 159.55 | 159.66 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 10:15:00 | 160.37 | 159.76 | 159.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 12:15:00 | 162.35 | 160.45 | 160.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 12:15:00 | 161.71 | 162.16 | 161.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 12:15:00 | 161.71 | 162.16 | 161.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 161.71 | 162.16 | 161.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:45:00 | 161.51 | 162.16 | 161.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 161.40 | 162.01 | 161.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:00:00 | 161.40 | 162.01 | 161.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 163.02 | 162.21 | 161.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:30:00 | 161.80 | 162.21 | 161.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 160.12 | 161.81 | 161.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 160.12 | 161.81 | 161.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 162.78 | 162.00 | 161.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 13:00:00 | 166.34 | 162.98 | 162.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 169.87 | 163.84 | 162.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 13:15:00 | 163.40 | 167.03 | 167.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 13:15:00 | 163.40 | 167.03 | 167.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2026-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 13:15:00 | 163.40 | 167.03 | 167.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 14:15:00 | 161.64 | 165.95 | 166.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 160.10 | 159.08 | 161.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:00:00 | 160.10 | 159.08 | 161.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 161.62 | 159.58 | 161.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 161.62 | 159.58 | 161.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 160.55 | 159.78 | 161.25 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 164.74 | 162.05 | 161.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 170.78 | 165.94 | 164.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 167.30 | 168.07 | 166.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 15:15:00 | 167.30 | 168.07 | 166.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 167.30 | 168.07 | 166.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 166.21 | 168.07 | 166.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 166.05 | 167.66 | 166.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 165.85 | 167.66 | 166.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 162.57 | 166.65 | 165.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:45:00 | 161.85 | 166.65 | 165.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 163.05 | 165.93 | 165.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 13:00:00 | 164.35 | 165.61 | 165.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 13:15:00 | 162.85 | 165.06 | 165.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 13:15:00 | 162.85 | 165.06 | 165.30 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 169.20 | 165.89 | 165.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 170.47 | 167.35 | 166.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 167.53 | 174.69 | 173.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 167.53 | 174.69 | 173.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 167.53 | 174.69 | 173.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 167.53 | 174.69 | 173.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 162.45 | 172.24 | 172.61 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 171.51 | 167.92 | 167.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 172.95 | 169.59 | 168.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 170.19 | 170.24 | 169.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 09:30:00 | 170.14 | 170.24 | 169.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 168.74 | 169.94 | 169.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 168.74 | 169.94 | 169.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 169.17 | 169.78 | 169.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:15:00 | 168.66 | 169.78 | 169.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 168.00 | 169.43 | 169.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:00:00 | 168.00 | 169.43 | 169.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 168.60 | 169.26 | 168.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 14:30:00 | 169.89 | 169.24 | 168.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 167.30 | 168.71 | 168.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 167.30 | 168.71 | 168.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 11:15:00 | 165.88 | 167.80 | 168.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 14:15:00 | 170.10 | 167.81 | 168.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 14:15:00 | 170.10 | 167.81 | 168.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 170.10 | 167.81 | 168.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:00:00 | 170.10 | 167.81 | 168.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 15:15:00 | 171.00 | 168.45 | 168.41 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 11:15:00 | 167.75 | 168.88 | 168.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-09 12:15:00 | 166.80 | 168.47 | 168.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 14:15:00 | 168.70 | 166.37 | 167.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 14:15:00 | 168.70 | 166.37 | 167.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 168.70 | 166.37 | 167.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 168.70 | 166.37 | 167.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 169.45 | 166.98 | 167.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 172.49 | 166.98 | 167.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 09:15:00 | 175.99 | 168.78 | 168.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 178.17 | 170.66 | 169.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 175.89 | 176.97 | 174.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-13 10:00:00 | 175.89 | 176.97 | 174.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 172.79 | 176.14 | 174.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:00:00 | 172.79 | 176.14 | 174.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 171.20 | 175.15 | 173.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:30:00 | 171.74 | 175.15 | 173.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 172.40 | 174.28 | 173.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:45:00 | 172.40 | 174.28 | 173.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 172.00 | 173.69 | 173.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:15:00 | 165.95 | 173.69 | 173.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 166.83 | 172.31 | 172.94 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 171.70 | 170.23 | 170.04 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 14:15:00 | 168.93 | 170.36 | 170.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 10:15:00 | 168.30 | 169.64 | 170.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 164.48 | 163.33 | 164.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 15:00:00 | 164.48 | 163.33 | 164.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 164.50 | 163.56 | 164.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 164.07 | 163.56 | 164.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 164.65 | 163.78 | 164.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:15:00 | 163.05 | 163.83 | 164.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 163.50 | 163.77 | 164.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:15:00 | 163.75 | 163.88 | 164.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:45:00 | 163.55 | 163.97 | 164.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 162.96 | 163.77 | 164.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 12:30:00 | 162.75 | 163.60 | 164.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 13:30:00 | 162.50 | 163.50 | 163.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 159.49 | 163.46 | 163.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 155.32 | 162.56 | 163.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 155.56 | 162.56 | 163.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 155.37 | 162.56 | 163.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 14:15:00 | 160.40 | 159.44 | 161.25 | SL hit (close>ema200) qty=0.50 sl=159.44 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 14:15:00 | 160.40 | 159.44 | 161.25 | SL hit (close>ema200) qty=0.50 sl=159.44 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 14:15:00 | 160.40 | 159.44 | 161.25 | SL hit (close>ema200) qty=0.50 sl=159.44 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 13:15:00 | 154.90 | 156.68 | 157.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 14:15:00 | 154.61 | 156.04 | 157.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 14:15:00 | 154.38 | 156.04 | 157.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 146.75 | 153.91 | 156.04 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 146.47 | 153.91 | 156.04 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 146.25 | 153.91 | 156.04 | Target hit (10%) qty=0.50 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 151.52 | 153.91 | 156.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 154.85 | 150.69 | 152.76 | SL hit (close>ema200) qty=0.50 sl=150.69 alert=retest2 |

### Cycle 24 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 157.30 | 154.06 | 153.88 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 154.02 | 154.77 | 154.86 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 155.95 | 154.93 | 154.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 14:15:00 | 157.91 | 155.53 | 155.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 154.48 | 155.79 | 155.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 154.48 | 155.79 | 155.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 154.48 | 155.79 | 155.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:45:00 | 154.34 | 155.79 | 155.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 154.52 | 155.54 | 155.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:30:00 | 155.18 | 155.35 | 155.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 153.75 | 155.03 | 155.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 153.75 | 155.03 | 155.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 152.93 | 154.61 | 154.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 14:15:00 | 156.30 | 154.95 | 155.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 14:15:00 | 156.30 | 154.95 | 155.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 156.30 | 154.95 | 155.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 156.30 | 154.95 | 155.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 15:15:00 | 155.88 | 155.13 | 155.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-16 12:15:00 | 156.51 | 155.69 | 155.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 160.15 | 161.65 | 160.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 160.15 | 161.65 | 160.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 160.15 | 161.65 | 160.20 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 158.75 | 159.59 | 159.65 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 161.39 | 159.93 | 159.73 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 156.10 | 159.16 | 159.40 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-23 15:15:00 | 161.91 | 159.63 | 159.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 11:15:00 | 164.11 | 161.43 | 160.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-24 14:15:00 | 162.27 | 162.45 | 161.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-24 14:30:00 | 162.85 | 162.45 | 161.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 160.58 | 162.07 | 161.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 09:45:00 | 162.48 | 162.47 | 161.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 163.08 | 162.78 | 162.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 10:15:00 | 162.61 | 162.64 | 162.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 153.94 | 160.96 | 161.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 153.94 | 160.96 | 161.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 153.94 | 160.96 | 161.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 153.94 | 160.96 | 161.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 152.81 | 159.33 | 160.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 156.30 | 154.24 | 157.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 156.30 | 154.24 | 157.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 156.30 | 154.24 | 157.12 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 161.44 | 158.36 | 158.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 163.90 | 161.13 | 159.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 13:15:00 | 167.75 | 167.91 | 165.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:00:00 | 167.75 | 167.91 | 165.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 194.14 | 192.73 | 187.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:00:00 | 198.75 | 194.25 | 188.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 197.63 | 194.27 | 189.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 204.00 | 194.43 | 190.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 12:15:00 | 198.02 | 201.90 | 201.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 13:15:00 | 201.56 | 201.87 | 201.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-17 15:15:00 | 199.37 | 200.76 | 200.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 15:15:00 | 199.37 | 200.76 | 200.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 15:15:00 | 199.37 | 200.76 | 200.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 15:15:00 | 199.37 | 200.76 | 200.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 15:15:00 | 199.37 | 200.76 | 200.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 14:15:00 | 197.70 | 199.49 | 200.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 205.45 | 200.35 | 200.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 205.45 | 200.35 | 200.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 205.45 | 200.35 | 200.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 205.45 | 200.35 | 200.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 210.29 | 202.34 | 201.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 11:15:00 | 214.46 | 204.76 | 202.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 14:15:00 | 217.87 | 218.68 | 215.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-23 15:00:00 | 217.87 | 218.68 | 215.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 217.30 | 218.42 | 216.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:45:00 | 215.98 | 218.42 | 216.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 216.80 | 218.10 | 216.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:30:00 | 216.50 | 218.10 | 216.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 215.73 | 217.62 | 216.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 13:30:00 | 215.51 | 217.62 | 216.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 218.35 | 217.77 | 216.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 219.79 | 217.80 | 216.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 09:15:00 | 220.23 | 217.02 | 216.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 11:30:00 | 219.64 | 217.99 | 217.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 215.10 | 217.42 | 217.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 215.10 | 217.42 | 217.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 215.10 | 217.42 | 217.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 215.10 | 217.42 | 217.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 213.40 | 216.62 | 217.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 214.92 | 214.50 | 215.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 214.92 | 214.50 | 215.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 214.92 | 214.50 | 215.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 215.87 | 214.50 | 215.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 215.20 | 214.64 | 215.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 216.21 | 214.64 | 215.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 217.09 | 215.13 | 215.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 217.92 | 215.13 | 215.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 217.83 | 215.67 | 215.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 217.99 | 215.67 | 215.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 221.80 | 216.90 | 216.40 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 212.01 | 217.38 | 218.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 09:15:00 | 205.17 | 212.49 | 215.36 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-11-24 11:15:00 | 156.59 | 2025-11-26 09:15:00 | 163.42 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2025-11-25 10:15:00 | 157.15 | 2025-11-26 09:15:00 | 163.42 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2025-11-25 13:15:00 | 157.21 | 2025-11-26 09:15:00 | 163.42 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2025-12-03 10:30:00 | 155.23 | 2025-12-04 13:15:00 | 147.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 10:30:00 | 155.23 | 2025-12-05 13:15:00 | 152.01 | STOP_HIT | 0.50 | 2.07% |
| SELL | retest2 | 2025-12-11 15:00:00 | 145.59 | 2025-12-19 10:15:00 | 146.13 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-12-12 09:30:00 | 144.70 | 2025-12-19 10:15:00 | 146.13 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-12-16 09:15:00 | 144.60 | 2025-12-19 10:15:00 | 146.13 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-12-16 10:00:00 | 145.14 | 2025-12-19 10:15:00 | 146.13 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-12-18 11:30:00 | 144.22 | 2025-12-19 10:15:00 | 146.13 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-12-18 12:00:00 | 143.78 | 2025-12-19 10:15:00 | 146.13 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-12-19 09:30:00 | 144.23 | 2025-12-19 10:15:00 | 146.13 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-01-06 11:15:00 | 155.40 | 2026-01-07 09:15:00 | 158.27 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-01-06 14:00:00 | 155.18 | 2026-01-07 09:15:00 | 158.27 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-01-09 09:15:00 | 160.98 | 2026-01-09 12:15:00 | 158.38 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-01-09 09:45:00 | 161.71 | 2026-01-09 12:15:00 | 158.38 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-01-09 10:15:00 | 161.20 | 2026-01-09 12:15:00 | 158.38 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-01-14 13:00:00 | 166.34 | 2026-01-19 13:15:00 | 163.40 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2026-01-16 09:15:00 | 169.87 | 2026-01-19 13:15:00 | 163.40 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2026-01-27 13:00:00 | 164.35 | 2026-01-27 13:15:00 | 162.85 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-02-04 14:30:00 | 169.89 | 2026-02-05 09:15:00 | 167.30 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-02-26 11:15:00 | 163.05 | 2026-03-02 09:15:00 | 155.32 | PARTIAL | 0.50 | 4.74% |
| SELL | retest2 | 2026-02-27 09:15:00 | 163.50 | 2026-03-02 09:15:00 | 155.56 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2026-02-27 10:15:00 | 163.75 | 2026-03-02 09:15:00 | 155.37 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2026-02-26 11:15:00 | 163.05 | 2026-03-02 14:15:00 | 160.40 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2026-02-27 09:15:00 | 163.50 | 2026-03-02 14:15:00 | 160.40 | STOP_HIT | 0.50 | 1.90% |
| SELL | retest2 | 2026-02-27 10:15:00 | 163.75 | 2026-03-02 14:15:00 | 160.40 | STOP_HIT | 0.50 | 2.05% |
| SELL | retest2 | 2026-02-27 10:45:00 | 163.55 | 2026-03-06 13:15:00 | 154.90 | PARTIAL | 0.50 | 5.29% |
| SELL | retest2 | 2026-02-27 12:30:00 | 162.75 | 2026-03-06 14:15:00 | 154.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 13:30:00 | 162.50 | 2026-03-06 14:15:00 | 154.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 10:45:00 | 163.55 | 2026-03-09 09:15:00 | 146.75 | TARGET_HIT | 0.50 | 10.28% |
| SELL | retest2 | 2026-02-27 12:30:00 | 162.75 | 2026-03-09 09:15:00 | 146.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-27 13:30:00 | 162.50 | 2026-03-09 09:15:00 | 146.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 159.49 | 2026-03-09 09:15:00 | 151.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 159.49 | 2026-03-10 09:15:00 | 154.85 | STOP_HIT | 0.50 | 2.91% |
| BUY | retest2 | 2026-03-13 11:30:00 | 155.18 | 2026-03-13 12:15:00 | 153.75 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-03-25 09:45:00 | 162.48 | 2026-03-30 09:15:00 | 153.94 | STOP_HIT | 1.00 | -5.26% |
| BUY | retest2 | 2026-03-27 09:15:00 | 163.08 | 2026-03-30 09:15:00 | 153.94 | STOP_HIT | 1.00 | -5.60% |
| BUY | retest2 | 2026-03-27 10:15:00 | 162.61 | 2026-03-30 09:15:00 | 153.94 | STOP_HIT | 1.00 | -5.33% |
| BUY | retest2 | 2026-04-13 12:00:00 | 198.75 | 2026-04-17 15:15:00 | 199.37 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2026-04-13 12:30:00 | 197.63 | 2026-04-17 15:15:00 | 199.37 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2026-04-15 09:15:00 | 204.00 | 2026-04-17 15:15:00 | 199.37 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2026-04-17 12:15:00 | 198.02 | 2026-04-17 15:15:00 | 199.37 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2026-04-27 09:15:00 | 219.79 | 2026-04-29 13:15:00 | 215.10 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2026-04-28 09:15:00 | 220.23 | 2026-04-29 13:15:00 | 215.10 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-04-28 11:30:00 | 219.64 | 2026-04-29 13:15:00 | 215.10 | STOP_HIT | 1.00 | -2.07% |
