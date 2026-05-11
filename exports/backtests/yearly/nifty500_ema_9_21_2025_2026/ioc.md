# Indian Oil Corporation Ltd. (IOC)

## Backtest Summary

- **Window:** 2026-01-19 09:15:00 → 2026-05-08 15:15:00 (518 bars)
- **Last close:** 144.88
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 17 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT2_SKIP | 7 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 15 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 9
- **Target hits / Stop hits / Partials:** 0 / 18 / 5
- **Avg / median % per leg:** 2.12% / 0.64%
- **Sum % (uncompounded):** 48.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 0 | 9 | 0 | -0.78% | -7.0% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.68% | -8.0% |
| BUY @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 0 | 6 | 0 | 0.17% | 1.0% |
| SELL (all) | 14 | 10 | 71.4% | 0 | 9 | 5 | 3.99% | 55.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 10 | 71.4% | 0 | 9 | 5 | 3.99% | 55.8% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.68% | -8.0% |
| retest2 (combined) | 20 | 14 | 70.0% | 0 | 15 | 5 | 2.84% | 56.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 161.76 | 159.00 | 158.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 163.90 | 159.98 | 159.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 162.84 | 162.90 | 161.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 162.84 | 162.90 | 161.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 162.84 | 162.90 | 161.89 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 160.18 | 161.82 | 161.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 159.90 | 161.44 | 161.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 161.33 | 160.79 | 161.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 161.33 | 160.79 | 161.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 161.33 | 160.79 | 161.25 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 164.42 | 161.92 | 161.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 164.88 | 162.51 | 161.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 11:15:00 | 174.58 | 174.63 | 172.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 12:00:00 | 174.58 | 174.63 | 172.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 175.61 | 178.81 | 178.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 175.61 | 178.81 | 178.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 177.40 | 178.53 | 178.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 177.50 | 178.53 | 178.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 177.57 | 178.34 | 178.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 176.69 | 177.65 | 177.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 176.69 | 177.65 | 177.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 173.91 | 176.48 | 177.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 175.18 | 175.15 | 176.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 175.18 | 175.15 | 176.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 175.55 | 174.47 | 175.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 175.55 | 174.47 | 175.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 175.75 | 174.73 | 175.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 176.74 | 174.73 | 175.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 177.43 | 175.64 | 175.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 177.80 | 176.61 | 176.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 175.72 | 177.02 | 176.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 175.72 | 177.02 | 176.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 175.72 | 177.02 | 176.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 175.72 | 177.02 | 176.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 176.79 | 176.97 | 176.47 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 174.03 | 176.09 | 176.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 173.84 | 175.64 | 175.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 174.24 | 174.14 | 174.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 174.24 | 174.14 | 174.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 175.60 | 174.27 | 174.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:15:00 | 176.00 | 174.27 | 174.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 176.54 | 174.72 | 174.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 176.54 | 174.72 | 174.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 175.79 | 175.19 | 175.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 176.40 | 175.58 | 175.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 179.81 | 185.41 | 184.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 179.81 | 185.41 | 184.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 179.81 | 185.41 | 184.32 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 179.36 | 183.35 | 183.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 178.24 | 182.33 | 183.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 174.65 | 173.36 | 176.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 174.65 | 173.36 | 176.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 161.73 | 162.36 | 165.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 14:00:00 | 160.05 | 161.43 | 164.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:45:00 | 160.45 | 160.81 | 163.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 15:00:00 | 160.33 | 160.88 | 162.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:30:00 | 160.38 | 160.55 | 161.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 152.05 | 155.91 | 158.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 152.43 | 155.91 | 158.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 152.31 | 155.91 | 158.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 152.36 | 155.91 | 158.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 147.39 | 147.37 | 149.86 | SL hit (close>ema200) qty=0.50 sl=147.37 alert=retest2 |

### Cycle 9 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 142.12 | 135.26 | 134.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 11:15:00 | 144.18 | 138.18 | 135.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 141.15 | 141.21 | 138.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 143.04 | 141.84 | 140.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 13:30:00 | 143.07 | 142.45 | 141.06 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 15:00:00 | 142.96 | 142.56 | 141.23 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 139.19 | 141.95 | 141.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 139.19 | 141.95 | 141.19 | SL hit (close<ema400) qty=1.00 sl=141.19 alert=retest1 |

### Cycle 10 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 140.14 | 140.80 | 140.81 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 14:15:00 | 140.96 | 140.83 | 140.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 145.08 | 141.74 | 141.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 12:15:00 | 144.35 | 144.50 | 143.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 13:00:00 | 144.35 | 144.50 | 143.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 144.28 | 144.43 | 143.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 144.64 | 144.41 | 143.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:45:00 | 144.56 | 144.57 | 143.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 13:15:00 | 145.56 | 146.56 | 146.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 145.56 | 146.56 | 146.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 145.41 | 146.33 | 146.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 145.39 | 144.30 | 145.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 145.39 | 144.30 | 145.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 145.39 | 144.30 | 145.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 146.04 | 144.30 | 145.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 145.73 | 144.59 | 145.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 145.69 | 144.59 | 145.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 146.22 | 145.40 | 145.35 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 144.78 | 145.31 | 145.37 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 146.37 | 145.58 | 145.48 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 144.45 | 145.42 | 145.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 142.35 | 144.81 | 145.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 143.12 | 142.90 | 143.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 143.12 | 142.90 | 143.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 143.12 | 142.90 | 143.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 142.17 | 143.00 | 143.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:30:00 | 142.35 | 142.76 | 143.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 141.20 | 142.76 | 143.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 14:30:00 | 142.37 | 142.13 | 142.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 144.08 | 142.56 | 142.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 144.14 | 143.08 | 143.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 144.14 | 143.08 | 143.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 146.75 | 143.93 | 143.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 145.23 | 146.26 | 145.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 145.23 | 146.26 | 145.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 145.23 | 146.26 | 145.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 145.05 | 146.26 | 145.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 145.48 | 146.10 | 145.43 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-02-12 13:15:00 | 177.50 | 2026-02-13 10:15:00 | 176.69 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-02-12 14:45:00 | 177.57 | 2026-02-13 10:15:00 | 176.69 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-03-10 14:00:00 | 160.05 | 2026-03-16 09:15:00 | 152.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:45:00 | 160.45 | 2026-03-16 09:15:00 | 152.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 15:00:00 | 160.33 | 2026-03-16 09:15:00 | 152.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:30:00 | 160.38 | 2026-03-16 09:15:00 | 152.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 14:00:00 | 160.05 | 2026-03-18 10:15:00 | 147.39 | STOP_HIT | 0.50 | 7.91% |
| SELL | retest2 | 2026-03-11 10:45:00 | 160.45 | 2026-03-18 10:15:00 | 147.39 | STOP_HIT | 0.50 | 8.14% |
| SELL | retest2 | 2026-03-11 15:00:00 | 160.33 | 2026-03-18 10:15:00 | 147.39 | STOP_HIT | 0.50 | 8.07% |
| SELL | retest2 | 2026-03-12 14:30:00 | 160.38 | 2026-03-18 10:15:00 | 147.39 | STOP_HIT | 0.50 | 8.10% |
| SELL | retest2 | 2026-03-23 09:15:00 | 140.83 | 2026-04-02 09:15:00 | 133.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 140.83 | 2026-04-02 14:15:00 | 134.38 | STOP_HIT | 0.50 | 4.58% |
| BUY | retest1 | 2026-04-10 09:30:00 | 143.04 | 2026-04-13 09:15:00 | 139.19 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest1 | 2026-04-10 13:30:00 | 143.07 | 2026-04-13 09:15:00 | 139.19 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest1 | 2026-04-10 15:00:00 | 142.96 | 2026-04-13 09:15:00 | 139.19 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2026-04-13 10:15:00 | 139.72 | 2026-04-13 13:15:00 | 140.14 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2026-04-13 10:45:00 | 139.70 | 2026-04-13 13:15:00 | 140.14 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2026-04-17 09:15:00 | 144.64 | 2026-04-23 13:15:00 | 145.56 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2026-04-17 09:45:00 | 144.56 | 2026-04-23 13:15:00 | 145.56 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2026-05-04 13:15:00 | 142.17 | 2026-05-06 11:15:00 | 144.14 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-05-04 14:30:00 | 142.35 | 2026-05-06 11:15:00 | 144.14 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-05-05 09:15:00 | 141.20 | 2026-05-06 11:15:00 | 144.14 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-05-05 14:30:00 | 142.37 | 2026-05-06 11:15:00 | 144.14 | STOP_HIT | 1.00 | -1.24% |
