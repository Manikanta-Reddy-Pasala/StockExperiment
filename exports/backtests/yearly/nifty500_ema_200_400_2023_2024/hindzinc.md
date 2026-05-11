# Hindustan Zinc Ltd. (HINDZINC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 634.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 3 |
| ALERT3 | 80 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 63 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 68 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 60
- **Target hits / Stop hits / Partials:** 0 / 68 / 8
- **Avg / median % per leg:** -0.67% / -1.33%
- **Sum % (uncompounded):** -50.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 0 | 0.0% | 0 | 44 | 0 | -1.88% | -82.8% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.04% | -3.0% |
| BUY @ 3rd Alert (retest2) | 43 | 0 | 0.0% | 0 | 43 | 0 | -1.86% | -79.8% |
| SELL (all) | 32 | 16 | 50.0% | 0 | 24 | 8 | 1.00% | 32.1% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.65% | -6.6% |
| SELL @ 3rd Alert (retest2) | 28 | 16 | 57.1% | 0 | 20 | 8 | 1.38% | 38.7% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.93% | -9.6% |
| retest2 (combined) | 71 | 16 | 22.5% | 0 | 63 | 8 | -0.58% | -41.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 09:15:00 | 331.30 | 312.77 | 312.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 09:15:00 | 333.40 | 315.27 | 314.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 13:15:00 | 317.20 | 318.38 | 316.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-21 14:00:00 | 317.20 | 318.38 | 316.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 14:15:00 | 318.10 | 318.38 | 316.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 14:30:00 | 316.00 | 318.38 | 316.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 313.40 | 318.33 | 316.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:30:00 | 313.85 | 318.33 | 316.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 10:15:00 | 313.50 | 318.28 | 316.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 10:45:00 | 313.45 | 318.28 | 316.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 14:15:00 | 316.65 | 318.18 | 316.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 15:00:00 | 316.65 | 318.18 | 316.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 314.25 | 318.14 | 316.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:45:00 | 314.00 | 318.14 | 316.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 314.00 | 318.10 | 316.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 11:00:00 | 314.00 | 318.10 | 316.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 318.35 | 318.12 | 316.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:45:00 | 316.30 | 318.12 | 316.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 09:15:00 | 318.10 | 318.99 | 316.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-03 09:30:00 | 317.55 | 318.99 | 316.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 10:15:00 | 318.70 | 318.99 | 316.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-03 10:30:00 | 317.40 | 318.99 | 316.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 15:15:00 | 318.10 | 318.95 | 317.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 09:15:00 | 319.95 | 318.95 | 317.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-07 15:15:00 | 316.55 | 318.85 | 317.09 | SL hit (close<static) qty=1.00 sl=316.75 alert=retest2 |

### Cycle 2 — SELL (started 2023-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 14:15:00 | 311.35 | 317.19 | 317.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 09:15:00 | 309.35 | 316.65 | 316.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-09 10:15:00 | 313.10 | 313.02 | 314.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-09 11:00:00 | 313.10 | 313.02 | 314.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 315.80 | 313.01 | 314.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:30:00 | 315.80 | 313.01 | 314.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 315.80 | 313.03 | 314.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 13:15:00 | 314.75 | 313.36 | 314.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 13:45:00 | 315.00 | 313.38 | 314.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 14:15:00 | 315.10 | 313.38 | 314.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 15:15:00 | 317.95 | 313.46 | 314.92 | SL hit (close>static) qty=1.00 sl=317.10 alert=retest2 |

### Cycle 3 — BUY (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 10:15:00 | 324.55 | 308.84 | 308.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 15:15:00 | 326.40 | 309.62 | 309.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 312.05 | 312.62 | 310.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-19 10:00:00 | 312.05 | 312.62 | 310.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 308.45 | 312.56 | 310.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:00:00 | 308.45 | 312.56 | 310.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 306.70 | 312.50 | 310.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 15:00:00 | 306.70 | 312.50 | 310.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 309.95 | 311.99 | 310.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:00:00 | 309.95 | 311.99 | 310.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 11:15:00 | 309.80 | 311.97 | 310.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:30:00 | 310.00 | 311.97 | 310.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 12:15:00 | 309.85 | 311.95 | 310.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-22 12:30:00 | 309.00 | 311.95 | 310.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 312.25 | 311.87 | 310.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-26 10:15:00 | 312.50 | 311.87 | 310.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-27 09:30:00 | 312.60 | 311.88 | 310.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-27 11:30:00 | 312.55 | 311.89 | 310.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 09:15:00 | 313.40 | 311.87 | 310.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 312.40 | 314.18 | 312.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 15:00:00 | 318.65 | 314.15 | 312.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 309.85 | 315.38 | 313.44 | SL hit (close<static) qty=1.00 sl=310.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 09:15:00 | 310.55 | 313.30 | 313.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 09:15:00 | 308.80 | 313.00 | 313.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-02 11:15:00 | 312.45 | 312.00 | 312.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 11:15:00 | 312.45 | 312.00 | 312.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 312.45 | 312.00 | 312.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 11:45:00 | 312.15 | 312.00 | 312.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 12:15:00 | 313.00 | 312.01 | 312.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:15:00 | 313.30 | 312.01 | 312.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 312.30 | 312.02 | 312.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 10:45:00 | 311.20 | 312.01 | 312.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 09:30:00 | 310.80 | 312.00 | 312.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 09:30:00 | 310.50 | 311.53 | 312.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 09:15:00 | 310.55 | 311.51 | 312.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 310.80 | 311.50 | 312.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 11:00:00 | 309.75 | 311.48 | 312.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 09:15:00 | 295.64 | 309.93 | 311.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 09:15:00 | 295.26 | 309.93 | 311.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 09:15:00 | 295.02 | 309.93 | 311.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 11:15:00 | 294.97 | 309.62 | 311.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 11:15:00 | 294.26 | 309.62 | 311.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-01 12:15:00 | 303.90 | 302.68 | 306.68 | SL hit (close>ema200) qty=0.50 sl=302.68 alert=retest2 |

### Cycle 5 — BUY (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 10:15:00 | 345.20 | 310.15 | 310.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 12:15:00 | 348.20 | 310.88 | 310.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 14:15:00 | 658.00 | 658.82 | 607.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 15:00:00 | 658.00 | 658.82 | 607.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 621.00 | 654.47 | 612.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 621.90 | 654.47 | 612.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 611.40 | 650.62 | 614.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 611.40 | 650.62 | 614.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 610.00 | 650.22 | 614.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 619.50 | 650.22 | 614.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 13:15:00 | 607.00 | 648.50 | 614.08 | SL hit (close<static) qty=1.00 sl=608.40 alert=retest2 |

### Cycle 6 — SELL (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 09:15:00 | 513.40 | 603.08 | 603.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 497.15 | 574.15 | 587.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 11:15:00 | 516.45 | 514.37 | 539.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 12:00:00 | 516.45 | 514.37 | 539.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 529.60 | 512.54 | 527.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:15:00 | 528.55 | 512.54 | 527.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 528.60 | 512.70 | 527.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:30:00 | 535.00 | 512.70 | 527.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 525.15 | 512.82 | 527.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:15:00 | 523.55 | 513.08 | 527.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 09:15:00 | 534.70 | 513.49 | 527.12 | SL hit (close>static) qty=1.00 sl=530.25 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 451.20 | 438.84 | 438.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 455.35 | 439.00 | 438.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 459.90 | 476.96 | 461.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 459.90 | 476.96 | 461.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 459.90 | 476.96 | 461.60 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 422.55 | 453.45 | 453.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 419.70 | 442.02 | 446.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 431.70 | 431.35 | 438.01 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 12:15:00 | 430.80 | 431.37 | 437.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 14:30:00 | 431.10 | 431.38 | 437.86 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 15:15:00 | 430.80 | 431.38 | 437.86 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 430.85 | 431.30 | 437.31 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 438.00 | 430.31 | 436.20 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 438.00 | 430.31 | 436.20 | SL hit (close>ema400) qty=1.00 sl=436.20 alert=retest1 |

### Cycle 9 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 459.90 | 439.28 | 439.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 09:15:00 | 465.95 | 442.46 | 440.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 476.55 | 480.39 | 467.03 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:30:00 | 482.65 | 479.88 | 467.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 468.00 | 479.26 | 468.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 468.00 | 479.26 | 468.95 | SL hit (close<ema400) qty=1.00 sl=468.95 alert=retest1 |

### Cycle 10 — SELL (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 14:15:00 | 547.35 | 591.31 | 591.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 09:15:00 | 531.10 | 590.27 | 590.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 557.60 | 548.24 | 565.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 557.60 | 548.24 | 565.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 557.60 | 548.24 | 565.06 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 621.05 | 573.70 | 573.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 630.75 | 584.55 | 579.52 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-07 09:15:00 | 319.95 | 2023-08-07 15:15:00 | 316.55 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2023-08-08 12:00:00 | 319.90 | 2023-08-11 09:15:00 | 316.55 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2023-08-09 12:00:00 | 319.20 | 2023-08-11 09:15:00 | 316.55 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-08-09 13:00:00 | 320.20 | 2023-08-11 09:15:00 | 316.55 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-08-30 09:15:00 | 316.85 | 2023-09-13 09:15:00 | 313.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2023-09-12 10:30:00 | 317.15 | 2023-09-13 09:15:00 | 313.30 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2023-09-13 09:15:00 | 316.45 | 2023-09-13 09:15:00 | 313.30 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-09-14 09:15:00 | 318.05 | 2023-09-20 10:15:00 | 316.15 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-09-14 11:15:00 | 320.85 | 2023-09-20 10:15:00 | 316.15 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2023-09-18 11:00:00 | 319.25 | 2023-09-20 10:15:00 | 316.15 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2023-09-18 12:00:00 | 319.05 | 2023-09-21 09:15:00 | 313.10 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2023-10-11 13:15:00 | 314.75 | 2023-10-11 15:15:00 | 317.95 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2023-10-11 13:45:00 | 315.00 | 2023-10-11 15:15:00 | 317.95 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2023-10-11 14:15:00 | 315.10 | 2023-10-11 15:15:00 | 317.95 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2023-10-18 15:00:00 | 314.10 | 2023-10-25 12:15:00 | 298.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 11:45:00 | 312.60 | 2023-10-25 12:15:00 | 296.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 15:00:00 | 314.10 | 2023-11-12 18:15:00 | 304.50 | STOP_HIT | 0.50 | 3.06% |
| SELL | retest2 | 2023-10-20 11:45:00 | 312.60 | 2023-11-12 18:15:00 | 304.50 | STOP_HIT | 0.50 | 2.59% |
| BUY | retest2 | 2023-12-26 10:15:00 | 312.50 | 2024-01-18 09:15:00 | 309.85 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2023-12-27 09:30:00 | 312.60 | 2024-01-18 09:15:00 | 309.85 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2023-12-27 11:30:00 | 312.55 | 2024-01-18 09:15:00 | 309.85 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-12-28 09:15:00 | 313.40 | 2024-01-18 09:15:00 | 309.85 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-01-10 15:00:00 | 318.65 | 2024-01-18 09:15:00 | 309.85 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2024-01-19 09:15:00 | 315.55 | 2024-01-23 12:15:00 | 309.85 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-01-29 09:30:00 | 314.00 | 2024-02-08 14:15:00 | 313.35 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2024-01-29 11:00:00 | 314.35 | 2024-02-08 14:15:00 | 313.35 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-01-30 10:15:00 | 318.35 | 2024-02-08 14:15:00 | 313.35 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-01-30 12:45:00 | 318.50 | 2024-02-09 09:15:00 | 312.60 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-01-30 15:00:00 | 318.20 | 2024-02-09 09:15:00 | 312.60 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-02-01 09:15:00 | 319.65 | 2024-02-09 09:15:00 | 312.60 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-02-06 14:30:00 | 316.95 | 2024-02-09 09:15:00 | 312.60 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-02-07 09:15:00 | 318.00 | 2024-02-12 11:15:00 | 310.65 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2024-02-07 10:45:00 | 317.25 | 2024-02-12 11:15:00 | 310.65 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-03-04 10:45:00 | 311.20 | 2024-03-15 09:15:00 | 295.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 09:30:00 | 310.80 | 2024-03-15 09:15:00 | 295.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 09:30:00 | 310.50 | 2024-03-15 09:15:00 | 295.02 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2024-03-11 09:15:00 | 310.55 | 2024-03-15 11:15:00 | 294.97 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2024-03-11 11:00:00 | 309.75 | 2024-03-15 11:15:00 | 294.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-04 10:45:00 | 311.20 | 2024-04-01 12:15:00 | 303.90 | STOP_HIT | 0.50 | 2.35% |
| SELL | retest2 | 2024-03-05 09:30:00 | 310.80 | 2024-04-01 12:15:00 | 303.90 | STOP_HIT | 0.50 | 2.22% |
| SELL | retest2 | 2024-03-07 09:30:00 | 310.50 | 2024-04-01 12:15:00 | 303.90 | STOP_HIT | 0.50 | 2.13% |
| SELL | retest2 | 2024-03-11 09:15:00 | 310.55 | 2024-04-01 12:15:00 | 303.90 | STOP_HIT | 0.50 | 2.14% |
| SELL | retest2 | 2024-03-11 11:00:00 | 309.75 | 2024-04-01 12:15:00 | 303.90 | STOP_HIT | 0.50 | 1.89% |
| BUY | retest2 | 2024-07-26 09:15:00 | 619.50 | 2024-07-26 13:15:00 | 607.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-07-29 09:45:00 | 618.75 | 2024-08-06 15:15:00 | 606.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-07-29 11:00:00 | 619.40 | 2024-08-06 15:15:00 | 606.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-07-29 13:15:00 | 620.80 | 2024-08-06 15:15:00 | 606.00 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-10-23 14:15:00 | 523.55 | 2024-10-24 09:15:00 | 534.70 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-10-25 09:30:00 | 519.35 | 2024-10-29 09:15:00 | 538.95 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2024-11-06 09:15:00 | 518.70 | 2024-11-13 09:15:00 | 492.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-06 09:15:00 | 518.70 | 2024-12-03 14:15:00 | 506.85 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest1 | 2025-08-21 12:15:00 | 430.80 | 2025-09-01 09:15:00 | 438.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest1 | 2025-08-21 14:30:00 | 431.10 | 2025-09-01 09:15:00 | 438.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest1 | 2025-08-21 15:15:00 | 430.80 | 2025-09-01 09:15:00 | 438.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest1 | 2025-08-26 09:30:00 | 430.85 | 2025-09-01 09:15:00 | 438.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-09-01 13:15:00 | 435.30 | 2025-09-02 09:15:00 | 441.10 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-09-04 09:15:00 | 436.00 | 2025-09-04 09:15:00 | 440.40 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-09-05 10:45:00 | 435.95 | 2025-09-05 14:15:00 | 440.05 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-05 11:15:00 | 436.30 | 2025-09-05 14:15:00 | 440.05 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-09-09 12:15:00 | 431.80 | 2025-09-11 11:15:00 | 441.80 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-09-10 12:30:00 | 432.00 | 2025-09-11 11:15:00 | 441.80 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-09-10 13:00:00 | 431.10 | 2025-09-11 11:15:00 | 441.80 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest1 | 2025-10-29 09:30:00 | 482.65 | 2025-11-06 09:15:00 | 468.00 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-11-18 11:30:00 | 474.00 | 2025-11-21 09:15:00 | 465.25 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-11-18 12:15:00 | 475.05 | 2025-11-21 09:15:00 | 465.25 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-11-20 13:15:00 | 473.80 | 2025-11-21 09:15:00 | 465.25 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-11-20 14:00:00 | 473.95 | 2025-11-21 09:15:00 | 465.25 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2026-02-02 15:15:00 | 613.50 | 2026-02-13 14:15:00 | 593.20 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2026-02-03 11:30:00 | 613.45 | 2026-02-13 14:15:00 | 593.20 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2026-02-05 14:15:00 | 612.40 | 2026-02-16 09:15:00 | 588.50 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2026-02-05 14:45:00 | 612.20 | 2026-02-16 09:15:00 | 588.50 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2026-02-06 11:45:00 | 604.95 | 2026-02-16 09:15:00 | 588.50 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2026-02-06 14:30:00 | 605.65 | 2026-02-16 09:15:00 | 588.50 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-02-25 09:15:00 | 609.70 | 2026-03-04 10:15:00 | 590.15 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2026-03-02 09:15:00 | 617.05 | 2026-03-04 10:15:00 | 590.15 | STOP_HIT | 1.00 | -4.36% |
| BUY | retest2 | 2026-03-11 09:15:00 | 600.30 | 2026-03-11 12:15:00 | 589.35 | STOP_HIT | 1.00 | -1.82% |
