# Himadri Speciality Chemical Ltd. (HSCL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5317 bars)
- **Last close:** 631.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 5 |
| ALERT3 | 51 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 44 |
| PARTIAL | 14 |
| TARGET_HIT | 1 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 34
- **Target hits / Stop hits / Partials:** 1 / 43 / 14
- **Avg / median % per leg:** 0.08% / -0.32%
- **Sum % (uncompounded):** 4.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 0 | 0.0% | 0 | 13 | 0 | -2.66% | -34.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 0 | 0.0% | 0 | 13 | 0 | -2.66% | -34.6% |
| SELL (all) | 45 | 24 | 53.3% | 1 | 30 | 14 | 0.87% | 39.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 24 | 53.3% | 1 | 30 | 14 | 0.87% | 39.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 58 | 24 | 41.4% | 1 | 43 | 14 | 0.08% | 4.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 12:15:00 | 485.25 | 555.05 | 555.22 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 583.50 | 551.34 | 551.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 588.75 | 554.24 | 552.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 15:15:00 | 561.95 | 562.97 | 557.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:15:00 | 561.45 | 562.97 | 557.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 561.10 | 564.76 | 559.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 561.10 | 564.76 | 559.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 557.40 | 564.96 | 559.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:00:00 | 557.40 | 564.96 | 559.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 557.90 | 564.89 | 559.48 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 525.90 | 555.50 | 555.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 524.00 | 552.78 | 554.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 443.60 | 437.23 | 465.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 15:00:00 | 443.60 | 437.23 | 465.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 460.40 | 436.38 | 459.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 460.40 | 436.38 | 459.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 460.90 | 436.63 | 459.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 471.95 | 436.63 | 459.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 454.45 | 436.80 | 459.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:30:00 | 457.70 | 436.80 | 459.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 457.80 | 437.21 | 459.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:30:00 | 463.40 | 437.21 | 459.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 460.10 | 437.44 | 459.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:00:00 | 460.10 | 437.44 | 459.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 462.20 | 437.68 | 459.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 14:00:00 | 459.35 | 438.11 | 459.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:15:00 | 436.38 | 438.45 | 459.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-04 10:15:00 | 438.85 | 438.46 | 459.17 | SL hit (close>ema200) qty=0.50 sl=438.46 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 476.50 | 454.67 | 454.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 481.20 | 458.42 | 456.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 470.45 | 471.19 | 464.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 470.45 | 471.19 | 464.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 470.45 | 471.19 | 464.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:30:00 | 486.75 | 461.26 | 460.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:45:00 | 475.00 | 494.58 | 484.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 15:15:00 | 461.85 | 491.78 | 484.02 | SL hit (close<static) qty=1.00 sl=462.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 472.10 | 478.29 | 478.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 467.95 | 477.46 | 477.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 483.50 | 466.84 | 471.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 483.50 | 466.84 | 471.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 483.50 | 466.84 | 471.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 483.50 | 466.84 | 471.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 474.80 | 466.92 | 471.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:30:00 | 471.85 | 467.19 | 471.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 470.45 | 467.37 | 471.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:00:00 | 471.20 | 467.21 | 470.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:00:00 | 472.25 | 467.36 | 470.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 472.00 | 467.40 | 470.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 472.85 | 467.40 | 470.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 473.15 | 467.46 | 470.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 473.15 | 467.46 | 470.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 469.30 | 467.48 | 470.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:30:00 | 468.10 | 467.72 | 470.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 448.26 | 466.15 | 469.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 446.93 | 466.15 | 469.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 447.64 | 466.15 | 469.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 448.64 | 466.15 | 469.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 471.35 | 466.20 | 469.78 | SL hit (close>ema200) qty=0.50 sl=466.20 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 483.95 | 467.42 | 467.35 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 452.60 | 467.30 | 467.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 10:15:00 | 449.80 | 461.61 | 464.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 454.50 | 451.26 | 457.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 454.50 | 451.26 | 457.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 454.50 | 451.26 | 457.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 456.30 | 451.26 | 457.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 467.80 | 451.43 | 457.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 467.80 | 451.43 | 457.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 463.15 | 451.54 | 457.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:15:00 | 458.15 | 451.54 | 457.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 474.50 | 452.11 | 457.51 | SL hit (close>static) qty=1.00 sl=467.90 alert=retest2 |

### Cycle 8 — BUY (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 15:15:00 | 484.00 | 460.94 | 460.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 486.35 | 461.20 | 461.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 13:15:00 | 473.10 | 474.03 | 468.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 13:30:00 | 472.90 | 474.03 | 468.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 474.35 | 474.06 | 468.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 470.25 | 474.06 | 468.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 469.30 | 473.94 | 468.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 15:15:00 | 475.00 | 473.94 | 468.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 464.35 | 473.86 | 468.95 | SL hit (close<static) qty=1.00 sl=467.50 alert=retest2 |

### Cycle 9 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 450.70 | 466.05 | 466.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 12:15:00 | 449.70 | 465.89 | 466.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 15:15:00 | 470.50 | 464.52 | 465.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 15:15:00 | 470.50 | 464.52 | 465.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 470.50 | 464.52 | 465.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 452.50 | 464.10 | 465.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 12:00:00 | 453.15 | 462.72 | 464.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:00:00 | 454.50 | 462.45 | 464.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 454.50 | 462.38 | 463.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 461.70 | 461.90 | 463.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 15:15:00 | 453.20 | 461.90 | 463.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 13:45:00 | 459.80 | 461.53 | 463.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 15:15:00 | 459.90 | 461.53 | 463.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 14:15:00 | 473.00 | 459.89 | 462.27 | SL hit (close>static) qty=1.00 sl=464.50 alert=retest2 |

### Cycle 10 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 486.65 | 464.37 | 464.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 488.55 | 464.83 | 464.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 469.65 | 470.29 | 467.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 469.65 | 470.29 | 467.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 469.65 | 470.29 | 467.56 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 435.50 | 464.96 | 465.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 426.50 | 457.79 | 461.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 13:15:00 | 457.20 | 455.47 | 459.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 13:30:00 | 457.00 | 455.47 | 459.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 461.15 | 452.62 | 457.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 461.00 | 452.62 | 457.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 460.40 | 452.69 | 457.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 460.75 | 452.69 | 457.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 454.20 | 453.25 | 457.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 455.35 | 453.25 | 457.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 452.50 | 453.24 | 457.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 452.50 | 453.24 | 457.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 458.60 | 451.93 | 456.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:15:00 | 454.95 | 451.93 | 456.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 457.60 | 451.99 | 456.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:45:00 | 451.20 | 452.06 | 456.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 450.90 | 452.06 | 456.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 428.64 | 451.74 | 456.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 428.35 | 451.74 | 456.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 451.95 | 450.59 | 455.59 | SL hit (close>ema200) qty=0.50 sl=450.59 alert=retest2 |

### Cycle 12 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 494.15 | 459.05 | 459.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 505.90 | 463.20 | 461.19 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-14 09:15:00 | 345.00 | 2024-05-28 09:15:00 | 340.35 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-05-14 11:00:00 | 344.00 | 2024-05-28 09:15:00 | 340.35 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-05-29 09:15:00 | 344.20 | 2024-05-30 09:15:00 | 336.45 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-11-06 09:15:00 | 589.80 | 2024-11-08 11:15:00 | 561.30 | STOP_HIT | 1.00 | -4.83% |
| BUY | retest2 | 2024-11-06 14:15:00 | 591.70 | 2024-11-08 11:15:00 | 561.30 | STOP_HIT | 1.00 | -5.14% |
| SELL | retest2 | 2025-04-03 14:00:00 | 459.35 | 2025-04-04 09:15:00 | 436.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 14:00:00 | 459.35 | 2025-04-04 10:15:00 | 438.85 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2025-04-25 09:30:00 | 452.90 | 2025-04-30 15:15:00 | 436.05 | PARTIAL | 0.50 | 3.72% |
| SELL | retest2 | 2025-04-29 11:45:00 | 459.00 | 2025-05-02 14:15:00 | 431.63 | PARTIAL | 0.50 | 5.96% |
| SELL | retest2 | 2025-04-30 09:15:00 | 454.35 | 2025-05-02 15:15:00 | 430.25 | PARTIAL | 0.50 | 5.30% |
| SELL | retest2 | 2025-04-30 11:45:00 | 448.00 | 2025-05-06 12:15:00 | 425.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:30:00 | 452.90 | 2025-05-09 09:15:00 | 413.10 | TARGET_HIT | 0.50 | 8.79% |
| SELL | retest2 | 2025-04-29 11:45:00 | 459.00 | 2025-05-12 14:15:00 | 440.80 | STOP_HIT | 0.50 | 3.97% |
| SELL | retest2 | 2025-04-30 09:15:00 | 454.35 | 2025-05-12 14:15:00 | 440.80 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2025-04-30 11:45:00 | 448.00 | 2025-05-12 14:15:00 | 440.80 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2025-05-14 13:30:00 | 447.10 | 2025-05-16 09:15:00 | 457.00 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-06-27 10:30:00 | 486.75 | 2025-08-01 15:15:00 | 461.85 | STOP_HIT | 1.00 | -5.12% |
| BUY | retest2 | 2025-07-31 10:45:00 | 475.00 | 2025-08-01 15:15:00 | 461.85 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2025-08-04 13:15:00 | 475.45 | 2025-08-06 10:15:00 | 460.60 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-08-04 14:30:00 | 474.65 | 2025-08-06 10:15:00 | 460.60 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-09-10 14:30:00 | 471.85 | 2025-09-24 10:15:00 | 448.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-12 09:15:00 | 470.45 | 2025-09-24 10:15:00 | 446.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 10:00:00 | 471.20 | 2025-09-24 10:15:00 | 447.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 10:00:00 | 472.25 | 2025-09-24 10:15:00 | 448.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 14:30:00 | 471.85 | 2025-09-24 11:15:00 | 471.35 | STOP_HIT | 0.50 | 0.11% |
| SELL | retest2 | 2025-09-12 09:15:00 | 470.45 | 2025-09-24 11:15:00 | 471.35 | STOP_HIT | 0.50 | -0.19% |
| SELL | retest2 | 2025-09-17 10:00:00 | 471.20 | 2025-09-24 11:15:00 | 471.35 | STOP_HIT | 0.50 | -0.03% |
| SELL | retest2 | 2025-09-18 10:00:00 | 472.25 | 2025-09-24 11:15:00 | 471.35 | STOP_HIT | 0.50 | 0.19% |
| SELL | retest2 | 2025-09-19 14:30:00 | 468.10 | 2025-09-24 12:15:00 | 487.65 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2025-09-24 15:00:00 | 468.10 | 2025-09-30 11:15:00 | 444.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:15:00 | 466.45 | 2025-09-30 13:15:00 | 443.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 10:00:00 | 464.60 | 2025-09-30 13:15:00 | 441.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 15:00:00 | 468.10 | 2025-10-06 11:15:00 | 463.30 | STOP_HIT | 0.50 | 1.03% |
| SELL | retest2 | 2025-09-25 09:15:00 | 466.45 | 2025-10-06 11:15:00 | 463.30 | STOP_HIT | 0.50 | 0.68% |
| SELL | retest2 | 2025-09-25 10:00:00 | 464.60 | 2025-10-06 11:15:00 | 463.30 | STOP_HIT | 0.50 | 0.28% |
| SELL | retest2 | 2025-10-14 15:15:00 | 464.00 | 2025-10-15 10:15:00 | 472.60 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-10-17 10:30:00 | 462.80 | 2025-10-21 13:15:00 | 469.65 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-12-04 12:15:00 | 458.15 | 2025-12-05 09:15:00 | 474.50 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2025-12-08 13:00:00 | 462.75 | 2025-12-11 15:15:00 | 464.25 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-12-10 15:15:00 | 459.75 | 2025-12-12 11:15:00 | 469.45 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-12-11 09:30:00 | 461.00 | 2025-12-12 11:15:00 | 469.45 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-11 12:30:00 | 457.25 | 2025-12-12 11:15:00 | 469.45 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2026-01-09 15:15:00 | 475.00 | 2026-01-12 09:15:00 | 464.35 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2026-01-13 09:15:00 | 475.00 | 2026-01-19 09:15:00 | 467.45 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-01-14 11:15:00 | 472.50 | 2026-01-19 09:15:00 | 467.45 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-01-16 09:15:00 | 472.65 | 2026-01-19 09:15:00 | 467.45 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-02-01 12:15:00 | 452.50 | 2026-02-16 14:15:00 | 473.00 | STOP_HIT | 1.00 | -4.53% |
| SELL | retest2 | 2026-02-05 12:00:00 | 453.15 | 2026-02-16 14:15:00 | 473.00 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2026-02-06 10:00:00 | 454.50 | 2026-02-16 14:15:00 | 473.00 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2026-02-06 11:15:00 | 454.50 | 2026-02-17 09:15:00 | 476.20 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2026-02-09 15:15:00 | 453.20 | 2026-02-17 09:15:00 | 476.20 | STOP_HIT | 1.00 | -5.08% |
| SELL | retest2 | 2026-02-10 13:45:00 | 459.80 | 2026-02-17 09:15:00 | 476.20 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2026-02-10 15:15:00 | 459.90 | 2026-02-17 09:15:00 | 476.20 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2026-04-01 13:45:00 | 451.20 | 2026-04-02 09:15:00 | 428.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 14:15:00 | 450.90 | 2026-04-02 09:15:00 | 428.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 13:45:00 | 451.20 | 2026-04-06 14:15:00 | 451.95 | STOP_HIT | 0.50 | -0.17% |
| SELL | retest2 | 2026-04-01 14:15:00 | 450.90 | 2026-04-06 14:15:00 | 451.95 | STOP_HIT | 0.50 | -0.23% |
| SELL | retest2 | 2026-04-07 15:00:00 | 452.00 | 2026-04-08 09:15:00 | 471.40 | STOP_HIT | 1.00 | -4.29% |
