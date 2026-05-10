# Himadri Speciality Chemical Ltd. (HSCL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 631.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 5 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 33 |
| PARTIAL | 9 |
| TARGET_HIT | 0 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 28
- **Target hits / Stop hits / Partials:** 0 / 33 / 9
- **Avg / median % per leg:** -0.61% / -1.10%
- **Sum % (uncompounded):** -25.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.50% | -20.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.50% | -20.0% |
| SELL (all) | 34 | 14 | 41.2% | 0 | 25 | 9 | -0.16% | -5.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 14 | 41.2% | 0 | 25 | 9 | -0.16% | -5.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 42 | 14 | 33.3% | 0 | 33 | 9 | -0.61% | -25.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 13:15:00)

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
| Stop hit — per-position SL triggered | 2025-08-01 15:15:00 | 461.85 | 491.78 | 484.02 | SL hit (close<static) qty=1.00 sl=462.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 13:15:00 | 475.45 | 490.84 | 483.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 14:30:00 | 474.65 | 490.50 | 483.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 460.60 | 488.62 | 482.97 | SL hit (close<static) qty=1.00 sl=462.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 460.60 | 488.62 | 482.97 | SL hit (close<static) qty=1.00 sl=462.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 474.00 | 479.61 | 478.92 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 472.10 | 478.29 | 478.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 467.95 | 477.46 | 477.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 483.50 | 466.84 | 471.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 483.50 | 466.84 | 471.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 483.50 | 466.84 | 471.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 483.50 | 466.84 | 471.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 474.80 | 466.92 | 471.43 | EMA400 retest candle locked (from downside) |
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
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 471.35 | 466.20 | 469.78 | SL hit (close>ema200) qty=0.50 sl=466.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 471.35 | 466.20 | 469.78 | SL hit (close>ema200) qty=0.50 sl=466.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 471.35 | 466.20 | 469.78 | SL hit (close>ema200) qty=0.50 sl=466.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 12:15:00 | 487.65 | 466.41 | 469.87 | SL hit (close>static) qty=1.00 sl=473.95 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:00:00 | 468.10 | 466.56 | 469.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 466.45 | 466.58 | 469.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 10:00:00 | 464.60 | 466.56 | 469.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 11:15:00 | 444.69 | 464.06 | 468.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 13:15:00 | 443.13 | 463.62 | 467.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 13:15:00 | 441.37 | 463.62 | 467.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 463.30 | 461.18 | 466.27 | SL hit (close>ema200) qty=0.50 sl=461.18 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 463.30 | 461.18 | 466.27 | SL hit (close>ema200) qty=0.50 sl=461.18 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 463.30 | 461.18 | 466.27 | SL hit (close>ema200) qty=0.50 sl=461.18 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 464.50 | 461.02 | 465.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 467.50 | 461.02 | 465.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 465.25 | 460.23 | 464.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:30:00 | 465.95 | 460.23 | 464.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 467.10 | 460.30 | 464.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 467.10 | 460.30 | 464.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 468.00 | 460.38 | 464.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 470.35 | 460.38 | 464.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 464.85 | 460.61 | 464.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:30:00 | 464.60 | 460.61 | 464.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 466.50 | 460.67 | 464.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:45:00 | 466.50 | 460.67 | 464.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 465.35 | 460.72 | 464.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 15:15:00 | 464.00 | 460.72 | 464.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 472.60 | 460.93 | 465.01 | SL hit (close>static) qty=1.00 sl=467.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:30:00 | 462.80 | 461.99 | 465.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 469.65 | 461.67 | 464.91 | SL hit (close>static) qty=1.00 sl=467.90 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 483.95 | 467.42 | 467.35 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-11-07 10:15:00)

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
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 13:00:00 | 462.75 | 454.11 | 458.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 15:15:00 | 459.75 | 455.21 | 458.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:30:00 | 461.00 | 455.31 | 458.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 457.60 | 455.33 | 458.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 12:30:00 | 457.25 | 455.38 | 458.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 15:15:00 | 464.25 | 455.55 | 458.56 | SL hit (close>static) qty=1.00 sl=461.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 11:15:00 | 469.45 | 455.89 | 458.69 | SL hit (close>static) qty=1.00 sl=467.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 11:15:00 | 469.45 | 455.89 | 458.69 | SL hit (close>static) qty=1.00 sl=467.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 11:15:00 | 469.45 | 455.89 | 458.69 | SL hit (close>static) qty=1.00 sl=467.90 alert=retest2 |

### Cycle 5 — BUY (started 2025-12-22 15:15:00)

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
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 475.00 | 473.46 | 468.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 11:15:00 | 472.50 | 473.59 | 469.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 472.65 | 473.52 | 469.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 467.45 | 473.40 | 469.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 467.45 | 473.40 | 469.34 | SL hit (close<static) qty=1.00 sl=467.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 467.45 | 473.40 | 469.34 | SL hit (close<static) qty=1.00 sl=467.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 467.45 | 473.40 | 469.34 | SL hit (close<static) qty=1.00 sl=467.50 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 465.25 | 473.40 | 469.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 466.00 | 473.32 | 469.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:45:00 | 465.40 | 473.32 | 469.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 457.25 | 469.78 | 467.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 457.25 | 469.78 | 467.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2026-01-28 11:15:00)

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
| Stop hit — per-position SL triggered | 2026-02-16 14:15:00 | 473.00 | 459.89 | 462.27 | SL hit (close>static) qty=1.00 sl=464.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 14:15:00 | 473.00 | 459.89 | 462.27 | SL hit (close>static) qty=1.00 sl=464.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 476.20 | 460.16 | 462.38 | SL hit (close>static) qty=1.00 sl=474.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 476.20 | 460.16 | 462.38 | SL hit (close>static) qty=1.00 sl=474.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 476.20 | 460.16 | 462.38 | SL hit (close>static) qty=1.00 sl=474.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 476.20 | 460.16 | 462.38 | SL hit (close>static) qty=1.00 sl=474.90 alert=retest2 |

### Cycle 7 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 486.65 | 464.37 | 464.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 488.55 | 464.83 | 464.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 469.65 | 470.29 | 467.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 469.65 | 470.29 | 467.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 469.65 | 470.29 | 467.56 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2026-03-09 10:15:00)

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
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 451.95 | 450.59 | 455.59 | SL hit (close>ema200) qty=0.50 sl=450.59 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 15:00:00 | 452.00 | 450.47 | 455.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 471.40 | 450.70 | 455.42 | SL hit (close>static) qty=1.00 sl=465.60 alert=retest2 |

### Cycle 9 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 494.15 | 459.05 | 459.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 505.90 | 463.20 | 461.19 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
