# Hindustan Petroleum Corporation Ltd. (HINDPETRO)

## Backtest Summary

- **Window:** 2024-07-09 09:15:00 → 2026-05-08 15:15:00 (3168 bars)
- **Last close:** 387.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 4 |
| TARGET_HIT | 7 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 15
- **Target hits / Stop hits / Partials:** 7 / 17 / 4
- **Avg / median % per leg:** 2.62% / -0.45%
- **Sum % (uncompounded):** 73.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 3 | 9 | 0 | 2.31% | 27.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 5 | 41.7% | 3 | 9 | 0 | 2.31% | 27.7% |
| SELL (all) | 16 | 8 | 50.0% | 4 | 8 | 4 | 2.85% | 45.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 8 | 50.0% | 4 | 8 | 4 | 2.85% | 45.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 13 | 46.4% | 7 | 17 | 4 | 2.62% | 73.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 10:15:00 | 394.70 | 410.55 | 410.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 393.80 | 410.22 | 410.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 398.30 | 396.35 | 401.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:45:00 | 397.80 | 396.35 | 401.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 403.80 | 396.42 | 401.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:00:00 | 403.80 | 396.42 | 401.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 402.35 | 396.48 | 401.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 401.85 | 396.48 | 401.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 399.15 | 396.51 | 401.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:30:00 | 398.60 | 396.52 | 401.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:45:00 | 398.10 | 396.61 | 401.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 14:30:00 | 398.50 | 396.67 | 401.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:30:00 | 398.70 | 396.72 | 401.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 402.20 | 396.88 | 401.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 406.80 | 397.28 | 401.48 | SL hit (close>static) qty=1.00 sl=402.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 406.80 | 397.28 | 401.48 | SL hit (close>static) qty=1.00 sl=402.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 406.80 | 397.28 | 401.48 | SL hit (close>static) qty=1.00 sl=402.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 406.80 | 397.28 | 401.48 | SL hit (close>static) qty=1.00 sl=402.90 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 14:15:00 | 423.05 | 404.64 | 404.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 435.70 | 406.34 | 405.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 13:15:00 | 465.00 | 467.20 | 450.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 14:00:00 | 465.00 | 467.20 | 450.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 452.35 | 465.03 | 452.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 450.20 | 465.03 | 452.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 452.60 | 464.91 | 452.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 456.80 | 464.91 | 452.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 450.00 | 464.65 | 452.39 | SL hit (close<static) qty=1.00 sl=451.45 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 11:30:00 | 453.15 | 464.52 | 452.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 449.65 | 464.38 | 452.37 | SL hit (close<static) qty=1.00 sl=451.45 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 13:30:00 | 453.10 | 463.27 | 452.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 451.05 | 463.15 | 452.27 | SL hit (close<static) qty=1.00 sl=451.45 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 12:00:00 | 452.70 | 459.87 | 451.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 451.75 | 459.79 | 451.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:30:00 | 450.50 | 459.79 | 451.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 452.70 | 459.72 | 451.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:15:00 | 450.90 | 459.72 | 451.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 448.60 | 459.61 | 451.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 448.60 | 459.61 | 451.80 | SL hit (close<static) qty=1.00 sl=451.45 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 448.60 | 459.61 | 451.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 447.85 | 459.50 | 451.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 450.50 | 459.50 | 451.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 15:00:00 | 450.00 | 459.05 | 451.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 450.75 | 458.95 | 451.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 445.50 | 458.81 | 451.74 | SL hit (close<static) qty=1.00 sl=446.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 445.50 | 458.81 | 451.74 | SL hit (close<static) qty=1.00 sl=446.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 445.50 | 458.81 | 451.74 | SL hit (close<static) qty=1.00 sl=446.70 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 451.20 | 458.17 | 451.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 450.55 | 458.01 | 451.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 449.00 | 458.01 | 451.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2025-12-31 13:15:00 | 496.32 | 465.67 | 458.28 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 464.90 | 472.85 | 463.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 464.90 | 472.85 | 463.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 458.50 | 472.71 | 463.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 458.50 | 472.71 | 463.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 456.15 | 472.54 | 463.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 456.15 | 472.54 | 463.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 421.20 | 457.24 | 457.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 418.25 | 456.85 | 457.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 447.70 | 447.14 | 451.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 10:00:00 | 447.70 | 447.14 | 451.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 452.45 | 447.15 | 451.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 452.45 | 447.15 | 451.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 453.65 | 447.22 | 451.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 453.50 | 447.22 | 451.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 450.90 | 447.27 | 451.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 15:00:00 | 450.90 | 447.27 | 451.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 463.25 | 447.47 | 451.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 463.25 | 447.47 | 451.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 460.20 | 447.59 | 451.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 11:30:00 | 459.05 | 447.71 | 451.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 14:30:00 | 459.10 | 448.02 | 451.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 464.60 | 448.31 | 451.93 | SL hit (close>static) qty=1.00 sl=464.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 464.60 | 448.31 | 451.93 | SL hit (close>static) qty=1.00 sl=464.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 12:30:00 | 457.50 | 448.64 | 452.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 15:00:00 | 458.90 | 448.82 | 452.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 466.45 | 449.09 | 452.21 | SL hit (close>static) qty=1.00 sl=464.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 466.45 | 449.09 | 452.21 | SL hit (close>static) qty=1.00 sl=464.30 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 453.85 | 452.33 | 453.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:00:00 | 453.85 | 452.33 | 453.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 453.85 | 452.35 | 453.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:30:00 | 455.20 | 452.35 | 453.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 451.85 | 452.34 | 453.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:45:00 | 452.50 | 452.34 | 453.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 451.90 | 452.34 | 453.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:30:00 | 452.90 | 452.34 | 453.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 454.30 | 451.88 | 453.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 454.30 | 451.88 | 453.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 454.70 | 451.90 | 453.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 456.30 | 451.90 | 453.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 439.00 | 448.71 | 451.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 435.40 | 448.61 | 451.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 437.90 | 447.70 | 450.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 13:15:00 | 438.05 | 447.63 | 450.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:45:00 | 436.50 | 447.31 | 450.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 416.00 | 446.74 | 449.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 416.15 | 446.74 | 449.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 413.63 | 445.19 | 449.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 414.67 | 445.19 | 449.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 391.86 | 437.78 | 444.77 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 394.11 | 437.78 | 444.77 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 394.25 | 437.78 | 444.77 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 392.85 | 437.78 | 444.77 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 383.80 | 372.00 | 385.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:30:00 | 383.15 | 372.00 | 385.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 393.15 | 372.21 | 385.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:45:00 | 392.95 | 372.21 | 385.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 398.15 | 372.47 | 385.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:30:00 | 401.50 | 372.47 | 385.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 386.45 | 375.08 | 385.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 389.20 | 375.08 | 385.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 387.50 | 375.20 | 385.67 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 11:45:00 | 386.15 | 2025-06-26 10:15:00 | 424.05 | TARGET_HIT | 1.00 | 9.81% |
| BUY | retest2 | 2025-06-16 10:15:00 | 385.50 | 2025-06-27 09:15:00 | 424.76 | TARGET_HIT | 1.00 | 10.19% |
| BUY | retest2 | 2025-08-18 12:30:00 | 385.30 | 2025-08-21 10:15:00 | 394.70 | STOP_HIT | 1.00 | 2.44% |
| BUY | retest2 | 2025-08-18 13:45:00 | 386.25 | 2025-08-21 10:15:00 | 394.70 | STOP_HIT | 1.00 | 2.19% |
| SELL | retest2 | 2025-09-11 13:30:00 | 398.60 | 2025-09-17 09:15:00 | 406.80 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-09-12 11:45:00 | 398.10 | 2025-09-17 09:15:00 | 406.80 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-09-12 14:30:00 | 398.50 | 2025-09-17 09:15:00 | 406.80 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-09-15 09:30:00 | 398.70 | 2025-09-17 09:15:00 | 406.80 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-12-02 09:15:00 | 456.80 | 2025-12-02 10:15:00 | 450.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-12-02 11:30:00 | 453.15 | 2025-12-02 12:15:00 | 449.65 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-03 13:30:00 | 453.10 | 2025-12-03 14:15:00 | 451.05 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-09 12:00:00 | 452.70 | 2025-12-09 14:15:00 | 448.60 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-10 09:15:00 | 450.50 | 2025-12-11 09:15:00 | 445.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-10 15:00:00 | 450.00 | 2025-12-11 09:15:00 | 445.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-11 09:15:00 | 450.75 | 2025-12-11 09:15:00 | 445.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-12-12 09:15:00 | 451.20 | 2025-12-31 13:15:00 | 496.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-04 11:30:00 | 459.05 | 2026-02-05 09:15:00 | 464.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-02-04 14:30:00 | 459.10 | 2026-02-05 09:15:00 | 464.60 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-02-05 12:30:00 | 457.50 | 2026-02-06 09:15:00 | 466.45 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-02-05 15:00:00 | 458.90 | 2026-02-06 09:15:00 | 466.45 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-02-25 11:15:00 | 435.40 | 2026-03-02 09:15:00 | 416.00 | PARTIAL | 0.50 | 4.45% |
| SELL | retest2 | 2026-02-26 11:30:00 | 437.90 | 2026-03-02 09:15:00 | 416.15 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2026-02-26 13:15:00 | 438.05 | 2026-03-04 09:15:00 | 413.63 | PARTIAL | 0.50 | 5.57% |
| SELL | retest2 | 2026-02-27 10:45:00 | 436.50 | 2026-03-04 09:15:00 | 414.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 435.40 | 2026-03-09 09:15:00 | 391.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 437.90 | 2026-03-09 09:15:00 | 394.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 13:15:00 | 438.05 | 2026-03-09 09:15:00 | 394.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-27 10:45:00 | 436.50 | 2026-03-09 09:15:00 | 392.85 | TARGET_HIT | 0.50 | 10.00% |
