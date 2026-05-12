# Dabur India Ltd. (DABUR)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 487.00
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
| ALERT2 | 9 |
| ALERT2_SKIP | 4 |
| ALERT3 | 54 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 37 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 28
- **Target hits / Stop hits / Partials:** 5 / 32 / 5
- **Avg / median % per leg:** 0.95% / -0.87%
- **Sum % (uncompounded):** 39.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 2 | 6 | 0 | 1.53% | 12.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 2 | 6 | 0 | 1.53% | 12.3% |
| SELL (all) | 34 | 12 | 35.3% | 3 | 26 | 5 | 0.81% | 27.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 12 | 35.3% | 3 | 26 | 5 | 0.81% | 27.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 42 | 14 | 33.3% | 5 | 32 | 5 | 0.95% | 40.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 557.00 | 527.31 | 527.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 559.50 | 528.20 | 527.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 12:15:00 | 626.35 | 627.46 | 608.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 13:00:00 | 626.35 | 627.46 | 608.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 609.45 | 626.48 | 609.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 609.45 | 626.48 | 609.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 605.60 | 626.27 | 609.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 605.60 | 626.27 | 609.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 605.15 | 626.06 | 609.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 605.15 | 626.06 | 609.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 608.00 | 625.88 | 609.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 605.05 | 625.88 | 609.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 606.30 | 625.45 | 609.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 610.50 | 624.44 | 609.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-17 09:15:00 | 671.55 | 645.23 | 630.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 14:15:00 | 567.50 | 626.59 | 626.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 563.85 | 599.22 | 610.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 518.20 | 514.82 | 532.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-03 11:00:00 | 518.20 | 514.82 | 532.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 523.30 | 514.32 | 529.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:30:00 | 527.15 | 514.32 | 529.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 530.05 | 515.37 | 527.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 530.05 | 515.37 | 527.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 530.05 | 515.51 | 527.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 529.45 | 515.51 | 527.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 529.50 | 515.65 | 527.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 12:30:00 | 529.95 | 515.65 | 527.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 526.30 | 515.88 | 527.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 09:15:00 | 523.60 | 515.99 | 527.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 11:00:00 | 524.60 | 516.15 | 527.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 12:30:00 | 523.80 | 516.31 | 527.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 13:00:00 | 524.90 | 516.81 | 527.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 525.05 | 517.83 | 526.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:45:00 | 525.95 | 517.83 | 526.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 528.80 | 518.13 | 526.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 528.80 | 518.13 | 526.75 | SL hit (close>static) qty=1.00 sl=527.85 alert=retest2 |

### Cycle 3 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 512.80 | 485.03 | 484.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 15:15:00 | 514.00 | 485.32 | 485.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 15:15:00 | 513.00 | 513.82 | 504.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:15:00 | 515.00 | 513.82 | 504.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 502.25 | 513.55 | 504.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:45:00 | 501.25 | 513.55 | 504.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 505.00 | 513.47 | 504.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 13:30:00 | 506.30 | 513.24 | 504.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 14:00:00 | 506.40 | 513.24 | 504.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 15:15:00 | 502.00 | 512.60 | 505.00 | SL hit (close<static) qty=1.00 sl=502.05 alert=retest2 |

### Cycle 4 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 491.40 | 515.91 | 516.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 489.10 | 514.79 | 515.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 09:15:00 | 505.60 | 505.15 | 509.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:00:00 | 505.60 | 505.15 | 509.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 509.95 | 505.20 | 509.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 509.95 | 505.20 | 509.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 509.45 | 505.24 | 509.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 12:15:00 | 505.80 | 505.41 | 509.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 12:45:00 | 505.00 | 505.41 | 509.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 505.50 | 505.40 | 509.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-21 14:30:00 | 504.80 | 505.40 | 509.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 508.75 | 505.43 | 509.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 508.90 | 505.43 | 509.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 508.70 | 505.46 | 509.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 510.10 | 505.46 | 509.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 511.45 | 505.52 | 509.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:45:00 | 511.90 | 505.52 | 509.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 512.30 | 505.59 | 509.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-23 12:15:00 | 512.30 | 505.59 | 509.63 | SL hit (close>static) qty=1.00 sl=511.65 alert=retest2 |

### Cycle 5 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 526.85 | 510.74 | 510.70 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 14:15:00 | 503.00 | 511.78 | 511.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 10:15:00 | 502.20 | 511.55 | 511.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 505.35 | 500.54 | 504.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 505.35 | 500.54 | 504.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 505.35 | 500.54 | 504.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 505.35 | 500.54 | 504.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 506.45 | 500.60 | 504.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:15:00 | 507.45 | 500.60 | 504.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 508.40 | 500.67 | 505.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 508.40 | 500.67 | 505.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 504.00 | 500.76 | 505.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:30:00 | 506.00 | 500.76 | 505.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 503.40 | 500.78 | 505.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 503.40 | 500.78 | 505.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 501.50 | 500.81 | 504.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:30:00 | 500.35 | 500.79 | 504.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 512.65 | 500.84 | 504.84 | SL hit (close>static) qty=1.00 sl=505.90 alert=retest2 |

### Cycle 7 — BUY (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 11:15:00 | 522.30 | 508.09 | 508.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 13:15:00 | 524.00 | 508.39 | 508.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 11:15:00 | 509.70 | 510.49 | 509.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 11:15:00 | 509.70 | 510.49 | 509.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 509.70 | 510.49 | 509.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 509.70 | 510.49 | 509.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 507.70 | 510.47 | 509.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:30:00 | 507.60 | 510.47 | 509.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 506.50 | 510.43 | 509.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 506.50 | 510.43 | 509.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 508.20 | 512.49 | 510.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 508.20 | 512.49 | 510.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 504.90 | 512.41 | 510.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:45:00 | 502.80 | 512.41 | 510.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 513.55 | 512.40 | 510.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:30:00 | 509.30 | 512.40 | 510.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 510.15 | 512.37 | 510.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-28 14:00:00 | 510.15 | 512.37 | 510.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 515.05 | 512.40 | 510.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 15:15:00 | 517.20 | 512.40 | 510.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 505.40 | 512.38 | 510.65 | SL hit (close<static) qty=1.00 sl=510.05 alert=retest2 |

### Cycle 8 — SELL (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 13:15:00 | 502.30 | 509.27 | 509.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 15:15:00 | 499.85 | 509.10 | 509.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 510.70 | 508.56 | 508.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 10:15:00 | 510.70 | 508.56 | 508.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 510.70 | 508.56 | 508.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 511.45 | 508.56 | 508.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 510.85 | 508.58 | 508.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 510.85 | 508.58 | 508.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 09:15:00 | 518.30 | 509.25 | 509.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 14:15:00 | 524.50 | 512.23 | 511.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 510.00 | 512.69 | 511.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 510.00 | 512.69 | 511.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 510.00 | 512.69 | 511.30 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 12:15:00 | 485.70 | 509.99 | 510.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 10:15:00 | 480.80 | 508.77 | 509.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 11:15:00 | 444.80 | 443.64 | 463.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-20 12:00:00 | 444.80 | 443.64 | 463.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 460.35 | 444.99 | 462.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:30:00 | 462.00 | 444.99 | 462.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 455.50 | 446.10 | 462.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:30:00 | 453.35 | 446.16 | 462.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:15:00 | 454.55 | 446.76 | 461.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:30:00 | 453.10 | 447.20 | 461.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 448.40 | 447.67 | 461.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 460.25 | 447.86 | 459.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:00:00 | 460.25 | 447.86 | 459.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 462.00 | 448.00 | 459.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 462.45 | 448.00 | 459.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 465.95 | 448.43 | 459.95 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 465.95 | 448.43 | 459.95 | SL hit (close>static) qty=1.00 sl=462.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-08-16 09:15:00 | 610.50 | 2024-09-17 09:15:00 | 671.55 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-20 09:15:00 | 523.60 | 2025-01-27 09:15:00 | 528.80 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-01-20 11:00:00 | 524.60 | 2025-01-27 09:15:00 | 528.80 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-01-20 12:30:00 | 523.80 | 2025-01-27 09:15:00 | 528.80 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-01-21 13:00:00 | 524.90 | 2025-01-27 09:15:00 | 528.80 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-01-27 12:15:00 | 526.40 | 2025-01-30 14:15:00 | 534.70 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-01-28 10:15:00 | 526.25 | 2025-01-30 14:15:00 | 534.70 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-02-01 11:45:00 | 523.85 | 2025-02-01 12:15:00 | 548.00 | STOP_HIT | 1.00 | -4.61% |
| SELL | retest2 | 2025-02-06 13:00:00 | 526.65 | 2025-02-10 09:15:00 | 531.80 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-02-07 10:15:00 | 521.05 | 2025-02-20 09:15:00 | 500.32 | PARTIAL | 0.50 | 3.98% |
| SELL | retest2 | 2025-02-11 12:15:00 | 521.35 | 2025-02-28 09:15:00 | 495.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:15:00 | 521.00 | 2025-02-28 09:15:00 | 494.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:45:00 | 520.45 | 2025-02-28 09:15:00 | 494.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 10:15:00 | 521.05 | 2025-03-21 09:15:00 | 504.70 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2025-02-11 12:15:00 | 521.35 | 2025-03-21 09:15:00 | 504.70 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2025-02-13 14:15:00 | 521.00 | 2025-03-21 09:15:00 | 504.70 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-02-14 10:45:00 | 520.45 | 2025-03-21 09:15:00 | 504.70 | STOP_HIT | 0.50 | 3.03% |
| SELL | retest2 | 2025-03-25 14:30:00 | 509.05 | 2025-03-27 14:15:00 | 513.50 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-03-25 15:00:00 | 508.80 | 2025-03-27 14:15:00 | 513.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-03-26 14:15:00 | 509.15 | 2025-03-27 14:15:00 | 513.50 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-03-27 11:15:00 | 509.30 | 2025-03-27 14:15:00 | 513.50 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-03-28 11:15:00 | 509.30 | 2025-04-03 09:15:00 | 458.37 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-28 12:00:00 | 508.30 | 2025-04-03 09:15:00 | 482.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 12:00:00 | 508.30 | 2025-04-03 09:15:00 | 458.24 | TARGET_HIT | 0.50 | 9.85% |
| SELL | retest2 | 2025-04-01 10:00:00 | 509.15 | 2025-04-07 09:15:00 | 457.47 | TARGET_HIT | 1.00 | 10.15% |
| BUY | retest2 | 2025-08-11 13:30:00 | 506.30 | 2025-08-12 15:15:00 | 502.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-08-11 14:00:00 | 506.40 | 2025-08-12 15:15:00 | 502.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-08-18 09:15:00 | 523.80 | 2025-09-04 09:15:00 | 576.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-20 12:15:00 | 505.80 | 2025-10-23 12:15:00 | 512.30 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-10-20 12:45:00 | 505.00 | 2025-10-23 12:15:00 | 512.30 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-10-21 13:45:00 | 505.50 | 2025-10-23 12:15:00 | 512.30 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-10-21 14:30:00 | 504.80 | 2025-10-23 12:15:00 | 512.30 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-10-24 09:15:00 | 502.35 | 2025-11-04 12:15:00 | 517.25 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2026-01-01 10:30:00 | 500.35 | 2026-01-02 09:15:00 | 512.65 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-01-28 15:15:00 | 517.20 | 2026-01-29 09:15:00 | 505.40 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-01-30 09:15:00 | 515.90 | 2026-01-30 11:15:00 | 509.60 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-30 09:45:00 | 515.85 | 2026-01-30 11:15:00 | 509.60 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-01-30 10:30:00 | 516.25 | 2026-01-30 11:15:00 | 509.60 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-04-24 10:30:00 | 453.35 | 2026-05-06 11:15:00 | 465.95 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2026-04-27 13:15:00 | 454.55 | 2026-05-06 11:15:00 | 465.95 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-04-29 09:30:00 | 453.10 | 2026-05-06 11:15:00 | 465.95 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-04-30 09:15:00 | 448.40 | 2026-05-06 11:15:00 | 465.95 | STOP_HIT | 1.00 | -3.91% |
