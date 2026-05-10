# Vardhman Textiles Ltd. (VTL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 583.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 6 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 1
- **Target hits / Stop hits / Partials:** 2 / 3 / 3
- **Avg / median % per leg:** 3.96% / 5.00%
- **Sum % (uncompounded):** 31.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 7 | 87.5% | 2 | 3 | 3 | 3.96% | 31.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 7 | 87.5% | 2 | 3 | 3 | 3.96% | 31.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 7 | 87.5% | 2 | 3 | 3 | 3.96% | 31.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 423.40 | 480.62 | 480.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 418.40 | 480.00 | 480.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 442.00 | 439.87 | 456.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 13:00:00 | 442.00 | 439.87 | 456.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 444.80 | 432.26 | 449.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 15:00:00 | 444.80 | 432.26 | 449.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 445.00 | 432.39 | 449.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 437.55 | 432.39 | 449.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:15:00 | 415.67 | 432.09 | 448.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 434.30 | 425.00 | 440.81 | SL hit (close>ema200) qty=0.50 sl=425.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 15:15:00 | 440.50 | 425.61 | 440.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 440.70 | 428.17 | 439.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 09:15:00 | 418.47 | 428.39 | 438.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 09:15:00 | 418.66 | 428.39 | 438.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-10-08 14:15:00 | 396.45 | 418.00 | 429.39 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-08 14:15:00 | 396.63 | 418.00 | 429.39 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:30:00 | 439.85 | 412.19 | 422.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 422.00 | 413.09 | 422.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 421.40 | 413.09 | 422.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 427.00 | 413.23 | 422.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 428.95 | 413.23 | 422.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 427.35 | 413.37 | 423.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 427.35 | 413.37 | 423.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 428.65 | 424.12 | 426.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 425.20 | 424.34 | 426.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 430.00 | 424.45 | 426.88 | SL hit (close>static) qty=1.00 sl=429.95 alert=retest2 |
| CROSSOVER_SKIP | 2025-11-13 12:15:00 | 451.85 | 429.04 | 428.98 | min_gap filter: gap=0.013% < 0.030% |
| Stop hit — per-position SL triggered | 2025-11-13 12:15:00 | 451.85 | 429.04 | 428.98 | Force close (TREND_INVERSION) qty=1.00 alert=retest2 |
| TREND_RESET | 2025-11-13 12:15:00 | 451.85 | 429.04 | 428.98 | EMA inversion without crossover edge (EMA200=429.04 EMA400=428.98) — end cycle |
| CROSSOVER_SKIP | 2026-01-09 13:15:00 | 410.70 | 435.79 | 435.90 | min_gap filter: gap=0.028% < 0.030% |
| CROSSOVER_SKIP | 2026-02-03 15:15:00 | 510.00 | 432.18 | 432.10 | min_gap filter: gap=0.017% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-29 09:15:00 | 437.55 | 2025-09-01 09:15:00 | 415.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-29 09:15:00 | 437.55 | 2025-09-10 09:15:00 | 434.30 | STOP_HIT | 0.50 | 0.74% |
| SELL | retest2 | 2025-09-10 15:15:00 | 440.50 | 2025-09-23 09:15:00 | 418.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 09:15:00 | 440.70 | 2025-09-23 09:15:00 | 418.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 15:15:00 | 440.50 | 2025-10-08 14:15:00 | 396.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-18 09:15:00 | 440.70 | 2025-10-08 14:15:00 | 396.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-23 12:30:00 | 439.85 | 2025-11-07 11:15:00 | 430.00 | STOP_HIT | 1.00 | 2.24% |
| SELL | retest2 | 2025-11-07 09:15:00 | 425.20 | 2025-11-13 12:15:00 | 451.85 | STOP_HIT | 1.00 | -6.27% |
