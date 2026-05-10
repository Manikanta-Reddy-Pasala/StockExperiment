# IIFL Finance Ltd. (IIFL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 460.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / Stop hits / Partials:** 0 / 6 / 3
- **Avg / median % per leg:** 0.14% / -0.64%
- **Sum % (uncompounded):** 1.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.14% | 1.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.14% | 1.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.14% | 1.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 437.90 | 459.83 | 459.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 435.55 | 458.34 | 459.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 10:15:00 | 451.75 | 450.88 | 454.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 11:00:00 | 451.75 | 450.88 | 454.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 449.55 | 450.87 | 454.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:30:00 | 454.25 | 450.87 | 454.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 454.30 | 450.32 | 454.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 454.30 | 450.32 | 454.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 448.00 | 450.30 | 454.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:30:00 | 445.45 | 450.74 | 453.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 11:00:00 | 446.00 | 450.69 | 453.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:30:00 | 447.25 | 450.62 | 453.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 423.18 | 448.36 | 452.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 423.70 | 448.36 | 452.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 424.89 | 448.36 | 452.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 448.85 | 446.92 | 451.42 | SL hit (close>ema200) qty=0.50 sl=446.92 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 448.85 | 446.92 | 451.42 | SL hit (close>ema200) qty=0.50 sl=446.92 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 448.85 | 446.92 | 451.42 | SL hit (close>ema200) qty=0.50 sl=446.92 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:00:00 | 447.20 | 446.92 | 451.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 455.00 | 446.99 | 451.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 447.40 | 446.97 | 451.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 14:30:00 | 449.45 | 447.09 | 451.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 465.90 | 447.33 | 451.32 | SL hit (close>static) qty=1.00 sl=456.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 465.90 | 447.33 | 451.32 | SL hit (close>static) qty=1.00 sl=455.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 465.90 | 447.33 | 451.32 | SL hit (close>static) qty=1.00 sl=455.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 12:15:00 | 495.45 | 454.74 | 454.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 504.20 | 463.17 | 459.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-21 10:15:00 | 611.00 | 613.45 | 582.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-21 10:45:00 | 611.85 | 613.45 | 582.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 593.30 | 613.40 | 583.64 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 503.60 | 564.07 | 564.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 502.85 | 562.92 | 563.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 13:15:00 | 469.20 | 465.09 | 489.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 14:00:00 | 469.20 | 465.09 | 489.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-23 09:30:00 | 445.45 | 2025-09-26 13:15:00 | 423.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 11:00:00 | 446.00 | 2025-09-26 13:15:00 | 423.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:30:00 | 447.25 | 2025-09-26 13:15:00 | 424.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 09:30:00 | 445.45 | 2025-09-30 09:15:00 | 448.85 | STOP_HIT | 0.50 | -0.76% |
| SELL | retest2 | 2025-09-23 11:00:00 | 446.00 | 2025-09-30 09:15:00 | 448.85 | STOP_HIT | 0.50 | -0.64% |
| SELL | retest2 | 2025-09-24 10:30:00 | 447.25 | 2025-09-30 09:15:00 | 448.85 | STOP_HIT | 0.50 | -0.36% |
| SELL | retest2 | 2025-09-30 11:00:00 | 447.20 | 2025-10-03 09:15:00 | 465.90 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2025-10-01 09:45:00 | 447.40 | 2025-10-03 09:15:00 | 465.90 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2025-10-01 14:30:00 | 449.45 | 2025-10-03 09:15:00 | 465.90 | STOP_HIT | 1.00 | -3.66% |
