# Radico Khaitan Ltd (RADICO)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 3481.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 4 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 4 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 2.29% / 3.56%
- **Sum % (uncompounded):** 13.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 4 | 66.7% | 0 | 4 | 2 | 2.29% | 13.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 0 | 4 | 2 | 2.29% | 13.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 4 | 66.7% | 0 | 4 | 2 | 2.29% | 13.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 11:15:00 | 2933.90 | 3139.49 | 3139.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 14:15:00 | 2903.30 | 3133.01 | 3136.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 14:15:00 | 2772.70 | 2747.51 | 2853.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 14:45:00 | 2756.10 | 2747.51 | 2853.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 2847.80 | 2750.18 | 2845.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 2847.80 | 2750.18 | 2845.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 2844.40 | 2751.12 | 2845.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:15:00 | 2870.90 | 2751.12 | 2845.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 2877.20 | 2752.37 | 2845.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 2877.20 | 2752.37 | 2845.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 2881.80 | 2753.66 | 2845.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 2815.70 | 2753.66 | 2845.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 2858.40 | 2758.54 | 2845.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 14:15:00 | 2715.48 | 2766.70 | 2841.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 2766.60 | 2766.43 | 2840.50 | SL hit (close>ema200) qty=0.50 sl=2766.43 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 11:15:00 | 3201.90 | 2821.63 | 2821.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 14:15:00 | 3255.00 | 2833.39 | 2827.32 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-03-12 09:15:00 | 2815.70 | 2026-03-17 14:15:00 | 2715.48 | PARTIAL | 0.50 | 3.56% |
| SELL | retest2 | 2026-03-12 09:15:00 | 2815.70 | 2026-03-18 09:15:00 | 2766.60 | STOP_HIT | 0.50 | 1.74% |
| SELL | retest2 | 2026-03-12 15:00:00 | 2858.40 | 2026-03-19 09:15:00 | 2674.91 | PARTIAL | 0.50 | 6.42% |
| SELL | retest2 | 2026-03-12 15:00:00 | 2858.40 | 2026-03-25 13:15:00 | 2734.80 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2026-04-15 10:45:00 | 2865.70 | 2026-04-15 12:15:00 | 2900.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-04-15 11:45:00 | 2867.90 | 2026-04-15 12:15:00 | 2900.00 | STOP_HIT | 1.00 | -1.12% |
