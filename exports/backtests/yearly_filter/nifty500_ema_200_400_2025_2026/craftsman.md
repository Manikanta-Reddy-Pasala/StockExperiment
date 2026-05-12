# Craftsman Automation Ltd. (CRAFTSMAN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 8990.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 0 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 0
- **Target hits / Stop hits / Partials:** 0 / 0 / 0
- **Avg / median % per leg:** 0.00% / 0.00%
- **Sum % (uncompounded):** 0.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 09:15:00 | 6936.00 | 7448.31 | 7448.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 11:15:00 | 6855.00 | 7437.44 | 7442.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 12:15:00 | 7269.50 | 7095.35 | 7233.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 12:15:00 | 7269.50 | 7095.35 | 7233.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 7269.50 | 7095.35 | 7233.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:00:00 | 7269.50 | 7095.35 | 7233.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 13:15:00 | 7378.50 | 7098.17 | 7234.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:45:00 | 7350.00 | 7098.17 | 7234.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 7388.50 | 7151.21 | 7250.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 7388.50 | 7151.21 | 7250.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 7708.00 | 7326.60 | 7326.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 13:15:00 | 7763.00 | 7411.62 | 7372.30 | Break + close above crossover candle high |

