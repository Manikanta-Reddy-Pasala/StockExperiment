# J.K. Cement Ltd. (JKCEMENT)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 5555.50
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
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 8
- **Target hits / Stop hits / Partials:** 1 / 10 / 3
- **Avg / median % per leg:** 0.49% / -1.26%
- **Sum % (uncompounded):** 6.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 6 | 42.9% | 1 | 10 | 3 | 0.49% | 6.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 6 | 42.9% | 1 | 10 | 3 | 0.49% | 6.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 6 | 42.9% | 1 | 10 | 3 | 0.49% | 6.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 15:15:00 | 6425.00 | 6534.96 | 6534.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 6399.00 | 6533.61 | 6534.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 14:15:00 | 5925.00 | 5888.49 | 6115.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 15:00:00 | 5925.00 | 5888.49 | 6115.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 5790.00 | 5638.46 | 5803.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:45:00 | 5791.00 | 5638.46 | 5803.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 5756.00 | 5639.63 | 5802.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:45:00 | 5798.00 | 5639.63 | 5802.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 5821.00 | 5648.33 | 5799.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 5821.00 | 5648.33 | 5799.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 5902.00 | 5650.86 | 5799.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 5902.00 | 5650.86 | 5799.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 5966.00 | 5653.99 | 5800.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 5966.00 | 5653.99 | 5800.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 5922.50 | 5670.72 | 5804.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:45:00 | 5916.00 | 5670.72 | 5804.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 5934.50 | 5673.35 | 5805.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 5934.50 | 5673.35 | 5805.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 5839.50 | 5688.55 | 5808.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:30:00 | 5820.50 | 5688.55 | 5808.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 5819.50 | 5689.85 | 5808.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:15:00 | 5784.50 | 5691.24 | 5808.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 10:15:00 | 5875.00 | 5693.91 | 5796.30 | SL hit (close>static) qty=1.00 sl=5850.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-09 14:15:00 | 5784.50 | 2026-01-16 10:15:00 | 5875.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-01-19 09:15:00 | 5763.00 | 2026-01-20 09:15:00 | 6054.00 | STOP_HIT | 1.00 | -5.05% |
| SELL | retest2 | 2026-01-19 15:00:00 | 5775.00 | 2026-01-20 09:15:00 | 6054.00 | STOP_HIT | 1.00 | -4.83% |
| SELL | retest2 | 2026-01-21 09:15:00 | 5654.50 | 2026-01-27 10:15:00 | 5371.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-21 09:15:00 | 5654.50 | 2026-02-04 09:15:00 | 5636.50 | STOP_HIT | 0.50 | 0.32% |
| SELL | retest2 | 2026-02-06 09:15:00 | 5671.00 | 2026-02-09 09:15:00 | 5849.00 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2026-02-06 12:00:00 | 5705.00 | 2026-02-09 09:15:00 | 5849.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2026-02-13 09:15:00 | 5704.00 | 2026-02-17 10:15:00 | 5787.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-02-13 09:45:00 | 5703.00 | 2026-02-17 10:15:00 | 5787.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-02-23 10:30:00 | 5735.00 | 2026-02-23 15:15:00 | 5807.50 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-02-24 09:15:00 | 5729.00 | 2026-03-02 09:15:00 | 5442.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 5729.00 | 2026-03-09 10:15:00 | 5156.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-24 09:15:00 | 5683.50 | 2026-04-30 09:15:00 | 5399.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 09:15:00 | 5683.50 | 2026-05-06 14:15:00 | 5525.50 | STOP_HIT | 0.50 | 2.78% |
