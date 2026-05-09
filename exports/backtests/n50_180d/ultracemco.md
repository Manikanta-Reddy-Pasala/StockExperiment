# ULTRACEMCO (ULTRACEMCO)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 11930.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 8 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 0
- **Avg / median % per leg:** -2.43% / -1.85%
- **Sum % (uncompounded):** -19.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.43% | -19.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.43% | -19.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.43% | -19.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 12304.00 | 11877.31 | 11875.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 12343.00 | 11881.94 | 11878.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 12669.00 | 12709.64 | 12461.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 09:45:00 | 12650.00 | 12709.64 | 12461.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 12469.00 | 12706.33 | 12468.32 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 11118.00 | 12296.55 | 12299.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 10762.00 | 12281.28 | 12291.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 11620.00 | 11358.66 | 11710.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:00:00 | 11620.00 | 11358.66 | 11710.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 11736.00 | 11362.41 | 11710.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 11736.00 | 11362.41 | 11710.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 11678.00 | 11365.55 | 11710.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:45:00 | 11616.00 | 11374.95 | 11709.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 11759.00 | 11409.18 | 11688.40 | SL hit (close>static) qty=1.00 sl=11746.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 11536.00 | 11704.82 | 11775.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 11749.00 | 11695.57 | 11767.93 | SL hit (close>static) qty=1.00 sl=11746.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 11604.00 | 11695.28 | 11767.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 14:15:00 | 11761.00 | 11695.83 | 11766.62 | SL hit (close>static) qty=1.00 sl=11746.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-02 09:30:00 | 11632.00 | 2026-01-01 09:15:00 | 11843.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-12-02 11:30:00 | 11640.00 | 2026-01-05 09:15:00 | 12039.00 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2025-12-02 12:00:00 | 11635.00 | 2026-01-05 09:15:00 | 12039.00 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2025-12-02 12:45:00 | 11636.00 | 2026-01-05 09:15:00 | 12039.00 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2025-12-30 09:15:00 | 11711.00 | 2026-01-05 09:15:00 | 12039.00 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2026-04-08 14:45:00 | 11616.00 | 2026-04-15 11:15:00 | 11759.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-04-30 09:15:00 | 11536.00 | 2026-05-04 10:15:00 | 11749.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-05-04 13:15:00 | 11604.00 | 2026-05-04 14:15:00 | 11761.00 | STOP_HIT | 1.00 | -1.35% |
