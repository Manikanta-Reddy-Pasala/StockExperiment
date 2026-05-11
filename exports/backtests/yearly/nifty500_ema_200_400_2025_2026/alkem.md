# Alkem Laboratories Ltd. (ALKEM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3122 bars)
- **Last close:** 5560.00
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
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 12
- **Target hits / Stop hits / Partials:** 0 / 12 / 0
- **Avg / median % per leg:** -2.81% / -2.35%
- **Sum % (uncompounded):** -33.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.81% | -33.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.81% | -33.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.81% | -33.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 5365.00 | 4964.93 | 4963.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 5404.50 | 4991.73 | 4977.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 15:15:00 | 5449.00 | 5472.95 | 5373.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:15:00 | 5426.00 | 5472.95 | 5373.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 5346.00 | 5471.02 | 5373.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 5346.00 | 5471.02 | 5373.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 5341.50 | 5469.73 | 5373.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 5341.50 | 5469.73 | 5373.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 5584.00 | 5639.25 | 5556.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:30:00 | 5567.00 | 5639.25 | 5556.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 5582.50 | 5635.66 | 5572.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 11:45:00 | 5641.50 | 5623.89 | 5571.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 13:45:00 | 5632.00 | 5624.10 | 5571.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 5647.00 | 5624.30 | 5572.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 5514.50 | 5617.18 | 5572.37 | SL hit (close<static) qty=1.00 sl=5528.00 alert=retest2 |

### Cycle 2 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 5369.50 | 5627.10 | 5628.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 5344.00 | 5560.11 | 5587.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 11:15:00 | 5450.50 | 5443.31 | 5515.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-25 12:00:00 | 5450.50 | 5443.31 | 5515.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 5440.00 | 5366.93 | 5450.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 15:00:00 | 5440.00 | 5366.93 | 5450.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 5463.00 | 5369.09 | 5447.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:00:00 | 5463.00 | 5369.09 | 5447.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 5474.00 | 5370.13 | 5448.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 11:00:00 | 5474.00 | 5370.13 | 5448.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 5478.00 | 5456.39 | 5480.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 13:15:00 | 5486.50 | 5456.39 | 5480.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 5509.50 | 5456.92 | 5480.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:00:00 | 5509.50 | 5456.92 | 5480.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 5502.00 | 5457.37 | 5480.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 5502.00 | 5457.37 | 5480.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 5539.00 | 5458.18 | 5480.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 5559.50 | 5458.18 | 5480.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 5455.50 | 5459.47 | 5481.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:45:00 | 5460.00 | 5459.47 | 5481.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 5419.50 | 5454.55 | 5478.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 5419.50 | 5454.55 | 5478.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 5509.50 | 5426.88 | 5458.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 5509.50 | 5426.88 | 5458.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 5580.50 | 5428.41 | 5458.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 5580.50 | 5428.41 | 5458.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-12-22 11:45:00 | 5641.50 | 2025-12-26 09:15:00 | 5514.50 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-12-22 13:45:00 | 5632.00 | 2025-12-26 09:15:00 | 5514.50 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-12-23 09:45:00 | 5647.00 | 2025-12-26 09:15:00 | 5514.50 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2026-01-06 12:45:00 | 5632.00 | 2026-02-01 14:15:00 | 5560.50 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-01-22 09:15:00 | 5703.50 | 2026-02-01 14:15:00 | 5560.50 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2026-01-28 13:00:00 | 5669.50 | 2026-02-01 14:15:00 | 5560.50 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-01-29 11:30:00 | 5670.00 | 2026-02-01 14:15:00 | 5560.50 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2026-01-29 13:00:00 | 5664.50 | 2026-02-02 10:15:00 | 5461.50 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2026-02-09 10:15:00 | 5752.00 | 2026-02-13 13:15:00 | 5530.50 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2026-02-09 11:30:00 | 5756.50 | 2026-02-13 13:15:00 | 5530.50 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2026-02-09 12:00:00 | 5771.00 | 2026-02-13 13:15:00 | 5530.50 | STOP_HIT | 1.00 | -4.17% |
| BUY | retest2 | 2026-02-09 14:45:00 | 5757.00 | 2026-02-13 13:15:00 | 5530.50 | STOP_HIT | 1.00 | -3.93% |
