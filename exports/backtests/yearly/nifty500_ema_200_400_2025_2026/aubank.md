# AU Small Finance Bank Ltd. (AUBANK)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3164 bars)
- **Last close:** 1051.00
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
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 0 / 10
- **Target hits / Stop hits / Partials:** 0 / 10 / 0
- **Avg / median % per leg:** -1.44% / -1.38%
- **Sum % (uncompounded):** -14.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.53% | -9.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.53% | -9.2% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.31% | -5.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.31% | -5.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.44% | -14.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 703.30 | 742.58 | 742.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 700.30 | 742.16 | 742.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 09:15:00 | 729.45 | 722.31 | 729.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 10:00:00 | 729.45 | 722.31 | 729.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 732.05 | 722.41 | 729.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 732.05 | 722.41 | 729.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 728.85 | 722.48 | 729.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:30:00 | 725.35 | 728.13 | 731.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:45:00 | 727.00 | 728.14 | 731.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 13:30:00 | 727.15 | 728.14 | 731.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 14:45:00 | 723.95 | 728.10 | 731.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 735.35 | 728.18 | 731.59 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 735.35 | 728.18 | 731.59 | SL hit (close>static) qty=1.00 sl=733.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 13:15:00 | 768.20 | 734.57 | 734.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 14:15:00 | 772.40 | 742.13 | 738.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 11:15:00 | 977.20 | 978.09 | 937.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 11:45:00 | 978.05 | 978.09 | 937.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 950.05 | 984.00 | 950.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:00:00 | 950.05 | 984.00 | 950.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 950.55 | 983.67 | 950.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:30:00 | 950.70 | 983.67 | 950.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 946.95 | 983.30 | 950.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:00:00 | 946.95 | 983.30 | 950.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 962.65 | 983.10 | 950.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 964.55 | 982.87 | 950.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 11:15:00 | 964.65 | 982.47 | 950.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 968.30 | 981.43 | 951.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 12:45:00 | 965.10 | 981.03 | 951.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-08 09:30:00 | 758.00 | 2025-08-08 13:15:00 | 744.60 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-08-18 09:45:00 | 760.85 | 2025-08-22 13:15:00 | 744.60 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-08-19 09:45:00 | 757.70 | 2025-08-22 13:15:00 | 744.60 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-08-20 12:15:00 | 758.05 | 2025-08-22 13:15:00 | 744.60 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-08-26 12:30:00 | 748.80 | 2025-08-26 13:15:00 | 742.50 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-08-26 13:15:00 | 749.50 | 2025-08-26 13:15:00 | 742.50 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-01 11:30:00 | 725.35 | 2025-10-03 09:15:00 | 735.35 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-10-01 12:45:00 | 727.00 | 2025-10-03 09:15:00 | 735.35 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-01 13:30:00 | 727.15 | 2025-10-03 09:15:00 | 735.35 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-10-01 14:45:00 | 723.95 | 2025-10-03 09:15:00 | 735.35 | STOP_HIT | 1.00 | -1.57% |
