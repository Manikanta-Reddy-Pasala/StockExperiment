# NTPC Green Energy Ltd. (NTPCGREEN)

## Backtest Summary

- **Window:** 2024-11-27 09:15:00 → 2026-05-08 15:15:00 (2501 bars)
- **Last close:** 107.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 7 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 8
- **Target hits / Stop hits / Partials:** 1 / 9 / 1
- **Avg / median % per leg:** 0.31% / -1.74%
- **Sum % (uncompounded):** 3.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 10 | 2 | 20.0% | 0 | 9 | 1 | -0.66% | -6.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 0 | 9 | 1 | -0.66% | -6.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 3 | 27.3% | 1 | 9 | 1 | 0.31% | 3.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 15:15:00 | 116.50 | 105.34 | 105.31 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 11:15:00 | 105.01 | 107.31 | 107.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 12:15:00 | 104.69 | 107.29 | 107.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 13:15:00 | 105.00 | 104.86 | 105.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 13:45:00 | 105.00 | 104.86 | 105.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 105.07 | 104.20 | 105.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:45:00 | 105.30 | 104.20 | 105.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 105.20 | 104.22 | 105.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 13:30:00 | 104.91 | 104.32 | 105.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 14:30:00 | 104.95 | 104.33 | 105.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 15:00:00 | 104.96 | 104.33 | 105.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:45:00 | 104.90 | 104.36 | 105.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 104.65 | 104.31 | 105.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:30:00 | 104.06 | 104.30 | 105.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:00:00 | 104.60 | 103.99 | 104.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 11:30:00 | 104.52 | 104.01 | 104.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 14:15:00 | 103.80 | 104.02 | 104.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 106.74 | 104.04 | 104.88 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 106.74 | 104.04 | 104.88 | SL hit (close>static) qty=1.00 sl=105.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 106.74 | 104.04 | 104.88 | SL hit (close>static) qty=1.00 sl=105.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 106.74 | 104.04 | 104.88 | SL hit (close>static) qty=1.00 sl=105.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 106.74 | 104.04 | 104.88 | SL hit (close>static) qty=1.00 sl=105.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 106.74 | 104.04 | 104.88 | SL hit (close>static) qty=1.00 sl=105.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 106.74 | 104.04 | 104.88 | SL hit (close>static) qty=1.00 sl=105.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 106.74 | 104.04 | 104.88 | SL hit (close>static) qty=1.00 sl=105.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 106.74 | 104.04 | 104.88 | SL hit (close>static) qty=1.00 sl=105.95 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 106.92 | 104.04 | 104.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 105.90 | 104.06 | 104.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:45:00 | 105.73 | 104.08 | 104.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 100.44 | 103.76 | 104.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 100.58 | 100.51 | 102.09 | SL hit (close>ema200) qty=0.50 sl=100.51 alert=retest2 |

### Cycle 3 — BUY (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 11:15:00 | 100.48 | 91.55 | 91.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 10:15:00 | 101.43 | 92.03 | 91.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 14:15:00 | 92.84 | 93.42 | 92.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 14:15:00 | 92.84 | 93.42 | 92.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 92.84 | 93.42 | 92.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 92.84 | 93.42 | 92.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 92.50 | 93.41 | 92.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 96.61 | 93.41 | 92.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 106.27 | 95.70 | 94.08 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-03 13:30:00 | 104.91 | 2025-09-16 09:15:00 | 106.74 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-09-03 14:30:00 | 104.95 | 2025-09-16 09:15:00 | 106.74 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-09-03 15:00:00 | 104.96 | 2025-09-16 09:15:00 | 106.74 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-09-04 11:45:00 | 104.90 | 2025-09-16 09:15:00 | 106.74 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-09-08 10:30:00 | 104.06 | 2025-09-16 09:15:00 | 106.74 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-09-15 10:00:00 | 104.60 | 2025-09-16 09:15:00 | 106.74 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-09-15 11:30:00 | 104.52 | 2025-09-16 09:15:00 | 106.74 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-09-15 14:15:00 | 103.80 | 2025-09-16 09:15:00 | 106.74 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-09-16 11:45:00 | 105.73 | 2025-09-26 09:15:00 | 100.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 11:45:00 | 105.73 | 2025-10-21 13:15:00 | 100.58 | STOP_HIT | 0.50 | 4.87% |
| BUY | retest2 | 2026-04-01 09:15:00 | 96.61 | 2026-04-15 09:15:00 | 106.27 | TARGET_HIT | 1.00 | 10.00% |
