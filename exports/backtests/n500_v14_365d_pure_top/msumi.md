# Motherson Sumi Wiring India Ltd. (MSUMI)

## Backtest Summary

- **Window:** 2025-01-15 09:15:00 → 2026-05-08 15:15:00 (2263 bars)
- **Last close:** 42.56
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
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 1 |
| TARGET_HIT | 5 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 14
- **Target hits / Stop hits / Partials:** 5 / 16 / 1
- **Avg / median % per leg:** 1.24% / -1.64%
- **Sum % (uncompounded):** 27.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 6 | 30.0% | 5 | 15 | 0 | 1.03% | 20.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 6 | 30.0% | 5 | 15 | 0 | 1.03% | 20.7% |
| SELL (all) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.29% | 6.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.29% | 6.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 8 | 36.4% | 5 | 16 | 1 | 1.24% | 27.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 37.27 | 39.60 | 39.61 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 12:15:00 | 42.05 | 39.62 | 39.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 42.60 | 39.71 | 39.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 13:15:00 | 45.82 | 45.82 | 43.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-30 13:45:00 | 45.86 | 45.82 | 43.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 44.54 | 45.68 | 44.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:15:00 | 45.00 | 45.68 | 44.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:45:00 | 44.89 | 45.66 | 44.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 13:00:00 | 44.90 | 45.65 | 44.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 15:00:00 | 46.53 | 45.50 | 44.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 45.30 | 46.40 | 45.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 11:00:00 | 45.77 | 46.39 | 45.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 11:30:00 | 45.91 | 46.39 | 45.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-18 09:15:00 | 49.38 | 46.69 | 45.68 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-11-18 09:15:00 | 49.39 | 46.69 | 45.68 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 45.85 | 47.13 | 46.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:45:00 | 45.85 | 47.04 | 46.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 46.18 | 46.97 | 46.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 46.26 | 46.97 | 46.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 46.13 | 46.96 | 46.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 46.09 | 46.96 | 46.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 46.20 | 46.95 | 46.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 13:30:00 | 46.41 | 46.95 | 46.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 15:00:00 | 46.43 | 46.91 | 46.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 45.96 | 46.89 | 46.16 | SL hit (close<static) qty=1.00 sl=46.06 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 45.96 | 46.89 | 46.16 | SL hit (close<static) qty=1.00 sl=46.06 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 44.84 | 46.76 | 46.14 | SL hit (close<static) qty=1.00 sl=45.12 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 44.84 | 46.76 | 46.14 | SL hit (close<static) qty=1.00 sl=45.12 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 44.84 | 46.76 | 46.14 | SL hit (close<static) qty=1.00 sl=45.12 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 44.84 | 46.76 | 46.14 | SL hit (close<static) qty=1.00 sl=45.12 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:45:00 | 46.55 | 46.22 | 45.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 11:15:00 | 45.97 | 46.22 | 45.95 | SL hit (close<static) qty=1.00 sl=46.06 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 11:30:00 | 46.32 | 46.22 | 45.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 46.14 | 46.22 | 45.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:30:00 | 46.20 | 46.22 | 45.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 45.67 | 46.21 | 45.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 45.67 | 46.21 | 45.95 | SL hit (close<static) qty=1.00 sl=46.06 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-12 14:00:00 | 45.67 | 46.21 | 45.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 45.90 | 46.21 | 45.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:30:00 | 45.74 | 46.21 | 45.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 46.01 | 46.21 | 45.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 45.74 | 46.21 | 45.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 46.42 | 46.21 | 45.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 46.52 | 46.21 | 45.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 45.44 | 46.20 | 45.96 | SL hit (close<static) qty=1.00 sl=45.64 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:30:00 | 46.53 | 46.06 | 45.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-19 14:15:00 | 49.50 | 46.07 | 45.92 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 14:15:00 | 46.50 | 46.13 | 45.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 46.50 | 46.13 | 45.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 45.50 | 46.12 | 45.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 45.50 | 46.12 | 45.95 | SL hit (close<static) qty=1.00 sl=45.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 45.50 | 46.12 | 45.95 | SL hit (close<static) qty=1.00 sl=45.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 45.50 | 46.12 | 45.95 | SL hit (close<static) qty=1.00 sl=45.64 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-23 12:00:00 | 45.50 | 46.12 | 45.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 45.38 | 46.12 | 45.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 45.38 | 46.12 | 45.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 45.45 | 46.09 | 45.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 45.45 | 46.09 | 45.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 46.17 | 45.94 | 45.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:45:00 | 46.69 | 45.95 | 45.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-05 13:15:00 | 51.18 | 46.76 | 46.32 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-01-05 13:15:00 | 51.36 | 46.76 | 46.32 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 12:00:00 | 46.70 | 47.54 | 46.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 13:30:00 | 46.53 | 47.52 | 46.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 10:15:00 | 46.84 | 47.50 | 46.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 45.99 | 47.48 | 46.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 45.99 | 47.48 | 46.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 45.73 | 47.46 | 46.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 45.73 | 47.46 | 46.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-13 12:15:00 | 45.62 | 47.45 | 46.80 | SL hit (close<static) qty=1.00 sl=45.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 12:15:00 | 45.62 | 47.45 | 46.80 | SL hit (close<static) qty=1.00 sl=45.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 12:15:00 | 45.62 | 47.45 | 46.80 | SL hit (close<static) qty=1.00 sl=45.68 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 43.32 | 46.32 | 46.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 42.94 | 46.29 | 46.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 44.95 | 44.91 | 45.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 12:00:00 | 44.95 | 44.91 | 45.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 45.53 | 44.92 | 45.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:00:00 | 45.53 | 44.92 | 45.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 45.35 | 44.93 | 45.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 45.35 | 44.93 | 45.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 45.35 | 44.93 | 45.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 44.69 | 44.93 | 45.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 15:15:00 | 42.46 | 44.28 | 45.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 43.98 | 43.88 | 44.68 | SL hit (close>ema200) qty=0.50 sl=43.88 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-13 10:15:00 | 45.00 | 2025-11-18 09:15:00 | 49.38 | TARGET_HIT | 1.00 | 9.73% |
| BUY | retest2 | 2025-10-13 11:45:00 | 44.89 | 2025-11-18 09:15:00 | 49.39 | TARGET_HIT | 1.00 | 10.02% |
| BUY | retest2 | 2025-10-13 13:00:00 | 44.90 | 2025-12-03 10:15:00 | 45.96 | STOP_HIT | 1.00 | 2.36% |
| BUY | retest2 | 2025-10-15 15:00:00 | 46.53 | 2025-12-03 10:15:00 | 45.96 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-11-11 11:00:00 | 45.77 | 2025-12-05 09:15:00 | 44.84 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-11-11 11:30:00 | 45.91 | 2025-12-05 09:15:00 | 44.84 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-11-27 09:30:00 | 45.85 | 2025-12-05 09:15:00 | 44.84 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-11-28 11:45:00 | 45.85 | 2025-12-05 09:15:00 | 44.84 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-12-01 13:30:00 | 46.41 | 2025-12-12 11:15:00 | 45.97 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-12-02 15:00:00 | 46.43 | 2025-12-12 13:15:00 | 45.67 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-12-12 09:45:00 | 46.55 | 2025-12-16 11:15:00 | 45.44 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-12-12 11:30:00 | 46.32 | 2025-12-19 14:15:00 | 49.50 | TARGET_HIT | 1.00 | 6.87% |
| BUY | retest2 | 2025-12-16 09:15:00 | 46.52 | 2025-12-23 11:15:00 | 45.50 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-12-19 13:30:00 | 46.53 | 2025-12-23 11:15:00 | 45.50 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-12-22 14:15:00 | 46.50 | 2025-12-23 11:15:00 | 45.50 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-12-23 09:45:00 | 46.50 | 2026-01-05 13:15:00 | 51.18 | TARGET_HIT | 1.00 | 10.07% |
| BUY | retest2 | 2025-12-30 10:45:00 | 46.69 | 2026-01-05 13:15:00 | 51.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-12 12:00:00 | 46.70 | 2026-01-13 12:15:00 | 45.62 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2026-01-12 13:30:00 | 46.53 | 2026-01-13 12:15:00 | 45.62 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-01-13 10:15:00 | 46.84 | 2026-01-13 12:15:00 | 45.62 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2026-02-05 09:15:00 | 44.69 | 2026-02-13 15:15:00 | 42.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-05 09:15:00 | 44.69 | 2026-02-23 09:15:00 | 43.98 | STOP_HIT | 0.50 | 1.59% |
