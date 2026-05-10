# Sammaan Capital Ltd. (SAMMAANCAP)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 148.78
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 4 |
| ALERT3 | 46 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 34 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 31
- **Target hits / Stop hits / Partials:** 1 / 34 / 4
- **Avg / median % per leg:** -1.18% / -1.56%
- **Sum % (uncompounded):** -45.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 1 | 8.3% | 1 | 11 | 0 | -1.72% | -20.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 1 | 8.3% | 1 | 11 | 0 | -1.72% | -20.7% |
| SELL (all) | 27 | 7 | 25.9% | 0 | 23 | 4 | -0.93% | -25.2% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 2.95% | 5.9% |
| SELL @ 3rd Alert (retest2) | 25 | 5 | 20.0% | 0 | 22 | 3 | -1.24% | -31.1% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 2.95% | 5.9% |
| retest2 (combined) | 37 | 6 | 16.2% | 1 | 33 | 3 | -1.40% | -51.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 13:15:00 | 127.25 | 122.69 | 122.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 128.06 | 122.83 | 122.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 12:15:00 | 123.83 | 124.63 | 123.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 12:15:00 | 123.83 | 124.63 | 123.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 123.83 | 124.63 | 123.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 123.75 | 124.63 | 123.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 123.40 | 124.62 | 123.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:45:00 | 123.40 | 124.62 | 123.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 123.75 | 124.61 | 123.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:30:00 | 123.65 | 124.61 | 123.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 123.14 | 124.59 | 123.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 123.45 | 124.59 | 123.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 122.73 | 124.57 | 123.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 123.10 | 124.57 | 123.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 121.63 | 124.55 | 123.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 121.63 | 124.55 | 123.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 123.35 | 123.94 | 123.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 124.25 | 123.92 | 123.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-25 11:15:00 | 136.68 | 124.28 | 123.69 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 12:15:00 | 124.62 | 129.90 | 127.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:00:00 | 124.00 | 129.84 | 127.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:45:00 | 123.99 | 129.78 | 127.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 122.86 | 129.71 | 126.98 | SL hit (close<static) qty=1.00 sl=123.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 122.86 | 129.71 | 126.98 | SL hit (close<static) qty=1.00 sl=123.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 122.86 | 129.71 | 126.98 | SL hit (close<static) qty=1.00 sl=123.01 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 125.50 | 128.90 | 126.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 125.07 | 128.90 | 126.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 126.61 | 128.85 | 126.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 126.61 | 128.85 | 126.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 126.35 | 128.83 | 126.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:45:00 | 125.99 | 128.83 | 126.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 123.19 | 128.77 | 126.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:45:00 | 122.99 | 128.77 | 126.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 131.25 | 128.77 | 126.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 10:30:00 | 132.55 | 128.79 | 126.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 12:15:00 | 132.54 | 128.80 | 126.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 13:45:00 | 131.92 | 128.87 | 126.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 125.33 | 128.80 | 126.94 | SL hit (close<static) qty=1.00 sl=126.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 125.33 | 128.80 | 126.94 | SL hit (close<static) qty=1.00 sl=126.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 125.33 | 128.80 | 126.94 | SL hit (close<static) qty=1.00 sl=126.56 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 132.40 | 128.31 | 126.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 128.26 | 130.30 | 128.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 128.26 | 130.30 | 128.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 126.68 | 130.27 | 128.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 126.68 | 130.27 | 128.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 126.78 | 130.23 | 128.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:30:00 | 127.11 | 130.23 | 128.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 126.44 | 130.16 | 128.20 | SL hit (close<static) qty=1.00 sl=126.56 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 127.70 | 130.05 | 128.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 127.70 | 130.05 | 128.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 127.34 | 130.02 | 128.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:00:00 | 127.34 | 130.02 | 128.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 128.14 | 129.15 | 127.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:30:00 | 127.23 | 129.15 | 127.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 127.00 | 129.13 | 127.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:00:00 | 127.00 | 129.13 | 127.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 128.79 | 129.13 | 127.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 11:15:00 | 129.73 | 129.06 | 127.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 125.69 | 129.02 | 127.90 | SL hit (close<static) qty=1.00 sl=126.94 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 116.72 | 126.94 | 126.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 11:15:00 | 115.56 | 126.83 | 126.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 125.38 | 124.53 | 125.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 125.38 | 124.53 | 125.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 125.38 | 124.53 | 125.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 125.38 | 124.53 | 125.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 125.66 | 124.54 | 125.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 125.66 | 124.54 | 125.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 124.45 | 124.54 | 125.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:30:00 | 123.80 | 124.52 | 125.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:00:00 | 122.89 | 124.51 | 125.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 117.61 | 123.76 | 125.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 11:15:00 | 116.75 | 123.63 | 124.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 124.28 | 123.41 | 124.83 | SL hit (close>ema200) qty=0.50 sl=123.41 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 124.28 | 123.41 | 124.83 | SL hit (close>ema200) qty=0.50 sl=123.41 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:30:00 | 123.61 | 123.50 | 124.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 12:15:00 | 125.91 | 123.55 | 124.83 | SL hit (close>static) qty=1.00 sl=125.87 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 12:15:00 | 138.44 | 126.01 | 125.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 139.38 | 127.20 | 126.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 13:15:00 | 171.00 | 174.36 | 162.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 14:00:00 | 171.00 | 174.36 | 162.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 158.38 | 174.20 | 162.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 158.38 | 174.20 | 162.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 158.48 | 174.05 | 162.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 154.25 | 174.05 | 162.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 164.08 | 173.82 | 162.67 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 12:15:00 | 149.13 | 157.24 | 157.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 147.09 | 156.11 | 156.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 149.30 | 149.12 | 152.25 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 15:15:00 | 148.51 | 149.12 | 152.23 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 151.81 | 149.18 | 152.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:45:00 | 152.35 | 149.18 | 152.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 141.08 | 148.70 | 151.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 147.17 | 144.31 | 147.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 147.17 | 144.31 | 147.96 | SL hit (close>ema200) qty=0.50 sl=144.31 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-01-29 13:45:00 | 147.70 | 144.31 | 147.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 147.44 | 144.34 | 147.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:30:00 | 147.87 | 144.34 | 147.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 148.40 | 144.38 | 147.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 145.00 | 144.38 | 147.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 149.38 | 144.47 | 147.96 | SL hit (close>static) qty=1.00 sl=148.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:45:00 | 146.13 | 144.85 | 148.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 12:15:00 | 148.61 | 144.91 | 147.82 | SL hit (close>static) qty=1.00 sl=148.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 146.52 | 145.00 | 147.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 10:15:00 | 147.10 | 145.02 | 147.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 148.85 | 145.06 | 147.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-04 10:15:00 | 148.85 | 145.06 | 147.83 | SL hit (close>static) qty=1.00 sl=148.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 10:15:00 | 148.85 | 145.06 | 147.83 | SL hit (close>static) qty=1.00 sl=148.60 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 148.85 | 145.06 | 147.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 149.60 | 145.11 | 147.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:00:00 | 149.60 | 145.11 | 147.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 149.10 | 145.20 | 147.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 149.10 | 145.20 | 147.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 149.79 | 145.25 | 147.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 148.75 | 145.25 | 147.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 12:15:00 | 148.86 | 145.42 | 147.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 14:15:00 | 149.00 | 145.50 | 147.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 15:15:00 | 148.50 | 145.53 | 147.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 147.92 | 145.67 | 147.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:30:00 | 148.41 | 145.67 | 147.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 147.35 | 145.69 | 147.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 15:15:00 | 147.00 | 145.71 | 147.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:30:00 | 146.41 | 145.73 | 147.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 148.50 | 145.76 | 147.76 | SL hit (close>static) qty=1.00 sl=148.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 148.50 | 145.76 | 147.76 | SL hit (close>static) qty=1.00 sl=148.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 10:15:00 | 146.87 | 145.90 | 147.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:30:00 | 146.72 | 145.93 | 147.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 147.30 | 145.83 | 147.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 148.22 | 145.83 | 147.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 146.71 | 145.84 | 147.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 147.66 | 145.86 | 147.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 147.35 | 145.88 | 147.56 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 149.25 | 145.99 | 147.57 | SL hit (close>static) qty=1.00 sl=148.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 149.25 | 145.99 | 147.57 | SL hit (close>static) qty=1.00 sl=148.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 150.63 | 146.22 | 147.63 | SL hit (close>static) qty=1.00 sl=149.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 150.63 | 146.22 | 147.63 | SL hit (close>static) qty=1.00 sl=149.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 150.63 | 146.22 | 147.63 | SL hit (close>static) qty=1.00 sl=149.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 150.63 | 146.22 | 147.63 | SL hit (close>static) qty=1.00 sl=149.90 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 13:15:00 | 154.32 | 148.80 | 148.78 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 141.00 | 148.72 | 148.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 138.82 | 147.75 | 148.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 09:15:00 | 147.25 | 147.07 | 147.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 147.25 | 147.07 | 147.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 147.25 | 147.07 | 147.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 142.40 | 147.03 | 147.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 135.28 | 144.24 | 146.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 153.70 | 143.03 | 145.32 | SL hit (close>ema200) qty=0.50 sl=143.03 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:45:00 | 143.60 | 143.20 | 145.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 12:15:00 | 150.37 | 143.53 | 145.47 | SL hit (close>static) qty=1.00 sl=149.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 142.82 | 144.42 | 145.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 150.03 | 144.83 | 145.85 | SL hit (close>static) qty=1.00 sl=149.20 alert=retest2 |

### Cycle 7 — BUY (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 14:15:00 | 154.10 | 146.78 | 146.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 156.65 | 146.95 | 146.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 10:15:00 | 148.76 | 148.77 | 147.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 11:00:00 | 148.76 | 148.77 | 147.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 146.14 | 148.76 | 147.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 146.14 | 148.76 | 147.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 147.06 | 148.74 | 147.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 147.51 | 148.74 | 147.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 12:45:00 | 147.57 | 148.71 | 147.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 13:15:00 | 147.60 | 148.71 | 147.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 145.63 | 148.61 | 147.82 | SL hit (close<static) qty=1.00 sl=145.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 145.63 | 148.61 | 147.82 | SL hit (close<static) qty=1.00 sl=145.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 145.63 | 148.61 | 147.82 | SL hit (close<static) qty=1.00 sl=145.68 alert=retest2 |

### Cycle 8 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 140.90 | 147.13 | 147.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 139.20 | 147.05 | 147.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 148.65 | 146.91 | 147.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 148.65 | 146.91 | 147.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 148.65 | 146.91 | 147.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 148.65 | 146.91 | 147.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 147.41 | 146.91 | 147.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 146.55 | 146.92 | 147.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:45:00 | 146.50 | 146.92 | 147.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 146.00 | 146.93 | 147.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:45:00 | 146.49 | 146.91 | 147.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 146.70 | 146.91 | 147.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:30:00 | 148.20 | 146.91 | 147.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 146.59 | 146.85 | 147.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 13:15:00 | 148.83 | 146.90 | 147.02 | SL hit (close>static) qty=1.00 sl=148.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 13:15:00 | 148.83 | 146.90 | 147.02 | SL hit (close>static) qty=1.00 sl=148.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 13:15:00 | 148.83 | 146.90 | 147.02 | SL hit (close>static) qty=1.00 sl=148.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 13:15:00 | 148.83 | 146.90 | 147.02 | SL hit (close>static) qty=1.00 sl=148.68 alert=retest2 |

### Cycle 9 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 151.74 | 147.18 | 147.16 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-24 09:15:00 | 124.25 | 2025-06-25 11:15:00 | 136.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-04 12:15:00 | 124.62 | 2025-07-04 14:15:00 | 122.86 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-07-04 13:00:00 | 124.00 | 2025-07-04 14:15:00 | 122.86 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-07-04 13:45:00 | 123.99 | 2025-07-04 14:15:00 | 122.86 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-07-10 10:30:00 | 132.55 | 2025-07-14 10:15:00 | 125.33 | STOP_HIT | 1.00 | -5.45% |
| BUY | retest2 | 2025-07-10 12:15:00 | 132.54 | 2025-07-14 10:15:00 | 125.33 | STOP_HIT | 1.00 | -5.44% |
| BUY | retest2 | 2025-07-10 13:45:00 | 131.92 | 2025-07-14 10:15:00 | 125.33 | STOP_HIT | 1.00 | -5.00% |
| BUY | retest2 | 2025-07-17 09:15:00 | 132.40 | 2025-07-25 13:15:00 | 126.44 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest2 | 2025-08-01 11:15:00 | 129.73 | 2025-08-01 14:15:00 | 125.69 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-08-21 13:30:00 | 123.80 | 2025-08-28 09:15:00 | 117.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 15:00:00 | 122.89 | 2025-08-28 11:15:00 | 116.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 13:30:00 | 123.80 | 2025-08-29 10:15:00 | 124.28 | STOP_HIT | 0.50 | -0.39% |
| SELL | retest2 | 2025-08-21 15:00:00 | 122.89 | 2025-08-29 10:15:00 | 124.28 | STOP_HIT | 0.50 | -1.13% |
| SELL | retest2 | 2025-09-01 10:30:00 | 123.61 | 2025-09-01 12:15:00 | 125.91 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest1 | 2026-01-02 15:15:00 | 148.51 | 2026-01-12 11:15:00 | 141.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-02 15:15:00 | 148.51 | 2026-01-29 13:15:00 | 147.17 | STOP_HIT | 0.50 | 0.90% |
| SELL | retest2 | 2026-01-30 09:15:00 | 145.00 | 2026-01-30 10:15:00 | 149.38 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2026-02-01 11:45:00 | 146.13 | 2026-02-03 12:15:00 | 148.61 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-02-04 09:15:00 | 146.52 | 2026-02-04 10:15:00 | 148.85 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-02-04 10:15:00 | 147.10 | 2026-02-04 10:15:00 | 148.85 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-02-05 09:15:00 | 148.75 | 2026-02-11 10:15:00 | 148.50 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2026-02-09 12:15:00 | 148.86 | 2026-02-11 10:15:00 | 148.50 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2026-02-09 14:15:00 | 149.00 | 2026-02-18 09:15:00 | 149.25 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2026-02-09 15:15:00 | 148.50 | 2026-02-18 09:15:00 | 149.25 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-02-10 15:15:00 | 147.00 | 2026-02-19 10:15:00 | 150.63 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-02-11 09:30:00 | 146.41 | 2026-02-19 10:15:00 | 150.63 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2026-02-12 10:15:00 | 146.87 | 2026-02-19 10:15:00 | 150.63 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-02-12 14:30:00 | 146.72 | 2026-02-19 10:15:00 | 150.63 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2026-03-12 09:15:00 | 142.40 | 2026-03-23 09:15:00 | 135.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 09:15:00 | 142.40 | 2026-03-25 09:15:00 | 153.70 | STOP_HIT | 0.50 | -7.94% |
| SELL | retest2 | 2026-03-25 12:45:00 | 143.60 | 2026-03-27 12:15:00 | 150.37 | STOP_HIT | 1.00 | -4.71% |
| SELL | retest2 | 2026-04-02 09:15:00 | 142.82 | 2026-04-08 09:15:00 | 150.03 | STOP_HIT | 1.00 | -5.05% |
| BUY | retest2 | 2026-04-22 11:15:00 | 147.51 | 2026-04-23 10:15:00 | 145.63 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-04-22 12:45:00 | 147.57 | 2026-04-23 10:15:00 | 145.63 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-04-22 13:15:00 | 147.60 | 2026-04-23 10:15:00 | 145.63 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-05-04 13:15:00 | 146.55 | 2026-05-06 13:15:00 | 148.83 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-05-04 14:45:00 | 146.50 | 2026-05-06 13:15:00 | 148.83 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-05-05 09:15:00 | 146.00 | 2026-05-06 13:15:00 | 148.83 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-05-05 11:45:00 | 146.49 | 2026-05-06 13:15:00 | 148.83 | STOP_HIT | 1.00 | -1.60% |
