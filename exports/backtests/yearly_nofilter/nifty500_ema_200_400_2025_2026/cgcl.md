# Capri Global Capital Ltd. (CGCL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 197.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 3 |
| TARGET_HIT | 4 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 4
- **Target hits / Stop hits / Partials:** 4 / 6 / 2
- **Avg / median % per leg:** 4.07% / 5.00%
- **Sum % (uncompounded):** 48.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| SELL (all) | 10 | 6 | 60.0% | 2 | 6 | 2 | 2.88% | 28.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 6 | 60.0% | 2 | 6 | 2 | 2.88% | 28.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 8 | 66.7% | 4 | 6 | 2 | 4.07% | 48.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 174.90 | 166.50 | 166.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 177.95 | 166.61 | 166.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 14:15:00 | 167.90 | 168.17 | 167.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 15:00:00 | 167.90 | 168.17 | 167.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 167.99 | 168.17 | 167.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 167.12 | 168.17 | 167.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 166.05 | 168.15 | 167.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:30:00 | 166.18 | 168.15 | 167.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 166.24 | 168.13 | 167.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 165.63 | 168.13 | 167.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 166.77 | 168.11 | 167.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 166.95 | 168.11 | 167.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 167.77 | 168.10 | 167.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:30:00 | 168.64 | 167.73 | 167.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 169.28 | 167.88 | 167.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-24 09:15:00 | 185.50 | 174.02 | 171.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 180.01 | 190.91 | 190.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 179.29 | 188.36 | 189.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 185.50 | 183.80 | 186.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 185.50 | 183.80 | 186.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 185.50 | 183.80 | 186.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 185.98 | 183.80 | 186.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 185.77 | 183.82 | 186.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 185.77 | 183.82 | 186.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 186.57 | 183.85 | 186.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 186.57 | 183.85 | 186.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 185.97 | 183.87 | 186.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:15:00 | 185.40 | 183.92 | 186.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:00:00 | 185.37 | 183.97 | 186.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 188.51 | 184.04 | 186.44 | SL hit (close>static) qty=1.00 sl=187.20 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 182.00 | 173.11 | 173.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 185.50 | 173.23 | 173.17 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-08 11:30:00 | 168.64 | 2025-07-24 09:15:00 | 185.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-10 09:15:00 | 169.28 | 2025-07-24 09:15:00 | 186.21 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-02 13:15:00 | 185.40 | 2026-01-05 11:15:00 | 188.51 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-01-05 10:00:00 | 185.37 | 2026-01-05 11:15:00 | 188.51 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-01-07 09:15:00 | 185.61 | 2026-01-07 14:15:00 | 187.48 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-01-07 10:00:00 | 185.62 | 2026-01-07 14:15:00 | 187.48 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-07 15:15:00 | 184.90 | 2026-01-20 09:15:00 | 175.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 10:45:00 | 185.60 | 2026-01-20 09:15:00 | 176.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 15:15:00 | 184.90 | 2026-01-22 10:15:00 | 167.04 | TARGET_HIT | 0.50 | 9.66% |
| SELL | retest2 | 2026-01-09 10:45:00 | 185.60 | 2026-01-23 14:15:00 | 166.41 | TARGET_HIT | 0.50 | 10.34% |
| SELL | retest2 | 2026-04-15 09:45:00 | 185.89 | 2026-04-17 15:15:00 | 182.00 | STOP_HIT | 1.00 | 2.09% |
| SELL | retest2 | 2026-04-15 13:00:00 | 185.92 | 2026-04-17 15:15:00 | 182.00 | STOP_HIT | 1.00 | 2.11% |
