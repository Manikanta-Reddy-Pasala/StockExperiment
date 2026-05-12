# Vedanta Ltd. (VEDL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 297.00
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
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 26 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 21
- **Target hits / Stop hits / Partials:** 1 / 25 / 0
- **Avg / median % per leg:** -1.78% / -1.74%
- **Sum % (uncompounded):** -46.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 4 | 26.7% | 1 | 14 | 0 | -2.03% | -30.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 4 | 26.7% | 1 | 14 | 0 | -2.03% | -30.4% |
| SELL (all) | 11 | 1 | 9.1% | 0 | 11 | 0 | -1.44% | -15.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 1 | 9.1% | 0 | 11 | 0 | -1.44% | -15.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 26 | 5 | 19.2% | 1 | 25 | 0 | -1.78% | -46.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 168.61 | 161.11 | 161.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 169.98 | 162.02 | 161.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 161.01 | 162.32 | 161.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 161.01 | 162.32 | 161.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 161.01 | 162.32 | 161.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 13:15:00 | 162.60 | 162.29 | 161.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 163.99 | 162.28 | 161.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:00:00 | 162.60 | 162.29 | 161.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 12:30:00 | 163.16 | 162.30 | 161.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 162.57 | 162.30 | 161.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:45:00 | 162.12 | 162.30 | 161.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 162.62 | 162.31 | 161.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 161.91 | 162.31 | 161.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 165.32 | 166.99 | 164.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 164.01 | 166.99 | 164.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 164.70 | 166.97 | 164.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 164.70 | 166.97 | 164.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 164.76 | 166.95 | 164.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 164.42 | 166.95 | 164.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 165.58 | 166.93 | 164.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 166.63 | 166.93 | 164.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 12:45:00 | 166.35 | 166.94 | 164.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 166.37 | 166.96 | 165.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 11:15:00 | 163.50 | 169.29 | 166.94 | SL hit (close<static) qty=1.00 sl=163.93 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 161.46 | 166.20 | 166.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 161.24 | 165.51 | 165.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 166.03 | 165.03 | 165.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 166.03 | 165.03 | 165.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 166.03 | 165.03 | 165.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:45:00 | 165.97 | 165.03 | 165.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 165.82 | 165.04 | 165.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:15:00 | 165.24 | 165.04 | 165.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 09:15:00 | 167.81 | 164.70 | 165.35 | SL hit (close>static) qty=1.00 sl=166.59 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 173.07 | 165.27 | 165.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 12:15:00 | 175.49 | 165.46 | 165.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 167.08 | 167.46 | 166.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 14:00:00 | 167.08 | 167.46 | 166.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 255.36 | 259.89 | 249.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 257.51 | 259.84 | 249.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 256.12 | 259.43 | 249.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 256.12 | 259.39 | 249.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 241.80 | 258.80 | 249.59 | SL hit (close<static) qty=1.00 sl=248.75 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-02 13:15:00 | 162.60 | 2025-07-09 11:15:00 | 163.50 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2025-06-03 09:15:00 | 163.99 | 2025-07-09 11:15:00 | 163.50 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-06-03 10:00:00 | 162.60 | 2025-07-09 11:15:00 | 163.50 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2025-06-03 12:30:00 | 163.16 | 2025-07-17 12:15:00 | 166.31 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest2 | 2025-06-20 10:15:00 | 166.63 | 2025-07-25 11:15:00 | 166.40 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-06-20 12:45:00 | 166.35 | 2025-07-28 12:15:00 | 163.18 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-06-26 09:15:00 | 166.37 | 2025-07-31 15:15:00 | 159.04 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2025-07-11 09:30:00 | 166.40 | 2025-07-31 15:15:00 | 159.04 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2025-07-17 09:15:00 | 168.03 | 2025-07-31 15:15:00 | 159.04 | STOP_HIT | 1.00 | -5.35% |
| BUY | retest2 | 2025-07-21 09:15:00 | 168.91 | 2025-07-31 15:15:00 | 159.04 | STOP_HIT | 1.00 | -5.84% |
| SELL | retest2 | 2025-08-13 11:15:00 | 165.24 | 2025-08-19 09:15:00 | 167.81 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-08-20 12:30:00 | 165.00 | 2025-08-20 14:15:00 | 166.78 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-08-26 09:15:00 | 161.95 | 2025-09-05 13:15:00 | 165.17 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-09-04 11:30:00 | 165.43 | 2025-09-05 13:15:00 | 165.17 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-09-04 14:15:00 | 164.08 | 2025-09-05 13:15:00 | 165.17 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-09-05 10:00:00 | 164.23 | 2025-09-05 13:15:00 | 165.17 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-09-05 10:45:00 | 164.23 | 2025-09-05 14:15:00 | 167.04 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-09-05 11:15:00 | 164.19 | 2025-09-05 14:15:00 | 167.04 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-09-11 12:30:00 | 163.16 | 2025-09-12 09:15:00 | 166.76 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-11 13:00:00 | 163.16 | 2025-09-12 09:15:00 | 166.76 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-11 13:30:00 | 163.11 | 2025-09-12 09:15:00 | 166.76 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2026-03-19 10:30:00 | 257.51 | 2026-03-23 09:15:00 | 241.80 | STOP_HIT | 1.00 | -6.10% |
| BUY | retest2 | 2026-03-20 09:15:00 | 256.12 | 2026-03-23 09:15:00 | 241.80 | STOP_HIT | 1.00 | -5.59% |
| BUY | retest2 | 2026-03-20 10:15:00 | 256.12 | 2026-03-23 09:15:00 | 241.80 | STOP_HIT | 1.00 | -5.59% |
| BUY | retest2 | 2026-04-01 09:45:00 | 256.31 | 2026-04-02 09:15:00 | 246.57 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2026-04-02 11:30:00 | 249.10 | 2026-04-08 09:15:00 | 274.01 | TARGET_HIT | 1.00 | 10.00% |
