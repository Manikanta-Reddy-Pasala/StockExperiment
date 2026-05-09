# HDFCLIFE (HDFCLIFE)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 619.60
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
| ALERT2_SKIP | 2 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 13
- **Target hits / Stop hits / Partials:** 1 / 17 / 5
- **Avg / median % per leg:** 1.48% / -0.61%
- **Sum % (uncompounded):** 33.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.26% | -8.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.26% | -8.8% |
| SELL (all) | 16 | 10 | 62.5% | 1 | 10 | 5 | 2.68% | 42.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 10 | 62.5% | 1 | 10 | 5 | 2.68% | 42.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 23 | 10 | 43.5% | 1 | 17 | 5 | 1.48% | 34.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 785.45 | 759.54 | 759.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 786.70 | 759.81 | 759.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 760.75 | 761.97 | 760.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 760.75 | 761.97 | 760.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 760.75 | 761.97 | 760.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 761.10 | 761.97 | 760.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 761.10 | 761.96 | 760.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 760.85 | 761.96 | 760.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 761.25 | 761.96 | 760.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:45:00 | 758.55 | 761.96 | 760.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 767.55 | 762.01 | 760.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 769.30 | 760.64 | 760.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 15:00:00 | 768.40 | 760.90 | 760.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 754.75 | 761.58 | 760.79 | SL hit (close<static) qty=1.00 sl=760.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 754.75 | 761.58 | 760.79 | SL hit (close<static) qty=1.00 sl=760.75 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 11:00:00 | 769.85 | 761.62 | 760.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:45:00 | 768.35 | 762.18 | 761.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 756.50 | 764.40 | 762.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 756.50 | 764.40 | 762.48 | SL hit (close<static) qty=1.00 sl=760.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 756.50 | 764.40 | 762.48 | SL hit (close<static) qty=1.00 sl=760.75 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 756.50 | 764.40 | 762.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 753.75 | 764.30 | 762.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:15:00 | 750.80 | 764.30 | 762.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 764.95 | 763.04 | 761.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 764.95 | 763.04 | 761.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 761.65 | 763.11 | 761.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 13:00:00 | 761.65 | 763.11 | 761.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 13:15:00 | 761.30 | 763.09 | 761.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:00:00 | 761.30 | 763.09 | 761.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 761.45 | 763.07 | 761.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 10:30:00 | 763.30 | 763.01 | 761.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:45:00 | 763.00 | 762.97 | 761.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 763.15 | 763.03 | 762.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 758.35 | 762.95 | 761.99 | SL hit (close<static) qty=1.00 sl=759.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 758.35 | 762.95 | 761.99 | SL hit (close<static) qty=1.00 sl=759.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 758.35 | 762.95 | 761.99 | SL hit (close<static) qty=1.00 sl=759.95 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 747.05 | 761.02 | 761.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 743.40 | 760.71 | 760.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 762.00 | 758.12 | 759.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 762.00 | 758.12 | 759.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 762.00 | 758.12 | 759.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 762.00 | 758.12 | 759.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 762.95 | 758.17 | 759.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 762.35 | 758.17 | 759.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 759.85 | 758.29 | 759.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 759.85 | 758.29 | 759.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 760.60 | 758.31 | 759.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 775.20 | 758.31 | 759.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 773.20 | 758.46 | 759.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 754.15 | 760.24 | 760.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:15:00 | 752.60 | 760.17 | 760.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 751.95 | 759.47 | 760.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 751.15 | 758.93 | 759.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 09:15:00 | 716.44 | 749.68 | 754.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 09:15:00 | 714.97 | 749.68 | 754.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 09:15:00 | 714.35 | 749.68 | 754.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 14:15:00 | 713.59 | 748.09 | 753.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 723.30 | 721.38 | 734.33 | SL hit (close>ema200) qty=0.50 sl=721.38 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 723.30 | 721.38 | 734.33 | SL hit (close>ema200) qty=0.50 sl=721.38 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 723.30 | 721.38 | 734.33 | SL hit (close>ema200) qty=0.50 sl=721.38 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 723.30 | 721.38 | 734.33 | SL hit (close>ema200) qty=0.50 sl=721.38 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 732.60 | 721.76 | 734.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 732.60 | 721.76 | 734.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 729.15 | 721.83 | 734.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:45:00 | 731.15 | 721.83 | 734.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 733.10 | 722.04 | 734.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:15:00 | 733.50 | 722.04 | 734.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 732.20 | 722.15 | 734.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 727.40 | 722.27 | 734.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:45:00 | 730.75 | 722.35 | 734.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 13:00:00 | 731.00 | 722.65 | 734.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 14:00:00 | 730.75 | 722.73 | 734.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 740.80 | 723.02 | 734.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 740.80 | 723.02 | 734.06 | SL hit (close>static) qty=1.00 sl=735.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 740.80 | 723.02 | 734.06 | SL hit (close>static) qty=1.00 sl=735.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 740.80 | 723.02 | 734.06 | SL hit (close>static) qty=1.00 sl=735.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 740.80 | 723.02 | 734.06 | SL hit (close>static) qty=1.00 sl=735.15 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 740.80 | 723.02 | 734.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 743.35 | 723.22 | 734.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:30:00 | 742.15 | 723.22 | 734.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 734.00 | 724.93 | 734.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 734.00 | 724.93 | 734.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 740.00 | 725.08 | 734.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 730.05 | 726.62 | 734.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 693.55 | 724.67 | 733.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 657.04 | 715.57 | 727.43 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-11-11 11:15:00 | 754.90 | 2025-11-11 11:15:00 | 759.15 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-11-20 09:30:00 | 750.85 | 2025-11-20 12:15:00 | 759.75 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-12-05 12:15:00 | 769.30 | 2025-12-09 09:15:00 | 754.75 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-12-05 15:00:00 | 768.40 | 2025-12-09 09:15:00 | 754.75 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-12-10 11:00:00 | 769.85 | 2025-12-17 09:15:00 | 756.50 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-12-11 10:45:00 | 768.35 | 2025-12-17 09:15:00 | 756.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-12-23 10:30:00 | 763.30 | 2025-12-24 13:15:00 | 758.35 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-12-23 14:45:00 | 763.00 | 2025-12-24 13:15:00 | 758.35 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-12-24 12:00:00 | 763.15 | 2025-12-24 13:15:00 | 758.35 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-01-09 09:15:00 | 754.15 | 2026-01-23 09:15:00 | 716.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 11:15:00 | 752.60 | 2026-01-23 09:15:00 | 714.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 09:15:00 | 751.95 | 2026-01-23 09:15:00 | 714.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 751.15 | 2026-01-23 14:15:00 | 713.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 09:15:00 | 754.15 | 2026-02-18 12:15:00 | 723.30 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2026-01-09 11:15:00 | 752.60 | 2026-02-18 12:15:00 | 723.30 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2026-01-13 09:15:00 | 751.95 | 2026-02-18 12:15:00 | 723.30 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2026-01-13 14:00:00 | 751.15 | 2026-02-18 12:15:00 | 723.30 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2026-02-20 09:15:00 | 727.40 | 2026-02-23 09:15:00 | 740.80 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-02-20 09:45:00 | 730.75 | 2026-02-23 09:15:00 | 740.80 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-02-20 13:00:00 | 731.00 | 2026-02-23 09:15:00 | 740.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-02-20 14:00:00 | 730.75 | 2026-02-23 09:15:00 | 740.80 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-02-27 09:15:00 | 730.05 | 2026-03-04 09:15:00 | 693.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 730.05 | 2026-03-09 09:15:00 | 657.04 | TARGET_HIT | 0.50 | 10.00% |
