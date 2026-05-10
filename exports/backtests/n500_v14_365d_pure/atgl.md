# Adani Total Gas Ltd. (ATGL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 632.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 20
- **Target hits / Stop hits / Partials:** 0 / 21 / 1
- **Avg / median % per leg:** -0.90% / -1.28%
- **Sum % (uncompounded):** -19.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.34% | -21.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.34% | -21.4% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 5 | 1 | 0.25% | 1.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 2 | 33.3% | 0 | 5 | 1 | 0.25% | 1.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 2 | 9.1% | 0 | 21 | 1 | -0.90% | -19.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 658.25 | 620.53 | 620.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 673.10 | 621.84 | 621.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 662.45 | 667.52 | 651.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 647.90 | 666.89 | 651.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 647.90 | 666.89 | 651.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 646.60 | 666.89 | 651.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 657.00 | 666.79 | 651.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 661.70 | 653.68 | 648.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:30:00 | 661.70 | 658.27 | 651.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:45:00 | 661.70 | 659.07 | 652.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 661.40 | 658.97 | 652.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 652.15 | 658.88 | 652.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 652.15 | 658.88 | 652.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 650.50 | 658.80 | 652.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:45:00 | 651.70 | 658.80 | 652.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 651.85 | 658.73 | 652.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:45:00 | 654.15 | 658.62 | 652.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 655.75 | 658.56 | 652.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 12:45:00 | 653.80 | 658.41 | 652.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 13:15:00 | 653.25 | 658.41 | 652.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 652.40 | 658.35 | 652.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:00:00 | 652.40 | 658.35 | 652.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 651.25 | 658.28 | 652.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 652.00 | 658.28 | 652.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 651.20 | 658.21 | 652.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 657.65 | 658.21 | 652.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 12:15:00 | 653.30 | 658.03 | 652.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:00:00 | 651.60 | 657.90 | 652.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 653.15 | 657.78 | 652.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 650.60 | 657.71 | 652.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 647.70 | 657.61 | 652.33 | SL hit (close<static) qty=1.00 sl=647.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 647.70 | 657.61 | 652.33 | SL hit (close<static) qty=1.00 sl=647.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 647.70 | 657.61 | 652.33 | SL hit (close<static) qty=1.00 sl=647.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 647.70 | 657.61 | 652.33 | SL hit (close<static) qty=1.00 sl=647.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 647.70 | 657.61 | 652.33 | SL hit (close<static) qty=1.00 sl=650.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 647.70 | 657.61 | 652.33 | SL hit (close<static) qty=1.00 sl=650.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 647.70 | 657.61 | 652.33 | SL hit (close<static) qty=1.00 sl=650.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 647.70 | 657.61 | 652.33 | SL hit (close<static) qty=1.00 sl=650.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 647.70 | 657.61 | 652.33 | SL hit (close<static) qty=1.00 sl=650.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 647.70 | 657.61 | 652.33 | SL hit (close<static) qty=1.00 sl=650.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 647.70 | 657.61 | 652.33 | SL hit (close<static) qty=1.00 sl=650.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 647.70 | 657.61 | 652.33 | SL hit (close<static) qty=1.00 sl=650.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:00:00 | 655.80 | 655.64 | 651.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:00:00 | 656.40 | 656.13 | 652.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 13:30:00 | 656.40 | 656.12 | 652.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:45:00 | 657.50 | 656.09 | 652.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 651.30 | 656.16 | 652.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:15:00 | 650.70 | 656.16 | 652.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 651.90 | 656.11 | 652.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:30:00 | 650.80 | 656.11 | 652.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 651.10 | 656.02 | 652.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:30:00 | 650.65 | 656.02 | 652.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 648.00 | 655.94 | 652.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 648.00 | 655.94 | 652.57 | SL hit (close<static) qty=1.00 sl=649.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 648.00 | 655.94 | 652.57 | SL hit (close<static) qty=1.00 sl=649.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 648.00 | 655.94 | 652.57 | SL hit (close<static) qty=1.00 sl=649.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 648.00 | 655.94 | 652.57 | SL hit (close<static) qty=1.00 sl=649.15 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 648.00 | 655.94 | 652.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 651.65 | 655.77 | 652.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 649.25 | 655.77 | 652.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 653.25 | 655.70 | 652.53 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 09:15:00 | 629.00 | 649.76 | 649.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 15:15:00 | 625.00 | 648.54 | 649.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 13:15:00 | 625.20 | 623.03 | 633.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-18 14:00:00 | 625.20 | 623.03 | 633.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 631.55 | 623.58 | 633.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:15:00 | 632.55 | 623.58 | 633.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 630.30 | 623.65 | 633.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 11:00:00 | 627.20 | 624.04 | 633.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 10:30:00 | 627.10 | 623.54 | 632.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:00:00 | 625.60 | 623.54 | 632.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 636.40 | 624.00 | 632.25 | SL hit (close>static) qty=1.00 sl=635.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 636.40 | 624.00 | 632.25 | SL hit (close>static) qty=1.00 sl=635.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 636.40 | 624.00 | 632.25 | SL hit (close>static) qty=1.00 sl=635.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:30:00 | 626.65 | 624.00 | 632.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 635.95 | 624.12 | 632.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 635.95 | 624.12 | 632.27 | SL hit (close>static) qty=1.00 sl=635.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-08-26 11:15:00 | 637.20 | 624.12 | 632.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 614.70 | 610.53 | 621.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 624.30 | 610.53 | 621.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 661.35 | 609.80 | 619.21 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 742.50 | 628.59 | 628.13 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 621.00 | 631.05 | 631.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 15:15:00 | 618.55 | 628.83 | 629.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 658.10 | 628.56 | 629.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 658.10 | 628.56 | 629.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 658.10 | 628.56 | 629.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 658.10 | 628.56 | 629.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 651.00 | 628.79 | 629.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:00:00 | 646.80 | 629.18 | 630.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 614.46 | 628.80 | 629.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 629.55 | 625.69 | 627.98 | SL hit (close>ema200) qty=0.50 sl=625.69 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 600.10 | 542.94 | 542.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 617.10 | 543.68 | 543.16 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-27 09:15:00 | 661.70 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-07-02 13:30:00 | 661.70 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-07-04 14:45:00 | 661.70 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-07-08 09:15:00 | 661.40 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-07-08 14:45:00 | 654.15 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-09 09:15:00 | 655.75 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-07-09 12:45:00 | 653.80 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-07-09 13:15:00 | 653.25 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-07-10 09:15:00 | 657.65 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-07-10 12:15:00 | 653.30 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-07-10 14:00:00 | 651.60 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-07-11 09:15:00 | 653.15 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-16 14:00:00 | 655.80 | 2025-07-22 14:15:00 | 648.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-07-18 11:00:00 | 656.40 | 2025-07-22 14:15:00 | 648.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-18 13:30:00 | 656.40 | 2025-07-22 14:15:00 | 648.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-21 09:45:00 | 657.50 | 2025-07-22 14:15:00 | 648.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-08-21 11:00:00 | 627.20 | 2025-08-26 09:15:00 | 636.40 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-08-25 10:30:00 | 627.10 | 2025-08-26 09:15:00 | 636.40 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-08-25 11:00:00 | 625.60 | 2025-08-26 09:15:00 | 636.40 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-08-26 09:30:00 | 626.65 | 2025-08-26 10:15:00 | 635.95 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-10-29 14:00:00 | 646.80 | 2025-11-07 09:15:00 | 614.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 14:00:00 | 646.80 | 2025-11-12 09:15:00 | 629.55 | STOP_HIT | 0.50 | 2.67% |
