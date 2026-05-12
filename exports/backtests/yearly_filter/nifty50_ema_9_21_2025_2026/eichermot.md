# EICHERMOT (EICHERMOT)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 7309.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 79 |
| ALERT1 | 50 |
| ALERT2 | 51 |
| ALERT2_SKIP | 34 |
| ALERT3 | 153 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 46 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 17 / 36
- **Target hits / Stop hits / Partials:** 1 / 48 / 4
- **Avg / median % per leg:** 0.27% / -0.66%
- **Sum % (uncompounded):** 14.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 9 | 40.9% | 1 | 21 | 0 | 0.30% | 6.7% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | 1.02% | 2.0% |
| BUY @ 3rd Alert (retest2) | 20 | 8 | 40.0% | 1 | 19 | 0 | 0.23% | 4.7% |
| SELL (all) | 31 | 8 | 25.8% | 0 | 27 | 4 | 0.25% | 7.7% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.01% | -3.0% |
| SELL @ 3rd Alert (retest2) | 28 | 8 | 28.6% | 0 | 24 | 4 | 0.38% | 10.7% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 5 | 0 | -0.20% | -1.0% |
| retest2 (combined) | 48 | 16 | 33.3% | 1 | 43 | 4 | 0.32% | 15.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 5500.00 | 5409.23 | 5406.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 5513.50 | 5444.13 | 5423.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 5439.50 | 5465.43 | 5443.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 10:15:00 | 5439.50 | 5465.43 | 5443.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 5439.50 | 5465.43 | 5443.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:00:00 | 5439.50 | 5465.43 | 5443.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 5468.00 | 5465.94 | 5445.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:45:00 | 5450.50 | 5465.94 | 5445.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 5414.50 | 5455.66 | 5442.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 5414.50 | 5455.66 | 5442.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 5407.00 | 5445.92 | 5439.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 5407.00 | 5445.92 | 5439.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 5440.00 | 5437.99 | 5436.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:45:00 | 5414.50 | 5437.99 | 5436.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 5451.50 | 5440.70 | 5438.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 5434.50 | 5440.70 | 5438.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 5453.50 | 5443.26 | 5439.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 5430.00 | 5443.26 | 5439.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 5445.00 | 5443.61 | 5440.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:00:00 | 5445.00 | 5443.61 | 5440.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 5445.50 | 5443.98 | 5440.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 15:00:00 | 5445.50 | 5443.98 | 5440.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 5447.00 | 5444.59 | 5441.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:15:00 | 5473.00 | 5444.59 | 5441.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 5428.00 | 5441.27 | 5439.94 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 10:15:00 | 5404.00 | 5433.82 | 5436.67 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 5479.50 | 5440.58 | 5439.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 5514.00 | 5455.27 | 5445.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 14:15:00 | 5531.50 | 5555.19 | 5525.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 15:00:00 | 5531.50 | 5555.19 | 5525.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 5528.00 | 5549.75 | 5525.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 5474.00 | 5549.75 | 5525.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 5470.50 | 5533.90 | 5520.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 5447.00 | 5533.90 | 5520.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 5495.50 | 5526.22 | 5518.46 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 5460.50 | 5513.08 | 5513.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 5445.50 | 5499.56 | 5507.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 5455.00 | 5450.01 | 5474.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 5455.00 | 5450.01 | 5474.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 5455.00 | 5450.01 | 5474.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 5456.50 | 5450.01 | 5474.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 5417.50 | 5394.62 | 5413.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 5417.50 | 5394.62 | 5413.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 5412.00 | 5398.09 | 5413.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 5417.00 | 5398.09 | 5413.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 5403.00 | 5399.08 | 5412.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 5411.00 | 5399.08 | 5412.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 5407.50 | 5400.76 | 5412.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:30:00 | 5419.50 | 5400.76 | 5412.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 5457.50 | 5411.45 | 5413.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:45:00 | 5462.50 | 5411.45 | 5413.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 5435.50 | 5416.26 | 5415.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 11:15:00 | 5469.00 | 5426.81 | 5420.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 13:15:00 | 5423.00 | 5427.36 | 5421.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 13:15:00 | 5423.00 | 5427.36 | 5421.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 5423.00 | 5427.36 | 5421.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:00:00 | 5423.00 | 5427.36 | 5421.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 5429.00 | 5427.69 | 5422.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:30:00 | 5421.50 | 5427.69 | 5422.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 5430.00 | 5428.15 | 5423.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 5407.50 | 5428.15 | 5423.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 5389.50 | 5420.42 | 5420.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 5382.00 | 5420.42 | 5420.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 5413.00 | 5418.94 | 5419.44 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 5440.00 | 5423.15 | 5421.31 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 5400.00 | 5418.90 | 5419.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 5357.00 | 5401.33 | 5411.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 13:15:00 | 5352.50 | 5340.20 | 5363.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:30:00 | 5351.50 | 5340.20 | 5363.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 5387.50 | 5349.66 | 5365.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 5387.50 | 5349.66 | 5365.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 5385.00 | 5356.73 | 5367.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 5396.00 | 5356.73 | 5367.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 5376.00 | 5364.22 | 5368.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 5376.00 | 5364.22 | 5368.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 5345.50 | 5360.48 | 5366.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 12:15:00 | 5344.50 | 5360.48 | 5366.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 10:00:00 | 5337.00 | 5345.61 | 5355.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 10:45:00 | 5344.50 | 5348.19 | 5356.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 5341.00 | 5348.19 | 5356.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 5351.00 | 5348.75 | 5355.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:45:00 | 5352.50 | 5348.75 | 5355.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 5348.00 | 5348.60 | 5354.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:45:00 | 5332.00 | 5352.08 | 5354.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:00:00 | 5317.50 | 5345.17 | 5351.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 09:15:00 | 5380.00 | 5349.59 | 5349.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 5380.00 | 5349.59 | 5349.45 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 10:15:00 | 5334.50 | 5346.57 | 5348.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 11:15:00 | 5318.50 | 5340.95 | 5345.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 11:15:00 | 5327.00 | 5320.95 | 5331.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 11:15:00 | 5327.00 | 5320.95 | 5331.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 5327.00 | 5320.95 | 5331.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 5327.00 | 5320.95 | 5331.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 5306.00 | 5317.08 | 5326.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:45:00 | 5316.00 | 5317.08 | 5326.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 5350.50 | 5321.52 | 5326.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 5360.00 | 5321.52 | 5326.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 5377.00 | 5332.61 | 5330.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 14:15:00 | 5396.50 | 5356.80 | 5343.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 5385.00 | 5385.03 | 5368.87 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:30:00 | 5399.50 | 5390.43 | 5372.79 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 5379.00 | 5393.45 | 5381.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-10 14:15:00 | 5379.00 | 5393.45 | 5381.84 | SL hit (close<ema400) qty=1.00 sl=5381.84 alert=retest1 |

### Cycle 12 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 5355.50 | 5375.86 | 5378.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 5313.00 | 5353.04 | 5364.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 5310.50 | 5301.68 | 5326.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 5310.50 | 5301.68 | 5326.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 5324.50 | 5306.24 | 5326.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 5324.50 | 5306.24 | 5326.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 5323.00 | 5309.59 | 5326.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 5340.50 | 5309.59 | 5326.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 5342.00 | 5316.07 | 5327.76 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 5372.00 | 5340.89 | 5337.30 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 5322.50 | 5338.58 | 5339.40 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-06-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 15:15:00 | 5344.00 | 5339.83 | 5339.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 09:15:00 | 5449.50 | 5361.77 | 5349.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 15:15:00 | 5385.00 | 5390.18 | 5372.64 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:15:00 | 5437.00 | 5390.18 | 5372.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 5578.00 | 5592.05 | 5572.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:45:00 | 5571.00 | 5592.05 | 5572.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 5576.50 | 5588.94 | 5573.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 5571.00 | 5588.94 | 5573.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 5569.00 | 5584.95 | 5572.77 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 5569.00 | 5584.95 | 5572.77 | SL hit (close<ema400) qty=1.00 sl=5572.77 alert=retest1 |

### Cycle 16 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 5638.50 | 5691.77 | 5697.57 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 5716.00 | 5680.59 | 5678.70 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 5662.50 | 5683.18 | 5683.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 15:15:00 | 5649.00 | 5665.17 | 5673.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 5591.00 | 5562.59 | 5592.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 5591.00 | 5562.59 | 5592.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 5591.00 | 5562.59 | 5592.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:15:00 | 5614.50 | 5562.59 | 5592.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 5618.50 | 5573.77 | 5595.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:45:00 | 5613.50 | 5573.77 | 5595.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 5632.50 | 5585.52 | 5598.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 5630.50 | 5585.52 | 5598.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 5585.00 | 5587.73 | 5597.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 5575.00 | 5589.81 | 5596.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:15:00 | 5573.00 | 5584.92 | 5593.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 5621.50 | 5592.45 | 5594.47 | SL hit (close>static) qty=1.00 sl=5601.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 14:15:00 | 5627.50 | 5599.46 | 5597.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 5649.50 | 5611.63 | 5603.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 10:15:00 | 5637.00 | 5638.93 | 5625.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 10:15:00 | 5637.00 | 5638.93 | 5625.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 5637.00 | 5638.93 | 5625.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 5642.00 | 5638.93 | 5625.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 5635.00 | 5638.14 | 5626.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 5613.50 | 5638.14 | 5626.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 5632.00 | 5637.50 | 5629.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:45:00 | 5621.00 | 5637.50 | 5629.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 5620.50 | 5634.10 | 5628.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 5567.00 | 5634.10 | 5628.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 5613.50 | 5629.98 | 5626.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 5580.50 | 5629.98 | 5626.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 5607.50 | 5625.48 | 5625.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:45:00 | 5597.50 | 5625.48 | 5625.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 5578.00 | 5615.99 | 5620.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 5569.00 | 5606.59 | 5616.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 5485.00 | 5457.70 | 5487.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 5485.00 | 5457.70 | 5487.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 5485.00 | 5457.70 | 5487.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:45:00 | 5490.00 | 5457.70 | 5487.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 5465.00 | 5459.16 | 5485.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:30:00 | 5487.00 | 5459.16 | 5485.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 5381.50 | 5448.09 | 5469.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 14:15:00 | 5372.50 | 5407.63 | 5428.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 15:15:00 | 5380.00 | 5404.61 | 5425.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 5474.00 | 5427.20 | 5425.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 5474.00 | 5427.20 | 5425.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 5479.50 | 5437.66 | 5430.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 13:15:00 | 5483.50 | 5484.13 | 5460.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 13:15:00 | 5483.50 | 5484.13 | 5460.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 5483.50 | 5484.13 | 5460.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:30:00 | 5498.50 | 5484.13 | 5460.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 5490.00 | 5485.52 | 5465.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 5415.00 | 5485.52 | 5465.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 5438.00 | 5476.02 | 5462.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:00:00 | 5458.00 | 5472.41 | 5462.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-18 09:15:00 | 6003.80 | 5793.53 | 5746.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 15:15:00 | 6795.00 | 6814.57 | 6817.02 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 6868.00 | 6825.26 | 6821.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 12:15:00 | 6877.50 | 6846.11 | 6832.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 6897.00 | 6901.67 | 6877.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 6897.00 | 6901.67 | 6877.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 6900.00 | 6899.47 | 6880.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 6906.50 | 6899.47 | 6880.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 6905.00 | 6899.59 | 6888.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 6976.50 | 7013.63 | 7016.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 14:15:00 | 6976.50 | 7013.63 | 7016.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 10:15:00 | 6967.50 | 6993.25 | 7005.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 11:15:00 | 7007.00 | 6996.00 | 7005.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 11:15:00 | 7007.00 | 6996.00 | 7005.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 7007.00 | 6996.00 | 7005.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 7007.00 | 6996.00 | 7005.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 7021.50 | 7001.10 | 7006.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:00:00 | 7021.50 | 7001.10 | 7006.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 6999.00 | 7000.68 | 7006.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 6992.50 | 7001.54 | 7005.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 7032.50 | 7011.73 | 7009.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 7032.50 | 7011.73 | 7009.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 7042.50 | 7017.88 | 7012.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 15:15:00 | 6988.00 | 7016.85 | 7013.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 15:15:00 | 6988.00 | 7016.85 | 7013.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 6988.00 | 7016.85 | 7013.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 6875.50 | 7016.85 | 7013.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 09:15:00 | 6864.00 | 6986.28 | 7000.26 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 13:15:00 | 6983.00 | 6951.88 | 6949.36 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 6900.00 | 6952.08 | 6953.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 6890.50 | 6939.77 | 6947.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 6957.50 | 6929.91 | 6941.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 10:15:00 | 6957.50 | 6929.91 | 6941.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 6957.50 | 6929.91 | 6941.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 6957.50 | 6929.91 | 6941.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 6914.50 | 6926.83 | 6938.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:00:00 | 6901.00 | 6921.66 | 6935.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:30:00 | 6899.50 | 6915.73 | 6931.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 12:15:00 | 6970.00 | 6934.33 | 6932.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 6970.00 | 6934.33 | 6932.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 09:15:00 | 7005.50 | 6962.52 | 6947.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 6951.00 | 6960.21 | 6948.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 10:15:00 | 6951.00 | 6960.21 | 6948.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 6951.00 | 6960.21 | 6948.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 6951.00 | 6960.21 | 6948.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 6950.00 | 6958.17 | 6948.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:15:00 | 6949.50 | 6958.17 | 6948.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 6946.50 | 6955.84 | 6948.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:45:00 | 6939.00 | 6955.84 | 6948.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 6915.00 | 6947.67 | 6945.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 6915.00 | 6947.67 | 6945.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 14:15:00 | 6912.00 | 6940.54 | 6942.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 6828.00 | 6914.74 | 6929.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 6883.50 | 6882.17 | 6904.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 6883.50 | 6882.17 | 6904.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 6883.50 | 6882.17 | 6904.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:30:00 | 6893.50 | 6882.17 | 6904.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 6929.50 | 6890.17 | 6904.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 6924.00 | 6890.17 | 6904.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 6904.00 | 6892.93 | 6904.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 6926.50 | 6892.93 | 6904.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 6903.00 | 6894.95 | 6904.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 6927.00 | 6894.95 | 6904.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 6900.50 | 6896.06 | 6904.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 6900.50 | 6896.06 | 6904.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 6886.00 | 6894.05 | 6902.36 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 7008.50 | 6917.02 | 6910.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 7069.50 | 6987.47 | 6954.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 12:15:00 | 7035.50 | 7045.18 | 7015.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 13:00:00 | 7035.50 | 7045.18 | 7015.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 7032.00 | 7042.55 | 7016.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 7015.50 | 7042.55 | 7016.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 7011.50 | 7036.34 | 7016.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:45:00 | 7022.50 | 7036.34 | 7016.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 7012.00 | 7031.47 | 7016.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 7025.00 | 7029.46 | 7017.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 6971.00 | 7017.77 | 7013.53 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 6964.00 | 7007.01 | 7009.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 6957.00 | 6997.01 | 7004.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 6865.00 | 6860.27 | 6903.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 10:15:00 | 6879.50 | 6860.27 | 6903.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 6916.00 | 6876.95 | 6897.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 6915.00 | 6876.95 | 6897.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 6908.50 | 6883.26 | 6898.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 15:15:00 | 6905.00 | 6883.26 | 6898.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 6963.00 | 6902.69 | 6905.23 | SL hit (close>static) qty=1.00 sl=6926.50 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 6975.00 | 6917.15 | 6911.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 14:15:00 | 7010.00 | 6953.83 | 6932.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 6953.00 | 6969.36 | 6951.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 6953.00 | 6969.36 | 6951.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 6953.00 | 6969.36 | 6951.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 6953.00 | 6969.36 | 6951.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 6948.50 | 6965.19 | 6951.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 6948.50 | 6965.19 | 6951.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 6978.50 | 6967.85 | 6953.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 6975.00 | 6967.85 | 6953.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 6923.00 | 6958.88 | 6951.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 6923.00 | 6958.88 | 6951.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 6929.50 | 6953.00 | 6949.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 6906.00 | 6953.00 | 6949.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 6895.50 | 6941.50 | 6944.26 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 7049.50 | 6940.99 | 6939.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 11:15:00 | 7080.00 | 6985.83 | 6961.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 11:15:00 | 7009.00 | 7016.94 | 6992.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-03 11:45:00 | 7003.00 | 7016.94 | 6992.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 6971.00 | 7008.92 | 6998.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 6940.50 | 7008.92 | 6998.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 6932.00 | 6993.54 | 6992.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 6932.00 | 6993.54 | 6992.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 6959.00 | 6986.63 | 6989.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 6926.50 | 6966.79 | 6978.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 6822.00 | 6816.17 | 6867.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:30:00 | 6832.50 | 6816.17 | 6867.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 6858.00 | 6824.54 | 6866.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 6855.00 | 6824.54 | 6866.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 6882.50 | 6836.13 | 6868.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 6882.50 | 6836.13 | 6868.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 6885.00 | 6845.90 | 6869.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 6820.50 | 6845.90 | 6869.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 6798.00 | 6853.88 | 6861.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:30:00 | 6877.00 | 6840.13 | 6846.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:00:00 | 6863.00 | 6849.96 | 6850.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 10:15:00 | 6880.00 | 6855.97 | 6852.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 6880.00 | 6855.97 | 6852.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 6930.00 | 6877.46 | 6864.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 6886.00 | 6895.45 | 6880.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 14:15:00 | 6886.00 | 6895.45 | 6880.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 6886.00 | 6895.45 | 6880.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 6886.00 | 6895.45 | 6880.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 6796.50 | 6875.66 | 6872.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 6792.00 | 6875.66 | 6872.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 6746.00 | 6849.73 | 6861.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 11:15:00 | 6690.00 | 6799.98 | 6835.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 6739.00 | 6738.86 | 6786.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 09:30:00 | 6733.50 | 6738.86 | 6786.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 6781.00 | 6752.83 | 6785.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 6793.50 | 6752.83 | 6785.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 6788.00 | 6759.87 | 6785.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:30:00 | 6784.00 | 6759.87 | 6785.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 6792.50 | 6766.39 | 6785.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:00:00 | 6792.50 | 6766.39 | 6785.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 6800.00 | 6773.11 | 6787.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:30:00 | 6798.00 | 6773.11 | 6787.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 6809.00 | 6780.29 | 6789.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 6799.00 | 6780.29 | 6789.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 6755.50 | 6775.33 | 6786.14 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 12:15:00 | 6829.50 | 6795.51 | 6793.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 13:15:00 | 6831.50 | 6802.71 | 6796.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 15:15:00 | 6785.00 | 6803.05 | 6798.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 15:15:00 | 6785.00 | 6803.05 | 6798.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 6785.00 | 6803.05 | 6798.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 6799.00 | 6803.05 | 6798.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 6827.00 | 6807.84 | 6800.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 14:00:00 | 6865.00 | 6822.83 | 6810.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 10:15:00 | 7051.50 | 7166.11 | 7180.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 7051.50 | 7166.11 | 7180.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 7005.50 | 7133.99 | 7164.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 14:15:00 | 7058.50 | 7047.26 | 7082.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 15:00:00 | 7058.50 | 7047.26 | 7082.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 7128.00 | 7063.05 | 7083.47 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 7230.00 | 7112.66 | 7101.23 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 7069.50 | 7106.24 | 7107.27 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 09:15:00 | 7148.50 | 7104.88 | 7099.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 7211.50 | 7151.05 | 7126.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 7148.50 | 7168.30 | 7146.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 12:15:00 | 7148.50 | 7168.30 | 7146.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 7148.50 | 7168.30 | 7146.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 7148.50 | 7168.30 | 7146.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 7122.50 | 7159.14 | 7143.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 7123.00 | 7159.14 | 7143.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 7140.50 | 7155.41 | 7143.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 11:45:00 | 7158.00 | 7143.43 | 7140.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 7163.50 | 7143.72 | 7142.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 7125.50 | 7215.70 | 7221.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 7125.50 | 7215.70 | 7221.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 11:15:00 | 7104.00 | 7179.25 | 7202.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 7151.50 | 7092.62 | 7123.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 7151.50 | 7092.62 | 7123.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 7151.50 | 7092.62 | 7123.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 7053.50 | 7113.28 | 7125.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:00:00 | 7074.50 | 7093.91 | 7110.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 7184.00 | 7113.81 | 7115.87 | SL hit (close>static) qty=1.00 sl=7176.50 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 7182.50 | 7127.55 | 7121.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 7219.50 | 7175.59 | 7149.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 7290.00 | 7295.66 | 7258.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 7261.00 | 7295.66 | 7258.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 7270.00 | 7290.53 | 7259.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:45:00 | 7303.00 | 7294.43 | 7264.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 7353.50 | 7308.16 | 7284.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:45:00 | 7313.00 | 7313.20 | 7292.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:45:00 | 7306.50 | 7308.22 | 7301.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 7285.00 | 7303.57 | 7300.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:45:00 | 7281.50 | 7303.57 | 7300.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 7261.50 | 7295.16 | 7296.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 13:15:00 | 7261.50 | 7295.16 | 7296.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 7230.50 | 7274.09 | 7285.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 7244.50 | 7223.75 | 7249.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 7244.50 | 7223.75 | 7249.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 7244.50 | 7223.75 | 7249.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 7244.50 | 7223.75 | 7249.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 7260.50 | 7231.10 | 7250.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 7260.50 | 7231.10 | 7250.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 7271.00 | 7239.08 | 7252.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 7285.50 | 7239.08 | 7252.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 7317.50 | 7263.15 | 7261.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 7348.50 | 7301.60 | 7282.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 14:15:00 | 7332.00 | 7341.67 | 7321.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 14:15:00 | 7332.00 | 7341.67 | 7321.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 7332.00 | 7341.67 | 7321.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:45:00 | 7317.50 | 7341.67 | 7321.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 7464.00 | 7366.51 | 7336.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 7476.50 | 7366.51 | 7336.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:30:00 | 7469.00 | 7403.05 | 7358.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 13:45:00 | 7469.00 | 7425.75 | 7377.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 14:15:00 | 7481.50 | 7425.75 | 7377.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 7514.50 | 7555.98 | 7541.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:45:00 | 7524.00 | 7555.98 | 7541.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 7508.00 | 7546.38 | 7538.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:00:00 | 7508.00 | 7546.38 | 7538.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-09 15:15:00 | 7510.00 | 7533.29 | 7533.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 15:15:00 | 7510.00 | 7533.29 | 7533.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 7403.00 | 7507.23 | 7521.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 7439.50 | 7429.19 | 7469.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 15:00:00 | 7439.50 | 7429.19 | 7469.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 7454.00 | 7436.36 | 7466.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 7429.00 | 7431.49 | 7461.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 7435.50 | 7432.29 | 7458.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:30:00 | 7402.00 | 7423.93 | 7452.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 7057.55 | 7209.44 | 7270.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 7063.72 | 7209.44 | 7270.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 7031.90 | 7184.15 | 7253.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 7266.00 | 7163.76 | 7205.95 | SL hit (close>ema200) qty=0.50 sl=7163.76 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 7147.50 | 7084.54 | 7084.39 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 09:15:00 | 6880.00 | 7043.63 | 7065.81 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 7071.50 | 7044.23 | 7042.91 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 7028.00 | 7040.99 | 7041.55 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 7060.00 | 7043.83 | 7042.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 7090.00 | 7053.17 | 7047.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 6965.00 | 7083.65 | 7069.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 6965.00 | 7083.65 | 7069.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 6965.00 | 7083.65 | 7069.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 6965.00 | 7083.65 | 7069.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 6930.00 | 7052.92 | 7057.03 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 7180.00 | 7017.83 | 7002.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 7208.50 | 7055.97 | 7021.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 7211.50 | 7245.64 | 7182.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 7211.50 | 7245.64 | 7182.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 7211.50 | 7245.64 | 7182.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 7211.50 | 7245.64 | 7182.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 7211.50 | 7229.00 | 7189.90 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 7151.00 | 7179.60 | 7180.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-09 09:15:00 | 7135.50 | 7165.21 | 7173.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 7194.50 | 7171.07 | 7175.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 10:15:00 | 7194.50 | 7171.07 | 7175.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 7194.50 | 7171.07 | 7175.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:45:00 | 7200.00 | 7171.07 | 7175.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 7200.00 | 7176.85 | 7177.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:45:00 | 7202.00 | 7176.85 | 7177.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 7203.00 | 7182.08 | 7179.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 15:15:00 | 7209.00 | 7188.10 | 7182.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 11:15:00 | 8009.50 | 8021.67 | 7907.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-16 11:45:00 | 7999.50 | 8021.67 | 7907.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 7939.00 | 7985.67 | 7943.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 7939.00 | 7985.67 | 7943.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 8016.00 | 7991.74 | 7949.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 13:30:00 | 8023.50 | 7996.49 | 7955.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 8019.50 | 8005.79 | 7963.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 8020.00 | 7992.09 | 7975.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 7900.50 | 7965.13 | 7971.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 7900.50 | 7965.13 | 7971.79 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 8040.00 | 7974.95 | 7966.95 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 13:15:00 | 7937.50 | 7971.57 | 7976.15 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 8074.00 | 7989.72 | 7980.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 12:15:00 | 8102.00 | 8047.92 | 8018.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 8087.00 | 8108.96 | 8061.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 8087.00 | 8108.96 | 8061.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 8087.00 | 8108.96 | 8061.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:45:00 | 8094.50 | 8108.96 | 8061.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 8083.50 | 8103.87 | 8063.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:45:00 | 8101.00 | 8098.10 | 8064.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:45:00 | 8101.50 | 8094.02 | 8068.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 14:15:00 | 8005.00 | 8076.22 | 8062.45 | SL hit (close<static) qty=1.00 sl=8056.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 7930.00 | 8037.02 | 8046.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 7780.00 | 7956.25 | 8006.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 7696.00 | 7683.04 | 7781.61 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:15:00 | 7666.50 | 7683.04 | 7781.61 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 7657.00 | 7677.83 | 7770.28 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 13:45:00 | 7669.50 | 7670.99 | 7743.58 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 7725.00 | 7681.80 | 7741.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 7784.00 | 7681.80 | 7741.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 7742.00 | 7693.84 | 7741.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 15:15:00 | 7742.00 | 7693.84 | 7741.90 | SL hit (close>ema400) qty=1.00 sl=7741.90 alert=retest1 |

### Cycle 63 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 7019.00 | 6916.24 | 6911.32 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 6801.00 | 6923.79 | 6933.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 6745.00 | 6859.50 | 6879.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 6830.00 | 6757.84 | 6803.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 6830.00 | 6757.84 | 6803.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 6830.00 | 6757.84 | 6803.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 6859.50 | 6757.84 | 6803.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 6822.00 | 6770.67 | 6805.62 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 6930.00 | 6841.65 | 6832.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 7050.00 | 6890.85 | 6857.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 6828.00 | 6952.10 | 6919.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 6828.00 | 6952.10 | 6919.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 6828.00 | 6952.10 | 6919.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 6828.00 | 6952.10 | 6919.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 6768.00 | 6888.86 | 6894.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 6747.50 | 6827.65 | 6860.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 6805.50 | 6701.23 | 6763.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 6805.50 | 6701.23 | 6763.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 6805.50 | 6701.23 | 6763.16 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 6825.00 | 6794.66 | 6791.52 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 6631.00 | 6765.98 | 6779.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 10:15:00 | 6599.00 | 6658.43 | 6703.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 6598.50 | 6558.48 | 6604.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 14:15:00 | 6598.50 | 6558.48 | 6604.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 6598.50 | 6558.48 | 6604.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 15:00:00 | 6598.50 | 6558.48 | 6604.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 6648.00 | 6576.39 | 6608.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:15:00 | 6931.50 | 6576.39 | 6608.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 7000.00 | 6661.11 | 6644.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 7049.00 | 6738.69 | 6681.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 7142.50 | 7295.36 | 7174.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 7142.50 | 7295.36 | 7174.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 7142.50 | 7295.36 | 7174.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:15:00 | 7136.00 | 7295.36 | 7174.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 7115.00 | 7259.28 | 7169.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 7115.00 | 7259.28 | 7169.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 7050.00 | 7132.80 | 7133.69 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 7165.00 | 7123.85 | 7118.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 14:15:00 | 7192.00 | 7151.89 | 7134.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 14:15:00 | 7239.00 | 7251.92 | 7217.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 15:00:00 | 7239.00 | 7251.92 | 7217.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 7220.00 | 7245.54 | 7217.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 7184.00 | 7245.54 | 7217.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 7165.00 | 7229.43 | 7212.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 7165.00 | 7229.43 | 7212.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 7140.00 | 7211.54 | 7206.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 11:00:00 | 7140.00 | 7211.54 | 7206.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 12:15:00 | 7191.50 | 7202.49 | 7202.78 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 7240.50 | 7210.09 | 7206.21 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 7093.50 | 7185.21 | 7196.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 12:15:00 | 7072.00 | 7150.22 | 7177.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 7120.00 | 7118.06 | 7151.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 11:15:00 | 7149.00 | 7123.28 | 7147.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 7149.00 | 7123.28 | 7147.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:30:00 | 7142.00 | 7123.28 | 7147.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 7105.50 | 7119.72 | 7143.85 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 7193.50 | 7146.52 | 7145.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 7225.00 | 7172.73 | 7159.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 7065.50 | 7155.73 | 7154.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 7065.50 | 7155.73 | 7154.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 7065.50 | 7155.73 | 7154.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 7065.50 | 7155.73 | 7154.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 7088.00 | 7142.18 | 7148.08 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 7229.50 | 7153.31 | 7147.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 7269.00 | 7176.45 | 7158.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 7188.00 | 7198.05 | 7174.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 14:15:00 | 7188.00 | 7198.05 | 7174.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 7188.00 | 7198.05 | 7174.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:30:00 | 7192.00 | 7198.05 | 7174.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 6982.50 | 7153.98 | 7158.85 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 7215.00 | 7139.75 | 7136.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 7308.50 | 7189.80 | 7161.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 10:15:00 | 7220.00 | 7282.65 | 7248.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 10:15:00 | 7220.00 | 7282.65 | 7248.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 7220.00 | 7282.65 | 7248.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 7220.00 | 7282.65 | 7248.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 7247.00 | 7275.52 | 7248.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:00:00 | 7268.00 | 7274.01 | 7250.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 7279.00 | 7265.81 | 7248.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-30 12:15:00 | 5344.50 | 2025-06-04 09:15:00 | 5380.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-06-02 10:00:00 | 5337.00 | 2025-06-04 09:15:00 | 5380.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-02 10:45:00 | 5344.50 | 2025-06-04 09:15:00 | 5380.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-06-02 11:15:00 | 5341.00 | 2025-06-04 09:15:00 | 5380.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-06-03 09:45:00 | 5332.00 | 2025-06-04 09:15:00 | 5380.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-06-03 11:00:00 | 5317.50 | 2025-06-04 09:15:00 | 5380.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest1 | 2025-06-10 09:30:00 | 5399.50 | 2025-06-10 14:15:00 | 5379.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-06-19 09:15:00 | 5437.00 | 2025-06-26 09:15:00 | 5569.00 | STOP_HIT | 1.00 | 2.43% |
| SELL | retest2 | 2025-07-16 09:15:00 | 5575.00 | 2025-07-16 13:15:00 | 5621.50 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-07-16 11:15:00 | 5573.00 | 2025-07-16 13:15:00 | 5621.50 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-07-28 14:15:00 | 5372.50 | 2025-07-29 14:15:00 | 5474.00 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-07-28 15:15:00 | 5380.00 | 2025-07-29 14:15:00 | 5474.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-07-31 11:00:00 | 5458.00 | 2025-08-18 09:15:00 | 6003.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-18 09:15:00 | 6906.50 | 2025-09-29 14:15:00 | 6976.50 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest2 | 2025-09-18 14:15:00 | 6905.00 | 2025-09-29 14:15:00 | 6976.50 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2025-10-01 09:45:00 | 6992.50 | 2025-10-01 11:15:00 | 7032.50 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-10-09 13:00:00 | 6901.00 | 2025-10-10 12:15:00 | 6970.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-10-09 13:30:00 | 6899.50 | 2025-10-10 12:15:00 | 6970.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-10-27 15:15:00 | 6905.00 | 2025-10-28 09:15:00 | 6963.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-10 09:15:00 | 6820.50 | 2025-11-12 10:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-11 09:15:00 | 6798.00 | 2025-11-12 10:15:00 | 6880.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-11-11 14:30:00 | 6877.00 | 2025-11-12 10:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-11-12 10:00:00 | 6863.00 | 2025-11-12 10:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-11-19 14:00:00 | 6865.00 | 2025-11-27 10:15:00 | 7051.50 | STOP_HIT | 1.00 | 2.72% |
| BUY | retest2 | 2025-12-09 11:45:00 | 7158.00 | 2025-12-15 09:15:00 | 7125.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-10 09:15:00 | 7163.50 | 2025-12-15 09:15:00 | 7125.50 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-12-18 09:30:00 | 7053.50 | 2025-12-19 09:15:00 | 7184.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-12-18 14:00:00 | 7074.50 | 2025-12-19 09:15:00 | 7184.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-12-24 10:45:00 | 7303.00 | 2025-12-29 13:15:00 | 7261.50 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-12-26 09:15:00 | 7353.50 | 2025-12-29 13:15:00 | 7261.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-12-26 11:45:00 | 7313.00 | 2025-12-29 13:15:00 | 7261.50 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-12-29 11:45:00 | 7306.50 | 2025-12-29 13:15:00 | 7261.50 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-01-05 10:15:00 | 7476.50 | 2026-01-09 15:15:00 | 7510.00 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2026-01-05 11:30:00 | 7469.00 | 2026-01-09 15:15:00 | 7510.00 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2026-01-05 13:45:00 | 7469.00 | 2026-01-09 15:15:00 | 7510.00 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2026-01-05 14:15:00 | 7481.50 | 2026-01-09 15:15:00 | 7510.00 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2026-01-13 10:30:00 | 7429.00 | 2026-01-21 09:15:00 | 7057.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 7435.50 | 2026-01-21 09:15:00 | 7063.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:30:00 | 7402.00 | 2026-01-21 10:15:00 | 7031.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 10:30:00 | 7429.00 | 2026-01-22 09:15:00 | 7266.00 | STOP_HIT | 0.50 | 2.19% |
| SELL | retest2 | 2026-01-13 12:00:00 | 7435.50 | 2026-01-22 09:15:00 | 7266.00 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2026-01-13 12:30:00 | 7402.00 | 2026-01-22 09:15:00 | 7266.00 | STOP_HIT | 0.50 | 1.84% |
| BUY | retest2 | 2026-02-17 13:30:00 | 8023.50 | 2026-02-19 14:15:00 | 7900.50 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-02-17 14:30:00 | 8019.50 | 2026-02-19 14:15:00 | 7900.50 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-02-18 15:15:00 | 8020.00 | 2026-02-19 14:15:00 | 7900.50 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-02-27 11:45:00 | 8101.00 | 2026-02-27 14:15:00 | 8005.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-27 13:45:00 | 8101.50 | 2026-02-27 14:15:00 | 8005.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-27 14:30:00 | 8097.50 | 2026-02-27 15:15:00 | 8014.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest1 | 2026-03-05 10:15:00 | 7666.50 | 2026-03-05 15:15:00 | 7742.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest1 | 2026-03-05 11:00:00 | 7657.00 | 2026-03-05 15:15:00 | 7742.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest1 | 2026-03-05 13:45:00 | 7669.50 | 2026-03-05 15:15:00 | 7742.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-03-06 09:15:00 | 7681.00 | 2026-03-09 10:15:00 | 7296.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 7681.00 | 2026-03-10 09:15:00 | 7383.50 | STOP_HIT | 0.50 | 3.87% |
