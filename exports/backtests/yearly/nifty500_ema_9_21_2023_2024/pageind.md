# Page Industries Ltd. (PAGEIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 37365.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 219 |
| ALERT1 | 149 |
| ALERT2 | 147 |
| ALERT2_SKIP | 97 |
| ALERT3 | 328 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 127 |
| PARTIAL | 10 |
| TARGET_HIT | 5 |
| STOP_HIT | 126 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 141 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 79
- **Target hits / Stop hits / Partials:** 5 / 126 / 10
- **Avg / median % per leg:** 0.87% / -0.38%
- **Sum % (uncompounded):** 123.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 20 | 32.8% | 0 | 61 | 0 | 0.47% | 28.7% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.86% | -2.6% |
| BUY @ 3rd Alert (retest2) | 58 | 20 | 34.5% | 0 | 58 | 0 | 0.54% | 31.3% |
| SELL (all) | 80 | 42 | 52.5% | 5 | 65 | 10 | 1.18% | 94.6% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.31% | 0.3% |
| SELL @ 3rd Alert (retest2) | 79 | 41 | 51.9% | 5 | 64 | 10 | 1.19% | 94.3% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -0.57% | -2.3% |
| retest2 (combined) | 137 | 61 | 44.5% | 5 | 122 | 10 | 0.92% | 125.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 10:15:00 | 42180.00 | 42492.31 | 42515.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 12:15:00 | 41777.10 | 42290.59 | 42416.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 10:15:00 | 42200.00 | 42058.88 | 42233.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 10:15:00 | 42200.00 | 42058.88 | 42233.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 10:15:00 | 42200.00 | 42058.88 | 42233.51 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 09:15:00 | 39263.10 | 38818.78 | 38798.74 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 12:15:00 | 38835.60 | 38864.58 | 38867.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-02 13:15:00 | 38747.10 | 38841.08 | 38856.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-05 13:15:00 | 38765.60 | 38631.65 | 38711.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 13:15:00 | 38765.60 | 38631.65 | 38711.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 13:15:00 | 38765.60 | 38631.65 | 38711.61 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 15:15:00 | 38750.00 | 38744.67 | 38743.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 10:15:00 | 38930.00 | 38790.33 | 38765.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 09:15:00 | 38967.00 | 39089.16 | 38961.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 10:15:00 | 39217.60 | 39114.85 | 38984.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 10:15:00 | 39217.60 | 39114.85 | 38984.84 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 12:15:00 | 38799.50 | 38996.29 | 39001.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 13:15:00 | 38570.10 | 38911.06 | 38962.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 38409.80 | 38328.67 | 38530.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 14:15:00 | 38451.40 | 38317.36 | 38440.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 14:15:00 | 38451.40 | 38317.36 | 38440.82 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 09:15:00 | 38801.60 | 38513.63 | 38480.03 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 13:15:00 | 38521.00 | 38633.54 | 38644.95 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 15:15:00 | 38690.00 | 38652.94 | 38652.29 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 09:15:00 | 38550.40 | 38632.43 | 38643.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 12:15:00 | 38370.00 | 38518.83 | 38583.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 37780.90 | 37729.67 | 37924.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 37780.90 | 37729.67 | 37924.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 37780.90 | 37729.67 | 37924.70 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 12:15:00 | 38020.20 | 37955.59 | 37948.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 09:15:00 | 38135.00 | 37999.64 | 37971.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 10:15:00 | 37994.90 | 37998.70 | 37973.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 11:15:00 | 37851.10 | 37969.18 | 37962.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 11:15:00 | 37851.10 | 37969.18 | 37962.47 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 14:15:00 | 37465.90 | 37876.23 | 37922.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-03 14:15:00 | 37240.20 | 37513.20 | 37658.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-06 14:15:00 | 36985.00 | 36731.05 | 36920.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 14:15:00 | 36985.00 | 36731.05 | 36920.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 14:15:00 | 36985.00 | 36731.05 | 36920.91 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 09:15:00 | 36760.00 | 36409.62 | 36381.79 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 36265.10 | 36532.48 | 36551.73 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 10:15:00 | 36795.00 | 36546.56 | 36521.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 09:15:00 | 37465.10 | 36835.64 | 36713.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 09:15:00 | 37250.00 | 37318.33 | 37082.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 13:15:00 | 36949.90 | 37187.59 | 37090.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 13:15:00 | 36949.90 | 37187.59 | 37090.70 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 10:15:00 | 37266.90 | 37395.33 | 37399.52 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 37913.70 | 37451.10 | 37416.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 10:15:00 | 38284.50 | 37617.78 | 37495.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-31 14:15:00 | 37801.00 | 37822.36 | 37649.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 14:15:00 | 37801.00 | 37822.36 | 37649.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 14:15:00 | 37801.00 | 37822.36 | 37649.30 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 09:15:00 | 38930.90 | 39085.38 | 39092.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 10:15:00 | 38734.00 | 39015.11 | 39060.30 | Break + close below crossover candle low |

### Cycle 18 — BUY (started 2023-08-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 14:15:00 | 39995.00 | 39174.98 | 39113.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 15:15:00 | 40235.00 | 39386.99 | 39215.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 09:15:00 | 39245.90 | 39358.77 | 39218.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 39245.90 | 39358.77 | 39218.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 39245.90 | 39358.77 | 39218.39 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 10:15:00 | 39845.50 | 40722.87 | 40798.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 11:15:00 | 39721.90 | 39903.01 | 40048.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 11:15:00 | 39600.00 | 39501.01 | 39725.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 12:15:00 | 39630.00 | 39526.80 | 39716.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 12:15:00 | 39630.00 | 39526.80 | 39716.94 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 11:15:00 | 39890.90 | 39703.51 | 39690.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 12:15:00 | 39942.80 | 39751.37 | 39713.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 13:15:00 | 39750.00 | 39751.09 | 39717.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 13:15:00 | 39750.00 | 39751.09 | 39717.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 13:15:00 | 39750.00 | 39751.09 | 39717.04 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 14:15:00 | 39821.60 | 39940.98 | 39941.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-04 13:15:00 | 39524.10 | 39796.62 | 39866.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-05 09:15:00 | 39739.10 | 39737.74 | 39817.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 09:15:00 | 39739.10 | 39737.74 | 39817.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 09:15:00 | 39739.10 | 39737.74 | 39817.74 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 10:15:00 | 40490.00 | 39828.16 | 39788.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 11:15:00 | 40781.90 | 40018.91 | 39878.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 12:15:00 | 41970.70 | 42006.89 | 41457.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 12:15:00 | 41666.20 | 41897.36 | 41665.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 12:15:00 | 41666.20 | 41897.36 | 41665.54 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 41210.10 | 41581.79 | 41595.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 13:15:00 | 41114.10 | 41434.01 | 41522.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-15 09:15:00 | 40230.00 | 40188.60 | 40454.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 12:15:00 | 40786.60 | 40302.83 | 40439.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 12:15:00 | 40786.60 | 40302.83 | 40439.21 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 09:15:00 | 41096.10 | 40538.48 | 40504.30 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 14:15:00 | 39900.40 | 40416.32 | 40465.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 13:15:00 | 39623.80 | 39979.02 | 40193.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 11:15:00 | 39594.10 | 39519.30 | 39706.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 11:15:00 | 39594.10 | 39519.30 | 39706.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 11:15:00 | 39594.10 | 39519.30 | 39706.23 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 10:15:00 | 39453.90 | 38993.20 | 38989.80 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 38977.40 | 39056.69 | 39058.17 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 14:15:00 | 39253.10 | 39061.74 | 39055.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 10:15:00 | 39412.10 | 39169.95 | 39110.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-05 13:15:00 | 39050.60 | 39173.25 | 39129.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 13:15:00 | 39050.60 | 39173.25 | 39129.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 13:15:00 | 39050.60 | 39173.25 | 39129.56 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 11:15:00 | 39114.60 | 39432.57 | 39451.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 15:15:00 | 39070.00 | 39282.28 | 39366.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 09:15:00 | 39325.00 | 39290.82 | 39362.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 09:15:00 | 39325.00 | 39290.82 | 39362.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 39325.00 | 39290.82 | 39362.68 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 13:15:00 | 37502.20 | 37334.78 | 37328.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 09:15:00 | 37750.00 | 37471.40 | 37397.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 14:15:00 | 37544.80 | 37672.48 | 37549.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-30 14:15:00 | 37544.80 | 37672.48 | 37549.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 14:15:00 | 37544.80 | 37672.48 | 37549.67 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-11-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 11:15:00 | 37435.00 | 37641.66 | 37643.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 11:15:00 | 37217.00 | 37453.59 | 37531.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 14:15:00 | 37580.20 | 37378.63 | 37468.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 14:15:00 | 37580.20 | 37378.63 | 37468.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 37580.20 | 37378.63 | 37468.07 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 11:15:00 | 37791.00 | 37538.41 | 37523.86 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 14:15:00 | 37433.10 | 37500.62 | 37508.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 15:15:00 | 37379.90 | 37476.48 | 37497.19 | Break + close below crossover candle low |

### Cycle 34 — BUY (started 2023-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 09:15:00 | 37650.10 | 37511.20 | 37511.09 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 09:15:00 | 37302.20 | 37492.02 | 37516.43 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 12:15:00 | 37646.20 | 37468.90 | 37461.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 13:15:00 | 37798.00 | 37534.72 | 37492.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 10:15:00 | 37664.10 | 37696.22 | 37596.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 12:15:00 | 38039.10 | 37784.75 | 37654.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 38039.10 | 37784.75 | 37654.87 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 11:15:00 | 37211.00 | 37802.05 | 37823.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 14:15:00 | 37000.00 | 37498.72 | 37667.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 09:15:00 | 37134.90 | 37081.07 | 37288.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 10:15:00 | 37309.40 | 37126.73 | 37290.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 10:15:00 | 37309.40 | 37126.73 | 37290.28 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 09:15:00 | 37730.00 | 37364.59 | 37344.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 09:15:00 | 37876.60 | 37586.56 | 37506.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-21 14:15:00 | 37701.00 | 37726.83 | 37621.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 15:15:00 | 37700.10 | 37721.48 | 37628.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 15:15:00 | 37700.10 | 37721.48 | 37628.73 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 09:15:00 | 37631.00 | 37710.57 | 37716.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 12:15:00 | 37499.40 | 37642.43 | 37681.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 09:15:00 | 37715.00 | 37602.33 | 37643.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 09:15:00 | 37715.00 | 37602.33 | 37643.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 37715.00 | 37602.33 | 37643.53 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 11:15:00 | 37773.10 | 37668.69 | 37668.54 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 13:15:00 | 37500.00 | 37636.82 | 37654.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 10:15:00 | 37462.00 | 37592.59 | 37628.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 12:15:00 | 37593.60 | 37575.72 | 37613.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 12:15:00 | 37593.60 | 37575.72 | 37613.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 12:15:00 | 37593.60 | 37575.72 | 37613.40 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 13:15:00 | 37714.90 | 37438.57 | 37434.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 14:15:00 | 37900.00 | 37530.86 | 37476.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 10:15:00 | 37500.00 | 37566.55 | 37510.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 10:15:00 | 37500.00 | 37566.55 | 37510.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 10:15:00 | 37500.00 | 37566.55 | 37510.32 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 10:15:00 | 37395.00 | 37498.21 | 37503.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 11:15:00 | 37339.10 | 37466.38 | 37488.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-05 12:15:00 | 37490.00 | 37471.11 | 37489.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 12:15:00 | 37490.00 | 37471.11 | 37489.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 37490.00 | 37471.11 | 37489.03 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-12-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 14:15:00 | 37400.00 | 37297.11 | 37288.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 37660.00 | 37394.15 | 37335.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 11:15:00 | 37357.90 | 37391.04 | 37344.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 11:15:00 | 37357.90 | 37391.04 | 37344.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 37357.90 | 37391.04 | 37344.79 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 09:15:00 | 37040.10 | 37296.95 | 37317.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 10:15:00 | 36978.40 | 37233.24 | 37286.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 12:15:00 | 37200.00 | 37191.66 | 37256.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 14:15:00 | 37317.20 | 37223.70 | 37260.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 14:15:00 | 37317.20 | 37223.70 | 37260.24 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 10:15:00 | 37374.10 | 37287.40 | 37282.06 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-12-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 12:15:00 | 37180.00 | 37291.92 | 37303.13 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 14:15:00 | 37661.80 | 37365.96 | 37334.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 12:15:00 | 37717.00 | 37533.62 | 37436.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 15:15:00 | 37755.00 | 37808.43 | 37677.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 09:15:00 | 38187.50 | 37884.25 | 37723.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 38187.50 | 37884.25 | 37723.96 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 37057.60 | 37724.68 | 37814.80 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 37627.90 | 37530.58 | 37527.42 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-26 10:15:00 | 37444.60 | 37513.39 | 37519.89 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 10:15:00 | 37587.90 | 37531.78 | 37524.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 13:15:00 | 37805.00 | 37586.77 | 37550.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 13:15:00 | 38460.90 | 38476.86 | 38228.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 10:15:00 | 38603.00 | 38630.76 | 38496.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 38603.00 | 38630.76 | 38496.10 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 38550.00 | 38837.23 | 38852.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 10:15:00 | 38300.00 | 38729.78 | 38802.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 09:15:00 | 37054.20 | 37040.20 | 37595.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 14:15:00 | 36893.60 | 36847.13 | 37005.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 14:15:00 | 36893.60 | 36847.13 | 37005.82 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 12:15:00 | 37600.00 | 37151.46 | 37100.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 09:15:00 | 38600.10 | 37619.29 | 37351.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 10:15:00 | 37930.00 | 38162.96 | 37885.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 10:15:00 | 37930.00 | 38162.96 | 37885.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 10:15:00 | 37930.00 | 38162.96 | 37885.75 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 37325.00 | 37738.60 | 37762.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 36974.90 | 37363.34 | 37513.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 15:15:00 | 36782.90 | 36642.27 | 36906.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 13:15:00 | 36566.40 | 36538.86 | 36747.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 13:15:00 | 36566.40 | 36538.86 | 36747.39 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 11:15:00 | 37280.40 | 36817.30 | 36809.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 14:15:00 | 37498.00 | 37086.42 | 36947.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 36948.60 | 37268.53 | 37148.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 14:15:00 | 36948.60 | 37268.53 | 37148.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 36948.60 | 37268.53 | 37148.75 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 13:15:00 | 37011.00 | 37127.03 | 37129.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 15:15:00 | 36950.00 | 37071.71 | 37102.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 09:15:00 | 37111.00 | 37079.57 | 37103.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 09:15:00 | 37111.00 | 37079.57 | 37103.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 37111.00 | 37079.57 | 37103.26 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 11:15:00 | 37166.10 | 37122.80 | 37120.41 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 12:15:00 | 37050.10 | 37108.26 | 37114.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 13:15:00 | 36907.60 | 37068.13 | 37095.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 09:15:00 | 37201.90 | 37057.89 | 37080.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 09:15:00 | 37201.90 | 37057.89 | 37080.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 37201.90 | 37057.89 | 37080.74 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 12:15:00 | 37184.20 | 37098.03 | 37095.10 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 14:15:00 | 36687.50 | 37034.35 | 37067.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 12:15:00 | 36618.50 | 36812.13 | 36934.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 11:15:00 | 36341.90 | 36303.12 | 36468.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 14:15:00 | 36705.00 | 36315.86 | 36428.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 14:15:00 | 36705.00 | 36315.86 | 36428.75 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 10:15:00 | 36350.00 | 36085.05 | 36082.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 12:15:00 | 36393.70 | 36178.48 | 36127.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 15:15:00 | 36156.60 | 36228.94 | 36168.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 15:15:00 | 36156.60 | 36228.94 | 36168.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 15:15:00 | 36156.60 | 36228.94 | 36168.47 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 36282.90 | 36419.48 | 36432.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 10:15:00 | 36129.40 | 36318.35 | 36381.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 09:15:00 | 36240.00 | 36176.02 | 36266.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 09:15:00 | 36240.00 | 36176.02 | 36266.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 36240.00 | 36176.02 | 36266.55 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-03-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 13:15:00 | 35641.15 | 35210.04 | 35153.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-05 14:15:00 | 35700.00 | 35569.50 | 35477.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 10:15:00 | 35474.85 | 35582.14 | 35509.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 10:15:00 | 35474.85 | 35582.14 | 35509.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 35474.85 | 35582.14 | 35509.40 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 11:15:00 | 35431.95 | 35674.22 | 35706.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 14:15:00 | 35155.55 | 35464.61 | 35593.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 14:15:00 | 34431.40 | 34228.34 | 34581.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 15:15:00 | 34301.85 | 34243.04 | 34555.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 34301.85 | 34243.04 | 34555.71 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 15:15:00 | 34970.00 | 34633.88 | 34609.86 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 10:15:00 | 34322.45 | 34598.72 | 34631.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 13:15:00 | 34039.90 | 34362.74 | 34504.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 11:15:00 | 33774.40 | 33647.87 | 33911.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 12:15:00 | 34169.00 | 33752.09 | 33934.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 12:15:00 | 34169.00 | 33752.09 | 33934.74 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-03-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 11:15:00 | 34400.00 | 34068.99 | 34031.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 11:15:00 | 34652.00 | 34391.27 | 34240.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 09:15:00 | 34453.95 | 34523.09 | 34376.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 10:15:00 | 34402.35 | 34498.94 | 34378.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 10:15:00 | 34402.35 | 34498.94 | 34378.81 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 12:15:00 | 34606.10 | 34771.42 | 34791.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 13:15:00 | 34566.10 | 34730.36 | 34771.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-08 12:15:00 | 34189.00 | 34179.52 | 34347.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-08 13:15:00 | 34502.20 | 34244.06 | 34361.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 13:15:00 | 34502.20 | 34244.06 | 34361.57 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 15:15:00 | 35089.00 | 34538.69 | 34482.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-09 09:15:00 | 35270.05 | 34684.96 | 34554.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 09:15:00 | 36296.70 | 36777.00 | 36195.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-12 10:15:00 | 36171.30 | 36777.00 | 36195.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 36226.70 | 36666.94 | 36198.02 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 35391.35 | 36027.21 | 36052.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 11:15:00 | 35070.40 | 35730.21 | 35906.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 15:15:00 | 35525.00 | 35407.58 | 35667.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 09:45:00 | 35525.00 | 35439.25 | 35658.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 35511.95 | 35453.79 | 35645.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:30:00 | 35516.45 | 35453.79 | 35645.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 35690.15 | 35501.06 | 35649.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 12:00:00 | 35690.15 | 35501.06 | 35649.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 12:15:00 | 35727.90 | 35546.43 | 35656.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 14:45:00 | 35505.95 | 35616.05 | 35671.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 09:30:00 | 35398.20 | 35583.14 | 35647.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 09:15:00 | 34945.30 | 35497.89 | 35561.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 10:15:00 | 35779.95 | 35496.38 | 35485.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 10:15:00 | 35779.95 | 35496.38 | 35485.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 12:15:00 | 35900.00 | 35663.32 | 35584.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 09:15:00 | 36200.00 | 36215.42 | 36011.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 09:45:00 | 36111.20 | 36215.42 | 36011.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 35893.75 | 36151.08 | 36000.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 35916.20 | 36151.08 | 36000.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 36200.00 | 36160.87 | 36018.89 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 15:15:00 | 35457.65 | 35864.24 | 35912.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 12:15:00 | 35295.00 | 35634.07 | 35779.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 35468.75 | 35246.30 | 35395.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 09:15:00 | 35468.75 | 35246.30 | 35395.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 35468.75 | 35246.30 | 35395.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:30:00 | 35502.30 | 35246.30 | 35395.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 35267.55 | 35250.55 | 35383.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:30:00 | 35508.70 | 35250.55 | 35383.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 35363.90 | 35273.22 | 35381.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 14:30:00 | 35026.65 | 35202.31 | 35325.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-10 10:15:00 | 34499.10 | 34487.36 | 34486.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-05-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 10:15:00 | 34499.10 | 34487.36 | 34486.90 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-05-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 11:15:00 | 34431.35 | 34476.16 | 34481.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 12:15:00 | 34322.80 | 34445.49 | 34467.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 13:15:00 | 34466.55 | 34449.70 | 34467.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 13:15:00 | 34466.55 | 34449.70 | 34467.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 34466.55 | 34449.70 | 34467.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:45:00 | 34498.80 | 34449.70 | 34467.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 34514.45 | 34462.65 | 34471.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 34514.45 | 34462.65 | 34471.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 34431.00 | 34456.32 | 34467.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:15:00 | 34341.70 | 34456.32 | 34467.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 34177.20 | 34400.50 | 34441.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 10:15:00 | 34147.90 | 34400.50 | 34441.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 11:15:00 | 34126.80 | 34353.52 | 34416.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 14:15:00 | 34987.05 | 34521.60 | 34475.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 14:15:00 | 34987.05 | 34521.60 | 34475.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 10:15:00 | 35215.00 | 34805.42 | 34629.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 10:15:00 | 34866.10 | 35064.95 | 34889.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 10:15:00 | 34866.10 | 35064.95 | 34889.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 34866.10 | 35064.95 | 34889.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:00:00 | 34866.10 | 35064.95 | 34889.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 34942.35 | 35040.43 | 34894.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:30:00 | 34897.15 | 35040.43 | 34894.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 34889.85 | 34987.85 | 34894.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 14:45:00 | 34934.90 | 34992.28 | 34905.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 12:15:00 | 35082.35 | 35421.65 | 35428.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 12:15:00 | 35082.35 | 35421.65 | 35428.62 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 12:15:00 | 35540.00 | 35391.64 | 35390.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 13:15:00 | 36106.15 | 35534.54 | 35455.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 09:15:00 | 35659.55 | 35803.82 | 35620.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-23 10:00:00 | 35659.55 | 35803.82 | 35620.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 35734.00 | 35802.24 | 35680.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:00:00 | 35734.00 | 35802.24 | 35680.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 35524.15 | 35746.62 | 35666.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:30:00 | 35250.25 | 35746.62 | 35666.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 35650.00 | 35727.30 | 35664.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 35150.25 | 35727.30 | 35664.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 09:15:00 | 35100.00 | 35601.84 | 35613.57 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 13:15:00 | 35799.95 | 35632.79 | 35622.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 09:15:00 | 36183.90 | 35733.57 | 35669.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 09:15:00 | 36231.70 | 36380.98 | 36106.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 10:00:00 | 36231.70 | 36380.98 | 36106.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 36206.95 | 36328.65 | 36129.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:45:00 | 36089.05 | 36328.65 | 36129.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 36300.00 | 36322.92 | 36144.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 36179.45 | 36322.92 | 36144.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 36350.10 | 36303.13 | 36165.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:30:00 | 36277.10 | 36303.13 | 36165.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 36265.00 | 36295.51 | 36174.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 36354.25 | 36295.51 | 36174.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 36611.50 | 36358.70 | 36214.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 11:00:00 | 36745.40 | 36436.04 | 36262.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 12:15:00 | 36657.65 | 36474.83 | 36295.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 10:15:00 | 36647.75 | 36469.09 | 36361.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 10:45:00 | 36736.55 | 36502.52 | 36386.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 36840.00 | 37120.16 | 36834.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:45:00 | 36740.00 | 37120.16 | 36834.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 36474.00 | 36990.93 | 36802.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:00:00 | 36474.00 | 36990.93 | 36802.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 36144.25 | 36821.59 | 36742.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:00:00 | 36144.25 | 36821.59 | 36742.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-31 14:15:00 | 35940.00 | 36560.07 | 36631.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 14:15:00 | 35940.00 | 36560.07 | 36631.96 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 12:15:00 | 36700.30 | 36508.72 | 36505.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 09:15:00 | 37931.45 | 36896.08 | 36695.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 11:15:00 | 38477.00 | 38534.33 | 38124.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 12:00:00 | 38477.00 | 38534.33 | 38124.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 12:15:00 | 38229.95 | 38473.45 | 38134.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 12:45:00 | 38300.00 | 38473.45 | 38134.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 38284.40 | 38391.70 | 38198.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:00:00 | 38284.40 | 38391.70 | 38198.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 38230.75 | 38359.51 | 38201.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:30:00 | 38031.70 | 38359.51 | 38201.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 11:15:00 | 38184.95 | 38324.60 | 38199.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 12:00:00 | 38184.95 | 38324.60 | 38199.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 12:15:00 | 38190.75 | 38297.83 | 38199.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 13:00:00 | 38190.75 | 38297.83 | 38199.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 13:15:00 | 38385.10 | 38315.28 | 38216.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 14:45:00 | 38564.40 | 38396.12 | 38261.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 38530.40 | 38416.89 | 38283.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 10:15:00 | 38464.05 | 38643.67 | 38654.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 10:15:00 | 38464.05 | 38643.67 | 38654.60 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 11:15:00 | 39077.55 | 38572.52 | 38546.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 09:15:00 | 39364.35 | 38916.20 | 38741.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 12:15:00 | 39927.40 | 40372.41 | 40194.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 12:15:00 | 39927.40 | 40372.41 | 40194.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 39927.40 | 40372.41 | 40194.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:00:00 | 39927.40 | 40372.41 | 40194.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 39800.00 | 40257.93 | 40158.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:45:00 | 39721.05 | 40257.93 | 40158.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 15:15:00 | 39700.00 | 40044.22 | 40072.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 13:15:00 | 39376.05 | 39757.45 | 39905.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 15:15:00 | 39150.00 | 39146.76 | 39403.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-02 09:15:00 | 39147.20 | 39146.76 | 39403.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 39441.95 | 39205.80 | 39407.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 39441.95 | 39205.80 | 39407.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 39523.40 | 39269.32 | 39417.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 11:15:00 | 39380.00 | 39269.32 | 39417.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 12:15:00 | 39530.00 | 38934.13 | 38890.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 12:15:00 | 39530.00 | 38934.13 | 38890.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 09:15:00 | 39840.00 | 39386.21 | 39191.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 13:15:00 | 39311.55 | 39493.72 | 39320.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 13:15:00 | 39311.55 | 39493.72 | 39320.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 39311.55 | 39493.72 | 39320.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 39311.55 | 39493.72 | 39320.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 39395.05 | 39473.98 | 39327.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 09:15:00 | 39547.40 | 39359.21 | 39321.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 11:15:00 | 39220.00 | 39386.72 | 39349.21 | SL hit (close<static) qty=1.00 sl=39300.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 09:15:00 | 39167.00 | 39322.59 | 39329.53 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 39500.00 | 39346.03 | 39338.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 13:15:00 | 39600.10 | 39425.11 | 39377.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 39291.40 | 39506.34 | 39436.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 39291.40 | 39506.34 | 39436.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 39291.40 | 39506.34 | 39436.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:00:00 | 39291.40 | 39506.34 | 39436.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 39737.50 | 39552.57 | 39463.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:30:00 | 39538.00 | 39552.57 | 39463.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 40257.50 | 40669.72 | 40377.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:00:00 | 40257.50 | 40669.72 | 40377.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 40312.75 | 40598.32 | 40371.19 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 14:15:00 | 40180.00 | 40250.43 | 40252.55 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 12:15:00 | 40372.00 | 40247.64 | 40243.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-19 13:15:00 | 40584.90 | 40315.09 | 40274.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 11:15:00 | 40467.50 | 40475.38 | 40379.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 11:15:00 | 40467.50 | 40475.38 | 40379.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 40467.50 | 40475.38 | 40379.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 40467.50 | 40475.38 | 40379.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 40344.65 | 40449.24 | 40376.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 40344.65 | 40449.24 | 40376.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 40400.05 | 40439.40 | 40378.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:15:00 | 40342.10 | 40439.40 | 40378.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 40194.90 | 40390.50 | 40361.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 40194.90 | 40390.50 | 40361.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 40359.00 | 40384.20 | 40361.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 09:15:00 | 40641.85 | 40384.20 | 40361.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 10:15:00 | 40422.20 | 40364.33 | 40354.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 11:15:00 | 41612.55 | 42090.28 | 42136.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 41612.55 | 42090.28 | 42136.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 41274.10 | 41927.05 | 42058.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 42137.95 | 41753.69 | 41912.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 42137.95 | 41753.69 | 41912.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 42137.95 | 41753.69 | 41912.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 42137.95 | 41753.69 | 41912.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 41863.00 | 41775.55 | 41907.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 41327.80 | 41638.11 | 41811.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 10:15:00 | 41287.95 | 40727.11 | 40657.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 10:15:00 | 41287.95 | 40727.11 | 40657.50 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 15:15:00 | 40700.00 | 40767.55 | 40774.93 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 40861.90 | 40782.46 | 40778.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 40922.90 | 40810.55 | 40791.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-16 13:15:00 | 40792.85 | 40807.01 | 40791.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-16 14:00:00 | 40792.85 | 40807.01 | 40791.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 40985.00 | 40842.61 | 40809.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:45:00 | 40794.30 | 40842.61 | 40809.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 40800.00 | 40874.47 | 40832.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:00:00 | 40800.00 | 40874.47 | 40832.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 40679.85 | 40835.54 | 40818.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:30:00 | 40654.00 | 40835.54 | 40818.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 11:15:00 | 40631.65 | 40794.77 | 40801.37 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 14:15:00 | 41055.10 | 40817.89 | 40807.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 15:15:00 | 41185.00 | 40891.31 | 40841.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 14:15:00 | 40910.00 | 40965.78 | 40911.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 14:15:00 | 40910.00 | 40965.78 | 40911.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 40910.00 | 40965.78 | 40911.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:30:00 | 40885.05 | 40965.78 | 40911.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 40900.00 | 40952.62 | 40910.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:15:00 | 40915.60 | 40952.62 | 40910.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 41224.00 | 41006.90 | 40938.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 41158.35 | 41006.90 | 40938.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 41829.95 | 41977.27 | 41757.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 41829.95 | 41977.27 | 41757.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 41816.20 | 41945.06 | 41762.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:30:00 | 41642.10 | 41945.06 | 41762.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 41641.10 | 41884.27 | 41751.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 41641.10 | 41884.27 | 41751.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 41480.00 | 41803.41 | 41726.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 41900.05 | 41803.41 | 41726.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 13:15:00 | 41500.00 | 41815.47 | 41834.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 13:15:00 | 41500.00 | 41815.47 | 41834.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 15:15:00 | 41470.65 | 41708.83 | 41780.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 13:15:00 | 41526.65 | 41475.73 | 41615.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-28 14:00:00 | 41526.65 | 41475.73 | 41615.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 41820.05 | 41494.66 | 41584.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:00:00 | 41820.05 | 41494.66 | 41584.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 41745.70 | 41544.87 | 41599.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:15:00 | 41842.45 | 41544.87 | 41599.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 41625.75 | 41580.69 | 41604.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 13:30:00 | 41680.00 | 41580.69 | 41604.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 41700.05 | 41604.56 | 41613.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 41700.05 | 41604.56 | 41613.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 15:15:00 | 41850.00 | 41653.65 | 41634.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 09:15:00 | 42126.05 | 41748.13 | 41679.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 12:15:00 | 42018.50 | 42259.89 | 42084.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 12:15:00 | 42018.50 | 42259.89 | 42084.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 42018.50 | 42259.89 | 42084.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:30:00 | 42010.95 | 42259.89 | 42084.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 41953.90 | 42198.69 | 42072.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:45:00 | 41990.70 | 42198.69 | 42072.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 41880.00 | 42134.95 | 42055.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:15:00 | 41813.55 | 42134.95 | 42055.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 41813.55 | 42070.67 | 42033.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 11:00:00 | 42026.85 | 42038.50 | 42023.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 11:15:00 | 41769.20 | 41984.64 | 42000.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 11:15:00 | 41769.20 | 41984.64 | 42000.57 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 13:15:00 | 42123.00 | 42026.59 | 42017.76 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 41921.30 | 42005.53 | 42008.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 41797.05 | 41948.56 | 41981.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 11:15:00 | 41504.55 | 41442.26 | 41623.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-05 11:30:00 | 41513.70 | 41442.26 | 41623.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 40561.55 | 40436.09 | 40665.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:45:00 | 40612.55 | 40436.09 | 40665.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 40688.15 | 40486.50 | 40667.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:00:00 | 40688.15 | 40486.50 | 40667.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 40564.90 | 40502.18 | 40657.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 12:45:00 | 40535.50 | 40517.75 | 40650.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 13:15:00 | 40520.30 | 40517.75 | 40650.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 14:00:00 | 40549.90 | 40524.18 | 40641.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 10:15:00 | 41414.15 | 40664.16 | 40661.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 41414.15 | 40664.16 | 40661.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 11:15:00 | 42368.35 | 41005.00 | 40816.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 13:15:00 | 42888.65 | 43284.21 | 42947.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 13:15:00 | 42888.65 | 43284.21 | 42947.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 42888.65 | 43284.21 | 42947.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:00:00 | 42888.65 | 43284.21 | 42947.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 43085.80 | 43244.53 | 42959.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 43327.25 | 43214.03 | 42972.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 10:00:00 | 43174.65 | 43206.15 | 42990.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 13:15:00 | 42786.70 | 43232.62 | 43197.52 | SL hit (close<static) qty=1.00 sl=42809.95 alert=retest2 |

### Cycle 103 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 42968.10 | 43158.00 | 43169.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 42777.80 | 43081.96 | 43133.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 43000.00 | 42990.37 | 43067.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 43000.00 | 42990.37 | 43067.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 43000.00 | 42990.37 | 43067.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:30:00 | 43019.65 | 42990.37 | 43067.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 42950.00 | 42982.30 | 43056.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 42955.85 | 42982.30 | 43056.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 43129.15 | 43011.67 | 43063.15 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 43164.90 | 43085.06 | 43084.18 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 12:15:00 | 43013.60 | 43076.35 | 43080.92 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 43118.10 | 43084.70 | 43084.30 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 14:15:00 | 43000.00 | 43067.76 | 43076.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 11:15:00 | 42795.00 | 43012.36 | 43047.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 42032.75 | 41831.26 | 42244.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-25 15:00:00 | 42032.75 | 41831.26 | 42244.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 42012.60 | 41767.79 | 42001.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 42012.60 | 41767.79 | 42001.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 42103.95 | 41835.03 | 42010.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 42358.75 | 41835.03 | 42010.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 42353.95 | 41938.81 | 42041.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 42406.00 | 41938.81 | 42041.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 42514.95 | 42054.04 | 42084.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:00:00 | 42514.95 | 42054.04 | 42084.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 42302.10 | 42091.88 | 42095.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:00:00 | 42302.10 | 42091.88 | 42095.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 13:15:00 | 42237.10 | 42120.93 | 42108.78 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 15:15:00 | 41952.60 | 42078.02 | 42090.81 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 09:15:00 | 42359.60 | 42134.33 | 42115.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 11:15:00 | 42625.10 | 42274.22 | 42184.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 09:15:00 | 42486.05 | 42530.82 | 42360.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 10:00:00 | 42486.05 | 42530.82 | 42360.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 42439.95 | 42512.65 | 42368.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 42298.30 | 42512.65 | 42368.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 42340.00 | 42562.50 | 42466.64 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 41954.80 | 42351.25 | 42381.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 41629.25 | 42206.85 | 42313.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 42127.05 | 41981.59 | 42150.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 42127.05 | 41981.59 | 42150.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 42127.05 | 41981.59 | 42150.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:00:00 | 42127.05 | 41981.59 | 42150.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 41393.70 | 41605.56 | 41858.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:30:00 | 41172.95 | 41496.35 | 41785.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 12:00:00 | 41144.90 | 41426.06 | 41727.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 12:15:00 | 42025.05 | 41552.33 | 41620.53 | SL hit (close>static) qty=1.00 sl=42006.45 alert=retest2 |

### Cycle 112 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 42168.30 | 41675.52 | 41670.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 42951.40 | 42112.53 | 41887.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 46260.75 | 46351.85 | 45995.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 09:45:00 | 46326.45 | 46351.85 | 45995.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 46061.10 | 46293.70 | 46001.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 46061.10 | 46293.70 | 46001.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 46005.10 | 46235.98 | 46001.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:00:00 | 46005.10 | 46235.98 | 46001.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 45753.90 | 46139.56 | 45978.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:45:00 | 45821.40 | 46139.56 | 45978.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 45463.05 | 46004.26 | 45932.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:00:00 | 45463.05 | 46004.26 | 45932.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 14:15:00 | 45378.70 | 45879.15 | 45881.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 45260.00 | 45721.85 | 45807.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 45721.10 | 45701.10 | 45768.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 13:15:00 | 45721.10 | 45701.10 | 45768.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 45721.10 | 45701.10 | 45768.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:30:00 | 45699.95 | 45701.10 | 45768.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 45249.80 | 45565.98 | 45686.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:45:00 | 45104.65 | 45483.80 | 45638.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:30:00 | 45080.00 | 45374.04 | 45574.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 44935.00 | 45374.04 | 45574.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 09:15:00 | 42849.42 | 43474.68 | 44012.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 14:15:00 | 43491.60 | 43262.70 | 43680.74 | SL hit (close>ema200) qty=0.50 sl=43262.70 alert=retest2 |

### Cycle 114 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 43314.45 | 43168.41 | 43167.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 43651.75 | 43265.08 | 43211.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 43086.25 | 43235.91 | 43216.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 43086.25 | 43235.91 | 43216.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 43086.25 | 43235.91 | 43216.30 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 42891.15 | 43171.29 | 43192.22 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 18:15:00 | 43410.00 | 43196.58 | 43189.36 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 43057.05 | 43168.68 | 43177.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 42769.85 | 43088.91 | 43140.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-04 11:15:00 | 43092.40 | 43089.61 | 43135.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-04 12:00:00 | 43092.40 | 43089.61 | 43135.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 42870.30 | 43045.75 | 43111.79 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 43429.00 | 43094.01 | 43088.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 43911.15 | 43306.40 | 43188.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 14:15:00 | 47340.05 | 47452.53 | 46648.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-11 14:45:00 | 47414.90 | 47452.53 | 46648.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 46528.85 | 47142.37 | 46961.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:30:00 | 46530.00 | 47142.37 | 46961.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 46053.50 | 46924.60 | 46879.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:45:00 | 46100.80 | 46924.60 | 46879.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 11:15:00 | 45970.00 | 46733.68 | 46796.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 09:15:00 | 45786.90 | 46151.49 | 46446.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 44944.40 | 44649.32 | 45171.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 44944.40 | 44649.32 | 45171.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 44944.40 | 44649.32 | 45171.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:30:00 | 44986.65 | 44649.32 | 45171.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 44575.00 | 44700.30 | 45006.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:30:00 | 44998.00 | 44700.30 | 45006.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 44169.70 | 44549.16 | 44881.92 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 45664.50 | 44967.47 | 44874.16 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 12:15:00 | 44928.95 | 45193.61 | 45224.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 44781.90 | 44959.38 | 45074.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 13:15:00 | 45116.70 | 44990.85 | 45077.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 13:15:00 | 45116.70 | 44990.85 | 45077.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 45116.70 | 44990.85 | 45077.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 45116.70 | 44990.85 | 45077.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 44794.85 | 44951.65 | 45052.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 13:45:00 | 44700.00 | 44858.77 | 44959.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 14:30:00 | 44676.70 | 44818.81 | 44931.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 45216.25 | 44865.90 | 44931.52 | SL hit (close>static) qty=1.00 sl=45135.20 alert=retest2 |

### Cycle 122 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 11:15:00 | 45964.00 | 45125.73 | 45040.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 12:15:00 | 46199.50 | 45340.49 | 45146.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 09:15:00 | 45277.55 | 45522.78 | 45314.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 45277.55 | 45522.78 | 45314.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 45277.55 | 45522.78 | 45314.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 45277.55 | 45522.78 | 45314.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 45638.95 | 45546.01 | 45344.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 12:30:00 | 45876.80 | 45622.32 | 45414.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 13:45:00 | 45894.65 | 45665.98 | 45452.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 12:15:00 | 45994.20 | 45781.02 | 45608.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 45941.55 | 46039.77 | 45900.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 46030.65 | 46037.94 | 45912.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 09:15:00 | 46290.10 | 46037.94 | 45912.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 15:15:00 | 46400.00 | 46151.74 | 46041.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 10:45:00 | 46064.55 | 46206.44 | 46096.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 14:15:00 | 48812.60 | 48995.17 | 49015.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 48812.60 | 48995.17 | 49015.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 13:15:00 | 48549.50 | 48847.18 | 48929.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 48853.05 | 48760.82 | 48862.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 48853.05 | 48760.82 | 48862.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 48853.05 | 48760.82 | 48862.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:15:00 | 49009.85 | 48760.82 | 48862.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 48970.50 | 48802.76 | 48872.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 12:00:00 | 48790.00 | 48800.21 | 48864.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 12:45:00 | 48775.85 | 48782.17 | 48850.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 15:15:00 | 48779.00 | 48794.98 | 48844.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 14:15:00 | 47448.75 | 47360.15 | 47350.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 47448.75 | 47360.15 | 47350.06 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 10:15:00 | 47279.65 | 47341.36 | 47345.02 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 47641.60 | 47363.16 | 47350.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 47769.95 | 47444.52 | 47388.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 47502.00 | 47520.57 | 47436.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 10:00:00 | 47502.00 | 47520.57 | 47436.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 47848.95 | 47586.25 | 47474.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:15:00 | 47858.25 | 47655.95 | 47526.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 09:30:00 | 47861.70 | 47934.03 | 47851.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 10:15:00 | 48029.65 | 47934.03 | 47851.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 11:15:00 | 47923.10 | 47878.61 | 47834.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 47906.45 | 47884.18 | 47840.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 47624.60 | 47884.18 | 47840.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 48096.75 | 47926.69 | 47864.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 48425.95 | 47973.43 | 47905.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 11:15:00 | 47758.25 | 48302.98 | 48266.70 | SL hit (close<static) qty=1.00 sl=47783.60 alert=retest2 |

### Cycle 127 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 47299.95 | 48102.37 | 48178.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 11:15:00 | 46932.95 | 47377.25 | 47665.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 46189.00 | 46121.06 | 46620.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 09:45:00 | 46358.10 | 46121.06 | 46620.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 46349.65 | 46260.41 | 46534.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:30:00 | 45797.15 | 46232.61 | 46459.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 10:45:00 | 46088.05 | 46140.32 | 46396.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 09:45:00 | 46016.00 | 45937.06 | 46157.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:45:00 | 45972.45 | 45979.04 | 46140.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 46019.45 | 45987.12 | 46129.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 12:30:00 | 46034.65 | 45987.12 | 46129.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 46015.95 | 45992.89 | 46119.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 13:45:00 | 46020.00 | 45992.89 | 46119.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 45689.95 | 45894.60 | 46040.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-20 14:15:00 | 46325.00 | 46053.64 | 46021.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 14:15:00 | 46325.00 | 46053.64 | 46021.62 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 15:15:00 | 45900.00 | 46080.77 | 46093.74 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 46384.65 | 46141.55 | 46120.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 10:15:00 | 46494.95 | 46212.23 | 46154.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 13:15:00 | 46491.00 | 46642.81 | 46485.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 13:15:00 | 46491.00 | 46642.81 | 46485.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 46491.00 | 46642.81 | 46485.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:00:00 | 46491.00 | 46642.81 | 46485.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 46369.90 | 46588.23 | 46475.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:30:00 | 46287.60 | 46588.23 | 46475.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 46321.60 | 46534.90 | 46461.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 45800.00 | 46534.90 | 46461.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 45590.00 | 46345.92 | 46382.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 45360.40 | 46148.82 | 46289.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 45999.10 | 44988.67 | 45242.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 45999.10 | 44988.67 | 45242.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 45999.10 | 44988.67 | 45242.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 45999.10 | 44988.67 | 45242.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 45802.35 | 45151.40 | 45293.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:30:00 | 45704.30 | 45355.84 | 45367.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 13:00:00 | 45708.65 | 45355.84 | 45367.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 13:15:00 | 45600.00 | 45404.67 | 45388.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 45600.00 | 45404.67 | 45388.48 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 15:15:00 | 44816.30 | 45273.74 | 45330.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 09:15:00 | 44240.10 | 45067.01 | 45231.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 10:15:00 | 45104.80 | 44642.45 | 44847.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 10:15:00 | 45104.80 | 44642.45 | 44847.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 45104.80 | 44642.45 | 44847.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 45104.80 | 44642.45 | 44847.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 44778.90 | 44669.74 | 44841.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 14:00:00 | 44720.00 | 44700.62 | 44827.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 09:15:00 | 44285.45 | 44697.40 | 44802.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 46430.00 | 44837.20 | 44809.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 12:15:00 | 46430.00 | 44837.20 | 44809.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 12:15:00 | 47200.40 | 46394.89 | 45756.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 09:15:00 | 45849.55 | 46581.17 | 46084.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 45849.55 | 46581.17 | 46084.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 45849.55 | 46581.17 | 46084.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 45849.55 | 46581.17 | 46084.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 46245.00 | 46513.94 | 46098.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 11:15:00 | 46540.40 | 46513.94 | 46098.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 12:00:00 | 46335.00 | 46478.15 | 46120.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 09:15:00 | 43748.65 | 45923.37 | 46201.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 09:15:00 | 43748.65 | 45923.37 | 46201.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 43424.00 | 44276.57 | 45070.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 09:15:00 | 43638.50 | 43447.05 | 44174.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 10:00:00 | 43638.50 | 43447.05 | 44174.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 43409.90 | 43372.06 | 43812.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 42803.45 | 43372.06 | 43812.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 40663.28 | 41352.70 | 41895.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 40931.70 | 40901.73 | 41291.51 | SL hit (close>ema200) qty=0.50 sl=40901.73 alert=retest2 |

### Cycle 136 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 41519.00 | 41022.75 | 40982.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 41545.85 | 41127.37 | 41033.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 40995.70 | 41152.66 | 41064.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 09:15:00 | 40995.70 | 41152.66 | 41064.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 40995.70 | 41152.66 | 41064.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:45:00 | 40976.45 | 41152.66 | 41064.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 41076.20 | 41137.37 | 41065.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 11:30:00 | 41220.85 | 41157.24 | 41081.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 12:45:00 | 41195.55 | 41168.64 | 41093.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 14:00:00 | 41196.45 | 41232.98 | 41189.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 15:00:00 | 41208.85 | 41228.16 | 41191.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 15:15:00 | 40900.00 | 41162.53 | 41164.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 15:15:00 | 40900.00 | 41162.53 | 41164.98 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 10:15:00 | 41342.80 | 41197.09 | 41180.17 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 41051.00 | 41167.87 | 41168.43 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 12:15:00 | 41184.65 | 41171.23 | 41169.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 13:15:00 | 41315.40 | 41200.06 | 41183.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 15:15:00 | 41062.05 | 41180.45 | 41177.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 15:15:00 | 41062.05 | 41180.45 | 41177.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 41062.05 | 41180.45 | 41177.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:30:00 | 41577.65 | 41248.31 | 41208.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-28 10:15:00 | 40958.05 | 41533.12 | 41533.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 40958.05 | 41533.12 | 41533.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 40396.80 | 41305.85 | 41430.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 40838.30 | 40665.98 | 40926.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 40838.30 | 40665.98 | 40926.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 40195.20 | 40230.77 | 40485.26 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 40991.35 | 40657.69 | 40616.32 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 12:15:00 | 40380.20 | 40622.89 | 40626.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 12:15:00 | 40101.55 | 40434.42 | 40528.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 39750.15 | 39708.38 | 39912.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 15:00:00 | 39750.15 | 39708.38 | 39912.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 39878.90 | 39749.14 | 39896.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:45:00 | 40005.50 | 39749.14 | 39896.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 39758.30 | 39750.97 | 39883.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:30:00 | 39700.00 | 39754.17 | 39873.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 12:15:00 | 39725.00 | 39754.17 | 39873.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 14:15:00 | 40178.95 | 39865.76 | 39896.30 | SL hit (close>static) qty=1.00 sl=39959.95 alert=retest2 |

### Cycle 144 — BUY (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 15:15:00 | 40145.00 | 39921.61 | 39918.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 09:15:00 | 40453.05 | 40149.20 | 40047.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 15:15:00 | 40050.00 | 40338.30 | 40222.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 15:15:00 | 40050.00 | 40338.30 | 40222.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 40050.00 | 40338.30 | 40222.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 10:15:00 | 40483.90 | 40334.73 | 40231.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 12:00:00 | 40500.00 | 40365.90 | 40263.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 13:00:00 | 40622.70 | 40417.26 | 40296.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 15:15:00 | 42215.45 | 42595.20 | 42602.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 42215.45 | 42595.20 | 42602.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 11:15:00 | 42102.85 | 42417.81 | 42512.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 42426.25 | 42376.65 | 42474.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 14:00:00 | 42426.25 | 42376.65 | 42474.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 146 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 44109.80 | 42723.28 | 42623.50 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 15:15:00 | 41000.00 | 42378.62 | 42475.91 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 42900.00 | 42577.46 | 42555.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 15:15:00 | 43075.00 | 42813.85 | 42696.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 42701.25 | 42791.33 | 42696.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 42701.25 | 42791.33 | 42696.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 42701.25 | 42791.33 | 42696.61 | EMA400 retest candle locked (from upside) |

### Cycle 149 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 42109.25 | 42589.09 | 42617.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 12:15:00 | 42029.60 | 42477.19 | 42563.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 42334.20 | 42248.09 | 42393.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:00:00 | 42334.20 | 42248.09 | 42393.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 42489.85 | 42296.44 | 42402.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:30:00 | 42476.40 | 42296.44 | 42402.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 42654.10 | 42367.98 | 42425.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:45:00 | 42621.35 | 42367.98 | 42425.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 42925.00 | 42550.86 | 42503.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 15:15:00 | 43150.00 | 42670.69 | 42561.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 42550.05 | 43103.79 | 42931.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 42550.05 | 43103.79 | 42931.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 42550.05 | 43103.79 | 42931.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 42550.05 | 43103.79 | 42931.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 42475.00 | 42978.03 | 42890.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 11:15:00 | 42346.00 | 42978.03 | 42890.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 42144.90 | 42811.41 | 42822.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 40990.00 | 42234.38 | 42518.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 41499.95 | 41490.67 | 41937.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 42199.00 | 41490.67 | 41937.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 41677.25 | 41527.98 | 41914.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:45:00 | 41453.95 | 41664.15 | 41831.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 13:15:00 | 42499.95 | 41992.66 | 41937.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 42499.95 | 41992.66 | 41937.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 14:15:00 | 42754.60 | 42145.05 | 42012.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 09:15:00 | 42116.00 | 42183.57 | 42055.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 09:15:00 | 42116.00 | 42183.57 | 42055.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 42116.00 | 42183.57 | 42055.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:45:00 | 42024.30 | 42183.57 | 42055.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 42353.75 | 42217.61 | 42082.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 11:45:00 | 42526.80 | 42342.96 | 42151.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 13:15:00 | 45055.00 | 45417.90 | 45455.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 13:15:00 | 45055.00 | 45417.90 | 45455.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 44570.00 | 45122.21 | 45300.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 44895.00 | 44743.07 | 44963.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 44895.00 | 44743.07 | 44963.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 44895.00 | 44743.07 | 44963.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 11:00:00 | 44520.00 | 44698.46 | 44923.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-28 14:15:00 | 45125.00 | 44792.50 | 44899.89 | SL hit (close>static) qty=1.00 sl=45105.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 45355.00 | 44997.16 | 44954.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 10:15:00 | 45820.00 | 45161.73 | 45032.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 45300.00 | 45347.20 | 45189.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 09:15:00 | 45400.00 | 45347.20 | 45189.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 45305.00 | 45338.76 | 45199.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 45305.00 | 45338.76 | 45199.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 45255.00 | 45322.01 | 45204.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:30:00 | 45320.00 | 45322.01 | 45204.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 45310.00 | 45319.61 | 45214.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:15:00 | 45065.00 | 45319.61 | 45214.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 45275.00 | 45310.68 | 45219.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:30:00 | 45235.00 | 45310.68 | 45219.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 45460.00 | 45340.55 | 45241.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 45555.00 | 45309.15 | 45243.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 14:15:00 | 45295.00 | 45849.60 | 45895.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 45295.00 | 45849.60 | 45895.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 44880.00 | 45655.68 | 45803.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 45430.00 | 45336.98 | 45540.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 14:15:00 | 45430.00 | 45336.98 | 45540.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 45430.00 | 45336.98 | 45540.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 45430.00 | 45336.98 | 45540.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 45650.00 | 45399.58 | 45550.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 46550.00 | 45399.58 | 45550.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 46860.00 | 45691.67 | 45669.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 47020.00 | 46453.52 | 46098.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 46525.00 | 46624.52 | 46338.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 46525.00 | 46624.52 | 46338.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 46375.00 | 46515.73 | 46372.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:45:00 | 47110.00 | 46478.04 | 46403.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:00:00 | 46840.00 | 46641.95 | 46496.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 15:15:00 | 47455.00 | 47642.81 | 47654.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 15:15:00 | 47455.00 | 47642.81 | 47654.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 47430.00 | 47600.25 | 47634.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 47730.00 | 47626.20 | 47642.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 10:15:00 | 47730.00 | 47626.20 | 47642.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 47730.00 | 47626.20 | 47642.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 47730.00 | 47626.20 | 47642.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 47920.00 | 47684.96 | 47668.00 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 47450.00 | 47692.13 | 47711.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 47325.00 | 47601.57 | 47665.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 46575.00 | 46184.84 | 46452.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 46575.00 | 46184.84 | 46452.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 46575.00 | 46184.84 | 46452.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 46575.00 | 46184.84 | 46452.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 46100.00 | 46167.87 | 46420.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 46805.00 | 46167.87 | 46420.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 46770.00 | 46288.30 | 46452.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 46770.00 | 46288.30 | 46452.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 46495.00 | 46329.64 | 46456.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 11:45:00 | 46310.00 | 46315.71 | 46438.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 46510.00 | 46085.78 | 46044.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 46510.00 | 46085.78 | 46044.14 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 14:15:00 | 45755.00 | 46010.81 | 46032.19 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 14:15:00 | 46100.00 | 46023.26 | 46019.61 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 15:15:00 | 45950.00 | 46008.60 | 46013.29 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 46250.00 | 46056.88 | 46034.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 46765.00 | 46312.17 | 46178.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 46650.00 | 46663.55 | 46459.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:15:00 | 47125.00 | 46663.55 | 46459.89 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 12:45:00 | 46815.00 | 46847.26 | 46629.40 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 46745.00 | 46826.81 | 46639.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 46765.00 | 46826.81 | 46639.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 46695.00 | 46780.16 | 46649.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 46760.00 | 46780.16 | 46649.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 46820.00 | 46788.13 | 46665.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 46605.00 | 46751.50 | 46659.86 | SL hit (close<ema400) qty=1.00 sl=46659.86 alert=retest1 |

### Cycle 165 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 46110.00 | 46609.65 | 46615.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 45750.00 | 46340.86 | 46482.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 46195.00 | 46078.41 | 46273.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 46195.00 | 46078.41 | 46273.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 46195.00 | 46078.41 | 46273.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 46195.00 | 46078.41 | 46273.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 46275.00 | 46117.73 | 46273.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 45870.00 | 46117.73 | 46273.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 45850.00 | 46064.18 | 46234.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 45485.00 | 45773.62 | 45897.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:00:00 | 45530.00 | 45724.90 | 45864.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 14:15:00 | 45440.00 | 45110.19 | 45093.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 45440.00 | 45110.19 | 45093.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 46495.00 | 45449.52 | 45254.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 45875.00 | 45888.24 | 45562.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 45920.00 | 45888.24 | 45562.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 48490.00 | 48686.64 | 48152.61 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 47390.00 | 48343.62 | 48461.54 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 48395.00 | 48197.99 | 48179.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 13:15:00 | 48685.00 | 48403.18 | 48291.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 13:15:00 | 48480.00 | 48626.36 | 48490.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 13:15:00 | 48480.00 | 48626.36 | 48490.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 48480.00 | 48626.36 | 48490.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:00:00 | 48480.00 | 48626.36 | 48490.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 48950.00 | 48691.09 | 48532.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 15:15:00 | 49025.00 | 48725.82 | 48627.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 13:15:00 | 48495.00 | 48603.89 | 48609.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 48495.00 | 48603.89 | 48609.78 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 48740.00 | 48612.12 | 48608.82 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 48370.00 | 48563.70 | 48587.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 15:15:00 | 48115.00 | 48389.51 | 48490.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 48180.00 | 48101.26 | 48259.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 48180.00 | 48101.26 | 48259.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 48180.00 | 48101.26 | 48259.11 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 48835.00 | 48342.45 | 48318.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 48880.00 | 48459.17 | 48376.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 48670.00 | 48814.21 | 48642.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 48670.00 | 48814.21 | 48642.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 48670.00 | 48814.21 | 48642.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 48670.00 | 48814.21 | 48642.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 48355.00 | 48722.37 | 48616.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 48355.00 | 48722.37 | 48616.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 47975.00 | 48572.89 | 48558.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 47975.00 | 48572.89 | 48558.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 12:15:00 | 47990.00 | 48456.31 | 48506.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 13:15:00 | 47440.00 | 48253.05 | 48409.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 11:15:00 | 46480.00 | 46424.75 | 46645.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 12:00:00 | 46480.00 | 46424.75 | 46645.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 46600.00 | 46472.64 | 46629.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:45:00 | 46570.00 | 46472.64 | 46629.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 47535.00 | 46685.11 | 46712.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 47535.00 | 46685.11 | 46712.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 47200.00 | 46788.09 | 46756.60 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 46420.00 | 46781.47 | 46808.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 14:15:00 | 46300.00 | 46574.99 | 46697.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 46655.00 | 46578.99 | 46677.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 46655.00 | 46578.99 | 46677.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 46655.00 | 46578.99 | 46677.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 46725.00 | 46578.99 | 46677.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 46720.00 | 46607.20 | 46681.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:30:00 | 46710.00 | 46607.20 | 46681.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 46515.00 | 46588.76 | 46666.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:30:00 | 46645.00 | 46588.76 | 46666.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 46460.00 | 46531.84 | 46618.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 15:00:00 | 46460.00 | 46531.84 | 46618.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 46850.00 | 46598.38 | 46633.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 46850.00 | 46598.38 | 46633.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 46880.00 | 46654.70 | 46656.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 46880.00 | 46654.70 | 46656.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 11:15:00 | 46995.00 | 46722.76 | 46686.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 47305.00 | 46839.21 | 46743.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 47910.00 | 48380.96 | 48014.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 47910.00 | 48380.96 | 48014.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 47910.00 | 48380.96 | 48014.23 | EMA400 retest candle locked (from upside) |

### Cycle 177 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 46920.00 | 47658.83 | 47751.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 46755.00 | 47478.06 | 47660.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 11:15:00 | 46195.00 | 45944.00 | 46300.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 12:00:00 | 46195.00 | 45944.00 | 46300.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 46165.00 | 46029.96 | 46279.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:30:00 | 46375.00 | 46029.96 | 46279.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 46315.00 | 46086.97 | 46283.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 46315.00 | 46086.97 | 46283.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 46160.00 | 46101.57 | 46271.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 46165.00 | 46101.57 | 46271.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 45860.00 | 46053.26 | 46234.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 44000.00 | 45853.35 | 46040.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 12:15:00 | 44515.00 | 44111.76 | 44076.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 44515.00 | 44111.76 | 44076.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 13:15:00 | 44605.00 | 44210.41 | 44124.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 10:15:00 | 45215.00 | 45275.06 | 44883.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:00:00 | 45215.00 | 45275.06 | 44883.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 45715.00 | 45873.29 | 45584.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 45505.00 | 45873.29 | 45584.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 45615.00 | 45807.73 | 45626.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:30:00 | 45555.00 | 45807.73 | 45626.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 45665.00 | 45779.18 | 45629.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 45665.00 | 45779.18 | 45629.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 45650.00 | 45753.34 | 45631.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:30:00 | 45640.00 | 45753.34 | 45631.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 45585.00 | 45719.68 | 45627.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 46530.00 | 45719.68 | 45627.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 46540.00 | 45883.74 | 45710.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:30:00 | 46710.00 | 46159.59 | 45873.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 13:15:00 | 46780.00 | 46257.68 | 45943.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 09:45:00 | 46860.00 | 46644.72 | 46256.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 15:15:00 | 45715.00 | 46057.84 | 46100.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 45715.00 | 46057.84 | 46100.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 44855.00 | 45817.27 | 45987.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 44420.00 | 44356.83 | 44909.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:45:00 | 44500.00 | 44356.83 | 44909.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 44520.00 | 44326.35 | 44676.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 44520.00 | 44326.35 | 44676.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 44685.00 | 44398.08 | 44677.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 44665.00 | 44398.08 | 44677.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 44620.00 | 44442.46 | 44671.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:15:00 | 44825.00 | 44442.46 | 44671.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 44840.00 | 44521.97 | 44687.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:45:00 | 44840.00 | 44521.97 | 44687.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 44830.00 | 44583.58 | 44700.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 44830.00 | 44583.58 | 44700.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 44530.00 | 44643.40 | 44700.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:30:00 | 44495.00 | 44635.72 | 44691.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 44815.00 | 44697.86 | 44712.05 | SL hit (close>static) qty=1.00 sl=44805.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 14:15:00 | 45245.00 | 44807.29 | 44760.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 45590.00 | 44988.26 | 44852.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 15:15:00 | 45225.00 | 45251.09 | 45074.43 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-04 09:15:00 | 45515.00 | 45251.09 | 45074.43 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 45050.00 | 45219.50 | 45091.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 45050.00 | 45219.50 | 45091.27 | SL hit (close<ema400) qty=1.00 sl=45091.27 alert=retest1 |

### Cycle 181 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 44610.00 | 44971.52 | 45003.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 44585.00 | 44842.58 | 44933.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 44455.00 | 44430.64 | 44597.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 13:00:00 | 44455.00 | 44430.64 | 44597.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 44270.00 | 44407.21 | 44558.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 44240.00 | 44407.21 | 44558.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:15:00 | 44245.00 | 44256.02 | 44397.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 12:15:00 | 44245.00 | 44372.74 | 44406.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:30:00 | 44260.00 | 44387.15 | 44406.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 44475.00 | 44404.72 | 44412.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 44475.00 | 44404.72 | 44412.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 44410.00 | 44405.78 | 44412.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 44375.00 | 44405.78 | 44412.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 44215.00 | 44367.62 | 44394.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:00:00 | 44110.00 | 44316.10 | 44368.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:00:00 | 44110.00 | 44256.30 | 44331.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 44500.00 | 44300.49 | 44291.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 44500.00 | 44300.49 | 44291.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 45100.00 | 44460.39 | 44365.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 14:15:00 | 45395.00 | 45397.98 | 45095.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:30:00 | 45355.00 | 45397.98 | 45095.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 44590.00 | 45214.31 | 45063.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 44590.00 | 45214.31 | 45063.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 44390.00 | 45049.44 | 45002.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 44360.00 | 45049.44 | 45002.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 44155.00 | 44870.56 | 44925.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 43930.00 | 44388.49 | 44637.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 42580.00 | 42577.38 | 42894.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 10:00:00 | 42580.00 | 42577.38 | 42894.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 42770.00 | 42649.12 | 42873.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:00:00 | 42390.00 | 42647.79 | 42822.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 42110.00 | 41496.85 | 41439.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 42110.00 | 41496.85 | 41439.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 42325.00 | 41882.59 | 41660.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 12:15:00 | 41915.00 | 41927.69 | 41740.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 12:15:00 | 41915.00 | 41927.69 | 41740.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 41915.00 | 41927.69 | 41740.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:00:00 | 41915.00 | 41927.69 | 41740.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 42125.00 | 42285.53 | 42141.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 42115.00 | 42285.53 | 42141.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 42100.00 | 42248.43 | 42137.63 | EMA400 retest candle locked (from upside) |

### Cycle 185 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 41665.00 | 42095.44 | 42108.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 41625.00 | 41850.46 | 41950.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 41860.00 | 41850.69 | 41932.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 10:15:00 | 41860.00 | 41850.69 | 41932.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 41860.00 | 41850.69 | 41932.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:30:00 | 41900.00 | 41850.69 | 41932.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 41715.00 | 41711.03 | 41829.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:45:00 | 41800.00 | 41711.03 | 41829.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 41790.00 | 41725.06 | 41815.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:15:00 | 41690.00 | 41733.64 | 41803.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 40975.00 | 40840.11 | 40834.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 40975.00 | 40840.11 | 40834.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 41190.00 | 41018.92 | 40949.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 41460.00 | 41671.74 | 41496.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 41460.00 | 41671.74 | 41496.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 41460.00 | 41671.74 | 41496.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 41460.00 | 41671.74 | 41496.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 41460.00 | 41629.39 | 41492.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 41385.00 | 41629.39 | 41492.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 41500.00 | 41603.51 | 41493.56 | EMA400 retest candle locked (from upside) |

### Cycle 187 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 41135.00 | 41412.48 | 41424.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 40900.00 | 41309.98 | 41377.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 40905.00 | 40867.35 | 41011.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 40905.00 | 40867.35 | 41011.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 40905.00 | 40867.35 | 41011.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 40905.00 | 40867.35 | 41011.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 41000.00 | 40893.88 | 41010.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 41190.00 | 40893.88 | 41010.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 40920.00 | 40899.10 | 41002.34 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 41190.00 | 41061.06 | 41054.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 41335.00 | 41115.85 | 41080.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 40790.00 | 41085.74 | 41074.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 40790.00 | 41085.74 | 41074.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 40790.00 | 41085.74 | 41074.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 40790.00 | 41085.74 | 41074.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 10:15:00 | 40925.00 | 41053.59 | 41061.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 12:15:00 | 40680.00 | 40938.30 | 41004.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 15:15:00 | 41000.00 | 40887.77 | 40959.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 15:15:00 | 41000.00 | 40887.77 | 40959.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 41000.00 | 40887.77 | 40959.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 40980.00 | 40887.77 | 40959.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 41020.00 | 40914.22 | 40964.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:30:00 | 41125.00 | 40914.22 | 40964.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 41030.00 | 40937.37 | 40970.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 41015.00 | 40937.37 | 40970.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 41185.00 | 40986.90 | 40990.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 41185.00 | 40986.90 | 40990.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 41160.00 | 41021.52 | 41005.68 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 40825.00 | 40995.38 | 41003.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 15:15:00 | 40600.00 | 40785.05 | 40879.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 39735.00 | 39734.24 | 39965.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 14:30:00 | 39700.00 | 39734.24 | 39965.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 39840.00 | 39757.91 | 39936.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 39900.00 | 39757.91 | 39936.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 40020.00 | 39810.33 | 39943.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 40020.00 | 39810.33 | 39943.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 40205.00 | 39889.26 | 39967.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 40205.00 | 39889.26 | 39967.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 40160.00 | 39943.41 | 39985.13 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 40150.00 | 40016.18 | 40012.91 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 39835.00 | 39986.96 | 40000.63 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 40085.00 | 40008.65 | 40008.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 40240.00 | 40059.63 | 40032.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 39720.00 | 40646.97 | 40517.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 14:15:00 | 39720.00 | 40646.97 | 40517.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 39720.00 | 40646.97 | 40517.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:45:00 | 39935.00 | 40646.97 | 40517.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 39350.00 | 40387.57 | 40411.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 38990.00 | 39549.18 | 39816.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 15:15:00 | 39375.00 | 39336.56 | 39563.32 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 09:15:00 | 39030.00 | 39336.56 | 39563.32 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 38910.00 | 38778.62 | 38882.09 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-21 14:15:00 | 38910.00 | 38778.62 | 38882.09 | SL hit (close>ema400) qty=1.00 sl=38882.09 alert=retest1 |

### Cycle 196 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 38920.00 | 38783.21 | 38776.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 39005.00 | 38827.57 | 38797.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 13:15:00 | 38800.00 | 38877.27 | 38845.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 13:15:00 | 38800.00 | 38877.27 | 38845.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 38800.00 | 38877.27 | 38845.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 38800.00 | 38877.27 | 38845.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 38975.00 | 38896.82 | 38857.18 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 38600.00 | 38823.56 | 38829.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 38410.00 | 38740.85 | 38791.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 12:15:00 | 37550.00 | 37519.28 | 37863.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 13:00:00 | 37550.00 | 37519.28 | 37863.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 37505.00 | 37289.86 | 37503.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:30:00 | 37150.00 | 37353.74 | 37425.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 36840.00 | 37222.27 | 37318.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 11:15:00 | 37115.00 | 37163.45 | 37271.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 12:15:00 | 37130.00 | 37169.76 | 37264.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 37280.00 | 37191.81 | 37266.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 37280.00 | 37191.81 | 37266.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 37265.00 | 37206.45 | 37265.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 37180.00 | 37206.45 | 37265.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 37080.00 | 37199.02 | 37244.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 11:15:00 | 35321.00 | 35703.87 | 35965.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 35730.00 | 35709.10 | 35944.36 | SL hit (close>ema200) qty=0.50 sl=35709.10 alert=retest2 |

### Cycle 198 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 36565.00 | 36031.06 | 36012.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 13:15:00 | 36700.00 | 36315.58 | 36159.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 36375.00 | 36460.90 | 36293.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:00:00 | 36375.00 | 36460.90 | 36293.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 36455.00 | 36459.72 | 36308.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 36350.00 | 36459.72 | 36308.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 36625.00 | 36575.65 | 36443.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:30:00 | 36510.00 | 36575.65 | 36443.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 36360.00 | 36566.16 | 36505.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 36360.00 | 36566.16 | 36505.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 36500.00 | 36552.93 | 36505.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:45:00 | 36535.00 | 36532.34 | 36500.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 36265.00 | 36442.22 | 36464.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 36265.00 | 36442.22 | 36464.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 36115.00 | 36376.78 | 36432.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 35900.00 | 35865.63 | 36084.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 35775.00 | 35865.63 | 36084.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 35655.00 | 35823.51 | 36045.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 35585.00 | 35850.84 | 35962.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:00:00 | 35535.00 | 35850.84 | 35962.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:45:00 | 35625.00 | 35764.94 | 35901.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 35570.00 | 35732.95 | 35874.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 35820.00 | 35709.07 | 35803.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:30:00 | 35860.00 | 35709.07 | 35803.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 35990.00 | 35765.26 | 35820.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:45:00 | 35955.00 | 35765.26 | 35820.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 35830.00 | 35798.96 | 35827.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 35635.00 | 35783.14 | 35814.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 33805.75 | 34149.31 | 34266.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 33758.25 | 34149.31 | 34266.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 33843.75 | 34149.31 | 34266.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 33791.50 | 34149.31 | 34266.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 33853.25 | 34149.31 | 34266.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 15:15:00 | 32026.50 | 33511.58 | 33861.28 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 200 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 33015.00 | 32628.19 | 32601.38 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 32685.00 | 32817.66 | 32818.54 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 33000.00 | 32820.47 | 32813.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 33605.00 | 32977.38 | 32885.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 09:15:00 | 35195.00 | 35295.72 | 34828.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:00:00 | 35195.00 | 35295.72 | 34828.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 34725.00 | 35126.26 | 34829.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 34725.00 | 35126.26 | 34829.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 34640.00 | 35029.01 | 34812.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:45:00 | 34490.00 | 35029.01 | 34812.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 34815.00 | 34958.37 | 34815.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:15:00 | 34850.00 | 34958.37 | 34815.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 34850.00 | 34936.69 | 34818.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 34920.00 | 34936.69 | 34818.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 34745.00 | 34898.35 | 34812.23 | SL hit (close<static) qty=1.00 sl=34780.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 13:15:00 | 34580.00 | 34733.31 | 34752.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 34305.00 | 34612.77 | 34689.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 13:15:00 | 34550.00 | 34453.12 | 34574.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 14:00:00 | 34550.00 | 34453.12 | 34574.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 34390.00 | 34440.50 | 34557.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 33925.00 | 34437.40 | 34545.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 12:15:00 | 33120.00 | 33015.57 | 33014.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 12:15:00 | 33120.00 | 33015.57 | 33014.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 33360.00 | 33097.25 | 33052.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 10:15:00 | 33085.00 | 33106.44 | 33065.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 11:00:00 | 33085.00 | 33106.44 | 33065.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 33030.00 | 33091.15 | 33061.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:00:00 | 33030.00 | 33091.15 | 33061.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 32870.00 | 33046.92 | 33044.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 32870.00 | 33046.92 | 33044.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — SELL (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 13:15:00 | 32725.00 | 32982.54 | 33015.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 15:15:00 | 32670.00 | 32883.62 | 32962.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 14:15:00 | 32095.00 | 32094.25 | 32350.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 14:45:00 | 32185.00 | 32094.25 | 32350.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 31270.00 | 31083.08 | 31361.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 31270.00 | 31083.08 | 31361.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 31490.00 | 31164.47 | 31373.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 31490.00 | 31164.47 | 31373.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 31580.00 | 31247.57 | 31392.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:30:00 | 31580.00 | 31247.57 | 31392.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 31440.00 | 31329.00 | 31398.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 31695.00 | 31329.00 | 31398.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 31485.00 | 31360.20 | 31406.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:30:00 | 31340.00 | 31390.16 | 31415.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 12:15:00 | 31695.00 | 31481.00 | 31453.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 31695.00 | 31481.00 | 31453.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 13:15:00 | 31725.00 | 31529.80 | 31477.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 31285.00 | 31503.94 | 31481.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 31285.00 | 31503.94 | 31481.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 31285.00 | 31503.94 | 31481.11 | EMA400 retest candle locked (from upside) |

### Cycle 207 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 31120.00 | 31427.15 | 31448.28 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 31610.00 | 31331.33 | 31321.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 10:15:00 | 31860.00 | 31466.45 | 31386.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 31480.00 | 31549.46 | 31460.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:00:00 | 31480.00 | 31549.46 | 31460.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 31745.00 | 31588.57 | 31486.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 31080.00 | 31588.57 | 31486.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 31015.00 | 31473.85 | 31443.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 31040.00 | 31473.85 | 31443.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 31040.00 | 31387.08 | 31407.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 12:15:00 | 30790.00 | 31211.33 | 31320.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 30220.00 | 30150.44 | 30437.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 30255.00 | 30150.44 | 30437.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 30445.00 | 30193.28 | 30406.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 30380.00 | 30193.28 | 30406.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 30470.00 | 30248.63 | 30412.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 30530.00 | 30248.63 | 30412.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 30515.00 | 30301.90 | 30421.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:00:00 | 30515.00 | 30301.90 | 30421.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 30615.00 | 30364.52 | 30439.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:00:00 | 30615.00 | 30364.52 | 30439.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 30645.00 | 30420.62 | 30457.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 30645.00 | 30420.62 | 30457.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 31190.00 | 30603.92 | 30534.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 31335.00 | 30750.13 | 30607.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 31265.00 | 31359.19 | 31036.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:30:00 | 31145.00 | 31359.19 | 31036.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 31335.00 | 31328.08 | 31076.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:45:00 | 31370.00 | 31335.47 | 31103.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 13:45:00 | 31355.00 | 31329.37 | 31121.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 31920.00 | 31279.20 | 31133.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 11:00:00 | 31470.00 | 31746.93 | 31562.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 31395.00 | 31633.83 | 31540.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:00:00 | 31395.00 | 31633.83 | 31540.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 31375.00 | 31571.85 | 31527.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 15:00:00 | 31375.00 | 31571.85 | 31527.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-23 15:15:00 | 31175.00 | 31492.48 | 31495.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 31175.00 | 31492.48 | 31495.53 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 31990.00 | 31576.71 | 31531.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 32125.00 | 31686.37 | 31585.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 32675.00 | 32746.57 | 32313.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 32675.00 | 32746.57 | 32313.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 32025.00 | 32578.81 | 32310.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 32025.00 | 32578.81 | 32310.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 32120.00 | 32487.04 | 32293.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 32075.00 | 32487.04 | 32293.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 32140.00 | 32381.31 | 32276.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 32140.00 | 32381.31 | 32276.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 31975.00 | 32300.05 | 32249.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 31975.00 | 32300.05 | 32249.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 31800.00 | 32200.04 | 32208.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 31525.00 | 31828.67 | 31994.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 15:15:00 | 32185.00 | 31880.95 | 31988.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 15:15:00 | 32185.00 | 31880.95 | 31988.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 32185.00 | 31880.95 | 31988.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 32535.00 | 31880.95 | 31988.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 32430.00 | 31990.76 | 32028.61 | EMA400 retest candle locked (from downside) |

### Cycle 214 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 32445.00 | 32081.61 | 32066.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 32730.00 | 32211.29 | 32126.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 35760.00 | 35789.37 | 35372.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 09:15:00 | 35945.00 | 35789.37 | 35372.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 36090.00 | 36092.79 | 35796.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 36230.00 | 36115.23 | 35834.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 36225.00 | 36136.19 | 35869.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 36435.00 | 36048.24 | 35911.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 10:15:00 | 36240.00 | 36067.59 | 35933.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 37940.00 | 38054.94 | 37904.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:30:00 | 37980.00 | 38054.94 | 37904.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 37810.00 | 37978.77 | 37904.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:00:00 | 37810.00 | 37978.77 | 37904.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 37765.00 | 37936.01 | 37892.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:00:00 | 37765.00 | 37936.01 | 37892.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 37825.00 | 37895.25 | 37880.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 37700.00 | 37895.25 | 37880.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 37725.00 | 37861.20 | 37865.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 37725.00 | 37861.20 | 37865.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 37415.00 | 37771.96 | 37824.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 13:15:00 | 37755.00 | 37679.36 | 37761.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 13:15:00 | 37755.00 | 37679.36 | 37761.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 37755.00 | 37679.36 | 37761.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:00:00 | 37755.00 | 37679.36 | 37761.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 37960.00 | 37735.49 | 37779.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 37960.00 | 37735.49 | 37779.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 38000.00 | 37788.39 | 37799.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 38070.00 | 37788.39 | 37799.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — BUY (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 09:15:00 | 38130.00 | 37856.71 | 37829.86 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 37665.00 | 37798.84 | 37809.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 10:15:00 | 37575.00 | 37689.56 | 37744.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 37770.00 | 37680.12 | 37729.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 12:15:00 | 37770.00 | 37680.12 | 37729.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 37770.00 | 37680.12 | 37729.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:45:00 | 37765.00 | 37680.12 | 37729.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 37685.00 | 37681.10 | 37725.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 15:15:00 | 37600.00 | 37685.88 | 37723.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:45:00 | 37635.00 | 37674.96 | 37710.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:30:00 | 37535.00 | 37636.97 | 37690.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:00:00 | 37450.00 | 37509.37 | 37603.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 37515.00 | 37508.99 | 37587.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:30:00 | 37305.00 | 37455.20 | 37555.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 37405.00 | 37048.76 | 37027.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 37405.00 | 37048.76 | 37027.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 37625.00 | 37271.47 | 37149.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 37370.00 | 37409.87 | 37276.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:00:00 | 37370.00 | 37409.87 | 37276.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 37285.00 | 37384.89 | 37277.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 37285.00 | 37384.89 | 37277.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 37285.00 | 37364.92 | 37278.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 13:00:00 | 37335.00 | 37358.93 | 37283.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 13:30:00 | 37335.00 | 37350.15 | 37286.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 37415.00 | 37324.69 | 37285.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 37120.00 | 37283.75 | 37270.33 | SL hit (close<static) qty=1.00 sl=37180.00 alert=retest2 |

### Cycle 219 — SELL (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 10:15:00 | 37000.00 | 37227.00 | 37245.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 12:15:00 | 36920.00 | 37128.48 | 37195.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 13:15:00 | 37235.00 | 37149.79 | 37199.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 13:15:00 | 37235.00 | 37149.79 | 37199.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 37235.00 | 37149.79 | 37199.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 37235.00 | 37149.79 | 37199.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 37335.00 | 37186.83 | 37211.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:45:00 | 37470.00 | 37186.83 | 37211.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-16 14:45:00 | 35505.95 | 2024-04-22 10:15:00 | 35779.95 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-04-18 09:30:00 | 35398.20 | 2024-04-22 10:15:00 | 35779.95 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-04-19 09:15:00 | 34945.30 | 2024-04-22 10:15:00 | 35779.95 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-04-30 14:30:00 | 35026.65 | 2024-05-10 10:15:00 | 34499.10 | STOP_HIT | 1.00 | 1.51% |
| SELL | retest2 | 2024-05-13 10:15:00 | 34147.90 | 2024-05-13 14:15:00 | 34987.05 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2024-05-13 11:15:00 | 34126.80 | 2024-05-13 14:15:00 | 34987.05 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-05-15 14:45:00 | 34934.90 | 2024-05-21 12:15:00 | 35082.35 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2024-05-29 11:00:00 | 36745.40 | 2024-05-31 14:15:00 | 35940.00 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-05-29 12:15:00 | 36657.65 | 2024-05-31 14:15:00 | 35940.00 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-05-30 10:15:00 | 36647.75 | 2024-05-31 14:15:00 | 35940.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-05-30 10:45:00 | 36736.55 | 2024-05-31 14:15:00 | 35940.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-06-10 14:45:00 | 38564.40 | 2024-06-14 10:15:00 | 38464.05 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-06-11 09:15:00 | 38530.40 | 2024-06-14 10:15:00 | 38464.05 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2024-07-02 11:15:00 | 39380.00 | 2024-07-05 12:15:00 | 39530.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-07-11 09:15:00 | 39547.40 | 2024-07-11 11:15:00 | 39220.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-07-23 09:15:00 | 40641.85 | 2024-08-05 11:15:00 | 41612.55 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest2 | 2024-07-23 10:15:00 | 40422.20 | 2024-08-05 11:15:00 | 41612.55 | STOP_HIT | 1.00 | 2.94% |
| SELL | retest2 | 2024-08-06 13:30:00 | 41327.80 | 2024-08-13 10:15:00 | 41287.95 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2024-08-26 09:15:00 | 41900.05 | 2024-08-27 13:15:00 | 41500.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-09-03 11:00:00 | 42026.85 | 2024-09-03 11:15:00 | 41769.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-09-10 12:45:00 | 40535.50 | 2024-09-11 10:15:00 | 41414.15 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-09-10 13:15:00 | 40520.30 | 2024-09-11 10:15:00 | 41414.15 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-09-10 14:00:00 | 40549.90 | 2024-09-11 10:15:00 | 41414.15 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-09-17 09:15:00 | 43327.25 | 2024-09-18 13:15:00 | 42786.70 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-09-17 10:00:00 | 43174.65 | 2024-09-18 13:15:00 | 42786.70 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-09-18 15:15:00 | 43219.00 | 2024-09-19 09:15:00 | 42968.10 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-10-07 10:30:00 | 41172.95 | 2024-10-08 12:15:00 | 42025.05 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-10-07 12:00:00 | 41144.90 | 2024-10-08 12:15:00 | 42025.05 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-10-21 10:45:00 | 45104.65 | 2024-10-24 09:15:00 | 42849.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 10:45:00 | 45104.65 | 2024-10-24 14:15:00 | 43491.60 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2024-10-21 11:30:00 | 45080.00 | 2024-10-25 10:15:00 | 42826.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:00:00 | 44935.00 | 2024-10-25 10:15:00 | 42688.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:30:00 | 45080.00 | 2024-10-25 11:15:00 | 43170.15 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2024-10-21 12:00:00 | 44935.00 | 2024-10-25 11:15:00 | 43170.15 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2024-11-29 13:45:00 | 44700.00 | 2024-12-02 09:15:00 | 45216.25 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-11-29 14:30:00 | 44676.70 | 2024-12-02 09:15:00 | 45216.25 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-12-03 12:30:00 | 45876.80 | 2024-12-20 14:15:00 | 48812.60 | STOP_HIT | 1.00 | 6.40% |
| BUY | retest2 | 2024-12-03 13:45:00 | 45894.65 | 2024-12-20 14:15:00 | 48812.60 | STOP_HIT | 1.00 | 6.36% |
| BUY | retest2 | 2024-12-04 12:15:00 | 45994.20 | 2024-12-20 14:15:00 | 48812.60 | STOP_HIT | 1.00 | 6.13% |
| BUY | retest2 | 2024-12-05 15:00:00 | 45941.55 | 2024-12-20 14:15:00 | 48812.60 | STOP_HIT | 1.00 | 6.25% |
| BUY | retest2 | 2024-12-06 09:15:00 | 46290.10 | 2024-12-20 14:15:00 | 48812.60 | STOP_HIT | 1.00 | 5.45% |
| BUY | retest2 | 2024-12-06 15:15:00 | 46400.00 | 2024-12-20 14:15:00 | 48812.60 | STOP_HIT | 1.00 | 5.20% |
| BUY | retest2 | 2024-12-09 10:45:00 | 46064.55 | 2024-12-20 14:15:00 | 48812.60 | STOP_HIT | 1.00 | 5.97% |
| SELL | retest2 | 2024-12-24 12:00:00 | 48790.00 | 2024-12-31 14:15:00 | 47448.75 | STOP_HIT | 1.00 | 2.75% |
| SELL | retest2 | 2024-12-24 12:45:00 | 48775.85 | 2024-12-31 14:15:00 | 47448.75 | STOP_HIT | 1.00 | 2.72% |
| SELL | retest2 | 2024-12-24 15:15:00 | 48779.00 | 2024-12-31 14:15:00 | 47448.75 | STOP_HIT | 1.00 | 2.73% |
| BUY | retest2 | 2025-01-02 13:15:00 | 47858.25 | 2025-01-08 11:15:00 | 47758.25 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-01-06 09:30:00 | 47861.70 | 2025-01-08 12:15:00 | 47299.95 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-01-06 10:15:00 | 48029.65 | 2025-01-08 12:15:00 | 47299.95 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-01-06 11:15:00 | 47923.10 | 2025-01-08 12:15:00 | 47299.95 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-01-07 09:15:00 | 48425.95 | 2025-01-08 12:15:00 | 47299.95 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-01-15 09:30:00 | 45797.15 | 2025-01-20 14:15:00 | 46325.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-01-15 10:45:00 | 46088.05 | 2025-01-20 14:15:00 | 46325.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-01-16 09:45:00 | 46016.00 | 2025-01-20 14:15:00 | 46325.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-01-16 11:45:00 | 45972.45 | 2025-01-20 14:15:00 | 46325.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-01-29 12:30:00 | 45704.30 | 2025-01-29 13:15:00 | 45600.00 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-01-29 13:00:00 | 45708.65 | 2025-01-29 13:15:00 | 45600.00 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-01-31 14:00:00 | 44720.00 | 2025-02-01 12:15:00 | 46430.00 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2025-02-01 09:15:00 | 44285.45 | 2025-02-01 12:15:00 | 46430.00 | STOP_HIT | 1.00 | -4.84% |
| BUY | retest2 | 2025-02-04 11:15:00 | 46540.40 | 2025-02-06 09:15:00 | 43748.65 | STOP_HIT | 1.00 | -6.00% |
| BUY | retest2 | 2025-02-04 12:00:00 | 46335.00 | 2025-02-06 09:15:00 | 43748.65 | STOP_HIT | 1.00 | -5.58% |
| SELL | retest2 | 2025-02-11 09:15:00 | 42803.45 | 2025-02-14 12:15:00 | 40663.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 42803.45 | 2025-02-17 14:15:00 | 40931.70 | STOP_HIT | 0.50 | 4.37% |
| BUY | retest2 | 2025-02-20 11:30:00 | 41220.85 | 2025-02-21 15:15:00 | 40900.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-02-20 12:45:00 | 41195.55 | 2025-02-21 15:15:00 | 40900.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-02-21 14:00:00 | 41196.45 | 2025-02-21 15:15:00 | 40900.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-02-21 15:00:00 | 41208.85 | 2025-02-21 15:15:00 | 40900.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-02-25 09:30:00 | 41577.65 | 2025-02-28 10:15:00 | 40958.05 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-03-12 11:30:00 | 39700.00 | 2025-03-12 14:15:00 | 40178.95 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-03-12 12:15:00 | 39725.00 | 2025-03-12 14:15:00 | 40178.95 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-03-18 10:15:00 | 40483.90 | 2025-03-26 15:15:00 | 42215.45 | STOP_HIT | 1.00 | 4.28% |
| BUY | retest2 | 2025-03-18 12:00:00 | 40500.00 | 2025-03-26 15:15:00 | 42215.45 | STOP_HIT | 1.00 | 4.24% |
| BUY | retest2 | 2025-03-18 13:00:00 | 40622.70 | 2025-03-26 15:15:00 | 42215.45 | STOP_HIT | 1.00 | 3.92% |
| SELL | retest2 | 2025-04-09 09:45:00 | 41453.95 | 2025-04-09 13:15:00 | 42499.95 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-04-11 11:45:00 | 42526.80 | 2025-04-24 13:15:00 | 45055.00 | STOP_HIT | 1.00 | 5.94% |
| SELL | retest2 | 2025-04-28 11:00:00 | 44520.00 | 2025-04-28 14:15:00 | 45125.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-05-05 09:15:00 | 45555.00 | 2025-05-08 14:15:00 | 45295.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-05-15 14:45:00 | 47110.00 | 2025-05-21 15:15:00 | 47455.00 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2025-05-16 10:00:00 | 46840.00 | 2025-05-21 15:15:00 | 47455.00 | STOP_HIT | 1.00 | 1.31% |
| SELL | retest2 | 2025-05-30 11:45:00 | 46310.00 | 2025-06-05 09:15:00 | 46510.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-06-11 09:15:00 | 47125.00 | 2025-06-12 10:15:00 | 46605.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest1 | 2025-06-11 12:45:00 | 46815.00 | 2025-06-12 10:15:00 | 46605.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-06-18 09:15:00 | 45485.00 | 2025-06-23 14:15:00 | 45440.00 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-06-18 10:00:00 | 45530.00 | 2025-06-23 14:15:00 | 45440.00 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-07-08 15:15:00 | 49025.00 | 2025-07-09 13:15:00 | 48495.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-08-08 09:15:00 | 44000.00 | 2025-08-18 12:15:00 | 44515.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-08-25 11:30:00 | 46710.00 | 2025-08-26 15:15:00 | 45715.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-08-25 13:15:00 | 46780.00 | 2025-08-26 15:15:00 | 45715.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-08-26 09:45:00 | 46860.00 | 2025-08-26 15:15:00 | 45715.00 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-09-02 11:30:00 | 44495.00 | 2025-09-02 13:15:00 | 44815.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2025-09-04 09:15:00 | 45515.00 | 2025-09-04 10:15:00 | 45050.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-09-08 15:15:00 | 44240.00 | 2025-09-15 15:15:00 | 44500.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-09-09 14:15:00 | 44245.00 | 2025-09-15 15:15:00 | 44500.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-09-10 12:15:00 | 44245.00 | 2025-09-15 15:15:00 | 44500.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-09-10 13:30:00 | 44260.00 | 2025-09-15 15:15:00 | 44500.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-09-11 11:00:00 | 44110.00 | 2025-09-15 15:15:00 | 44500.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-09-11 13:00:00 | 44110.00 | 2025-09-15 15:15:00 | 44500.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-09-25 15:00:00 | 42390.00 | 2025-10-01 12:15:00 | 42110.00 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2025-10-10 12:15:00 | 41690.00 | 2025-10-16 13:15:00 | 40975.00 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest1 | 2025-11-19 09:15:00 | 39030.00 | 2025-11-21 14:15:00 | 38910.00 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-11-24 11:45:00 | 38800.00 | 2025-11-26 13:15:00 | 38920.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-11-24 13:30:00 | 38850.00 | 2025-11-26 13:15:00 | 38920.00 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-11-24 15:15:00 | 38835.00 | 2025-11-26 13:15:00 | 38920.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-11-26 11:15:00 | 38865.00 | 2025-11-26 13:15:00 | 38920.00 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-12-08 10:30:00 | 37150.00 | 2025-12-19 11:15:00 | 35321.00 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-12-08 10:30:00 | 37150.00 | 2025-12-19 12:15:00 | 35730.00 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2025-12-09 09:15:00 | 36840.00 | 2025-12-22 10:15:00 | 36565.00 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2025-12-09 11:15:00 | 37115.00 | 2025-12-22 10:15:00 | 36565.00 | STOP_HIT | 1.00 | 1.48% |
| SELL | retest2 | 2025-12-09 12:15:00 | 37130.00 | 2025-12-22 10:15:00 | 36565.00 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2025-12-09 14:15:00 | 37180.00 | 2025-12-22 10:15:00 | 36565.00 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2025-12-10 10:45:00 | 37080.00 | 2025-12-22 10:15:00 | 36565.00 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2025-12-26 11:45:00 | 36535.00 | 2025-12-29 09:15:00 | 36265.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-01-01 09:30:00 | 35585.00 | 2026-01-20 09:15:00 | 33805.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 10:00:00 | 35535.00 | 2026-01-20 09:15:00 | 33758.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 11:45:00 | 35625.00 | 2026-01-20 09:15:00 | 33843.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 12:30:00 | 35570.00 | 2026-01-20 09:15:00 | 33791.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 35635.00 | 2026-01-20 09:15:00 | 33853.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 09:30:00 | 35585.00 | 2026-01-20 15:15:00 | 32026.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-01 10:00:00 | 35535.00 | 2026-01-20 15:15:00 | 31981.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-01 11:45:00 | 35625.00 | 2026-01-20 15:15:00 | 32062.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-01 12:30:00 | 35570.00 | 2026-01-20 15:15:00 | 32013.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 35635.00 | 2026-01-20 15:15:00 | 32071.50 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-10 09:15:00 | 34920.00 | 2026-02-10 09:15:00 | 34745.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-02-12 09:15:00 | 33925.00 | 2026-02-24 12:15:00 | 33120.00 | STOP_HIT | 1.00 | 2.37% |
| SELL | retest2 | 2026-03-06 09:30:00 | 31340.00 | 2026-03-06 12:15:00 | 31695.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-03-19 12:45:00 | 31370.00 | 2026-03-23 15:15:00 | 31175.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-03-19 13:45:00 | 31355.00 | 2026-03-23 15:15:00 | 31175.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-03-20 09:15:00 | 31920.00 | 2026-03-23 15:15:00 | 31175.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-03-23 11:00:00 | 31470.00 | 2026-03-23 15:15:00 | 31175.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-04-13 11:15:00 | 36230.00 | 2026-04-23 09:15:00 | 37725.00 | STOP_HIT | 1.00 | 4.13% |
| BUY | retest2 | 2026-04-13 11:45:00 | 36225.00 | 2026-04-23 09:15:00 | 37725.00 | STOP_HIT | 1.00 | 4.14% |
| BUY | retest2 | 2026-04-15 09:15:00 | 36435.00 | 2026-04-23 09:15:00 | 37725.00 | STOP_HIT | 1.00 | 3.54% |
| BUY | retest2 | 2026-04-15 10:15:00 | 36240.00 | 2026-04-23 09:15:00 | 37725.00 | STOP_HIT | 1.00 | 4.10% |
| SELL | retest2 | 2026-04-27 15:15:00 | 37600.00 | 2026-05-05 14:15:00 | 37405.00 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2026-04-28 09:45:00 | 37635.00 | 2026-05-05 14:15:00 | 37405.00 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2026-04-28 10:30:00 | 37535.00 | 2026-05-05 14:15:00 | 37405.00 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2026-04-28 15:00:00 | 37450.00 | 2026-05-05 14:15:00 | 37405.00 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2026-04-29 10:30:00 | 37305.00 | 2026-05-05 14:15:00 | 37405.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-05-07 13:00:00 | 37335.00 | 2026-05-08 09:15:00 | 37120.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2026-05-07 13:30:00 | 37335.00 | 2026-05-08 09:15:00 | 37120.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2026-05-08 09:15:00 | 37415.00 | 2026-05-08 09:15:00 | 37120.00 | STOP_HIT | 1.00 | -0.79% |
