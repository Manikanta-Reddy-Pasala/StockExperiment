# Indian Hotels Co. Ltd. (INDHOTEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 638.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -53.60
- **Avg P&L per closed trade:** -17.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 726.90 | 786.68 | 786.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 721.80 | 786.03 | 786.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 14:15:00 | 757.75 | 753.30 | 766.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-06 11:15:00 | 747.40 | 753.30 | 766.07 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-13 10:15:00 | 763.70 | 751.91 | 763.31 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 09:15:00 | 844.05 | 772.04 | 771.69 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 13:15:00 | 762.50 | 785.02 | 785.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 755.45 | 779.85 | 782.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 10:15:00 | 779.85 | 777.33 | 780.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-27 09:15:00 | 765.25 | 777.10 | 780.42 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-02 14:15:00 | 784.00 | 774.38 | 778.42 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 13:15:00 | 807.85 | 760.98 | 760.76 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 709.30 | 765.47 | 765.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 708.85 | 742.01 | 748.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 728.25 | 726.81 | 737.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-24 15:15:00 | 718.00 | 727.19 | 736.41 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-27 10:15:00 | 736.55 | 727.53 | 735.87 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-06 11:15:00 | 747.40 | 2025-03-13 10:15:00 | 763.70 | EXIT_EMA400 | -16.30 |
| SELL | 2025-05-27 09:15:00 | 765.25 | 2025-06-02 14:15:00 | 784.00 | EXIT_EMA400 | -18.75 |
| SELL | 2025-11-24 15:15:00 | 718.00 | 2025-11-27 10:15:00 | 736.55 | EXIT_EMA400 | -18.55 |
