# Persistent Systems Ltd. (PERSISTENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4804.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 1
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 206.03
- **Avg P&L per closed trade:** 51.51

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 13:15:00 | 5501.65 | 6002.47 | 6003.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 11:15:00 | 5439.90 | 5885.79 | 5939.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 12:15:00 | 5475.00 | 5448.30 | 5637.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-01 09:15:00 | 5369.20 | 5477.46 | 5625.54 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-24 09:15:00 | 5308.00 | 5058.08 | 5305.09 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 09:15:00 | 5764.50 | 5411.24 | 5410.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 5936.00 | 5551.39 | 5499.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 5873.00 | 5890.37 | 5744.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-08 12:15:00 | 5946.00 | 5886.11 | 5752.97 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 5787.00 | 5884.09 | 5753.28 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-09 09:15:00 | 5736.00 | 5881.49 | 5753.27 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 5174.00 | 5681.19 | 5681.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 5159.00 | 5670.87 | 5676.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 5346.00 | 5342.19 | 5460.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-05 09:15:00 | 5154.50 | 5354.26 | 5431.14 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 5391.50 | 5309.17 | 5398.28 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-10 13:15:00 | 5402.00 | 5310.09 | 5398.30 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 5851.40 | 5368.54 | 5366.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 5918.10 | 5391.71 | 5378.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 12:15:00 | 6140.50 | 6182.49 | 5952.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-11 15:15:00 | 6225.50 | 6175.79 | 5959.85 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-21 09:15:00 | 6120.00 | 6317.90 | 6187.86 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 5723.00 | 6125.83 | 6127.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 15:15:00 | 5703.00 | 6102.95 | 6115.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 5039.00 | 4963.44 | 5276.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-24 10:15:00 | 4907.80 | 5185.40 | 5290.09 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-01 09:15:00 | 5369.20 | 2025-04-04 09:15:00 | 4600.17 | TARGET | 769.03 |
| BUY | 2025-07-08 12:15:00 | 5946.00 | 2025-07-09 09:15:00 | 5736.00 | EXIT_EMA400 | -210.00 |
| SELL | 2025-09-05 09:15:00 | 5154.50 | 2025-09-10 13:15:00 | 5402.00 | EXIT_EMA400 | -247.50 |
| BUY | 2025-12-11 15:15:00 | 6225.50 | 2026-01-21 09:15:00 | 6120.00 | EXIT_EMA400 | -105.50 |
