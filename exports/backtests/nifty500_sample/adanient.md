# Adani Enterprises Ltd. (ADANIENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 2408.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Total realized P&L (per unit):** -160.20
- **Avg P&L per closed trade:** -26.70

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2024-07-29 12:15:00 | ENTRY1 | SELL | 3089.90 | 3099.53 | 3119.59 | Sell entry 1 (retest1 break) |
| 2024-07-30 11:15:00 | EXIT | SELL | 3119.60 | 3099.04 | 3118.74 | Close above EMA400 |
| 2024-10-11 13:15:00 | CROSSOVER | BUY | 3128.80 | 3077.40 | 3077.34 | EMA200 above EMA400 |
| 2024-10-18 10:15:00 | CROSSOVER | SELL | 3006.60 | 3077.15 | 3077.50 | EMA200 below EMA400 |
| 2024-10-18 12:15:00 | ALERT1 | SELL | 2998.90 | 3075.78 | 3076.81 | Break + close below crossover candle low |
| 2024-10-30 11:15:00 | ALERT2 | SELL | 2980.00 | 2975.07 | 3020.06 | EMA200 retest candle locked |
| 2024-10-30 15:15:00 | ENTRY1 | SELL | 2959.00 | 2974.78 | 3019.02 | Sell entry 1 (retest1 break) |
| 2024-11-06 11:15:00 | EXIT | SELL | 3022.65 | 2964.08 | 3008.34 | Close above EMA400 |
| 2025-04-24 10:15:00 | CROSSOVER | BUY | 2451.50 | 2342.97 | 2342.53 | EMA200 above EMA400 |
| 2025-05-05 11:15:00 | ALERT1 | BUY | 2478.70 | 2347.46 | 2345.34 | Break + close above crossover candle high |
| 2025-05-06 15:15:00 | ALERT2 | BUY | 2349.00 | 2353.29 | 2348.48 | EMA200 retest candle locked |
| 2025-05-09 15:15:00 | CROSSOVER | SELL | 2255.00 | 2344.08 | 2344.24 | EMA200 below EMA400 |
| 2025-05-12 09:15:00 | CROSSOVER | BUY | 2387.20 | 2344.51 | 2344.45 | EMA200 above EMA400 |
| 2025-05-12 10:15:00 | ALERT1 | BUY | 2416.70 | 2345.23 | 2344.81 | Break + close above crossover candle high |
| 2025-06-04 09:15:00 | ALERT2 | BUY | 2454.80 | 2463.35 | 2420.16 | EMA200 retest candle locked |
| 2025-06-04 10:15:00 | ENTRY1 | BUY | 2478.00 | 2463.50 | 2420.45 | Buy entry 1 (retest1 break) |
| 2025-06-18 12:15:00 | ALERT3 | BUY | 2459.00 | 2501.24 | 2456.09 | EMA400 retest candle locked |
| 2025-06-18 13:15:00 | EXIT | BUY | 2452.50 | 2500.76 | 2456.07 | Close below EMA400 |
| 2025-08-07 09:15:00 | CROSSOVER | SELL | 2265.20 | 2507.17 | 2507.39 | EMA200 below EMA400 |
| 2025-08-07 10:15:00 | ALERT1 | SELL | 2240.50 | 2504.52 | 2506.06 | Break + close below crossover candle low |
| 2025-09-11 09:15:00 | ALERT2 | SELL | 2381.20 | 2343.88 | 2393.71 | EMA200 retest candle locked |
| 2025-09-24 14:15:00 | CROSSOVER | BUY | 2616.50 | 2422.51 | 2422.42 | EMA200 above EMA400 |
| 2025-11-20 10:15:00 | CROSSOVER | SELL | 2456.80 | 2469.85 | 2469.85 | EMA200 below EMA400 |
| 2025-11-20 15:15:00 | ALERT1 | SELL | 2446.10 | 2468.80 | 2469.32 | Break + close below crossover candle low |
| 2026-01-02 13:15:00 | ALERT2 | SELL | 2279.60 | 2277.23 | 2331.21 | EMA200 retest candle locked |
| 2026-01-06 10:15:00 | ENTRY1 | SELL | 2265.00 | 2277.60 | 2328.52 | Sell entry 1 (retest1 break) |
| 2026-02-03 09:15:00 | ALERT3 | SELL | 2182.00 | 2128.62 | 2214.71 | EMA400 retest candle locked |
| 2026-02-03 11:15:00 | EXIT | SELL | 2215.90 | 2130.28 | 2214.69 | Close above EMA400 |
| 2026-04-27 14:15:00 | CROSSOVER | BUY | 2323.20 | 2101.36 | 2100.47 | EMA200 above EMA400 |
| 2026-04-28 09:15:00 | ALERT1 | BUY | 2394.20 | 2106.45 | 2103.03 | Break + close above crossover candle high |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| BUY | 2024-01-29 09:15:00 | 3046.70 | 2024-03-13 09:15:00 | 3033.35 | -13.35 |
| SELL | 2024-05-16 13:15:00 | 2981.00 | 2024-05-17 09:15:00 | 3058.10 | -77.10 |
| SELL | 2024-07-29 12:15:00 | 3089.90 | 2024-07-30 11:15:00 | 3119.60 | -29.70 |
| SELL | 2024-10-30 15:15:00 | 2959.00 | 2024-11-06 11:15:00 | 3022.65 | -63.65 |
| BUY | 2025-06-04 10:15:00 | 2478.00 | 2025-06-18 13:15:00 | 2452.50 | -25.50 |
| SELL | 2026-01-06 10:15:00 | 2265.00 | 2026-02-03 11:15:00 | 2215.90 | 49.10 |
