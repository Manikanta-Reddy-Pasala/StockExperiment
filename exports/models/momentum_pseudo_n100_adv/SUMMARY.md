# momentum_pseudo_n100_adv — SUMMARY

V1 lookahead variant. Monthly rotation on top-100 ADV-ranked stocks from Nifty 500.

## Stock pick logic (plain English)

1. **Universe build**: take all 500 stocks in NSE Nifty 500 list
2. **Compute liquidity**: 20-day ADV = avg(close × volume) per stock
3. **Rank by ADV**: sort descending, take **top 100** as 'pseudo-N100' universe
4. **Rebuild yearly**: at each year-start (2023-05-15, 2024-05-13, 2025-05-13) using prior-data ADV
5. **Rank by return**: within the 100 picked, compute 30-day return, sort highest-first
6. **Pick top-1**: hold the single best-performing for the next month
7. **Rebalance**: 1st trading day of each month

**Unique filter**: ADV ranking instead of NSE official membership. Includes liquid mid-caps NSE excludes from real Nifty 100 (BSE, MAZDOCK, NETWEB, COCHINSHIP, GRSE, IRFC, IDEA, ITI, NBCC, PAYTM, COFORGE, HFCL, GROWW etc.). 47 of 100 stocks differ from real N100.

⚠️ **Yearly-PIT bias**: universe built with year-end ADV applied to year-start = lookahead. Real-time strategy would not match these results.

| Key knob | Value |
|---|---|
| Universe size | 100 (ADV rank 1-100 from N500) |
| ADV window | 20 days |
| Lookback (signal) | 30 days |
| Position | top-1 (max_concurrent=1) |
| **Rebalance period** | **Monthly (1st trading day)** |
| Universe rebuild | **Yearly (at year-start)** |
| Exit rule | Rotation only |

## Headline result (3-year backtest, ₹10L start)

| Metric | Value |
|---|---:|
| Final NAV (cash+open MTM) | **₹13,210,187** |
| Total return | **+1221.02%** |
| **3-yr CAGR** | **+136.39%/yr** |
| Max DD (cash NAV) | 16.15% |
| Trades | 30 (+1 open) |
| WR | 86.7% (26W / 4L) |

## Returns by NSE cap segment

| Cap segment | Trades | Wins | Losses | WR | Total PnL ₹ | Avg PnL/trade ₹ |
|---|---:|---:|---:|---:|---:|---:|
| **Large** | 15 | 14 | 1 | 93% | +8,756,351 | +583,757 |
| **Mid** | 12 | 10 | 2 | 83% | +3,505,564 | +292,130 |
| **Small** | 3 | 2 | 1 | 67% | -226,439 | -75,480 |

## Full trade ledger — every entry with price, invested ₹, exit, gain/loss

| # | Symbol | Cap | Index | Entry Date | Entry ₹ | Qty | **Invested** | Exit Date | Exit ₹ | **PnL ₹** | Ret % | Reason |
|--:|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| 1 | RVNL | **Mid** | Nifty Midcap 150 | 2023-05-15 | 120.70 | 8,285 | ₹1,000,000 | 2023-07-03 | 121.85 | +9,528 | +0.95% | rotate |
| 2 | PFC | **Large** | Nifty 100 | 2023-07-03 | 175.80 | 5,742 | ₹1,009,444 | 2023-08-01 | 207.08 | +179,610 | +17.79% | rotate |
| 3 | ENGINERSIN | **Small** | Nifty Smallcap 250 | 2023-08-01 | 155.50 | 7,647 | ₹1,189,108 | 2023-09-01 | 157.75 | +17,206 | +1.45% | rotate |
| 4 | IRFC | **Large** | Nifty 100 | 2023-09-01 | 55.75 | 21,638 | ₹1,206,318 | 2023-11-01 | 72.95 | +372,174 | +30.85% | rotate |
| 5 | RECLTD | **Large** | Nifty 100 | 2023-11-01 | 282.85 | 5,580 | ₹1,578,303 | 2023-12-01 | 374.00 | +508,617 | +32.23% | rotate |
| 6 | PFC | **Large** | Nifty 100 | 2023-12-01 | 365.15 | 5,715 | ₹2,086,832 | 2024-01-01 | 395.05 | +170,879 | +8.19% | rotate |
| 7 | ADANIGREEN | **Large** | Nifty 100 | 2024-01-01 | 1,598.40 | 1,412 | ₹2,256,941 | 2024-02-01 | 1,665.95 | +95,381 | +4.23% | rotate |
| 8 | IRFC | **Large** | Nifty 100 | 2024-02-01 | 169.90 | 13,851 | ₹2,353,285 | 2024-03-01 | 147.40 | -311,647 | -13.24% | rotate |
| 9 | ETERNAL | **Large** | Nifty 100 | 2024-03-01 | 166.50 | 12,262 | ₹2,041,623 | 2024-04-01 | 184.50 | +220,716 | +10.81% | rotate |
| 10 | ABB | **Large** | Nifty 100 | 2024-04-01 | 6,504.65 | 347 | ₹2,257,114 | 2024-05-02 | 6,682.50 | +61,714 | +2.73% | rotate |
| 11 | INDUSTOWER | **Mid** | Nifty Midcap 150 | 2024-05-02 | 352.95 | 6,584 | ₹2,323,823 | 2024-06-03 | 364.00 | +72,753 | +3.13% | rotate |
| 12 | COCHINSHIP | **Mid** | Nifty Midcap 150 | 2024-06-03 | 2,013.00 | 1,190 | ₹2,395,470 | 2024-08-01 | 2,580.30 | +675,087 | +28.18% | rotate |
| 13 | RVNL | **Mid** | Nifty Midcap 150 | 2024-08-01 | 595.50 | 5,158 | ₹3,071,589 | 2024-09-02 | 601.20 | +29,401 | +0.96% | rotate |
| 14 | TRENT | **Large** | Nifty 100 | 2024-09-02 | 7,148.20 | 433 | ₹3,095,171 | 2024-10-01 | 7,612.70 | +201,128 | +6.50% | rotate |
| 15 | BSE | **Mid** | Nifty Midcap 150 | 2024-10-01 | 1,286.30 | 2,567 | ₹3,301,932 | 2024-11-01 | 1,487.88 | +517,456 | +15.67% | rotate |
| 16 | NATIONALUM | **Mid** | Nifty Midcap 150 | 2024-11-01 | 229.15 | 16,670 | ₹3,819,930 | 2024-12-02 | 243.30 | +235,880 | +6.17% | rotate |
| 17 | COFORGE | **Mid** | Nifty Midcap 150 | 2024-12-02 | 1,742.14 | 2,328 | ₹4,055,702 | 2025-01-01 | 1,925.71 | +427,351 | +10.54% | rotate |
| 18 | MCX | **Mid** | Nifty Midcap 150 | 2025-01-01 | 6,286.10 | 713 | ₹4,481,989 | 2025-02-01 | 5,774.75 | -364,593 | -8.13% | rotate |
| 19 | KOTAKBANK | **Large** | Nifty 100 | 2025-02-01 | 1,903.10 | 2,164 | ₹4,118,308 | 2025-03-03 | 1,914.60 | +24,886 | +0.60% | rotate |
| 20 | SHRIRAMFIN | **Large** | Nifty 100 | 2025-03-03 | 621.30 | 6,669 | ₹4,143,450 | 2025-04-01 | 637.45 | +107,704 | +2.60% | rotate |
| 21 | MAZDOCK | **Large** | Nifty 100 | 2025-04-01 | 2,578.55 | 1,648 | ₹4,249,450 | 2025-05-02 | 2,996.60 | +688,946 | +16.21% | rotate |
| 22 | BSE | **Mid** | Nifty Midcap 150 | 2025-05-02 | 2,102.17 | 2,350 | ₹4,940,100 | 2025-06-02 | 2,693.30 | +1,389,156 | +28.12% | rotate |
| 23 | GRSE | **Small** | Nifty Smallcap 250 | 2025-06-02 | 2,966.00 | 2,133 | ₹6,326,478 | 2025-07-01 | 2,983.20 | +36,688 | +0.58% | rotate |
| 24 | MCX | **Mid** | Nifty Midcap 150 | 2025-07-01 | 9,060.50 | 702 | ₹6,360,471 | 2025-08-01 | 7,595.50 | -1,028,430 | -16.17% | rotate |
| 25 | PAYTM | **Mid** | Nifty Midcap 150 | 2025-08-01 | 1,076.40 | 4,958 | ₹5,336,791 | 2025-09-01 | 1,235.80 | +790,305 | +14.81% | rotate |
| 26 | ETERNAL | **Large** | Nifty 100 | 2025-09-01 | 321.10 | 19,084 | ₹6,127,872 | 2025-10-01 | 329.00 | +150,764 | +2.46% | rotate |
| 27 | IDEA | **Mid** | Nifty Midcap 150 | 2025-10-01 | 8.52 | 736,931 | ₹6,278,652 | 2025-11-03 | 9.54 | +751,670 | +11.97% | rotate |
| 28 | SHRIRAMFIN | **Large** | Nifty 100 | 2025-11-03 | 796.45 | 8,827 | ₹7,030,264 | 2026-03-02 | 1,052.50 | +2,260,153 | +32.15% | rotate |
| 29 | DATAPATTNS | **Small** | Nifty Smallcap 250 | 2026-03-02 | 3,204.30 | 2,899 | ₹9,289,266 | 2026-04-01 | 3,107.60 | -280,333 | -3.02% | rotate |
| 30 | ADANIPOWER | **Large** | Nifty 100 | 2026-04-01 | 157.11 | 57,349 | ₹9,010,101 | 2026-05-04 | 227.30 | +4,025,326 | +44.68% | rotate |
| OPEN | ADANIGREEN | **Large** | Nifty 100 | 2026-05-04 | 1,290.70 | 10,099 | ₹13,034,779 | OPEN | 1,308.00 | +174,713 | +1.34% | open |

**Caveats**: Lookahead bias — universe captures stocks that became high-ADV AFTER 2023. Not deployable. Use Model 1 (`momentum_n100_top5_max1`) for live.
