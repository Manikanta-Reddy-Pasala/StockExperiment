# FinNifty IC — Weekly MTM Trace per Trade

Source: `/app/logs/FINNIFTY_monthly_IC_OTM4_w300_lots5/trades.csv`
Total trades: 35

Per-trade weekly mark-to-market during life of the iron condor.
Each Friday between entry and exit, recompute the cost-to-close
the position using leg-level option closes.

Columns:
- **date**: Friday in trade life (or exit date if not Friday)
- **CE px / PE px**: closes of the sold (body) strikes
- **WCE px / WPE px**: closes of the bought (wing) strikes
- **close_cost**: ₹ per unit to close right now (body buy − wings sell)
- **MTM/unit**: net_credit − close_cost (positive = profit so far)
- **MTM total**: MTM/unit × lot × lots (real ₹ P&L)

## Trade 2023-05-15 → 2023-05-30  (expiry 2023-05-30)

- Spot at entry: 19583.2
- Body strikes: SELL CE 20350 / PE 18800
- Wing strikes: BUY CE 20650 / PE 18500
- Net credit: ₹1026.66 per unit (40×5 = 200 qty)
- Final P&L: ₹205332.9 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2023-05-19 | 136.6 | 20.4 | None | 9.2 | — | None | — |
| 2023-05-26 | 0.55 | 3.3 | None | 1.85 | — | None | — |
| 2023-05-30 | 0.05 | 0.05 | None | 0.05 | — | None | — |

## Trade 2023-06-05 → 2023-06-15  (expiry 2023-06-27)

- Spot at entry: 19438.5
- Body strikes: SELL CE 20200 / PE 18650
- Wing strikes: BUY CE 20500 / PE 18350
- Net credit: ₹80.32 per unit (40×5 = 200 qty)
- Final P&L: ₹-36940.2 (SL)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2023-06-09 | 182.45 | 114.1 | 143.6 | 67.55 | 85.4 | -5.08 | ₹-1,016 |
| 2023-06-15 | 182.45 | 114.1 | 25.0 | 9.15 | 262.4 | -182.08 | ₹-36,416 |

**Trade extremes:** worst MTM ₹-36,416 on 2023-06-15, best MTM ₹-1,016 on 2023-06-09

## Trade 2023-07-10 → 2023-07-25  (expiry 2023-07-25)

- Spot at entry: 20057.3
- Body strikes: SELL CE 20850 / PE 19250
- Wing strikes: BUY CE 21150 / PE 18950
- Net credit: ₹120.22 per unit (40×5 = 200 qty)
- Final P&L: ₹24043.5 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2023-07-14 | 141.8 | 202.75 | 90.9 | 6.65 | 247.0 | -126.78 | ₹-25,356 |
| 2023-07-21 | 20.5 | 1.25 | 3.25 | 1.25 | 17.25 | 102.97 | ₹20,594 |
| 2023-07-25 | 0.1 | 0.1 | 0.1 | 0.1 | 0.0 | 120.22 | ₹24,044 |

**Trade extremes:** worst MTM ₹-25,356 on 2023-07-14, best MTM ₹24,044 on 2023-07-25

## Trade 2023-08-21 → 2023-08-29  (expiry 2023-08-29)

- Spot at entry: 19571.5
- Body strikes: SELL CE 20350 / PE 18800
- Wing strikes: BUY CE 20650 / PE 18500
- Net credit: ₹684.01 per unit (40×5 = 200 qty)
- Final P&L: ₹136801.5 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2023-08-25 | 1.15 | 1.35 | 0.8 | 0.75 | 0.95 | 683.06 | ₹136,612 |
| 2023-08-29 | 0.1 | 0.05 | 0.05 | 0.05 | 0.05 | 683.96 | ₹136,792 |

**Trade extremes:** worst MTM ₹136,612 on 2023-08-25, best MTM ₹136,792 on 2023-08-29

## Trade 2023-09-04 → 2023-09-26  (expiry 2023-09-26)

- Spot at entry: 19787.6
- Body strikes: SELL CE 20600 / PE 19000
- Wing strikes: BUY CE 20900 / PE 18700
- Net credit: ₹511.0 per unit (40×5 = 200 qty)
- Final P&L: ₹102200.6 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2023-09-08 | 133.95 | 100.0 | 113.75 | 284.0 | -163.8 | 674.8 | ₹134,960 |
| 2023-09-15 | 116.45 | 4.8 | 36.0 | 5.1 | 80.15 | 430.85 | ₹86,170 |
| 2023-09-22 | 1.75 | 1.4 | 1.35 | 0.8 | 1.0 | 510.0 | ₹102,000 |
| 2023-09-26 | 0.05 | 0.05 | 0.05 | 0.05 | 0.0 | 511.0 | ₹102,200 |

**Trade extremes:** worst MTM ₹86,170 on 2023-09-15, best MTM ₹134,960 on 2023-09-08

## Trade 2023-10-09 → 2023-10-10  (expiry 2023-10-31)

- Spot at entry: 19594.7
- Body strikes: SELL CE 20400 / PE 18800
- Wing strikes: BUY CE 20700 / PE 18500
- Net credit: ₹50.33 per unit (40×5 = 200 qty)
- Final P&L: ₹-24859.1 (SL)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2023-10-10 | 57.75 | 169.45 | 35.3 | 19.0 | 172.9 | -122.57 | ₹-24,514 |

**Trade extremes:** worst MTM ₹-24,514 on 2023-10-10, best MTM ₹-24,514 on 2023-10-10

## Trade 2023-11-13 → 2023-11-28  (expiry 2023-11-28)

- Spot at entry: 19542.2
- Body strikes: SELL CE 20300 / PE 18750
- Wing strikes: BUY CE 20600 / PE 18450
- Net credit: ₹114.15 per unit (40×5 = 200 qty)
- Final P&L: ₹22829.5 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2023-11-17 | 8.0 | 8.05 | 4.25 | 4.7 | 7.1 | 107.05 | ₹21,410 |
| 2023-11-24 | 0.45 | 0.45 | 0.35 | 0.3 | 0.25 | 113.9 | ₹22,780 |
| 2023-11-28 | 0.05 | 0.05 | 0.05 | 0.1 | -0.05 | 114.2 | ₹22,840 |

**Trade extremes:** worst MTM ₹21,410 on 2023-11-17, best MTM ₹22,840 on 2023-11-28

## Trade 2023-12-04 → 2023-12-07  (expiry 2023-12-26)

- Spot at entry: 20862.9
- Body strikes: SELL CE 21700 / PE 20050
- Wing strikes: BUY CE 22000 / PE 19750
- Net credit: ₹176.42 per unit (40×5 = 200 qty)
- Final P&L: ₹-85592.5 (SL)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2023-12-07 | 103.6 | 526.95 | 22.35 | 9.8 | 598.4 | -421.98 | ₹-84,396 |

**Trade extremes:** worst MTM ₹-84,396 on 2023-12-07, best MTM ₹-84,396 on 2023-12-07

## Trade 2024-01-29 → 2024-01-30  (expiry 2024-01-30)

- Spot at entry: 20413.3
- Body strikes: SELL CE 21250 / PE 19600
- Wing strikes: BUY CE 21550 / PE 19300
- Net credit: ₹0.53 per unit (40×5 = 200 qty)
- Final P&L: ₹105.1 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2024-01-30 | 0.05 | 0.05 | 0.05 | 0.05 | 0.0 | 0.53 | ₹106 |

**Trade extremes:** worst MTM ₹106 on 2024-01-30, best MTM ₹106 on 2024-01-30

## Trade 2024-02-05 → 2024-02-27  (expiry 2024-02-27)

- Spot at entry: 20315.8
- Body strikes: SELL CE 21150 / PE 19500
- Wing strikes: BUY CE 21450 / PE 19200
- Net credit: ₹165.66 per unit (40×5 = 200 qty)
- Final P&L: ₹33132.3 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2024-02-09 | 153.4 | 459.1 | 83.8 | 352.55 | 176.15 | -10.49 | ₹-2,098 |
| 2024-02-16 | 20.65 | 459.1 | 18.35 | 352.55 | 108.85 | 56.81 | ₹11,362 |
| 2024-02-23 | 8.6 | 1.05 | 1.8 | 0.5 | 7.35 | 158.31 | ₹31,662 |
| 2024-02-27 | 0.05 | 0.05 | 0.05 | 0.1 | -0.05 | 165.71 | ₹33,142 |

**Trade extremes:** worst MTM ₹-2,098 on 2024-02-09, best MTM ₹33,142 on 2024-02-27

## Trade 2024-03-04 → 2024-03-26  (expiry 2024-03-26)

- Spot at entry: 20927.2
- Body strikes: SELL CE 21750 / PE 20100
- Wing strikes: BUY CE 22050 / PE 19800
- Net credit: ₹88.09 per unit (40×5 = 200 qty)
- Final P&L: ₹17618.3 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2024-03-08 | 89.3 | 215.25 | 52.55 | 158.75 | 93.25 | -5.16 | ₹-1,032 |
| 2024-03-15 | 8.8 | 36.8 | 4.1 | 158.75 | -117.25 | 205.34 | ₹41,068 |
| 2024-03-22 | 0.35 | 1.25 | 0.35 | 0.75 | 0.5 | 87.59 | ₹17,518 |
| 2024-03-26 | 0.1 | 0.05 | 0.05 | 0.05 | 0.05 | 88.04 | ₹17,608 |

**Trade extremes:** worst MTM ₹-1,032 on 2024-03-08, best MTM ₹41,068 on 2024-03-15

## Trade 2024-04-08 → 2024-04-30  (expiry 2024-04-30)

- Spot at entry: 21604.5
- Body strikes: SELL CE 22450 / PE 20750
- Wing strikes: BUY CE 22750 / PE 20450
- Net credit: ₹163.78 per unit (40×5 = 200 qty)
- Final P&L: ₹32755.9 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2024-04-12 | 76.0 | 408.4 | 30.5 | 282.15 | 171.75 | -7.97 | ₹-1,594 |
| 2024-04-19 | 5.85 | 116.6 | 4.15 | 64.65 | 53.65 | 110.13 | ₹22,026 |
| 2024-04-26 | 1.1 | 6.35 | 0.75 | 1.9 | 4.8 | 158.98 | ₹31,796 |
| 2024-04-30 | 0.1 | 0.1 | 0.1 | 0.05 | 0.05 | 163.73 | ₹32,746 |

**Trade extremes:** worst MTM ₹-1,594 on 2024-04-12, best MTM ₹32,746 on 2024-04-30

## Trade 2024-05-06 → 2024-05-28  (expiry 2024-05-28)

- Spot at entry: 21743.7
- Body strikes: SELL CE 22600 / PE 20850
- Wing strikes: BUY CE 22900 / PE 20550
- Net credit: ₹80.53 per unit (40×5 = 200 qty)
- Final P&L: ₹16105.7 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2024-05-10 | 15.0 | 76.2 | 172.35 | 40.9 | -122.05 | 202.58 | ₹40,516 |
| 2024-05-17 | 10.15 | 76.2 | 172.35 | 27.9 | -113.9 | 194.43 | ₹38,886 |
| 2024-05-24 | 2.85 | 2.85 | 1.45 | 1.85 | 2.4 | 78.13 | ₹15,626 |
| 2024-05-28 | 0.05 | 0.05 | 0.05 | 0.05 | 0.0 | 80.53 | ₹16,106 |

**Trade extremes:** worst MTM ₹15,626 on 2024-05-24, best MTM ₹40,516 on 2024-05-10

## Trade 2024-06-10 → 2024-06-25  (expiry 2024-06-25)

- Spot at entry: 22154.8
- Body strikes: SELL CE 23050 / PE 21250
- Wing strikes: BUY CE 23350 / PE 20950
- Net credit: ₹123.24 per unit (40×5 = 200 qty)
- Final P&L: ₹-35951.2 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2024-06-14 | 117.0 | 124.2 | 9.15 | 71.4 | 160.65 | -37.41 | ₹-7,482 |
| 2024-06-21 | 77.75 | 0.65 | 15.75 | 0.45 | 62.2 | 61.04 | ₹12,208 |
| 2024-06-25 | 455.85 | 0.1 | 147.25 | 0.1 | 308.6 | -185.36 | ₹-37,072 |

**Trade extremes:** worst MTM ₹-37,072 on 2024-06-25, best MTM ₹12,208 on 2024-06-21

## Trade 2024-07-01 → 2024-07-30  (expiry 2024-07-30)

- Spot at entry: 23631.0
- Body strikes: SELL CE 24600 / PE 22700
- Wing strikes: BUY CE 24900 / PE 22400
- Net credit: ₹347.85 per unit (40×5 = 200 qty)
- Final P&L: ₹69569.1 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2024-07-05 | 128.95 | 1027.5 | 98.25 | 856.65 | 201.55 | 146.3 | ₹29,260 |
| 2024-07-12 | 100.0 | 1027.5 | 98.25 | 856.65 | 172.6 | 175.25 | ₹35,050 |
| 2024-07-19 | 48.45 | 77.45 | 98.25 | 856.65 | -829.0 | 1176.85 | ₹235,370 |
| 2024-07-26 | 0.8 | 7.5 | 0.65 | 3.3 | 4.35 | 343.5 | ₹68,700 |
| 2024-07-30 | 0.1 | 0.1 | 0.1 | 0.05 | 0.05 | 347.8 | ₹69,560 |

**Trade extremes:** worst MTM ₹29,260 on 2024-07-05, best MTM ₹235,370 on 2024-07-19

## Trade 2024-08-12 → 2024-08-27  (expiry 2024-08-27)

- Spot at entry: 23028.5
- Body strikes: SELL CE 23950 / PE 22100
- Wing strikes: BUY CE 24250 / PE 21800
- Net credit: ₹202.32 per unit (40×5 = 200 qty)
- Final P&L: ₹40463.2 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2024-08-16 | 337.95 | 160.35 | 244.65 | 504.8 | -251.15 | 453.47 | ₹90,694 |
| 2024-08-23 | 2.5 | 1.3 | 1.0 | 0.8 | 2.0 | 200.32 | ₹40,064 |
| 2024-08-27 | 0.1 | 0.1 | 0.1 | 0.1 | 0.0 | 202.32 | ₹40,464 |

**Trade extremes:** worst MTM ₹40,064 on 2024-08-23, best MTM ₹90,694 on 2024-08-16

## Trade 2024-09-02 → 2024-09-24  (expiry 2024-09-24)

- Spot at entry: 23727.5
- Body strikes: SELL CE 24700 / PE 22800
- Wing strikes: BUY CE 25000 / PE 22500
- Net credit: ₹689.38 per unit (40×5 = 200 qty)
- Final P&L: ₹100779.6 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2024-09-06 | 696.75 | 524.9 | 16.35 | 429.9 | 775.4 | -86.02 | ₹-17,204 |
| 2024-09-13 | 696.75 | 524.9 | 8.6 | 429.9 | 783.15 | -93.77 | ₹-18,754 |
| 2024-09-20 | 101.15 | 2.7 | 19.9 | 1.9 | 82.05 | 607.33 | ₹121,466 |
| 2024-09-24 | 176.25 | 0.05 | 0.6 | 0.05 | 175.65 | 513.73 | ₹102,746 |

**Trade extremes:** worst MTM ₹-18,754 on 2024-09-13, best MTM ₹121,466 on 2024-09-20

## Trade 2024-09-30 → 2024-10-29  (expiry 2024-10-29)

- Spot at entry: 24480.3
- Body strikes: SELL CE 25450 / PE 23500
- Wing strikes: BUY CE 25750 / PE 23200
- Net credit: ₹132.02 per unit (65×5 = 325 qty)
- Final P&L: ₹42907.15 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2024-10-04 | 452.45 | 200.0 | 346.65 | 192.0 | 113.8 | 18.22 | ₹5,922 |
| 2024-10-11 | 452.45 | 186.85 | 346.65 | 165.0 | 127.65 | 4.37 | ₹1,420 |
| 2024-10-18 | 452.45 | 65.7 | 346.65 | 45.0 | 126.5 | 5.52 | ₹1,794 |
| 2024-10-25 | 1.0 | 52.45 | 346.65 | 18.05 | -311.25 | 443.27 | ₹144,063 |
| 2024-10-29 | 0.1 | 0.1 | 0.1 | 0.1 | 0.0 | 132.02 | ₹42,906 |

**Trade extremes:** worst MTM ₹1,420 on 2024-10-11, best MTM ₹144,063 on 2024-10-25

## Trade 2024-11-04 → 2024-11-26  (expiry 2024-11-26)

- Spot at entry: 23660.2
- Body strikes: SELL CE 24600 / PE 22700
- Wing strikes: BUY CE 24900 / PE 22400
- Net credit: ₹282.08 per unit (65×5 = 325 qty)
- Final P&L: ₹91676.32 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2024-11-08 | 723.8 | 99.95 | 172.2 | 355.95 | 295.6 | -13.52 | ₹-4,394 |
| 2024-11-15 | 30.1 | 99.95 | 69.75 | 355.95 | -295.65 | 577.73 | ₹187,762 |
| 2024-11-22 | 1.65 | 9.6 | 0.95 | 4.65 | 5.65 | 276.43 | ₹89,840 |
| 2024-11-26 | 0.1 | 0.05 | 0.1 | 0.1 | -0.05 | 282.13 | ₹91,692 |

**Trade extremes:** worst MTM ₹-4,394 on 2024-11-08, best MTM ₹187,762 on 2024-11-15

## Trade 2024-12-02 → 2024-12-31  (expiry 2024-12-31)

- Spot at entry: 24072.7
- Body strikes: SELL CE 25050 / PE 23100
- Wing strikes: BUY CE 25350 / PE 22800
- Net credit: ₹77.93 per unit (65×5 = 325 qty)
- Final P&L: ₹25328.71 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2024-12-06 | 268.4 | 45.45 | 162.5 | 32.0 | 119.35 | -41.42 | ₹-13,462 |
| 2024-12-13 | 241.45 | 14.0 | 134.85 | 9.35 | 111.25 | -33.32 | ₹-10,829 |
| 2024-12-20 | 21.2 | 86.35 | 14.55 | 40.85 | 52.15 | 25.78 | ₹8,378 |
| 2024-12-27 | 0.7 | 3.0 | 0.55 | 1.7 | 1.45 | 76.48 | ₹24,856 |
| 2024-12-31 | 0.05 | 0.1 | 0.1 | 0.05 | 0.0 | 77.93 | ₹25,327 |

**Trade extremes:** worst MTM ₹-13,462 on 2024-12-06, best MTM ₹25,327 on 2024-12-31

## Trade 2025-01-06 → 2025-01-28  (expiry 2025-01-28)

- Spot at entry: 23317.8
- Body strikes: SELL CE 24250 / PE 22400
- Wing strikes: BUY CE 24550 / PE 22100
- Net credit: ₹92.58 per unit (65×5 = 325 qty)
- Final P&L: ₹30087.36 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2025-01-10 | 31.15 | 211.1 | 18.45 | 129.75 | 94.05 | -1.47 | ₹-478 |
| 2025-01-17 | 15.5 | 183.45 | 10.0 | 103.6 | 85.35 | 7.23 | ₹2,350 |
| 2025-01-24 | 1.95 | 143.2 | 1.9 | 56.8 | 86.45 | 6.13 | ₹1,992 |
| 2025-01-28 | 0.8 | 37.35 | 0.75 | 10.05 | 27.35 | 65.23 | ₹21,200 |

**Trade extremes:** worst MTM ₹-478 on 2025-01-10, best MTM ₹21,200 on 2025-01-28

## Trade 2025-02-03 → 2025-02-25  (expiry 2025-02-25)

- Spot at entry: 23132.5
- Body strikes: SELL CE 24050 / PE 22200
- Wing strikes: BUY CE 24350 / PE 21900
- Net credit: ₹48.79 per unit (65×5 = 325 qty)
- Final P&L: ₹15855.61 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2025-02-07 | None | 31.2 | None | 18.3 | — | None | — |
| 2025-02-14 | None | 33.65 | None | 15.75 | — | None | — |
| 2025-02-21 | None | 4.4 | None | 1.6 | — | None | — |
| 2025-02-25 | None | 1.75 | None | 1.45 | — | None | — |

## Trade 2025-03-10 → 2025-03-19  (expiry 2025-03-25)

- Spot at entry: 23056.8
- Body strikes: SELL CE 24000 / PE 22150
- Wing strikes: BUY CE 24300 / PE 21850
- Net credit: ₹48.96 per unit (65×5 = 325 qty)
- Final P&L: ₹-40348.75 (SL)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2025-03-14 | 34.35 | None | 12.25 | None | — | None | — |
| 2025-03-19 | 309.05 | None | 138.55 | None | — | None | — |

## Trade 2025-04-07 → 2025-04-17  (expiry 2025-04-24)

- Spot at entry: 23908.5
- Body strikes: SELL CE 24850 / PE 22950
- Wing strikes: BUY CE 25150 / PE 22650
- Net credit: ₹93.96 per unit (65×5 = 325 qty)
- Final P&L: ₹-71925.43 (SL)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2025-04-11 | 270.0 | 22.05 | 166.75 | 11.65 | 113.65 | -19.69 | ₹-6,399 |
| 2025-04-17 | 1239.4 | 7.55 | 928.3 | 6.5 | 312.15 | -218.19 | ₹-70,912 |

**Trade extremes:** worst MTM ₹-70,912 on 2025-04-17, best MTM ₹-6,399 on 2025-04-11

## Trade 2025-05-05 → 2025-05-29  (expiry 2025-05-29)

- Spot at entry: 26164.9
- Body strikes: SELL CE 27200 / PE 25100
- Wing strikes: BUY CE 27500 / PE 24800
- Net credit: ₹83.0 per unit (65×5 = 325 qty)
- Final P&L: ₹26975.65 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2025-05-09 | 41.4 | 324.45 | 25.85 | 241.85 | 98.15 | -15.15 | ₹-4,924 |
| 2025-05-16 | 82.9 | 30.65 | 45.35 | 21.8 | 46.4 | 36.6 | ₹11,895 |
| 2025-05-23 | 39.85 | 11.5 | 17.2 | 7.45 | 26.7 | 56.3 | ₹18,298 |
| 2025-05-29 | 0.15 | 0.25 | 0.15 | 0.2 | 0.05 | 82.95 | ₹26,959 |

**Trade extremes:** worst MTM ₹-4,924 on 2025-05-09, best MTM ₹26,959 on 2025-05-29

## Trade 2025-06-02 → 2025-06-26  (expiry 2025-06-26)

- Spot at entry: 26448.4
- Body strikes: SELL CE 27500 / PE 25400
- Wing strikes: BUY CE 27800 / PE 25100
- Net credit: ₹89.95 per unit (65×5 = 325 qty)
- Final P&L: ₹29234.56 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2025-06-06 | 172.4 | 46.6 | 104.25 | 33.9 | 80.85 | 9.1 | ₹2,958 |
| 2025-06-13 | 33.0 | 67.15 | 19.4 | 41.15 | 39.6 | 50.35 | ₹16,364 |
| 2025-06-20 | 11.15 | 15.4 | 5.15 | 11.1 | 10.3 | 79.65 | ₹25,886 |
| 2025-06-26 | 0.2 | 0.25 | 0.3 | 0.1 | 0.05 | 89.9 | ₹29,218 |

**Trade extremes:** worst MTM ₹2,958 on 2025-06-06, best MTM ₹29,218 on 2025-06-26

## Trade 2025-06-30 → 2025-07-31  (expiry 2025-07-31)

- Spot at entry: 27174.5
- Body strikes: SELL CE 28250 / PE 26100
- Wing strikes: BUY CE 28550 / PE 25800
- Net credit: ₹790.81 per unit (65×5 = 325 qty)
- Final P&L: ₹257013.41 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2025-07-04 | 342.25 | 787.4 | 30.25 | 67.45 | 1031.95 | -241.14 | ₹-78,370 |
| 2025-07-11 | 342.25 | 787.4 | 18.6 | 38.6 | 1072.45 | -281.64 | ₹-91,533 |
| 2025-07-18 | 342.25 | 787.4 | 7.3 | 42.95 | 1079.4 | -288.59 | ₹-93,792 |
| 2025-07-25 | 342.25 | 15.0 | 2.55 | 7.15 | 347.55 | 443.26 | ₹144,060 |
| 2025-07-31 | 0.15 | 0.2 | 0.1 | 0.15 | 0.1 | 790.71 | ₹256,981 |

**Trade extremes:** worst MTM ₹-93,792 on 2025-07-18, best MTM ₹256,981 on 2025-07-31

## Trade 2025-08-04 → 2025-08-28  (expiry 2025-08-28)

- Spot at entry: 26476.6
- Body strikes: SELL CE 27550 / PE 25400
- Wing strikes: BUY CE 27850 / PE 25100
- Net credit: ₹442.01 per unit (65×5 = 325 qty)
- Final P&L: ₹143654.22 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2025-08-08 | 15.35 | 462.15 | 11.35 | 34.25 | 431.9 | 10.11 | ₹3,286 |
| 2025-08-15 | 15.95 | 462.15 | 8.1 | 17.2 | 452.8 | -10.79 | ₹-3,507 |
| 2025-08-22 | 2.75 | 5.65 | 1.85 | 3.15 | 3.4 | 438.61 | ₹142,548 |
| 2025-08-28 | 0.25 | 0.15 | 0.3 | 0.15 | -0.05 | 442.06 | ₹143,670 |

**Trade extremes:** worst MTM ₹-3,507 on 2025-08-15, best MTM ₹143,670 on 2025-08-28

## Trade 2025-09-22 → 2025-09-30  (expiry 2025-09-30)

- Spot at entry: 26528.4
- Body strikes: SELL CE 27600 / PE 25450
- Wing strikes: BUY CE 27900 / PE 25150
- Net credit: ₹3.61 per unit (65×5 = 325 qty)
- Final P&L: ₹1174.55 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2025-09-26 | 2.5 | 8.4 | 2.0 | 3.7 | 5.2 | -1.59 | ₹-517 |
| 2025-09-30 | 0.25 | 0.2 | 0.2 | 0.1 | 0.15 | 3.46 | ₹1,124 |

**Trade extremes:** worst MTM ₹-517 on 2025-09-26, best MTM ₹1,124 on 2025-09-30

## Trade 2025-10-06 → 2025-10-16  (expiry 2025-10-28)

- Spot at entry: 26712.0
- Body strikes: SELL CE 27800 / PE 25650
- Wing strikes: BUY CE 28100 / PE 25350
- Net credit: ₹101.69 per unit (65×5 = 325 qty)
- Final P&L: ₹-73893.95 (SL)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2025-10-10 | 34.8 | 302.6 | 18.3 | 207.8 | 111.3 | -9.61 | ₹-3,123 |
| 2025-10-16 | 87.65 | 302.6 | 42.0 | 22.45 | 325.8 | -224.11 | ₹-72,836 |

**Trade extremes:** worst MTM ₹-72,836 on 2025-10-16, best MTM ₹-3,123 on 2025-10-10

## Trade 2025-11-03 → 2025-11-25  (expiry 2025-11-25)

- Spot at entry: 27306.2
- Body strikes: SELL CE 28400 / PE 26200
- Wing strikes: BUY CE 28700 / PE 25900
- Net credit: ₹1011.25 per unit (65×5 = 325 qty)
- Final P&L: ₹328656.41 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2025-11-07 | 34.45 | 47.1 | 17.9 | 26.8 | 36.85 | 974.4 | ₹316,680 |
| 2025-11-14 | 25.1 | 18.85 | 13.3 | 11.95 | 18.7 | 992.55 | ₹322,579 |
| 2025-11-21 | 4.4 | 3.15 | 2.65 | 2.55 | 2.35 | 1008.9 | ₹327,892 |
| 2025-11-25 | 0.05 | 0.1 | 0.1 | 0.15 | -0.1 | 1011.35 | ₹328,689 |

**Trade extremes:** worst MTM ₹316,680 on 2025-11-07, best MTM ₹328,689 on 2025-11-25

## Trade 2025-12-15 → 2025-12-30  (expiry 2025-12-30)

- Spot at entry: 27603.2
- Body strikes: SELL CE 28700 / PE 26500
- Wing strikes: BUY CE 29000 / PE 26200
- Net credit: ₹9.49 per unit (65×5 = 325 qty)
- Final P&L: ₹3083.44 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2025-12-19 | 7.1 | 15.2 | 4.9 | 9.9 | 7.5 | 1.99 | ₹647 |
| 2025-12-26 | 1.3 | 2.4 | 1.2 | 1.65 | 0.85 | 8.64 | ₹2,808 |
| 2025-12-30 | 0.1 | 0.15 | 0.1 | 0.1 | 0.05 | 9.44 | ₹3,068 |

**Trade extremes:** worst MTM ₹647 on 2025-12-19, best MTM ₹3,068 on 2025-12-30

## Trade 2026-02-02 → 2026-02-13  (expiry 2026-02-24)

- Spot at entry: 26799.0
- Body strikes: SELL CE 27850 / PE 25750
- Wing strikes: BUY CE 28150 / PE 25450
- Net credit: ₹111.71 per unit (60×5 = 300 qty)
- Final P&L: ₹-76840.8 (SL)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2026-02-06 | 305.15 | 23.6 | 171.45 | 8.55 | 148.75 | -37.04 | ₹-11,112 |
| 2026-02-13 | 599.9 | 10.4 | 238.3 | 7.8 | 364.2 | -252.49 | ₹-75,747 |

**Trade extremes:** worst MTM ₹-75,747 on 2026-02-13, best MTM ₹-11,112 on 2026-02-06

## Trade 2026-04-15 → 2026-04-28  (expiry 2026-04-28)

- Spot at entry: 27564.1
- Body strikes: SELL CE 28650 / PE 26450
- Wing strikes: BUY CE 28950 / PE 26150
- Net credit: ₹2249.68 per unit (60×5 = 300 qty)
- Final P&L: ₹584003.7 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2026-04-17 | 19.7 | 334.95 | 11.6 | 252.0 | 91.05 | 2158.63 | ₹647,589 |
| 2026-04-24 | 19.7 | 312.0 | 11.6 | 186.6 | 133.5 | 2116.18 | ₹634,854 |
| 2026-04-28 | 19.7 | 509.0 | 11.6 | 186.8 | 330.3 | 1919.38 | ₹575,814 |

**Trade extremes:** worst MTM ₹575,814 on 2026-04-28, best MTM ₹647,589 on 2026-04-17

## Trade 2026-05-04 → 2026-05-26  (expiry 2026-05-26)

- Spot at entry: 25814.4
- Body strikes: SELL CE 26850 / PE 24800
- Wing strikes: BUY CE 27150 / PE 24500
- Net credit: ₹148.84 per unit (60×5 = 300 qty)
- Final P&L: ₹44651.85 (EXPIRY)

| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |
|---|---|---|---|---|---|---|---|
| 2026-05-08 | 166.4 | 96.9 | 101.5 | 65.6 | 96.2 | 52.64 | ₹15,792 |
| 2026-05-15 | 36.8 | 138.85 | 20.35 | 117.5 | 37.8 | 111.04 | ₹33,312 |
| 2026-05-22 | 8.9 | 102.85 | 5.0 | 55.55 | 51.2 | 97.64 | ₹29,292 |
| 2026-05-26 | 8.9 | 102.85 | 5.0 | 55.55 | 51.2 | 97.64 | ₹29,292 |

**Trade extremes:** worst MTM ₹15,792 on 2026-05-08, best MTM ₹33,312 on 2026-05-15
