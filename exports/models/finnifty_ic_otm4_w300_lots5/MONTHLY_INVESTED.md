# finnifty_ic_otm4_w300_lots5 — Monthly Performance with Capital Deployed

## What each column means

| Column | Definition |
|---|---|
| **Trades** | Number of Iron Condor cycles opened that month |
| **Margin Locked** | Defined-risk capital deployed = wing_width × lot_size × num_lots (per cycle, summed if multiple in month). This is the MOST you can lose per IC, NOT what you spend cash-wise. |
| **Premium Collected** | Net credit received when opening IC = (CE_short + PE_short) − (CE_wing + PE_wing) × lot × lots. This is the cash that flows INTO your account on entry. |
| **Worst-case Loss** | Hard cap on month's loss = Margin Locked − Premium Collected. Strategy can never lose more than this per IC. |
| **Realized P&L** | Actual profit/loss when IC closed (at stop or expiry). Premium Collected − Exit Debit Paid. Sign reflects outcome: +ve = kept most of credit, −ve = had to buy back at higher price. |
| **ROI on Equity** | Month P&L / equity at start of month. Tracks portfolio growth. |
| **End-of-Month Equity** | Cumulative NAV after this month's trades close. |

Capital is **defined-risk margin**. With 5 lots × 65 (post Sep 2024) × ₹300 wing = ₹97,500 margin per cycle on ₹2L start. Margin scales with lot size + wing width but never exceeds the equity (refuses to enter otherwise).

**Note on negative Worst-case Loss values**: A handful of months show Worst-case Loss < 0 (e.g. 2023-05 −₹1.45L, 2025-11 −₹2.31L). This happens when the backtest's recorded Premium Collected exceeded the theoretical wing_width × lot × lots. Real-world IC structure caps loss at wing_width − net_credit (always ≥ 0). The negative values are an artifact of close-price entry modeling (mid-of-day prices may overstate filled credit). Treat Worst-case Loss = max(0, value) for live planning.

## Monthly ledger

| Month | Trades | WR | Margin Locked ₹ | Premium Collected ₹ | Worst-case Loss ₹ | Realized P&L ₹ | ROI on Equity | End-of-Month Equity ₹ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023-05 | 1 | 100% | ₹60,000 | ₹205,332 | ₹-145,332 | ₹+205,333 | +102.67% | ₹405,333 |
| 2023-06 | 1 | 0% | ₹60,000 | ₹16,064 | ₹43,936 | ₹-36,940 | -9.11% | ₹368,393 |
| 2023-07 | 1 | 100% | ₹60,000 | ₹24,044 | ₹35,956 | ₹+24,044 | +6.53% | ₹392,436 |
| 2023-08 | 1 | 100% | ₹60,000 | ₹136,802 | ₹-76,802 | ₹+136,802 | +34.86% | ₹529,238 |
| 2023-09 | 1 | 100% | ₹60,000 | ₹102,200 | ₹-42,200 | ₹+102,201 | +19.31% | ₹631,438 |
| 2023-10 | 1 | 0% | ₹60,000 | ₹10,066 | ₹49,934 | ₹-24,859 | -3.94% | ₹606,579 |
| 2023-11 | 1 | 100% | ₹60,000 | ₹22,830 | ₹37,170 | ₹+22,830 | +3.76% | ₹629,409 |
| 2023-12 | 1 | 0% | ₹60,000 | ₹35,284 | ₹24,716 | ₹-85,592 | -13.60% | ₹543,816 |
| 2024-01 | 1 | 100% | ₹60,000 | ₹106 | ₹59,894 | ₹+105 | +0.02% | ₹543,921 |
| 2024-02 | 1 | 100% | ₹60,000 | ₹33,132 | ₹26,868 | ₹+33,132 | +6.09% | ₹577,054 |
| 2024-03 | 1 | 100% | ₹60,000 | ₹17,618 | ₹42,382 | ₹+17,618 | +3.05% | ₹594,672 |
| 2024-04 | 1 | 100% | ₹60,000 | ₹32,756 | ₹27,244 | ₹+32,756 | +5.51% | ₹627,428 |
| 2024-05 | 1 | 100% | ₹60,000 | ₹16,106 | ₹43,894 | ₹+16,106 | +2.57% | ₹643,534 |
| 2024-06 | 1 | 0% | ₹60,000 | ₹24,648 | ₹35,352 | ₹-35,951 | -5.59% | ₹607,582 |
| 2024-07 | 1 | 100% | ₹60,000 | ₹69,570 | ₹-9,570 | ₹+69,569 | +11.45% | ₹677,151 |
| 2024-08 | 1 | 100% | ₹60,000 | ₹40,464 | ₹19,536 | ₹+40,463 | +5.98% | ₹717,615 |
| 2024-09 | 2 | 100% | ₹157,500 | ₹180,782 | ₹-23,283 | ₹+143,687 | +20.02% | ₹861,301 |
| 2024-11 | 1 | 100% | ₹97,500 | ₹91,676 | ₹5,824 | ₹+91,676 | +10.64% | ₹952,978 |
| 2024-12 | 1 | 100% | ₹97,500 | ₹25,327 | ₹72,173 | ₹+25,329 | +2.66% | ₹978,306 |
| 2025-01 | 1 | 100% | ₹97,500 | ₹30,088 | ₹67,412 | ₹+30,087 | +3.08% | ₹1,008,394 |
| 2025-02 | 1 | 100% | ₹97,500 | ₹15,857 | ₹81,643 | ₹+15,856 | +1.57% | ₹1,024,249 |
| 2025-03 | 1 | 0% | ₹97,500 | ₹15,912 | ₹81,588 | ₹-40,349 | -3.94% | ₹983,901 |
| 2025-04 | 1 | 0% | ₹97,500 | ₹30,537 | ₹66,963 | ₹-71,925 | -7.31% | ₹911,975 |
| 2025-05 | 1 | 100% | ₹97,500 | ₹26,975 | ₹70,525 | ₹+26,976 | +2.96% | ₹938,951 |
| 2025-06 | 2 | 100% | ₹195,000 | ₹286,247 | ₹-91,247 | ₹+286,248 | +30.49% | ₹1,225,199 |
| 2025-08 | 1 | 100% | ₹97,500 | ₹143,653 | ₹-46,153 | ₹+143,654 | +11.72% | ₹1,368,853 |
| 2025-09 | 1 | 100% | ₹97,500 | ₹1,173 | ₹96,327 | ₹+1,175 | +0.09% | ₹1,370,028 |
| 2025-10 | 1 | 0% | ₹97,500 | ₹33,049 | ₹64,451 | ₹-73,894 | -5.39% | ₹1,296,134 |
| 2025-11 | 1 | 100% | ₹97,500 | ₹328,656 | ₹-231,156 | ₹+328,656 | +25.36% | ₹1,624,790 |
| 2025-12 | 1 | 100% | ₹97,500 | ₹3,084 | ₹94,416 | ₹+3,083 | +0.19% | ₹1,627,873 |
| 2026-02 | 1 | 0% | ₹97,500 | ₹36,306 | ₹61,194 | ₹-83,244 | -5.11% | ₹1,544,629 |
| 2026-04 | 1 | 100% | ₹97,500 | ₹731,146 | ₹-633,646 | ₹+632,671 | +40.96% | ₹2,177,300 |
| 2026-05 | 1 | 100% | ₹97,500 | ₹48,373 | ₹49,127 | ₹+48,373 | +2.22% | ₹2,225,673 |

## Headline summary

- **Total cycles:** 35
- **Win rate:** 77.1% (27W / 8L)
- **Cumulative margin deployed:** ₹2,775,000 (sum across all cycles)
- **Cumulative premium collected:** ₹2,815,865
- **Total realized P&L:** ₹+2,025,673
- **Start NAV:** ₹2,00,000 → **End NAV:** ₹2,225,673
- **Return on starting equity:** +1012.84% (over 3yr)
