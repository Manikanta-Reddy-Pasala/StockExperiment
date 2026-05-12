# Scale-up to ₹10,00,000 (₹10L)

_Generated: 2026-05-12_

## Result: ROI% unchanged, absolute profit 5×

Same backtest window (May 2025 → May 2026), same strategy + universe
(EMA 200/400 + selector top-10 N500), only capital changes:

| Capital | Final₹ | Profit₹ | ROI% | MaxDD% | Trades | OpenEnd |
|--------:|-------:|--------:|-----:|-------:|-------:|--------:|
| ₹2L     | ₹2,42,606 | +₹42,606 | +21.30 | 9.60 | 28 | 0 |
| **₹10L** | **₹12,18,456** | **+₹2,18,456** | **+21.85** | **9.58** | 28 | 0 |

Tiny ROI difference (21.30 → 21.85) explained by lot-size rounding —
larger slot alloc fits more shares per entry, reducing waste.

## Pattern: ROI is invariant to capital (within reason)

Strategy is fully-invested per signal (slot allocation = available_cash /
remaining_slots). Doubling/5×-ing capital just scales each position's
absolute INR. ROI% stays the same.

This breaks at extremes:
- **Too small:** ₹50K — slot is ₹25K, can't buy 1 share of ₹3000 stocks
  (AMBER, IKS) → signals skipped → ROI drops
- **Too big:** ₹5cr+ — slippage starts mattering, market impact > 0.1%
  per fill, real returns < paper returns

Sweet spot for selector top-10 universe (median stock ~₹500-2000):
**₹2L – ₹50L** with no material ROI change.

## What changes with capital, what doesn't

| Aspect | Changes? |
|---|---|
| ROI % | No (within sweet spot) |
| MaxDD % | No |
| Number of trades | No (same signals) |
| Win rate | No |
| Absolute profit ₹ | Yes (linear) |
| Slot allocation ₹ | Yes (linear) |
| Per-share count | Yes (more shares per entry) |
| Slippage risk | Yes (grows with size) |

## Updated production config

```yaml
capital_inr: 1000000          # ₹10,00,000 locked
max_concurrent: 2              # ₹5L per slot
max_per_trade_inr: 500000      # ₹5L max single position
max_daily_loss_pct: -5.0       # ₹50K kill-switch
min_price: 50
min_adv_lakh: 100              # ₹1cr/day liquidity (unchanged — still fine)
```

## Expected outcomes at ₹10L

| Metric | Value |
|---|---|
| Year-end equity (backtest) | ~₹12.18L |
| Year profit (backtest) | ~₹2.18L |
| Worst drawdown ₹ | ~₹96K |
| Daily kill-switch trigger ₹ | -₹50K |
| Avg position size | ~₹3-4L |

## Slippage caveat at ₹10L

For Indian cash equity at ₹10L per trade, slippage is typically
< 0.05% on liquid stocks (ADV > ₹1cr/day, which is our filter floor).
Should NOT materially affect ROI. But:

- For stocks with ADV ₹1-5cr/day, slippage may be 0.1-0.2%
- For market orders during opening/closing 5 min, slippage spikes
- Use limit orders ≥ 0.1% buffer for entries during volatile periods

The current `fyers_executor.py` uses market orders. Future: add limit
order mode for cleaner fills.

## Cron usage at ₹10L (no env change needed — defaults updated)

```bash
./tools/live/run_daily.sh selector    # monthly
./tools/live/run_daily.sh prefetch    # daily 09:00 IST
./tools/live/run_daily.sh signals     # daily 09:30 IST
./tools/live/run_daily.sh paper       # daily 09:35 IST

# Override capital if you want (e.g., scale up further)
CAPITAL_INR=2000000 ./tools/live/run_daily.sh paper
```
