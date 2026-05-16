"""Data pulls for FinNifty IC OTM3 w500 lots=4. Shares cache with FinNifty IC OTM4 model."""
import logging
log = logging.getLogger(__name__)


def noop():
    """Data already pulled by tools/models/finnifty_ic_otm4_w300_lots5/."""
    log.info("finnifty_ic_otm3_w500_lots4: shares bhav cache with FinNifty IC OTM4")
