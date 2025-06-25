import logging
class TradeExecutor:
    def __init__(self, real_trading=False, headless_browser=False, quotex=None):
        self.real_trading = real_trading
        self.headless_browser = headless_browser
        self.quotex = quotex

        if self.real_trading and self.quotex is None:
            raise ValueError("❌ TradeExecutor requires a valid QuotexExecutor instance if real_trading=True.")
        else:
            print("✅ TradeExecutor initialized.")

    def close(self):
        if self.quotex:
            self.quotex.close()
