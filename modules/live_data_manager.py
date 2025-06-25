class LiveDataManager:
    """
    Manages live data streaming simulation.
    """
    def __init__(self, initial_data):
        self.df = initial_data.copy()
    
    def append_candle(self, candle_dict):
        import pandas as pd
        new_row = pd.DataFrame([candle_dict])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
    
    def get_latest_data(self):
        return self.df.copy()
