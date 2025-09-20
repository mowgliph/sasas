import numpy as np
import pandas as pd
from datetime import datetime
from pandas import DataFrame
from typing import Optional
import logging

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
)

import talib.abstract as ta

logger = logging.getLogger(__name__)


class SASAS(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    startup_candle_count = 120
    timeframe = '15m'

    minimal_roi = {
        "0": 0.10,
    }
    stoploss = 0.10

    sup_period = IntParameter(7, 30, default=14, space='strategy', optimize=True)  
    sup_multiplier = DecimalParameter(1.0, 3.0, default=1.8, decimals=2, space='strategy', optimize=True)  

    ash_length = IntParameter(5, 40, default=16, space='strategy', optimize=True)  
    ash_smooth = IntParameter(2, 10, default=4, space='strategy', optimize=True)  
    ash_mode = CategoricalParameter(['RSI', 'STOCHASTIC', 'ADX'], default='RSI', space='strategy', optimize=True)  
    ash_ma_type = CategoricalParameter(['ALMA', 'EMA', 'WMA', 'SMA', 'SMMA', 'HMA'], default='EMA', space='strategy', optimize=True)  
    ash_alma_offset = DecimalParameter(0, 1.5, default=0.85, decimals=2, space='strategy', optimize=True)  
    ash_alma_sigma = IntParameter(1, 10, default=6, space='strategy', optimize=True)  

    atr_period = IntParameter(5, 20, default=14, space="strategy", optimize=True)  

    def version(self) -> str:  
        return "SASAS_v0.3.20"  

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  
        dataframe = supertrend(  
            dataframe, period=int(self.sup_period.value), multiplier=float(self.sup_multiplier.value)  
        )  
        dataframe = ash_with_color(  
            dataframe,  
            length=int(self.ash_length.value),  
            smooth=int(self.ash_smooth.value),  
            src="close",  
            mode=self.ash_mode.value,  
            ma_type=self.ash_ma_type.value,  
            alma_offset=float(self.ash_alma_offset.value),  
            alma_sigma=int(self.ash_alma_sigma.value)  
        )  
        dataframe['atr'] = ta.ATR(
            dataframe['high'], 
            dataframe['low'], 
            dataframe['close'], 
            timeperiod=int(self.atr_period.value)
        )
          
        return dataframe  

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  
        dataframe['enter_long'] = (  
            (dataframe['supertrend_direction'] == 1) &  
            dataframe['ash_bullish_signal']  
        ).astype(int)  

        dataframe['enter_short'] = (  
            (dataframe['supertrend_direction'] == -1) &  
            dataframe['ash_bearish_signal']  
        ).astype(int)  
        return dataframe  

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  
        return dataframe  
          
    def process_entry(self, trade: 'Trade', order: dict, **kwargs) -> None:
        """
        Called when a new trade is opened.
        Store ATR at entry in trade.user_data for later use in TP/SL calculations.
        """
        try:
            # Get the dataframe up to the entry time
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            
            # Find the candle where the trade was opened
            entry_candle = dataframe[dataframe['date'] <= trade.open_date_utc].iloc[-1]
            atr_at_entry = entry_candle['atr']
            
            # Initialize user_data if it doesn't exist
            if not hasattr(trade, 'user_data') or trade.user_data is None:
                trade.user_data = {}
                
            # Store ATR at entry
            trade.user_data['atr_at_entry'] = float(atr_at_entry)
            
            logger.info(
                f"Trade {trade.id} ({trade.pair}): "
                f"Stored ATR at entry: {atr_at_entry:.5f}"
            )
            
        except Exception as e:
            logger.error(f"Error in process_entry for trade {trade.id}: {e}")

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic using ATR at entry.
        Returns stoploss in absolute value (current price * relative stoploss).
        """
        # Get ATR at entry from trade.user_data
        atr_at_entry = trade.user_data.get('atr_at_entry')
        
        if atr_at_entry is None or atr_at_entry <= 0:
            logger.warning(f"No valid ATR at entry for {pair}, using default stoploss")
            return 1  # Fallback to default stoploss

        # Calculate stoploss price based on ATR at entry
        if trade.is_short:
            stop_price = trade.open_rate + (atr_at_entry * 1.5)  # 1.5x ATR for shorts
            # Convert to relative stoploss value
            stoploss = 1 - (stop_price / current_rate)
        else:  # Long
            stop_price = trade.open_rate - (atr_at_entry * 1.5)  # 1.5x ATR for longs
            # Convert to relative stoploss value
            stoploss = (current_rate - stop_price) / current_rate

        # Ensure stoploss is within reasonable bounds
        stoploss = min(0.9, max(0.02, abs(stoploss)))  # Between 2% and 90%
        
        logger.info(
            f"Custom stoploss for {pair} {trade.entry_side.upper()}: "
            f"ATR={atr_at_entry:.5f}, Stop Price={stop_price:.5f}, Stoploss={stoploss*100:.2f}%"
        )
        return stoploss

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        Custom exit logic based on ATR take profit.
        Uses ATR at entry stored in trade.user_data.
        """
        # Get ATR at entry from trade.user_data
        atr_at_entry = trade.user_data.get('atr_at_entry')
        
        if atr_at_entry is None or atr_at_entry <= 0:
            logger.warning(f"No valid ATR at entry for {pair}, skipping TP check")
            return None
            
        # Check take profit condition for long positions
        if trade.is_long and current_rate >= (trade.open_rate + atr_at_entry * 1.0):
            logger.info(
                f"Take profit hit for {pair} LONG: "
                f"Entry={trade.open_rate:.5f}, Current={current_rate:.5f}, "
                f"ATR at entry={atr_at_entry:.5f}, TP={trade.open_rate + atr_at_entry * 1.0:.5f}"
            )
            return 'take_profit_atr'
        
        # Check take profit condition for short positions
        if trade.is_short and current_rate <= (trade.open_rate - atr_at_entry * 1.0):
            logger.info(
                f"Take profit hit for {pair} SHORT: "
                f"Entry={trade.open_rate:.5f}, Current={current_rate:.5f}, "
                f"ATR at entry={atr_at_entry:.5f}, TP={trade.open_rate - atr_at_entry * 1.0:.5f}"
            )
            return 'take_profit_atr'

        return None  

    def leverage(self, pair: str, current_time: datetime,  
                 current_rate: float, proposed_leverage: float,  
                 max_leverage: float, side: str, **kwargs) -> float:  
        return 25.0


# === Funciones auxiliares ===

def supertrend(df: pd.DataFrame, period: int = 14, multiplier: float = 1.8) -> pd.DataFrame:
    atr = ta.ATR(df, timeperiod=period)
    hl2 = (df['high'] + df['low']) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    trend = np.full(len(df), np.nan)
    direction = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        prev_trend = trend[i - 1] if not np.isnan(trend[i - 1]) else lowerband[i - 1]
        prev_dir = direction[i - 1] if not np.isnan(direction[i - 1]) else 1
        if df['close'].iloc[i] > upperband.iloc[i - 1]:
            trend[i] = lowerband.iloc[i]
            direction[i] = 1
        elif df['close'].iloc[i] < lowerband.iloc[i - 1]:
            trend[i] = upperband.iloc[i]
            direction[i] = -1
        else:
            trend[i] = prev_trend
            direction[i] = prev_dir
        if direction[i] == 1 and lowerband.iloc[i] > trend[i]:
            trend[i] = lowerband.iloc[i]
        if direction[i] == -1 and upperband.iloc[i] < trend[i]:
            trend[i] = upperband.iloc[i]
    df['supertrend'] = trend
    df['supertrend_direction'] = direction
    return df

def ash_with_color(df: pd.DataFrame, length: int = 16, smooth: int = 4, src: str = "close",
                   mode: str = "RSI", ma_type: str = "EMA",
                   alma_offset: float = 0.85, alma_sigma: int = 6) -> pd.DataFrame:
    def ma_func(arr, period: int):
        series = pd.Series(arr, index=df.index)
        if period == 1:
            return series
        if ma_type == "SMA":
            return pd.Series(ta.SMA(series, timeperiod=period), index=df.index)
        elif ma_type == "EMA":
            return pd.Series(ta.EMA(series, timeperiod=period), index=df.index)
        elif ma_type == "WMA":
            return pd.Series(ta.WMA(series, timeperiod=period), index=df.index)
        elif ma_type == "SMMA":
            out = series.rolling(period, min_periods=period).mean().copy()
            for i in range(period, len(series)):
                out.iloc[i] = (out.iloc[i - 1] * (period - 1) + series.iloc[i]) / period
            return out
        elif ma_type == "HMA":
            wma1 = ta.WMA(series, timeperiod=int(period / 2))
            wma2 = ta.WMA(series, timeperiod=period)
            return pd.Series(ta.WMA(2 * wma1 - wma2, timeperiod=int(np.sqrt(period))), index=df.index)
        elif ma_type == "ALMA":
            from technical.indicators import alma
            return pd.Series(alma(series, window=period, offset=alma_offset, sigma=alma_sigma), index=df.index)
        return pd.Series(ta.EMA(series, timeperiod=period), index=df.index)

    prices = pd.Series(df[src], index=df.index)  
    price1 = ma_func(prices, 1)  
    price2 = ma_func(prices.shift(1), 1)  

    Bulls0 = 0.5 * (abs(price1 - price2) + (price1 - price2))  
    Bears0 = 0.5 * (abs(price1 - price2) - (price1 - price2))  
    Bulls1 = price1 - prices.rolling(length).min()  
    Bears1 = prices.rolling(length).max() - price1  
    Bulls2 = 0.5 * (abs(df['high'] - df['high'].shift(1)) + (df['high'] - df['high'].shift(1)))  
    Bears2 = 0.5 * (abs(df['low'].shift(1) - df['low']) + (df['low'].shift(1) - df['low']))  

    Bulls, Bears = (Bulls0, Bears0) if mode == "RSI" else (Bulls1, Bears1) if mode == "STOCHASTIC" else (Bulls2, Bears2)  
    AvgBulls = ma_func(Bulls, length)  
    AvgBears = ma_func(Bears, length)  
    SmthBulls = ma_func(AvgBulls, smooth)  
    SmthBears = ma_func(AvgBears, smooth)  
    difference = pd.Series(abs(SmthBulls - SmthBears), index=df.index)  

    ash_color = pd.Series(['gray'] * len(df), index=df.index)  
    ash_color[(difference > SmthBulls) & (SmthBears < SmthBears.shift(1))] = 'orange'  
    ash_color[(difference > SmthBulls) & ~(SmthBears < SmthBears.shift(1))] = 'red'  
    ash_color[(difference > SmthBears) & (SmthBulls < SmthBulls.shift(1))] = 'lime'  
    ash_color[(difference > SmthBears) & ~(SmthBulls < SmthBulls.shift(1))] = 'green'  

    ash_bullish_signal = ash_color.isin(['green', 'lime']) & (  
        (ash_color.shift(1) == 'gray') | (ash_color != ash_color.shift(1))  
    )  
    ash_bearish_signal = ash_color.isin(['red', 'orange']) & (  
        (ash_color.shift(1) == 'gray') | (ash_color != ash_color.shift(1))  
    )  

    df['ash_bulls'] = AvgBulls  
    df['ash_bears'] = AvgBears  
    df['ash_diff'] = difference  
    df['ash_color'] = ash_color  
    df['ash_bullish_signal'] = ash_bullish_signal  
    df['ash_bearish_signal'] = ash_bearish_signal  
    return df