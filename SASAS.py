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

    minimal_roi = {"0": 0.1}  
    stoploss = -0.1  

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
        return "SASAS_v0.3.16_debug"  

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
        dataframe["atr"] = ta.ATR(  
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=int(self.atr_period.value)  
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
          
    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:  
        try:  
            # Para obtener amount / rate / time_in_force si están disponibles:  
            amount = kwargs.get('amount', None)  
            rate = kwargs.get('rate', None)  
            time_in_force = kwargs.get('time_in_force', None)  

            dataframe, last_updated = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                logger.warning(f"[custom_exit - order_filled] DataFrame vacío para trade {trade.id} ({pair})")
                return

            last_candle = dataframe.iloc[-2]  
            atr = last_candle.get("atr", None)  

            if atr is not None and np.isfinite(atr) and atr > 0:  
                tp_multiplier = 1.0  
                sl_multiplier = 1.5  

                if trade.nr_of_successful_entries == 0 and (order.ft_order_side == trade.entry_side):  
                    if trade.entry_side == 'long':  
                        tp_level = trade.open_rate + atr * tp_multiplier  
                        sl_level = trade.open_rate - atr * sl_multiplier  
                    else:  
                        tp_level = trade.open_rate - atr * tp_multiplier  
                        sl_level = trade.open_rate + atr * sl_multiplier  

                    trade.set_custom_data("tp_level", float(tp_level))  
                    trade.set_custom_data("sl_level", float(sl_level))  

                    logger.info(  
                        f"[custom_exit - order_filled] trade {trade.id} ({pair}): tp_level={tp_level:.5f}, sl_level={sl_level:.5f}, side={trade.entry_side}"  
                    )  
            else:  
                logger.warning(f"[custom_exit - order_filled] ATR inválido en trade {trade.id} ({pair})")  
        except Exception as e:  
            logger.error(f"[custom_exit - order_filled] Exception en trade {trade.id} ({pair}): {e}")  
              
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,  
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:  
        tp_level = trade.get_custom_data("tp_level", None)  
        sl_level = trade.get_custom_data("sl_level", None)  

        logger.debug(  
            f"[custom_exit] trade {trade.id} ({pair}): current_rate={current_rate:.5f}, tp_level={tp_level}, sl_level={sl_level}, entry_side={trade.entry_side}"  
        )  

        if tp_level is None or sl_level is None:  
            logger.debug(f"[custom_exit] trade {trade.id}: Niveles TP/SL no definidos")  
            return None  

        exit_reason = None  

        if trade.entry_side == 'long':  
            if current_rate >= tp_level:  
                exit_reason = "tp_reached"  
            elif current_rate <= sl_level:  
                exit_reason = "sl_reached"  
        else:  
            if current_rate <= tp_level:  
                exit_reason = "tp_reached"  
            elif current_rate >= sl_level:  
                exit_reason = "sl_reached"  

        if exit_reason:  
            logger.info(f"[custom_exit] trade {trade.id} ({pair}): salida {exit_reason} a precio {current_rate:.5f}")  
            return exit_reason  

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