# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade, 
    Order,
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta

class SASAS(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    startup_candle_count = 120
    timeframe = '15m'

    # ROI y SL internos por defecto en 10%
    minimal_roi = {"0": 0.2}
    stoploss = -0.2
    
    # Activando Custom ROI y STOPLOSS
    use_custom_roi = True
    use_custom_stoploss = True  # Activado para stoploss dinámico basado en ATR
    
    # Trailing stop: DESACTIVADO para evitar cierres prematuros
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = None
    trailing_only_offset_is_reached = False

    # Parámetros optimizables
    sup_period = IntParameter(7, 30, default=14, space='buy', optimize=True)
    sup_multiplier = DecimalParameter(1.0, 3.0, default=1.8, decimals=2, space='buy', optimize=True)

    ash_length = IntParameter(5, 40, default=16, space='buy', optimize=True)
    ash_smooth = IntParameter(2, 10, default=4, space='buy', optimize=True)
    ash_mode = CategoricalParameter(['RSI', 'STOCHASTIC', 'ADX'], default='RSI', space='buy', optimize=True)
    ash_ma_type = CategoricalParameter(['ALMA', 'EMA', 'WMA', 'SMA', 'SMMA', 'HMA'], default='EMA', space='buy', optimize=True)
    ash_alma_offset = DecimalParameter(0, 1.5, default=0.85, decimals=2, space='buy', optimize=True)
    ash_alma_sigma = IntParameter(1, 10, default=6, space='buy', optimize=True)

    atr_stoploss_multiplier = 1.5  # Multiplicador para stoploss dinámico sobre ATR
    atr_takeprofit_multiplier = 1.5 # Multiplicador para takeprofit sobre ATR

    def version(self) -> str:
        return "SASAS_v0.3.0_tester"

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
        # ATR estándar TA-Lib para ROI dinámico
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
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
        # Método vacío para cumplir con la estructura esperada
        return dataframe
        

    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        """
        Callback llamado cuando se llena una orden.
        Guarda el ATR de la última vela en el momento de la primera entrada para calcular ROI y Stoploss dinámicos.
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()
            # Solo guardar ATR en la primera entrada del trade y si la orden es de entrada
            if (trade.nr_of_successful_entries == 1) and (order.ft_order_side == trade.entry_side):
                atr_value = last_candle.get("atr", None)
                if atr_value is not None and np.isfinite(atr_value):
                    trade.set_custom_data(key="entry_atr", value=float(atr_value))
        except Exception:
            pass  # En caso de error no hacer nada para evitar crash

    def custom_roi(self, pair: str, trade: Trade, current_time: datetime, trade_duration: int,
                   entry_tag: str | None, side: str, **kwargs) -> float | None:
        """
        ROI dinámico basado en el ATR guardado en la entrada multiplicado por atr_takeprofit_multiplier.
        """
        try:
            entry_atr = trade.get_custom_data("entry_atr")
            if entry_atr is None:
                return 0.1  # Fallback al 10% si no hay ATR
            
            entry_price = trade.open_rate
            if entry_price and entry_price > 0:
                entry_atr = float(entry_atr)
                if np.isfinite(entry_atr) and entry_atr > 0:
                    roi_value = (entry_atr * self.atr_takeprofit_multiplier) / entry_price
                    return max(roi_value, 0.05)  # Mínimo 5% de ROI
        except Exception:
            return 0.05  # Fallback seguro al 5%
        return 0.05   
        
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> float | None:
        """
        Stoploss dinámico basado en ATR - sin trailing, salida completa al tocar el stop.
        """
        try:
            entry_atr = trade.get_custom_data("entry_atr")
            if entry_atr is None:
                return None  # Usar stoploss fijo por defecto
            
            entry_price = trade.open_rate
            if entry_price and entry_price > 0:
                entry_atr = float(entry_atr)
                if np.isfinite(entry_atr) and entry_atr > 0:
                    # Stoploss fijo basado en ATR (sin trailing)
                    stoploss_distance = entry_atr * self.atr_stoploss_multiplier
                    stoploss_ratio = -stoploss_distance / entry_price
                    return stoploss_ratio
        except Exception:
            return None  # Usar stoploss por defecto en caso de error
        return None
        
    def leverage(self, pair: str, current_time: datetime,
                 current_rate: float, proposed_leverage: float,
                 max_leverage: float, side: str, **kwargs) -> float:
        # Configurar apalancamiento deseado, por ejemplo 10x
        return 20.0


# ==== Indicadores auxiliares ====
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
