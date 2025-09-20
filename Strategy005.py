# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, IntParameter
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa
from datetime import datetime

class Strategy005(IStrategy):
    """
    Strategy 005 adaptada para futuros con leverage 25 y órdenes limit para Hyperliquid.
    author@: Gerald Lonlas
    github@: https://github.com/freqtrade/freqtrade-strategies
    """

    INTERFACE_VERSION = 3
    can_short = True
    minimal_roi = {
        "1440": 0.01,
        "80": 0.02,
        "40": 0.03,
        "20": 0.04,
        "0":  0.05
    }
    stoploss = -0.10
    timeframe = '5m'
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    buy_volumeAVG = IntParameter(low=50, high=300, default=70, space='buy', optimize=True)
    buy_rsi = IntParameter(low=1, high=100, default=30, space='buy', optimize=True)
    buy_fastd = IntParameter(low=1, high=100, default=30, space='buy', optimize=True)
    buy_fishRsiNorma = IntParameter(low=1, high=100, default=30, space='buy', optimize=True)

    sell_rsi = IntParameter(low=1, high=100, default=70, space='sell', optimize=True)
    sell_minusDI = IntParameter(low=1, high=100, default=50, space='sell', optimize=True)
    sell_fishRsiNorma = IntParameter(low=1, high=100, default=50, space='sell', optimize=True)
    sell_trigger = CategoricalParameter(["rsi-macd-minusdi", "sar-fisherRsi"],
                                        default="rsi-macd-minusdi", space='sell', optimize=True)

    buy_params = {
        "buy_fastd": 1,
        "buy_fishRsiNorma": 5,
        "buy_rsi": 26,
        "buy_volumeAVG": 150,
    }
    sell_params = {
        "sell_fishRsiNorma": 30,
        "sell_minusDI": 4,
        "sell_rsi": 74,
        "sell_trigger": "rsi-macd-minusdi",
    }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)
        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > 0.00000200) &
                (dataframe['volume'] > dataframe['volume'].rolling(self.buy_volumeAVG.value).mean() * 4) &
                (dataframe['close'] < dataframe['sma']) &
                (dataframe['fastd'] > dataframe['fastk']) &
                (dataframe['rsi'] > self.buy_rsi.value) &
                (dataframe['fastd'] > self.buy_fastd.value) &
                (dataframe['fisher_rsi_norma'] < self.buy_fishRsiNorma.value)
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        if self.sell_trigger.value == 'rsi-macd-minusdi':
            conditions.append(qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value))
            conditions.append(dataframe['macd'] < 0)
            conditions.append(dataframe['minus_di'] > self.sell_minusDI.value)
        if self.sell_trigger.value == 'sar-fisherRsi':
            conditions.append(dataframe['sar'] > dataframe['close'])
            conditions.append(dataframe['fisher_rsi'] > self.sell_fishRsiNorma.value)
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'exit_long'] = 1
        return dataframe

    def leverage(self, pair: str, current_time: datetime, **kwargs) -> float:
        return 25.0
