# Estrategia SASAS v0.3.0

## 📊 Descripción

SASAS es una estrategia de trading algorítmico para Freqtrade que combina indicadores técnicos avanzados para generar señales de entrada y salida precisas. La estrategia utiliza **Supertrend** y **ASH (Absolute Strength Histogram)** como indicadores principales, con gestión de riesgo dinámica basada en **ATR (Average True Range)**.

## 🎯 Características Principales

- **Indicadores Combinados**: Supertrend + ASH para señales robustas
- **ROI Dinámico**: Basado en ATR para adaptarse a la volatilidad del mercado
- **Stoploss Dinámico**: Calculado con ATR para gestión de riesgo consistente
- **Sin Trailing Stop**: Salidas limpias al 100% al tocar ROI o Stoploss
- **Apalancamiento**: Configurado para 20x (ajustable)
- **Timeframe**: Optimizado para 15 minutos
- **Trading Bidireccional**: Soporta posiciones Long y Short

## 🔧 Configuración

### Parámetros Principales

```python
# Configuración básica
timeframe = '15m'
stoploss = -0.02  # 2% stoploss fijo como respaldo
minimal_roi = {"0": 0.1}  # 10% ROI mínimo como respaldo

# Gestión dinámica activada
use_custom_roi = True
use_custom_stoploss = True

# Trailing stop desactivado
trailing_stop = False

# Apalancamiento
leverage = 20.0
```

### Multiplicadores ATR

```python
atr_stoploss_multiplier = 1.5    # Stoploss = ATR × 1.5
atr_takeprofit_multiplier = 1.5  # ROI = ATR × 1.5
```

## 📈 Indicadores Utilizados

### 1. Supertrend
- **Período**: 7-30 (optimizable, default: 14)
- **Multiplicador**: 1.0-3.0 (optimizable, default: 1.8)
- **Función**: Determina la tendencia principal del mercado

### 2. ASH (Absolute Strength Histogram)
- **Longitud**: 5-40 (optimizable, default: 16)
- **Suavizado**: 2-10 (optimizable, default: 4)
- **Modo**: RSI, STOCHASTIC, ADX (optimizable, default: RSI)
- **Tipo MA**: ALMA, EMA, WMA, SMA, SMMA, HMA (optimizable, default: EMA)
- **Función**: Genera señales de entrada precisas

### 3. ATR (Average True Range)
- **Período**: 14 (fijo)
- **Función**: Calcula volatilidad para ROI y Stoploss dinámicos

## 🚀 Lógica de Trading

### Señales de Entrada

**Entrada Long:**
```python
(supertrend_direction == 1) & ash_bullish_signal
```

**Entrada Short:**
```python
(supertrend_direction == -1) & ash_bearish_signal
```

### Gestión de Salidas

**ROI Dinámico:**
- Calculado como: `(ATR × 1.5) / precio_entrada`
- Mínimo: 3%
- Fallback: 5% si no hay datos de ATR

**Stoploss Dinámico:**
- Calculado como: `-(ATR × 1.5) / precio_entrada`
- Fallback: -2% (stoploss fijo)

## 📋 Instalación y Uso

### Requisitos

```bash
# Instalar dependencias
pip install freqtrade
pip install talib-binary
pip install technical
```

### Configuración en Freqtrade

1. Copiar `SASAS.py` a la carpeta `user_data/strategies/`
2. Configurar en `config.json`:

```json
{
  "strategy": "SASAS",
  "timeframe": "15m",
  "trading_mode": "futures",
  "margin_mode": "isolated",
  "max_open_trades": 5,
  "stake_amount": "unlimited",
  "tradable_balance_ratio": 0.1
}
```

### Comandos de Ejecución

```bash
# Backtesting
freqtrade backtesting --strategy SASAS --timerange 20240101-20241201

# Optimización de hiperparámetros
freqtrade hyperopt --strategy SASAS --hyperopt-loss SharpeHyperOptLoss --epochs 100

# Dry run (simulación)
freqtrade trade --strategy SASAS --dry-run

# Trading en vivo
freqtrade trade --strategy SASAS
```

## ⚙️ Parámetros Optimizables

| Parámetro | Tipo | Rango | Default | Descripción |
|-----------|------|-------|---------|-------------|
| `sup_period` | Int | 7-30 | 14 | Período del Supertrend |
| `sup_multiplier` | Decimal | 1.0-3.0 | 1.8 | Multiplicador del Supertrend |
| `ash_length` | Int | 5-40 | 16 | Longitud del ASH |
| `ash_smooth` | Int | 2-10 | 4 | Suavizado del ASH |
| `ash_mode` | Categorical | RSI/STOCHASTIC/ADX | RSI | Modo de cálculo del ASH |
| `ash_ma_type` | Categorical | ALMA/EMA/WMA/SMA/SMMA/HMA | EMA | Tipo de media móvil |
| `ash_alma_offset` | Decimal | 0-1.5 | 0.85 | Offset para ALMA |
| `ash_alma_sigma` | Int | 1-10 | 6 | Sigma para ALMA |

## 📊 Rendimiento Esperado

### Características de Riesgo/Beneficio
- **Relación R:R**: 1:1 (ambos multiplicadores en 1.5)
- **Adaptabilidad**: Alta (se ajusta a la volatilidad del mercado)
- **Drawdown**: Controlado por stoploss dinámico
- **Frecuencia**: Media (señales de calidad sobre cantidad)

### Mercados Recomendados
- **Criptomonedas**: BTC/USDT, ETH/USDT, principales altcoins
- **Forex**: Pares mayores (EUR/USD, GBP/USD, etc.)
- **Índices**: S&P500, NASDAQ, etc.

## 🛠️ Personalización

### Ajustar Multiplicadores ATR

```python
# Para mayor conservadurismo
atr_stoploss_multiplier = 1.0
atr_takeprofit_multiplier = 2.0

# Para mayor agresividad
atr_stoploss_multiplier = 2.0
atr_takeprofit_multiplier = 1.0
```

### Cambiar Apalancamiento

```python
def leverage(self, pair: str, current_time: datetime,
             current_rate: float, proposed_leverage: float,
             max_leverage: float, side: str, **kwargs) -> float:
    return 10.0  # Cambiar a 10x
```

## ⚠️ Advertencias y Consideraciones

1. **Backtesting Obligatorio**: Siempre realizar backtesting antes del trading en vivo
2. **Gestión de Capital**: No arriesgar más del 1-2% del capital por operación
3. **Monitoreo**: Supervisar el rendimiento regularmente
4. **Optimización**: Re-optimizar parámetros periódicamente
5. **Condiciones de Mercado**: La estrategia puede requerir ajustes en mercados muy volátiles o laterales

## 📝 Changelog

### v0.3.0
- ✅ Implementado stoploss dinámico basado en ATR
- ✅ Eliminado trailing stop para salidas más limpias
- ✅ Optimizados multiplicadores ATR (1.5x para ambos)
- ✅ Limpieza de código (eliminadas variables no utilizadas)
- ✅ Mejorada gestión de errores

### v0.2.0
- ✅ Añadido ROI dinámico basado en ATR
- ✅ Implementado callback order_filled
- ✅ Añadidos parámetros optimizables

### v0.1.0
- ✅ Implementación inicial con Supertrend + ASH
- ✅ Configuración básica de entrada y salida

## 📞 Soporte

Para preguntas, sugerencias o reportar problemas:
- Revisar la documentación oficial de [Freqtrade](https://www.freqtrade.io/)
- Consultar el [repositorio de estrategias](https://github.com/freqtrade/freqtrade-strategies)

---

**⚠️ Disclaimer**: Esta estrategia es solo para fines educativos. El trading conlleva riesgos significativos. Nunca inviertas más de lo que puedes permitirte perder.