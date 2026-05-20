# Sistema de Apuestas OU 2.5 — Arquitectura Conceptual

## 1. Objetivo

Este sistema es un framework cuantitativo de apuestas diseñado para identificar apuestas de valor en el mercado Over/Under 2.5 goles en partidos de fútbol.

El objetivo es estimar probabilidades justas utilizando un modelo de goles basado en Poisson, calibrar dichas probabilidades con información del mercado y evaluar la rentabilidad mediante backtesting histórico.

---

# 2. Pipeline de extremo a extremo

El pipeline completo es:

DATA → FEATURES → MODEL → CALIBRATION → BET SELECTION → BACKTEST

## Componentes

1. Ingesta de datos de partidos
2. Ingeniería de variables
3. Modelo de expectativa de goles
4. Calibración con mercado
5. Selección de apuestas
6. Evaluación de desempeño

---

# 3. Formulación matemática

## 3.1 Modelo de goles

Los goles esperados se estiman por separado:

- λ_home
- λ_away

Total de goles esperados:

$$
\lambda_{total} = \lambda_{home} + \lambda_{away}
$$

El modelo asume que el total de goles sigue un proceso Poisson.

---

## 3.2 Probabilidad de Under 2.5

El Under 2.5 ocurre cuando el total de goles es:

- 0
- 1
- 2

$$
P(\text{Under 2.5}) = \sum_{k=0}^{2} \frac{e^{-\lambda}\lambda^k}{k!}
$$

## 3.3 Probabilidad de Over 2.5


$$
P(\text{Over 2.5}) = 1 - P(\text{Under 2.5})
$$



---

## 3.4 Valor esperado (EV)

Para cualquier lado:

$$
EV = p \cdot odds - 1
$$

Un EV positivo indica valor teórico.

---

## 3.5 Lambda implícito del mercado

La probabilidad del mercado se convierte en goles esperados implícitos:

$$
P(\text{Over 2.5}) = 1 - \sum_{k=0}^{2} \frac{e^{-\lambda_{market}}\lambda_{market}^k}{k!}
$$

Se resuelve numéricamente usando Brent root finding.

---

## 3.6 Calibración

Se combina el modelo y el mercado:


$$
\lambda_{calibrated} = (1-\alpha)\lambda_{model} + \alpha\lambda_{market}
$$
	​

Donde α controla cuánto peso se asigna a la información del mercado.

---

# 4. Decisiones de diseño

## 4.1 Poisson

Se eligió porque:

- es interpretable
- es eficiente computacionalmente
- es una base sólida para goles en fútbol
- es compatible con inversión de probabilidades de mercado

## 4.2 Calibración con mercado

Se utiliza porque las cuotas de la casa de apuestas contienen información latente no capturada por modelos basados únicamente en históricos.

## 4.3 Kelly fraccional

Se utiliza para:

- reducir varianza
- controlar drawdowns
- mejorar estabilidad del bankroll

## 4.4 Selección del mejor lado

Solo se selecciona un lado:

- Over O Under

según el mayor EV positivo.

---

# 5. Hipótesis principales

El sistema se basa en las siguientes hipótesis.

## H1

La expectativa de goles mediante Poisson aproxima suficientemente la distribución real para OU 2.5.

## H2

Las cuotas del mercado contienen información adicional a la estadística histórica de los equipos.

## H3

La combinación del modelo con el mercado mejora la calibración.

## H4

Las oportunidades de EV positivo pueden monetizarse en muestras grandes.

---

# 6. Supuestos

## Supuestos estadísticos

- Los goles son eventos independientes
- El total de goles sigue distribución Poisson
- El histórico es representativo
- Las cuotas de cierre son proxies eficientes del mercado

## Supuestos operativos

- Las cuotas históricas son correctas
- La liquidación de apuestas es correcta
- No existen costos de transacción
- No existen restricciones de liquidez

---

# 7. Limitaciones actuales

Limitación principal actual:

No se cuenta aún con cuotas históricas reales de Over/Under.

Por lo tanto:

El backtest actual utiliza cuotas simuladas y no es válido para conclusiones de producción.

Este es el principal cuello de botella.

---

# 8. Interpretación de resultados

Los resultados actuales del backtest deben interpretarse únicamente como:

- validación del pipeline
- verificación lógica
- prueba de infraestructura

NO como evidencia de rentabilidad.

La validación real del modelo requiere:

- cuotas históricas reales
- walk-forward testing
- optimización de parámetros

---

# 9. Extensiones futuras

Mejoras planeadas:

- cuotas históricas reales OU
- optimización de alpha
- optimización de scaling
- walk-forward validation
- CLV tracking
- OU 3.5
- Asian totals
- live odds ingestion

---

# 10. Módulos fuente

Principales módulos implementados:

- src/markets/ou.py
- src/models/calibration_ou.py
- src/betting/selector_ou.py
- src/backtest/backtest_ou.py
- notebooks/analysis.ipynb

