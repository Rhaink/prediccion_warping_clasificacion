# SESION 53 - Fill Rate Tradeoff

## Objetivo
Evaluar el trade-off entre fill rate, accuracy y robustez en JPEG usando
variantes del dataset warpeado.

## Resultados (validados)
| Dataset | Accuracy | Fill Rate | Robustness (JPEG Q50) |
|---------|----------|-----------|----------------------|
| Original 100% | 98.84% | 100% | 16.14% |
| Warped 47% | 98.02% | 47% | 0.53% |
| Warped 99% | 98.73% | 99% | 7.34% |
| Warped 96% (recomendado) | 99.10% | 96% | 3.06% |

## Conclusion
Warped 96% es el mejor balance entre accuracy y robustez.

## Referencias
- GROUND_TRUTH.json (fill_rate_tradeoff, robustness)
- README.md (tablas de resultados)
