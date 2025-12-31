# Objetivos y Criterios de Evidencia

## Objetivo Principal

Demostrar que la alineacion geometrica (warping) mejora la discriminacion
entre clases en un pipeline simple (PCA + Fisher + clasificador simple).

Como lo dijo el asesor:
> "Lo que hace la diferencia es que hayas alineado esas canijas imagenes...
> van a funcionar mejor incluso con un clasificador tan chafa o tan simple
> como un KNN. Eso es lo que tienes que demostrar."

## Pregunta Central

¿Las imagenes alineadas (warped) producen caracteristicas mas separables
que las imagenes no alineadas (originales), manteniendo el mismo split
y protocolo experimental?

## Hipotesis

Si el warping es efectivo, entonces:
1. Las caracteristicas extraidas de imagenes warped tendran mayor Fisher ratio
2. Un clasificador KNN sobre imagenes warped tendra mayor accuracy que sobre originales
3. La mejora sera observable incluso con un clasificador simple (sin CNN)

## Criterios de Evidencia

### Evidencia Numerica

1. **Comparacion directa**: Original vs Warped con el mismo split
   - Mismas imagenes (por ID)
   - Mismo numero de componentes PCA
   - Mismo K en KNN

2. **Metricas robustas al desbalance**:
   - Accuracy
   - F1 Macro
   - Balanced Accuracy
   - Matriz de confusion

3. **Mejora esperada**: Warped > Original en todas las metricas

### Evidencia Visual

1. **Eigenfaces**: Mostrar las K eigenfaces principales
2. **Varianza explicada**: Curva acumulativa
3. **Distribucion de caracteristicas**: Histogramas por clase
4. **Fisher ratios**: Grafica de barras ordenada
5. **Matriz de confusion**: Para ambos escenarios

## Alcance

### Dataset Manual (957 imagenes con coordenadas)
- Se usa SIN balanceo adicional
- Util para desarrollo y depuracion

### Dataset Completo (15,153 imagenes)
- Se usa CON balanceo por split
- Escenario principal de evaluacion

### Escenarios de Clases

1. **Principal (2 clases)**: Enfermo (COVID + Viral_Pneumonia) vs Normal
2. **Secundario (3 clases)**: COVID vs Viral_Pneumonia vs Normal

## Principios de Trabajo

1. **Sin cajas negras**: Explicar matematicas y geometria paso a paso
2. **Clasificador simple**: Priorizar KNN como evidencia principal
3. **Reproducibilidad**: Documentar seeds, splits, parametros
4. **Transparencia**: Mostrar tanto exitos como fracasos

## No-Objetivos

- NO es objetivo obtener la mayor accuracy posible
- NO es objetivo superar el estado del arte
- NO es objetivo usar arquitecturas complejas (CNN, transformers)

El objetivo es demostrar que el WARPING aporta valor, no que tengamos
el mejor clasificador.
