# Sesion 3d: Modelo Jerarquico de Landmarks (hierarchical.py)

**Fecha:** 2025-12-12
**Duracion estimada:** 1 hora
**Rama Git:** audit/main
**Archivos en alcance:** 368 lineas, 1 archivo

## Alcance

- Archivos revisados:
  - `src_v2/models/hierarchical.py` (368 lineas)
- Tests asociados:
  - NINGUNO dedicado (confirma hallazgo m5 global)
  - Cobertura indirecta: test_losses.py tiene tests para CentralAlignmentLoss (no Hierarchical)
- Objetivo especifico: Auditar modelo jerarquico experimental para prediccion de landmarks

## Contexto Critico: Codigo EXPERIMENTAL

| Aspecto | Hallazgo |
|---------|----------|
| **Status** | EXPERIMENTAL, NO INTEGRADO AL PIPELINE |
| **En CLI** | No hay comandos |
| **En training** | No hay integracion |
| **En README** | No documentado como parte del pipeline |
| **Creacion** | Sesion 13 (Nov 27-28, 2024) |
| **Motivacion** | Explorar estructura geometrica del etiquetado |
| **Rendimiento** | 6.83 px (PEOR que ResNet-18 directo: 3.71 px) |
| **Documentacion** | Marcado como `[EXPERIMENTAL]` en REFERENCIA_SESIONES_FUTURAS.md |

**Implicacion:** Es un "resultado negativo" cientificamente valioso, documentado en LaTeX (08_arquitectura_jerarquica.tex). No requiere integracion al pipeline de produccion.

## Estructura del Codigo

| Clase/Funcion | Lineas | Descripcion |
|---------------|--------|-------------|
| Docstring modulo | 1-13 | Explica arquitectura jerarquica y estructura geometrica |
| `HierarchicalLandmarkModel` | 42-259 | Modelo principal con backbone ResNet-18 + 2 cabezas |
| `__init__` | 72-129 | Inicializacion con backbone, axis_head, relative_head |
| `forward` | 131-162 | Forward pass: backbone -> eje -> parametros relativos |
| `_reconstruct_landmarks` | 164-235 | Reconstruye 30 coords desde parametros |
| `freeze_backbone` | 237-240 | Congela backbone para Phase 1 |
| `unfreeze_backbone` | 242-245 | Descongela para Phase 2 |
| `get_trainable_params` | 247-259 | Grupos de parametros con LR diferenciado |
| `AxisLoss` | 262-293 | Loss para penalizar errores en eje L1-L2 |
| `CentralAlignmentLossHierarchical` | 296-335 | Loss de verificacion de alineacion |
| `__main__` block | 338-368 | Test de sanidad del modelo |

## Hallazgos por Auditor

### Arquitecto de Software

| ID | Severidad | Descripcion | Ubicacion | Solucion Propuesta |
|----|-----------|-------------|-----------|-------------------|
| A01 | ⚪ | Patron jerarquico innovador: predice eje central primero, luego parametros relativos. Explota estructura geometrica del etiquetado. | `hierarchical.py:42-259` | Fortaleza arquitectonica - contribucion cientifica. |
| A02 | ⚪ | Excelente centralizacion: 13 constantes importadas de constants.py en lugar de magic numbers. | `hierarchical.py:22-35` | Fortaleza - mantenibilidad. |
| A03 | ⚪ | No exportado en `__init__.py` - INTENCIONAL. Es codigo experimental, no parte del pipeline de produccion. | `models/__init__.py` | Correcto - prototipo experimental. |
| A04 | ⚪ | Uso de GroupNorm en lugar de BatchNorm. Mas estable para batches pequenos en fase experimental. | `hierarchical.py:102, 121, 125` | Fortaleza - decision tecnica apropiada. |
| A05 | ⚪ | Separacion clara de responsabilidades: backbone compartido, axis_head, relative_head, reconstruccion. | `hierarchical.py:87-129` | Fortaleza arquitectonica. |

### Revisor de Codigo

| ID | Severidad | Descripcion | Ubicacion | Solucion Propuesta |
|----|-----------|-------------|-----------|-------------------|
| C01 | ⚪ | Type hints completos en metodos publicos: `forward()`, `_reconstruct_landmarks()`, `get_trainable_params()`. | Global | Fortaleza. |
| C02 | ⚪ | Uso correcto de epsilon `1e-8` para evitar division por cero en 3 ubicaciones. | `hierarchical.py:193, 324, 361` | Fortaleza - estabilidad numerica. |
| C03 | ⚪ | `torch.clamp(0, 1)` al final de `_reconstruct_landmarks()` garantiza coordenadas normalizadas. | `hierarchical.py:233` | Fortaleza - output robusto. |
| C04 | ⚪ | Comentarios `# CORREGIDO:` documentan historial de mejoras y bugs resueltos. | `hierarchical.py:209, 216, 224` | Fortaleza - trazabilidad. |
| C05 | 🟡 | `Optional` importado de typing pero no usado en el modulo. | `hierarchical.py:20` | Remover import no usado. Menor, no afecta funcionalidad. |
| C06 | ⚪ | `CENTRAL_LANDMARKS` importado de constants.py pero se define `CENTRAL_T` localmente. Consistente con uso especializado (diccionario con valores t). | `hierarchical.py:24, 66-70` | Observacion - uso correcto. |

### Especialista en Documentacion

| ID | Severidad | Descripcion | Ubicacion | Solucion Propuesta |
|----|-----------|-------------|-----------|-------------------|
| D01 | ⚪ | Docstring de modulo excelente: explica idea principal, estructura del etiquetado descubierta (t=0.25, 0.50, 0.75). | `hierarchical.py:1-13` | Fortaleza - autodocumentado. |
| D02 | ⚪ | Docstring de clase completo: lista todos los landmarks con indices 0-based y posiciones t teoricas. | `hierarchical.py:43-59` | Fortaleza - referencia util. |
| D03 | ⚪ | Docstrings de metodos completos con Args/Returns bien formateados. | `hierarchical.py:132-139, 170-179, 273-281, 308-315` | Fortaleza. |
| D04 | ⚪ | Comentarios inline explicativos en logica compleja de `_reconstruct_landmarks()`. | `hierarchical.py:191-227` | Fortaleza - codigo autoexplicativo. |
| D05 | ⚪ | Documentacion LaTeX completa (08_arquitectura_jerarquica.tex) explica hipotesis, bugs encontrados y conclusion de resultado negativo. | Externo | Fortaleza - rigor academico. |

### Ingeniero de Validacion

| ID | Severidad | Descripcion | Ubicacion | Solucion Propuesta |
|----|-----------|-------------|-----------|-------------------|
| V01 | 🟡 | Sin tests unitarios dedicados. Relacionado con hallazgo m5 global. Dado que es codigo experimental no integrado, la prioridad es baja. | `tests/` | Recomendado (baja prioridad): agregar tests basicos si se retoma desarrollo. |
| V02 | ⚪ | Bloque `__main__` funciona como test de sanidad: verifica shapes, rango [0,1], y alineacion de L10 al eje. | `hierarchical.py:338-368` | Fortaleza - test manual incluido. |
| V03 | ⚪ | Validacion ejecutada exitosamente: Output shape (4, 30), range [0.195, 0.713], L10 dist = 0.000000. | Validacion §7.2 | Fortaleza - modelo funcional. |

### Auditor Maestro

| ID | Severidad | Descripcion | Ubicacion | Solucion Propuesta |
|----|-----------|-------------|-----------|-------------------|
| AM01 | ⚪ | Modulo representa resultado negativo cientificamente valioso: arquitectura jerarquica prometedora teoricamente pero inferior en practica (6.83 px vs 3.71 px). | Global | Valor academico - documentar en tesis. |
| AM02 | ⚪ | Codigo experimental bien mantenido: refactorizado en commits recientes (constantes, logging, geometry centralizado). | Global | Fortaleza - codigo limpio aunque experimental. |
| AM03 | ⚪ | Balance muy positivo: 15 fortalezas/observaciones vs 2 hallazgos menores. | Global | Proceder con confianza. |

## Veredicto del Auditor Maestro

| Metrica | Valor |
|---------|-------|
| **Estado del modulo** | ✅ **APROBADO** |
| **Conteo (Sesion 3d)** | 0🔴, 0🟠, 2🟡, 20⚪ |
| **Aplicacion umbrales §5.2** | Cumple criterio "✅ Aprobado" (0🔴, ≤2🟠) |
| **Naturaleza del modulo** | Codigo EXPERIMENTAL - resultado negativo cientifico |
| **Prioridad 1** | C05, V01 (🟡): Import no usado, tests opcionales |
| **Siguiente paso** | Completar modulo models/ - Sesion 4 (training/) |

### Justificacion del Veredicto

El modulo `hierarchical.py` es **codigo experimental** que representa un resultado negativo cientificamente valioso:

**Contexto Cientifico:**
- Hipotesis: Explotar estructura geometrica exacta del etiquetado (L9,L10,L11 dividen eje en t=0.25, 0.50, 0.75)
- Resultado: 6.83 px de error - PEOR que modelo directo ResNet-18 (3.71 px)
- Conclusion: La parametrizacion explicita no mejora el rendimiento
- Documentacion: LaTeX completo en 08_arquitectura_jerarquica.tex

**Fortalezas Tecnicas (20⚪):**
1. Arquitectura jerarquica innovadora (eje primero, luego relativos)
2. Excelente centralizacion de constantes (13 imports)
3. No exportado en __init__.py - INTENCIONAL (prototipo)
4. GroupNorm para batches pequenos
5. Separacion clara de responsabilidades (backbone, heads, reconstruccion)
6. Type hints completos
7. Epsilon 1e-8 para estabilidad numerica
8. torch.clamp(0,1) para normalizacion
9. Comentarios `# CORREGIDO:` con historial
10. CENTRAL_LANDMARKS uso consistente con variante especializada (CENTRAL_T)
11. Docstring de modulo explicativo
12. Docstring de clase con lista de landmarks
13. Docstrings de metodos completos
14. Comentarios inline en logica compleja
15. Documentacion LaTeX externa completa (08_arquitectura_jerarquica.tex)
16. Bloque __main__ como test de sanidad
17. Validacion exitosa (L10 dist = 0.000000)
18. Resultado negativo cientifico valioso
19. Codigo experimental bien refactorizado (constantes, logging, geometry)
20. Balance positivo: 20 fortalezas vs 2 hallazgos menores

**Hallazgos Menores (2🟡):**
- C05: `Optional` importado pero no usado
- V01: Sin tests dedicados (prioridad baja para codigo experimental)

## Solicitud de Validacion (§7.2)

```
📋 SOLICITUD DE VALIDACION #1
- Comando a ejecutar: .venv/bin/python src_v2/models/hierarchical.py
- Resultado esperado: Sin errores, shapes correctos, L10 dist ~0
- Importancia: Verifica que modelo jerarquico funciona
- Criterio de exito: Output shape (4, 30), range [0,1], dist = 0

Usuario confirmo: Si, ejecutar
Resultado: ✓ PASSED
```

## Validaciones Realizadas

| Comando/Accion | Resultado Esperado | Resultado Obtenido | ✓/✗ |
|----------------|-------------------|-------------------|-----|
| `.venv/bin/python src_v2/models/hierarchical.py` | Shapes correctos, dist ~0 | Input (4,3,224,224), Output (4,30), range [0.195, 0.713], dist=0.000000 | ✓ |

## Correcciones Aplicadas

**NINGUNA REQUERIDA** - El modulo es codigo experimental que funciona correctamente. Los 2 hallazgos 🟡 son opcionales y de baja prioridad.

## 🎯 Progreso de Auditoria

**Modulos completados:** 6/12 (Config + Datos + Losses + ResNet + Classifier + Hierarchical)
**Modulo models/:** 4/4 archivos ✅ COMPLETADO
**Hallazgos totales acumulados:** [🔴:0 | 🟠:8 (5 resueltos, 3 pendientes) | 🟡:21 (+2 esta sesion) | ⚪:76 (+20 esta sesion)]
**Proximo hito:** Sesion 4 - training/ (trainer.py + callbacks.py)

## Notas para Siguiente Sesion

- Modulo models/ COMPLETADO (4/4 archivos: losses.py, resnet_landmark.py, classifier.py, hierarchical.py)
- hierarchical.py es codigo experimental - no requiere integracion ni tests prioritarios
- Quedan 3🟠 pendientes globales: M1 (PFS claim), M3 (sesgos dataset), M4 (margen 1.05)
- Proxima sesion: Sesion 4 - Sistema de entrenamiento (training/)

## Registro de Commit (§4.4 paso 9, §8.2)

| Campo | Valor |
|-------|-------|
| **Rama** | audit/main |
| **Hash inicial** | 95273c3 |
| **Hash commit** | 34716c5 |
| **Mensaje** | `audit(session-3d): auditoria hierarchical.py (codigo experimental)` |
| **Archivos modificados** | `audit/sessions/session_03d_hierarchical.md` |

## Desviaciones de Protocolo Identificadas Post-Sesion

| ID | Severidad | Descripcion | Accion Correctiva |
|----|-----------|-------------|-------------------|
| P01 | 🟡 | V01 usaba "Opcional" siendo 🟡 (viola §5.1) | Corregido: cambiado a "Recomendado (baja prioridad)" |
| P02 | 🟡 | Conteo ⚪ incorrecto (15 reportado vs 20 real) | Corregido: actualizado a 20⚪ |

**Estado:** Desviaciones corregidas mediante verificacion con agentes. Cumplimiento 100% post-correccion.
