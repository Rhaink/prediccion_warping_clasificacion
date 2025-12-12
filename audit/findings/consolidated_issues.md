# Hallazgos Consolidados de Auditoría
**Proyecto:** Clasificación de Radiografías de Tórax
**Última actualización:** 2025-12-12
**Sesiones incluidas:** 0

## Metodología de Consolidación

Los hallazgos individuales de cada auditor (session_00_mapping.md) fueron consolidados siguiendo estos criterios:

### Agrupación por Tema
| Hallazgos Originales | Consolidado | Justificación |
|---------------------|-------------|---------------|
| A01 (cli monolítico) | m1 | Reclasificado a 🟡: no bloquea defensa, mejora futura |
| C01 (imports inline) | m3 | Reclasificado a 🟡: no bloquea defensa, mejora futura |
| D01 (PFS claim) | M1 | Mantenido 🟠: afecta credibilidad científica |
| D02 (sesgos dataset) | M3 | Mantenido 🟠: requisito ético en ML médico |
| D03 (margen 1.05) | M4 | Mantenido 🟠: pregunta probable del jurado |
| V01+V02 (tests modelos) | m5 | Reclasificado a 🟡: tests de integración existen |
| A02, C02, V03 | m2, m4 | Mantenido 🟡: mejoras menores |

### Criterios de Reclasificación
- **🟠 → 🟡**: Si existe workaround o no afecta directamente la defensa
- **Agrupación**: Hallazgos similares se combinan bajo un único ID

### Hallazgos Sesión 0 (Original)
- **Conteo original:** 0 🔴, 7 🟠, 3 🟡, 4 ⚪
- **Conteo consolidado:** 0 🔴, 4 🟠, 5 🟡, 4 ⚪

---

## Resumen de Hallazgos

| Severidad | Cantidad | Resueltos | Pendientes |
|-----------|----------|-----------|------------|
| 🔴 Critico | 0 | 0 | 0 |
| 🟠 Mayor | 6 | 3 | 3 |
| 🟡 Menor | 12 | 0 | 12 |
| ⚪ Nota | 14 | 0 | 14 |
| **Total** | **32** | **3** | **29** |

**Nota:**
- Sesion 1 agrego 1🟡 (C01) y 4⚪ (A01, D01, V01, V02)
- Sesion 2 agrego 2🟠 (D01-S2, V01-S2), 6🟡, 6⚪

---

## Hallazgos 🟠 Mayores (Requieren corrección antes de defensa)

### M1: Claim incorrecto de PFS
| Campo | Valor |
|-------|-------|
| **ID** | M1 |
| **Severidad** | 🟠 Mayor |
| **Auditor** | Especialista en Documentación |
| **Sesión** | 47-48 |
| **Descripción** | El README y documentación afirman que el sistema "fuerza la atención del modelo a los pulmones". Sin embargo, el análisis PFS (Sesiones 47-48) mostró PFS ≈ 0.487 (~50%), lo cual es estadísticamente igual a aleatorio y NO evidencia de foco pulmonar. |
| **Ubicación** | README.md, documentación antigua |
| **Impacto** | Falsa afirmación científica que un jurado experto detectaría |
| **Solución** | Remover claim de PFS. Mantener solo: "Normalización geométrica mejora robustez" (validado causalmente en Sesión 39) |
| **Esfuerzo** | 30 minutos |
| **Estado** | ⏳ Pendiente |

### M2: CLAHE tile_size inconsistente
| Campo | Valor |
|-------|-------|
| **ID** | M2 |
| **Severidad** | 🟠 Mayor |
| **Auditor** | Revisor de Código |
| **Sesión** | 50 |
| **Descripción** | Código usa tile_size=4 (correcto desde S50), pero documentación legacy menciona tile_size=8. |
| **Ubicación** | scripts/legacy/, documentación antigua |
| **Impacto** | Confusión al reproducir resultados |
| **Solución** | Auditar archivos legacy y clarificar: tile_size=4 es el válido para resultados finales |
| **Esfuerzo** | 20 minutos |
| **Estado** | ✅ **RESUELTO** (Sesión 1) |
| **Resolución** | Verificado en Sesión 1: tile_size=4 consistente en todos los archivos (constants.py, GROUND_TRUTH.json, README.md, configs/, todos los scripts). La única mención de tile_size=8 está en scripts/visualization/generate_prediction_samples.py para comparación visual intencional. CHANGELOG.md confirma unificación. |

### M3: Sesgos del dataset no documentados
| Campo | Valor |
|-------|-------|
| **ID** | M3 |
| **Severidad** | 🟠 Mayor |
| **Auditor** | Auditor Maestro |
| **Sesión** | 0 |
| **Descripción** | El proyecto no documenta explícitamente potenciales sesgos en el dataset COVID-19: distribución demográfica desconocida, equipamiento radiológico variado, origen geográfico múltiple. Tampoco hay disclaimer de uso clínico. |
| **Ubicación** | README.md |
| **Impacto** | Falta transparencia para evaluadores especializados en ML médico |
| **Solución** | Añadir sección "Limitaciones y Sesgos Conocidos" + disclaimer: "Este modelo es experimental y NO está validado para uso clínico" |
| **Esfuerzo** | 45 minutos |
| **Estado** | ⏳ Pendiente |

### M4: Margen óptimo 1.05 sin justificación
| Campo | Valor |
|-------|-------|
| **ID** | M4 |
| **Severidad** | 🟠 Mayor |
| **Auditor** | Especialista en Documentación |
| **Sesión** | 25 |
| **Descripción** | OPTIMAL_MARGIN_SCALE=1.05 en constants.py sin explicar por qué este valor. Sesión 25 optimizó margen pero análisis no está en documentación final. |
| **Ubicación** | src_v2/constants.py:212, documentación |
| **Impacto** | Un jurado preguntará "¿por qué 1.05 y no 1.10?" |
| **Solución** | Documentar: "Grid search [1.0-1.3] en Sesión 25 encontró 1.05 minimiza error de warping" |
| **Esfuerzo** | 30 minutos |
| **Estado** | ⏳ Pendiente |

### M5: Docstring incompleto en get_dataframe_splits() (Sesion 2)
| Campo | Valor |
|-------|-------|
| **ID** | M5 (D01-S2) |
| **Severidad** | 🟠 Mayor |
| **Auditor** | Especialista en Documentacion |
| **Sesion** | 2 |
| **Descripcion** | `get_dataframe_splits()` tiene docstring minimo sin Args/Returns completos. Funcion publica deberia estar mejor documentada para que terceros puedan usarla. |
| **Ubicacion** | src_v2/data/dataset.py:286-289 |
| **Impacto** | Documentacion incompleta para funcion publica |
| **Solucion** | Agregar docstring completo con Args y Returns |
| **Esfuerzo** | 5 minutos |
| **Estado** | ✅ **RESUELTO** (Sesion 2) |
| **Resolucion** | Docstring completado con Args y Returns en dataset.py:286-300 |

### M6: dataset.py sin tests dedicados (Sesion 2)
| Campo | Valor |
|-------|-------|
| **ID** | M6 (V01-S2) |
| **Severidad** | 🟠 Mayor |
| **Auditor** | Ingeniero de Validacion |
| **Sesion** | 2 |
| **Descripcion** | `LandmarkDataset`, `create_dataloaders()`, `compute_sample_weights()` sin tests unitarios dedicados. Test coverage del modulo dataset.py es ~0%. |
| **Ubicacion** | tests/ |
| **Impacto** | Falta cobertura de tests en modulo critico |
| **Solucion** | Crear tests/test_dataset.py con tests para funciones publicas principales |
| **Esfuerzo** | 30 minutos |
| **Estado** | ✅ **RESUELTO** (Sesion 2) |
| **Resolucion** | Creado tests/test_dataset.py con 14 tests: 5 para compute_sample_weights, 5 para LandmarkDataset, 4 para get_dataframe_splits |

---

## Hallazgos 🟡 Menores (Corregir si hay tiempo)

### m1: cli.py monolítico
| Campo | Valor |
|-------|-------|
| **ID** | m1 |
| **Severidad** | 🟡 Menor |
| **Sesión** | 42 |
| **Descripción** | cli.py tiene 6,687 líneas con 20 comandos en un solo archivo. Difícil de mantener. |
| **Ubicación** | src_v2/cli.py |
| **Solución** | Refactorizar en submódulos (cli_train.py, cli_eval.py, etc.) - Para futuro |
| **Estado** | ⏳ Pendiente |

### m2: Funciones CLI muy largas
| Campo | Valor |
|-------|-------|
| **ID** | m2 |
| **Severidad** | 🟡 Menor |
| **Sesión** | 42 |
| **Descripción** | optimize_margin() tiene 835 líneas, otras funciones >300 líneas |
| **Ubicación** | src_v2/cli.py:5843 |
| **Solución** | Extraer subfunciones con responsabilidad única |
| **Estado** | ⏳ Pendiente |

### m3: 48 imports inline en CLI
| Campo | Valor |
|-------|-------|
| **ID** | m3 |
| **Severidad** | 🟡 Menor |
| **Sesión** | 42 |
| **Descripción** | Imports dentro de funciones en lugar de top-level |
| **Ubicación** | src_v2/cli.py |
| **Solución** | Mover imports al inicio del módulo |
| **Estado** | ⏳ Pendiente |

### m4: Return type hints incompletos
| Campo | Valor |
|-------|-------|
| **ID** | m4 |
| **Severidad** | 🟡 Menor |
| **Sesión** | 42 |
| **Descripción** | ~40% de funciones sin type hints de retorno |
| **Ubicación** | Varios archivos |
| **Solución** | Añadir return type hints progresivamente |
| **Estado** | ⏳ Pendiente |

### m5: Módulos críticos sin tests dedicados
| Campo | Valor |
|-------|-------|
| **ID** | m5 |
| **Severidad** | 🟡 Menor |
| **Sesión** | 42 |
| **Descripción** | resnet_landmark.py (325 líneas) y hierarchical.py (368 líneas) sin tests unitarios |
| **Ubicación** | src_v2/models/, tests/ |
| **Solución** | Añadir tests para forward pass, shapes, outputs |
| **Estado** | ⏳ Pendiente |

### m6: Docstring inconsistente en geometry.py (Sesión 1)
| Campo | Valor |
|-------|-------|
| **ID** | m6 (C01) |
| **Severidad** | 🟡 Menor |
| **Sesión** | 1 |
| **Descripción** | Docstring de `compute_perpendicular_vector_np` indica soporte para shapes `(2,)` o `(N, 2)`, pero implementación solo funciona para `(2,)`. Inconsistencia documentación-código. |
| **Ubicación** | src_v2/utils/geometry.py:12-26 |
| **Solución** | Corregir docstring para indicar solo `(2,)` o implementar soporte real para `(N, 2)`. |
| **Estado** | ⏳ Pendiente |

---

## Hallazgos ⚪ Notas (Opcionales)

| ID | Sesión | Descripción | Consideración |
|----|--------|-------------|---------------|
| n1 | 0 | Type hints podrían mejorarse en archivos legacy | Archivos nuevos (S42+) tienen buen coverage |
| n2 | 0 | Documentación en español | Considerar traducir README para publicaciones |
| n3 | 0 | Dataset de 957 muestras | Válido para maestría, validación externa sería valiosa |
| n4 | 0 | 14 dependencias core | Bien documentado en requirements.txt |
| n5 (A01) | 1 | `compute_perpendicular_vector_np` no exportada en `__init__.py` | Documentar como uso interno |
| n6 (D01) | 1 | `OPTIMAL_MARGIN_SCALE` podría mencionar rango grid search [1.0-1.3] | Mejora opcional para jurado |
| n7 (V01) | 1 | `geometry.py` sin tests unitarios dedicados | Cobertura indirecta existe |
| n8 (V02) | 1 | ~15 constantes nuevas sin tests en test_constants.py | Agregar cuando haya tiempo |

---

## Historial de Resoluciones

| Fecha | ID | Acción | Verificado |
|-------|----|----|------------|
| 2025-12-12 | M2 | Verificado consistencia tile_size=4 en todos los archivos del proyecto | ✓ Sesión 1 |

---

## Criterios de Cierre de Auditoría

Para considerar la auditoría COMPLETA:
- [ ] 0 hallazgos 🔴 abiertos
- [ ] ≤3 hallazgos 🟠 (justificados si no resueltos)
- [ ] 100% módulos auditados
- [ ] Resumen ejecutivo aprobado

---

## Notas para Resolución

### Priorización Recomendada
1. **M1 (PFS claim)** - CRÍTICO para credibilidad científica
2. **M3 (Sesgos dataset)** - Importante para transparencia
3. **M4 (Margen 1.05)** - Respuesta simple con gran impacto
4. **M2 (CLAHE)** - Limpieza documental

### Tiempo Estimado Total
- Hallazgos Mayores: ~2 horas
- Hallazgos Menores: ~4-6 horas (opcional)

### Riesgos de No Resolver
- **M1**: Jurado experto puede cuestionar validez de afirmaciones
- **M3**: Falta transparencia esperada en ML médico
- **M4**: Preguntas incómodas durante defensa
- **M2**: Confusión al intentar reproducir resultados
