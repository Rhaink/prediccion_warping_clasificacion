# Hallazgos Consolidados de Auditoría
**Proyecto:** Clasificación de Radiografías de Tórax
**Última actualización:** 2025-12-11
**Sesiones incluidas:** 0-50

## Resumen de Hallazgos

| Severidad | Cantidad | Resueltos | Pendientes |
|-----------|----------|-----------|------------|
| 🔴 Crítico | 0 | 0 | 0 |
| 🟠 Mayor | 4 | 0 | 4 |
| 🟡 Menor | 5 | 0 | 5 |
| ⚪ Nota | 4 | 0 | 4 |
| **Total** | **13** | **0** | **13** |

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
| **Estado** | ⏳ Pendiente |

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

---

## Hallazgos ⚪ Notas (Opcionales)

| ID | Descripción | Consideración |
|----|-------------|---------------|
| n1 | Type hints podrían mejorarse en archivos legacy | Archivos nuevos (S42+) tienen buen coverage |
| n2 | Documentación en español | Considerar traducir README para publicaciones |
| n3 | Dataset de 957 muestras | Válido para maestría, validación externa sería valiosa |
| n4 | 14 dependencias core | Bien documentado en requirements.txt |

---

## Historial de Resoluciones

| Fecha | ID | Acción | Verificado |
|-------|----|----|------------|
| - | - | Sin resoluciones aún | - |

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
