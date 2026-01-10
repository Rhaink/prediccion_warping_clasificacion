# Prompt para Sesion 19 - Validacion de Comandos CLI

Copia y pega este prompt para continuar:

---

## CONTEXTO DE SESION 19

Continuacion del proyecto COVID-19 Landmark Detection.

**Estado actual:**
- Session 18 completada: Se implementaron 3 nuevos comandos CLI
- CLI tiene ahora 12 comandos
- 244 tests pasando
- Documentacion en docs/sesiones/SESION_18_CLI_NUEVOS_COMANDOS.md

**Comandos implementados en Session 18 (sin validar con datos reales):**
1. `cross-evaluate` - Evaluacion cruzada Original↔Warped
2. `evaluate-external` - Validacion en dataset externo FedCOVIDx
3. `test-robustness` - Pruebas de robustez (JPEG, blur, ruido)
4. DenseNet-121 agregado como backbone en train-classifier

## OBJETIVO SESION 19

Validar que los comandos implementados funcionan correctamente con datos reales y producen resultados consistentes con los experimentos originales.

### TAREAS DE VALIDACION:

1. **CROSS-EVALUATE** (Prioridad Alta):
   ```bash
   python -m src_v2 cross-evaluate \
       outputs/classifier_comparison/resnet18_original/best_model.pt \
       outputs/classifier_comparison/resnet18_warped/best_model.pt \
       --data-a outputs/full_warped_dataset \
       --data-b outputs/full_warped_dataset \
       --output-dir outputs/session19_validation/cross_eval
   ```

   **Verificar:**
   - Comando ejecuta sin errores
   - Resultados cercanos a Session 30: Original→Warped ~73%, Warped→Original ~95%
   - Gap ratio ~11x (modelo warped generaliza mejor)
   - JSON de salida tiene estructura correcta

2. **EVALUATE-EXTERNAL** (Prioridad Alta):
   ```bash
   python -m src_v2 evaluate-external \
       outputs/classifier_comparison/resnet18_warped/best_model.pt \
       --external-data outputs/external_validation/dataset3 \
       --output outputs/session19_validation/external_eval.json
   ```

   **Verificar:**
   - Mapeo 3→2 clases funciona (COVID=positive)
   - Metricas binarias correctas (sensitivity, specificity, AUC)
   - Accuracy cercana a ~53-57% (esperado por domain shift)

3. **TEST-ROBUSTNESS** (Prioridad Media):
   ```bash
   python -m src_v2 test-robustness \
       outputs/classifier_comparison/resnet18_warped/best_model.pt \
       --data-dir outputs/full_warped_dataset \
       --output outputs/session19_validation/robustness.json
   ```

   **Verificar:**
   - Perturbaciones se aplican correctamente
   - Degradacion medida vs baseline
   - Modelo warped debe ser mas robusto que original

4. **DENSENET-121** (Prioridad Baja):
   - Verificar que ya existe modelo entrenado: `outputs/classifier_comparison/densenet121_warped/best_model.pt`
   - Si no, probar entrenamiento corto (5 epochs) para validar que funciona

### PUNTOS CRITICOS A REVISAR:

1. **Orden de clases:** Verificar que el orden en checkpoints coincide con el esperado [COVID, Normal, Viral_Pneumonia]

2. **Estructura de datasets:**
   - Warped: `outputs/full_warped_dataset/test/` (ImageFolder)
   - Original: `data/dataset/COVID-19_Radiography_Dataset/` (estructura diferente)
   - External: `outputs/external_validation/dataset3/test/` (positive/negative)

3. **Posibles bugs:**
   - Carga de datasets con diferentes estructuras
   - Indice de clase COVID en mapeo 3→2
   - Funciones de perturbacion en test-robustness

### RESULTADOS ESPERADOS (Referencia):

| Comando | Metrica | Valor Esperado | Tolerancia |
|---------|---------|----------------|------------|
| cross-evaluate A→B | Accuracy | ~73% | ±5% |
| cross-evaluate B→A | Accuracy | ~95% | ±3% |
| cross-evaluate | Gap Ratio | ~11x | ±3x |
| evaluate-external | Accuracy | ~55% | ±10% |
| test-robustness original | Accuracy | ~98% | ±2% |
| test-robustness jpeg_q50 | Degradacion | <5% | - |

### ARCHIVOS CLAVE:

- `src_v2/cli.py` - Comandos implementados (lineas 2362-3154)
- `src_v2/models/classifier.py` - Clasificador con DenseNet-121
- `docs/sesiones/SESION_18_CLI_NUEVOS_COMANDOS.md` - Documentacion
- `docs/REFERENCIA_EXPERIMENTOS_ORIGINALES.md` - Resultados de referencia

### RESTRICCIONES:

- NO modificar codigo a menos que se encuentre un bug
- Documentar cualquier discrepancia con resultados esperados
- Si un comando falla, analizar el error antes de modificar
- Crear outputs/session19_validation/ para guardar resultados

### ENTREGABLES:

1. Reporte de validacion de cada comando
2. Comparacion con resultados originales
3. Lista de bugs encontrados (si hay)
4. Documentacion de Session 19 en docs/sesiones/

---

**Usa ultrathink para analizar errores si los hay.**
