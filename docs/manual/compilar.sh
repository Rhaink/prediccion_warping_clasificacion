#!/bin/bash
# Script para compilar el Manual de Usuario
# Autor: Sistema de Detección COVID-19 v15
# Fecha: Enero 2026

echo "========================================"
echo "  Compilación del Manual de Usuario"
echo "  Sistema de Detección COVID-19 v15"
echo "========================================"
echo ""

# Verificar que pdflatex está instalado
if ! command -v pdflatex &> /dev/null; then
    echo "❌ ERROR: pdflatex no está instalado"
    echo ""
    echo "Para instalar en Ubuntu/Debian:"
    echo "  sudo apt-get install texlive-full"
    echo ""
    echo "Para instalar en Fedora/RHEL:"
    echo "  sudo dnf install texlive-scheme-full"
    echo ""
    exit 1
fi

# Verificar que estamos en el directorio correcto
if [ ! -f "manual_usuario.tex" ]; then
    echo "❌ ERROR: No se encuentra manual_usuario.tex"
    echo "Por favor, ejecute este script desde el directorio docs/manual/"
    exit 1
fi

echo "📄 Archivo fuente encontrado: manual_usuario.tex"
echo ""

# Limpiar archivos temporales previos
echo "🧹 Limpiando archivos temporales previos..."
rm -f manual_usuario.aux manual_usuario.log manual_usuario.out manual_usuario.toc
echo "   ✓ Archivos temporales eliminados"
echo ""

# Primera compilación
echo "📝 Primera compilación (generando estructura)..."
pdflatex -interaction=nonstopmode manual_usuario.tex > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Primera compilación exitosa"
else
    echo "   ⚠️  Primera compilación con advertencias (normal)"
fi
echo ""

# Segunda compilación
echo "🔄 Segunda compilación (actualizando referencias)..."
pdflatex -interaction=nonstopmode manual_usuario.tex > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Segunda compilación exitosa"
else
    echo "   ⚠️  Segunda compilación con advertencias (normal)"
fi
echo ""

# Tercera compilación
echo "✨ Tercera compilación (finalizando)..."
pdflatex -interaction=nonstopmode manual_usuario.tex > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Tercera compilación exitosa"
else
    echo "   ⚠️  Tercera compilación con advertencias (normal)"
fi
echo ""

# Verificar que el PDF fue generado
if [ -f "manual_usuario.pdf" ]; then
    FILESIZE=$(du -h "manual_usuario.pdf" | cut -f1)
    PAGES=$(pdfinfo manual_usuario.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')

    echo "========================================"
    echo "  ✅ COMPILACIÓN COMPLETADA"
    echo "========================================"
    echo ""
    echo "📄 Archivo generado: manual_usuario.pdf"
    echo "📊 Tamaño: $FILESIZE"
    if [ ! -z "$PAGES" ]; then
        echo "📖 Páginas: $PAGES"
    fi
    echo ""

    # Verificar si hay imágenes
    IMG_COUNT=$(find imagenes/ -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
    if [ $IMG_COUNT -eq 0 ]; then
        echo "⚠️  ADVERTENCIA: No se encontraron imágenes en imagenes/"
        echo "   El PDF fue generado pero las imágenes aparecerán como espacios vacíos."
        echo "   Consulta INSTRUCCIONES_IMAGENES.md para capturar las imágenes."
        echo ""
    else
        echo "📸 Imágenes encontradas: $IMG_COUNT de 16"
        if [ $IMG_COUNT -lt 16 ]; then
            echo "   (Faltan $((16 - IMG_COUNT)) imágenes)"
        else
            echo "   ✓ Todas las imágenes están disponibles"
        fi
        echo ""
    fi

    # Ofrecer abrir el PDF
    echo "¿Desea abrir el PDF? (s/n)"
    read -r respuesta
    if [ "$respuesta" = "s" ] || [ "$respuesta" = "S" ]; then
        if command -v xdg-open &> /dev/null; then
            xdg-open manual_usuario.pdf &
            echo "✓ Abriendo PDF..."
        elif command -v open &> /dev/null; then
            open manual_usuario.pdf &
            echo "✓ Abriendo PDF..."
        else
            echo "❌ No se pudo abrir automáticamente. Abra manualmente: manual_usuario.pdf"
        fi
    fi

else
    echo "========================================"
    echo "  ❌ ERROR EN LA COMPILACIÓN"
    echo "========================================"
    echo ""
    echo "El archivo PDF no fue generado."
    echo "Revise el archivo manual_usuario.log para más detalles."
    echo ""
    echo "Errores comunes:"
    echo "  - Paquetes LaTeX faltantes"
    echo "  - Errores de sintaxis en el .tex"
    echo "  - Imágenes referenciadas pero no encontradas"
    echo ""
    exit 1
fi

# Limpiar archivos temporales si el usuario lo desea
echo ""
echo "¿Desea limpiar archivos temporales (.aux, .log, .out, .toc)? (s/n)"
read -r respuesta_limpiar
if [ "$respuesta_limpiar" = "s" ] || [ "$respuesta_limpiar" = "S" ]; then
    rm -f manual_usuario.aux manual_usuario.log manual_usuario.out manual_usuario.toc
    echo "✓ Archivos temporales eliminados"
else
    echo "✓ Archivos temporales conservados (útiles para debugging)"
fi

echo ""
echo "========================================"
echo "  ✨ Proceso completado"
echo "========================================"
