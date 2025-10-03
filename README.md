# Segmentacion_imagenes
Pipeline completo para segmentación binaria Permeable / No permeable sobre ortofotos PNOA usando PyTorch, TorchGeo y Segmentation Models PyTorch (SMP). Incluye datasets, transforms, entrenamiento, búsqueda de hiperparámetros (Optuna/Hyperopt), y exportación a SHP desde predicciones raster.


📂 Datos
El proyecto utiliza una estructura estándar de directorios para organizar insumos y datos procesados:

INSUMOS/   # Archivos de entrada (ortofotos, vectores, etc.)

DATA/      # Datos procesados y salidas de modelos

🔗 Opción 1: Datos ya organizados en Google Drive

Puedes acceder directamente a la estructura completa (INSUMOS y DATA) en el siguiente enlace:

👉 https://drive.google.com/drive/folders/1bAoQoRNwwrQ80NHqpTCJstQLO003U4NS


🔗 Opción 2: Descarga desde fuentes oficiales

Si prefieres armar la estructura desde cero:

* Ortofotos PNOA → disponibles en la web oficial del Instituto Geográfico Nacional:

      https://pnoa.ign.es/pnoa-imagen/productos-a-descarga

* Datos de cobertura del suelo SIOSE → descargables desde el Centro de Descargas del CNIG:

      https://centrodedescargas.cnig.es/CentroDescargas/siose


🧩 Estructura (scripts clave)

* datasets.py – Datasets TorchGeo para imágenes y etiquetas (mask TIFF).
    Define PNOAImages (ortofotos) y SidewalkLabels (máscaras), compatibles con samplers Geo. 

* transforms.py – Aumentos y normalización con Albumentations.
    TrainTransform (flip + Normalize) y TestValTransform (Normalize) aplicadas en tensor CHW↔HWC. 

* CNN.py – Entrenador general (arquitectura/backbone/loss configurables).
    Incluye: creación de modelos SMP, pérdidas (CE, Focal, Dice, Jaccard, combinadas), loops de train/val, logging a TensorBoard y utilidades de checkpoint. 

* CNN_deeplabv3plus_mitb4_v3.py – Receta lista para DeepLabV3+ + MiT-B4, con mezcla de pérdidas (CE+Dice), AMP y gradient accumulation. Parámetros para EPOCHS, LR, BATCH_SIZE, GRAD_ACCUM_STEPS, PATIENCE, SCHED_FACTOR, etc. 

* segmentation_optuna.py – Búsqueda bayesiana de hiperparámetros (Optuna) sobre arquitectura/backbone/loss/lr/batch, integrándose con los datasets y transforms anteriores; registra a TensorBoard y guarda el mejor checkpoint.

* segmentation_hyperopt.py – Alternativa de HPO con Hyperopt (estructura análoga a Optuna, mismo flujo de datos/entrenamiento). 

* CNN_models_export_shp.py – Inferencia por tiles con solape (promedio de probabilidades), visualización (RGB + predicción) y exportación a Shapefile con rasterio.features.shapes (filtro por min_area). Incluye mapeo de clases { "Permeable": 1, "No permeable": 0 }. 

🗂️ Organización de datos (rutas por defecto)

Se asume esta estructura relativa (ajustable en los scripts):

* INSUMOS/ORTOFOTOS/            # .tif ortofotos
* DATA/OUTPUT/MASCARAS/train/   # .tif máscaras de entrenamiento
* DATA/OUTPUT/MASCARAS/val/     # .tif máscaras de validación
* DATA/OUTPUT/MASCARAS/test/    # .tif máscaras de test
* DATA/OUTPUT/CNN/              # checkpoints, figuras y salidas

Los scripts usan PNOAImages y SidewalkLabels y los combinan con imgs & labels para construir el dataset geoespacial.

⚙️ Entorno

Para recrear el entorno de ejecución:

1. Instalar PyTorch acorde a la GPU (ej. CUDA 11.8 en RTX 3060):
2. Instalar paquetes:
     pip install torchgeo matplotlib numpy tqdm albumentations tensorboard torchmetrics segmentation-models-pytorch optuna hyperopt rasterio geopandas shapely
     Nota: SMP requiere timm. Si no se instaló como dependencia, añadir:
      pip install timm

📚 Documentaciones

* Segmentation Models PyTorch (SMP): https://smp.readthedocs.io/en/latest/

* TorchGeo: https://torchgeo.readthedocs.io/en/latest/

📒 TensorBoard

Al ejecutar los scripts de entrenamiento se escribirán logs en ./runs.
Acceder a TensorBoard a través de la terminal con:
  tensorboard --logdir runs 

Luego abrir http://localhost:6006. (Los scripts llaman a SummaryWriter y registran pérdidas, IoU y figuras de predicción de validación).

🚀 Uso rápido
1) Entrenamiento básico (configurable)
   python CNN.py

  * Configurar arquitectura (UNet, UNet++, FPN, DeepLabV3(+), etc.), backbone y loss desde el script.
  * Se aplican transforms (flip + normalize en train; normalize en val/test). 
  * Guardar el mejor checkpoint y registra métricas/figuras. 
2) Receta DeepLabV3+ + MiT-B4 (lista para ejecutar)
    python CNN_deeplabv3plus_mitb4_v3.py
    * Hiperparámetros establecidos (p. ej., EPOCHS, LR, BATCH_SIZE, GRAD_ACCUM_STEPS, PATIENCE, SCHED_FACTOR, LOSS=focal+dice o CE+Dice).
    * AMP + acumulación de gradientes para batch efectivo mayor y entrenamiento estable.
3) Búsqueda de hiperparámetros
* Optuna:
    python segmentation_optuna.py
    Explora lr, batch_size, arquitectura, backbone y pérdidas; escribe best.ckpt.
* Hyperopt:
    python segmentation_hyperopt.py
    Alternativa con HPO estilo TPE; mismo pipeline y logging.
  
4) Inferencia + Exportación a SHP
    python CNN_models_export_shp.py
    * Inferencia por tiles con stride/overlap y promedio de probabilidades (softmax).

    * Visualización (predicción + RGB) y exportación de polígonos a Shapefile (min_area en unidades del CRS).

🧪 Detalles útiles

* Normalización: usa medias/STD escaladas (0–1) calculadas sobre PNOA; se aplican en los transforms.

* Pérdidas: CE, Focal, Dice, Jaccard y combinaciones CE+Dice con pesos configurables; recomendable para binario con multiclass (2 clases) en SMP. 

* Checkpoints: utilidades para cargar/guardar (clave model_state_dict), fáciles de reusar en inferencia. 

* Samplers Geo: soporte para Grid/Random Geo Samplers (TorchGeo) si requieres muestreo espacial. 

* Clases y colores (visualización): paleta consistente en exportación y plots (Permeable = verde, No permeable = gris claro). 
