# GeoIA-Pasto

**Inteligencia Artificial Geoespacial para San Juan de Pasto, Colombia**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/geoai/blob/main/GeoIA-Pasto/notebooks/)

---

## Descripcion

Este modulo contiene ejemplos practicos de inteligencia artificial aplicada a datos geoespaciales, especificamente diseñados para la ciudad de **San Juan de Pasto, Nariño, Colombia**.

### Ubicacion Geografica

| Parametro | Valor |
|-----------|-------|
| **Ciudad** | San Juan de Pasto |
| **Departamento** | Nariño |
| **Pais** | Colombia |
| **Latitud** | 1.2136° N |
| **Longitud** | -77.2811° O |
| **Altitud** | ~2,527 m.s.n.m. |
| **Area Urbana** | ~1,181 km² |

---

## Notebooks Disponibles

Todos los notebooks estan diseñados para ejecutarse directamente en **Google Colab** sin necesidad de configuracion local.

| # | Notebook | Descripcion | Abrir en Colab |
|---|----------|-------------|----------------|
| 1 | [01_Introduccion_GeoIA_Pasto.ipynb](notebooks/01_Introduccion_GeoIA_Pasto.ipynb) | Introduccion y configuracion del entorno | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/geoai/blob/main/GeoIA-Pasto/notebooks/01_Introduccion_GeoIA_Pasto.ipynb) |
| 2 | [02_Descarga_Imagenes_Satelitales_Pasto.ipynb](notebooks/02_Descarga_Imagenes_Satelitales_Pasto.ipynb) | Descarga de imagenes Sentinel-2 y Landsat | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/geoai/blob/main/GeoIA-Pasto/notebooks/02_Descarga_Imagenes_Satelitales_Pasto.ipynb) |
| 3 | [03_Analisis_NDVI_Vegetacion_Pasto.ipynb](notebooks/03_Analisis_NDVI_Vegetacion_Pasto.ipynb) | Analisis de vegetacion con indices espectrales | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/geoai/blob/main/GeoIA-Pasto/notebooks/03_Analisis_NDVI_Vegetacion_Pasto.ipynb) |
| 4 | [04_Deteccion_Edificios_Pasto.ipynb](notebooks/04_Deteccion_Edificios_Pasto.ipynb) | Deteccion automatica de edificios con IA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/geoai/blob/main/GeoIA-Pasto/notebooks/04_Deteccion_Edificios_Pasto.ipynb) |
| 5 | [05_Segmentacion_Uso_Suelo_Pasto.ipynb](notebooks/05_Segmentacion_Uso_Suelo_Pasto.ipynb) | Clasificacion de cobertura y uso del suelo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/geoai/blob/main/GeoIA-Pasto/notebooks/05_Segmentacion_Uso_Suelo_Pasto.ipynb) |
| 6 | [06_Deteccion_Cambios_Urbanos_Pasto.ipynb](notebooks/06_Deteccion_Cambios_Urbanos_Pasto.ipynb) | Deteccion de cambios multi-temporales | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/geoai/blob/main/GeoIA-Pasto/notebooks/06_Deteccion_Cambios_Urbanos_Pasto.ipynb) |
| 7 | [07_Visualizacion_Interactiva_Pasto.ipynb](notebooks/07_Visualizacion_Interactiva_Pasto.ipynb) | Mapas interactivos y visualizacion de datos | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/geoai/blob/main/GeoIA-Pasto/notebooks/07_Visualizacion_Interactiva_Pasto.ipynb) |
| 8 | [08_Agentes_IA_Geoespacial_Pasto.ipynb](notebooks/08_Agentes_IA_Geoespacial_Pasto.ipynb) | Agentes IA para analisis geoespacial automatizado | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/geoai/blob/main/GeoIA-Pasto/notebooks/08_Agentes_IA_Geoespacial_Pasto.ipynb) |

---

## Requisitos

### Opcion 1: Google Colab (Recomendado)

Simplemente haz clic en el boton "Open in Colab" de cualquier notebook. Las dependencias se instalaran automaticamente.

### Opcion 2: Instalacion Local

```bash
# Clonar el repositorio
git clone https://github.com/opengeos/geoai.git
cd geoai

# Instalar dependencias
pip install geoai-py

# O instalar con todas las dependencias opcionales
pip install geoai-py[extra,agents]
```

---

## Datos Utilizados

Los notebooks utilizan las siguientes fuentes de datos:

| Fuente | Descripcion | Resolucion |
|--------|-------------|------------|
| **Sentinel-2** | Imagenes multiespectrales de la ESA | 10-60m |
| **Landsat 8/9** | Imagenes multiespectrales de NASA/USGS | 30m |
| **Overture Maps** | Datos vectoriales de edificios y calles | Vectorial |
| **OpenStreetMap** | Datos geograficos abiertos | Vectorial |
| **Google Open Buildings** | Huellas de edificios detectadas con IA | Vectorial |

---

## Estructura del Proyecto

```
GeoIA-Pasto/
├── README.md                    # Este archivo
├── notebooks/                   # Jupyter notebooks
│   ├── 01_Introduccion_GeoIA_Pasto.ipynb
│   ├── 02_Descarga_Imagenes_Satelitales_Pasto.ipynb
│   ├── 03_Analisis_NDVI_Vegetacion_Pasto.ipynb
│   ├── 04_Deteccion_Edificios_Pasto.ipynb
│   ├── 05_Segmentacion_Uso_Suelo_Pasto.ipynb
│   ├── 06_Deteccion_Cambios_Urbanos_Pasto.ipynb
│   ├── 07_Visualizacion_Interactiva_Pasto.ipynb
│   └── 08_Agentes_IA_Geoespacial_Pasto.ipynb
├── data/                        # Datos de ejemplo
└── assets/                      # Imagenes y recursos
```

---

## Coordenadas de Referencia

Para facilitar el uso de los notebooks, aqui estan las coordenadas de areas de interes en Pasto:

```python
# Centro de Pasto
PASTO_CENTER = {
    "lat": 1.2136,
    "lon": -77.2811
}

# Bounding Box del area urbana de Pasto
PASTO_BBOX = {
    "min_lon": -77.35,
    "min_lat": 1.15,
    "max_lon": -77.22,
    "max_lat": 1.28
}

# Puntos de interes
LUGARES_INTERES = {
    "Plaza_de_Narino": (1.2136, -77.2811),
    "Universidad_de_Narino": (1.2175, -77.2783),
    "Volcan_Galeras": (1.2208, -77.3592),
    "Laguna_de_la_Cocha": (1.1167, -77.1500),
    "Aeropuerto_Antonio_Narino": (1.3961, -77.2914),
    "Centro_Historico": (1.2140, -77.2815),
    "Parque_Bolivar": (1.2130, -77.2820)
}
```

---

## Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/NuevoEjemplo`)
3. Commit tus cambios (`git commit -m 'Agregar nuevo ejemplo para Pasto'`)
4. Push a la rama (`git push origin feature/NuevoEjemplo`)
5. Abre un Pull Request

---

## Licencia

Este proyecto esta bajo la Licencia MIT. Ver el archivo [LICENSE](../LICENSE) para mas detalles.

---

## Creditos

- **GeoAI**: [https://opengeoai.org](https://opengeoai.org)
- **Autor Original**: Qiusheng Wu
- **Adaptacion para Pasto**: Contribucion comunitaria

---

## Contacto

Para preguntas o sugerencias sobre los ejemplos de Pasto, por favor abre un issue en el repositorio.
