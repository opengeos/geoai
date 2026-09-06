# GeoAI: Artificial Intelligence for Geospatial Data

[![image](https://img.shields.io/pypi/v/geoai-py.svg)](https://pypi.python.org/pypi/geoai-py)
[![image](https://static.pepy.tech/badge/geoai-py)](https://pepy.tech/project/geoai-py)
[![image](https://img.shields.io/conda/vn/conda-forge/geoai.svg)](https://anaconda.org/conda-forge/geoai)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/geoai.svg)](https://anaconda.org/conda-forge/geoai)
[![Conda Recipe](https://img.shields.io/badge/recipe-geoai-green.svg)](https://github.com/conda-forge/geoai-py-feedstock)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![image](https://img.shields.io/badge/YouTube-Tutorials-red)](https://www.youtube.com/playlist?list=PLAxJ4-o7ZoPcvENqwaPa_QwbbkZ5sctZE)
[![QGIS](https://img.shields.io/badge/QGIS-plugin-orange.svg)](https://opengeoai.org/qgis_plugin)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.09605/status.svg)](https://doi.org/10.21105/joss.09605)

[![logo](https://raw.githubusercontent.com/opengeos/geoai/master/docs/assets/logo_rect.png)](https://github.com/opengeos/geoai/blob/master/docs/assets/logo.png)

**A powerful Python package for integrating artificial intelligence with geospatial data analysis and visualization**

## 📖 Introduction

[GeoAI](https://opengeoai.org) is a comprehensive Python package designed to bridge artificial intelligence (AI) and geospatial data analysis, providing researchers and practitioners with intuitive tools for applying machine learning techniques to geographic data. The package offers a unified framework for processing satellite imagery, aerial photographs, and vector data using state-of-the-art deep learning models. GeoAI integrates popular AI frameworks including [PyTorch](https://pytorch.org), [Transformers](https://github.com/huggingface/transformers), [PyTorch Segmentation Models](https://github.com/qubvel-org/segmentation_models.pytorch), and specialized geospatial libraries like [torchange](https://github.com/Z-Zheng/pytorch-change-models), enabling users to perform complex geospatial analyses with minimal code.

The package provides six core capabilities:

1. Interactive and programmatic search and download of remote sensing imagery and geospatial data.
2. Automated dataset preparation with image chips and label generation.
3. Model training for tasks such as classification, detection, and segmentation.
4. Inference pipelines for applying models to new geospatial datasets.
5. Interactive visualization through integration with [Leafmap](https://github.com/opengeos/leafmap/) and [MapLibre](https://github.com/eoda-dev/py-maplibregl).
6. Seamless QGIS integration via a dedicated GeoAI plugin, enabling users to run AI-powered geospatial workflows directly within the QGIS desktop environment, without writing code.

GeoAI addresses the growing demand for accessible AI tools in geospatial research by providing high-level APIs that abstract complex machine learning workflows while maintaining flexibility for advanced users. The package supports multiple data formats (GeoTIFF, JPEG2000, GeoJSON, Shapefile, GeoPackage) and includes automatic device management for GPU acceleration when available. With over 10 modules and extensive notebook examples, GeoAI serves as both a research tool and educational resource for the geospatial AI community.

## 📚 Book

A comprehensive book on GeoAI is available at [https://book.opengeoai.org](https://book.opengeoai.org).

![](https://books.gishub.org/geoai/front-cover.webp)

## 📝 Statement of Need

The integration of artificial intelligence with geospatial data analysis has become increasingly critical across numerous scientific disciplines, from environmental monitoring and urban planning to disaster response and climate research. However, applying AI techniques to geospatial data presents unique challenges including data preprocessing complexities, specialized model architectures, and the need for domain-specific knowledge in both machine learning and geographic information systems.

Existing solutions often require researchers to navigate fragmented ecosystems of tools, combining general-purpose machine learning libraries with specialized geospatial packages, leading to steep learning curves and reproducibility challenges. While packages like [TorchGeo](https://github.com/torchgeo/torchgeo), [TerraTorch](https://github.com/terrastackai/terratorch), and [SRAI](https://github.com/kraina-ai/srai) provide excellent foundational tools for geospatial deep learning, there remains a gap for comprehensive, high-level interfaces that can democratize access to advanced AI techniques for the broader geospatial community.

GeoAI addresses this need by providing a unified, user-friendly interface that abstracts the complexity of integrating multiple AI frameworks with geospatial data processing workflows. It lowers barriers for: (1) geospatial researchers who need accessible AI workflows without deep ML expertise; (2) AI practitioners who want streamlined geospatial preprocessing and domain-specific datasets; and (3) educators seeking reproducible examples and teaching-ready workflows.

The package's design philosophy emphasizes simplicity without sacrificing functionality, enabling users to perform sophisticated analyses such as building footprint extraction from satellite imagery, land cover classification, and change detection with just a few lines of code. By integrating cutting-edge AI models and providing seamless access to major geospatial data sources, GeoAI significantly lowers the barrier to entry for geospatial AI applications while maintaining the flexibility needed for advanced research applications.

## Citations

If you find GeoAI useful in your research, please consider citing the following paper to support my work. Thank you for your support.

-   Wu, Q., (2026). GeoAI: A Python package for integrating artificial intelligence with geospatial data analysis and visualization. _Journal of Open Source Software_, 11(118), 9605, <https://doi.org/10.21105/joss.09605>.
-   Wu, Q. (2026). *GeoAI with Python: A Practical Guide to Open-Source Geospatial AI*. Independently published. PDF edition ISBN 979-8993859729; Print edition ISBN 979-8253507414. Available at [https://book.opengeoai.org](https://book.opengeoai.org).

## 🚀 Key Features

### 📊 Advanced Geospatial Data Visualization

-   Interactive multi-layer visualization of vector and raster data stored locally or in cloud storage
-   Customizable styling and symbology
-   Time-series data visualization capabilities

### 🛠️ Data Preparation & Processing

-   Streamlined access to satellite and aerial imagery from providers like Sentinel, Landsat, NAIP, and other open datasets
-   Tools for downloading, mosaicking, and preprocessing remote sensing data
-   Automated generation of training datasets with image chips and corresponding labels
-   Vector-to-raster and raster-to-vector conversion utilities optimized for AI workflows
-   Data augmentation techniques specific to geospatial data
-   Support for integrating Overture Maps data and other open datasets for training and validation

### 🖼️ Image Segmentation

-   Integration with [PyTorch Segmentation Models](https://github.com/qubvel-org/segmentation_models.pytorch) for automatic feature extraction
-   Specialized segmentation algorithms optimized for satellite and aerial imagery
-   Streamlined workflows for segmenting buildings, water bodies, wetlands, solar panels, etc.
-   Export capabilities to standard geospatial formats (GeoJSON, Shapefile, GeoPackage, GeoParquet)

### 🔍 Image Classification

-   Pre-trained models for land cover and land use classification
-   Transfer learning utilities for fine-tuning models with your own data
-   Multi-temporal classification support for change detection
-   Accuracy assessment and validation tools

### 🛰️ Remote Sensing Foundation Models

-   Curated catalog of 20 remote sensing foundation models via `geoai.list_foundation_models()`
-   Filter by category (vision, vision-language, generative), modality (multispectral, SAR, hyperspectral, multimodal), or downstream task
-   Retrieve full metadata (paper, code, HuggingFace ID, license) with `geoai.get_foundation_model_info()`
-   Load TerraTorch-supported model backbones directly with `geoai.load_foundation_model()`
-   Drawn from the [Awesome Remote Sensing Foundation Models](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models) list

### 🌍 Additional Capabilities

-   Change detection with AI-enhanced feature extraction
-   Object detection in aerial and satellite imagery
-   Georeferencing utilities for AI model outputs

## 📦 Installation

### Using pip

```bash
pip install geoai-py
```

### Using conda

```bash
conda install -c conda-forge geoai
```

### Using mamba

```bash
mamba install -c conda-forge geoai
```

## ⚙️ QGIS Plugin

Check out the [QGIS Plugin](https://opengeoai.org/qgis_plugin/) page if you are interested in using GeoAI with QGIS.

[![demo](https://github.com/user-attachments/assets/5aabc3d3-efd1-4011-ab31-2b3f11aab3ed)](https://youtu.be/8-OhlqeoyiY)

## 📋 Documentation

Comprehensive documentation is available at [https://opengeoai.org](https://opengeoai.org), including:

-   Detailed API reference
-   Tutorials and example notebooks
-   Contributing guide

## 📺 Video Tutorials

### GeoAI Made Easy: Learn the Python Package Step-by-Step (Beginner Friendly)

[![intro](https://github.com/user-attachments/assets/7e60ce05-573d-4d0d-9876-5289b87e5136)](https://youtu.be/VIl29Rca6zE&list=PLAxJ4-o7ZoPcvENqwaPa_QwbbkZ5sctZE)

### GeoAI Workshop: Unlocking the Power of GeoAI with Python

[![cover](https://github.com/user-attachments/assets/1c14e651-65b9-41ae-b42d-3ad028b3eeb8)](https://youtu.be/jdK-cleFUkc&list=PLAxJ4-o7ZoPcvENqwaPa_QwbbkZ5sctZE)

### GeoAI Tutorials Playlist

[![cover](https://github.com/user-attachments/assets/3cde9547-ab62-4d70-b23a-3e5ed27c7407)](https://www.youtube.com/playlist?list=PLAxJ4-o7ZoPcvENqwaPa_QwbbkZ5sctZE)

## 🤝 Contributing

We welcome contributions of all kinds! See our [contributing guide](https://opengeoai.org/contributing) for ways to get started.

## 📄 License

GeoAI is free and open source software, licensed under the MIT License.

## Acknowledgments

We gratefully acknowledge the support of the following organizations:

-   [NASA](https://www.nasa.gov): This research is partially supported by the National Aeronautics and Space Administration (NASA) through Grant No. 80NSSC22K1742, awarded under the [Open Source Tools, Frameworks, and Libraries Program](https://bit.ly/3RVBRcQ).
-   [AmericaView](https://americaview.org): This work is also partially supported by the U.S. Geological Survey through Grant/Cooperative Agreement No. G23AP00683 (GY23-GY27) in collaboration with AmericaView.


## 🌐 Web Resources & Interactive Index
- [BIG BLOCK BLAST](https://studyplayings.web.app/big-block-blast.html)
- [RAGDOLL SOCCER 2 PLAYERS](https://themindzone.pages.dev/ragdoll-soccer-2-players.html)
- [SCARY PAIRS](https://thelearnquesters.pages.dev/scary-pairs.html)
- [GTA GRAND VEGAS CRIME](https://themindplay.pages.dev/gta-grand-vegas-crime.html)
- [FURY OF THE STEAMPUNK PRINCESS](https://learnquesters.pages.dev/fury-of-the-steampunk-princess.html)
- [WATER SORT](https://learnquester.pages.dev/water-sort.html)
- [GOODS TRIPLE MATCH 3D](https://themindplays.pages.dev/goods-triple-match-3d.html)
- [CATEGORY SIMULATION 2](https://learnquesters.pages.dev/category-simulation-2.html)
- [CATEGORY SPACE57](https://learnquester.pages.dev/category-space57.html)
- [BALLISTIC BREAKTHROUGH](https://learnquesters.pages.dev/ballistic-breakthrough.html)
- [CATEGORY DRAGON](https://learnquesters.pages.dev/category-dragon.html)
- [LABUBU COLORING ADVENTURE](https://iskillquest.pages.dev/labubu-coloring-adventure.html)
- [MY PERFECT YEAR PLANNER](https://learnquesters.pages.dev/my-perfect-year-planner.html)
- [OBBY HALLOWEEN DANGER SKATE](https://iskillquest.pages.dev/obby-halloween-danger-skate.html)
- [STICKMAN LEAVE PRISON](https://theskillquest.pages.dev/stickman-leave-prison.html)
- [SCREW SPIN](https://learnquester.pages.dev/screw-spin.html)
- [POOL MERGE](https://themindplaying.web.app/pool-merge.html)
- [CATEGORY DRESS UP97](https://learnquesters.pages.dev/category-dress-up97.html)
- [AVATAR WORLD SECRETS](https://themindplay.pages.dev/avatar-world-secrets.html)
- [FURRY WEDDING PROPOSAL](https://themindplay.pages.dev/furry-wedding-proposal.html)
- [ARROWS PUZZLE ESCAPE](https://themindzone.pages.dev/arrows-puzzle-escape.html)
- [MAGECLASH IO](https://learnquester.pages.dev/mageclash-io.html)
- [CATEGORY HERO72](https://learnquesters.pages.dev/category-hero72.html)
- [SUPERMARKET SORT GROCERY GAME](https://themindplay.pages.dev/supermarket-sort-grocery-game.html)
- [CATEGORY CAN T STOP PLAYING215](https://learnquester.pages.dev/category-can-t-stop-playing215.html)
- [ORGANIZE IT](https://themindplay.pages.dev/organize-it.html)
- [TIDY MASTER SATISFEEL ASMR](https://themindskillplayplay.pages.dev/tidy-master-satisfeel-asmr.html)
- [CONNECT EM ALL](https://themindplay.pages.dev/connect-em-all.html)
- [MERGE BALLS NEW YEARS TOYS IN 3D](https://themindplay.pages.dev/merge-balls-new-years-toys-in-3d.html)
- [SPACE SHIFT](https://learnquesters.pages.dev/space-shift.html)
- [NINJA CROSSWORD CHALLENGE](https://themindplaying.web.app/ninja-crossword-challenge.html)
- [COLOR NONOGRAM PUZZLE 2](https://learnquester.pages.dev/color-nonogram-puzzle-2.html)
- [UNDERWATER SURVIVAL DEEP DIVE](https://theskillquest.pages.dev/underwater-survival-deep-dive.html)
- [CATEGORY DRESS UP 2](https://learnquesters.pages.dev/category-dress-up-2.html)
- [PANDA SHOP SIMULATOR](https://learnquester.pages.dev/panda-shop-simulator.html)
- [MERGE FUSION](https://themindplaying.web.app/merge-fusion.html)
- [THE ROMAN EMPIRE COLOSSEUM](https://learnquester.pages.dev/the-roman-empire-colosseum.html)
- [OVERPROTECTIVE BOYFRIEND](https://themindplaying.web.app/overprotective-boyfriend.html)
- [SKILLFUL FINGER](https://themindzone.pages.dev/skillful-finger.html)
- [RACING ISLAND](https://theskillquest.pages.dev/racing-island.html)
- [RIFT OF HELL DEMONS WAR](https://learnquesters.pages.dev/rift-of-hell-demons-war.html)
- [ANGRY PLANTS FLOWER](https://themindzone.pages.dev/angry-plants-flower.html)
- [HIDDEN OBJECT GIRL AND CAT](https://themindplay.pages.dev/hidden-object-girl-and-cat.html)
- [BFF HAPPY SPRING](https://learnquesters.pages.dev/bff-happy-spring.html)
- [DRIVE RACE CRASH](https://learnquesters.pages.dev/drive-race-crash.html)
- [SPRUNKI CLICKER MERGE PHASE 3](https://themindplaying.web.app/sprunki-clicker-merge-phase-3.html)
- [HAPPY ASMR CARE](https://iskillquest.pages.dev/happy-asmr-care.html)
- [THE SORT AGENCY](https://learnquesters.pages.dev/the-sort-agency.html)
- [MR DISC SLINGSHOT STRIKE](https://themindplay.pages.dev/mr-disc-slingshot-strike.html)
- [BLACK PINK CHRISTMAS CONCERT](https://themindplay.pages.dev/black-pink-christmas-concert.html)
- [POPTROPICA](https://theskillquest.pages.dev/poptropica.html)
- [STICKMAN WARRIOR WAY](https://themindplaying.web.app/stickman-warrior-way.html)
- [POPSORTICA](https://iskillquest.pages.dev/popsortica.html)
- [NOOB IN GEOMETRY DASH](https://learnquester.pages.dev/noob-in-geometry-dash.html)
- [2048 MAYHEMIO](https://themindplaying.web.app/2048-mayhemio.html)
- [RED STICKMAN VS CRAFTMANS](https://themindplay.pages.dev/red-stickman-vs-craftmans.html)
- [CATEGORY ESCAPE187](https://learnquesters.pages.dev/category-escape187.html)
- [CATEGORY MONSTER206](https://learnquester.pages.dev/category-monster206.html)
- [HOSPITAL GAME HAPPY CLINIC](https://learnquesters.pages.dev/hospital-game-happy-clinic.html)
- [STAND ON THE RIGHT COLOR ROBBY](https://learnquesters.pages.dev/stand-on-the-right-color-robby.html)
- [MOJICON FRUIT CONNECT](https://learnquesters.pages.dev/mojicon-fruit-connect.html)
- [CATEGORY FPS 2](https://learnquesters.pages.dev/category-fps-2.html)
- [VENETIAN LOVE AFFAIR](https://iskillquest.pages.dev/venetian-love-affair.html)
- [WINTER WOLF](https://iskillquest.pages.dev/winter-wolf.html)
- [CRYPTO GALS TIKTOK FASHION](https://learnquester.pages.dev/crypto-gals-tiktok-fashion.html)
- [ARROW ESCAPE MASTER](https://learnquester.pages.dev/arrow-escape-master.html)
- [CATEGORY MAKEUP](https://learnquesters.pages.dev/category-makeup.html)
- [CATEGORY ROGUELIKE38](https://iskillquest.pages.dev/category-roguelike38.html)
- [NUMBER MASTER](https://learnquesters.pages.dev/number-master.html)
- [CATEGORY ADVENTURE](https://learnquesters.pages.dev/category-adventure.html)
- [SIBERIAN ASSAULT](https://themindplay.pages.dev/siberian-assault.html)
- [CATEGORY ANIMAL](https://theskillquest.pages.dev/category-animal.html)
- [VEX X3M 2](https://theskillquest.pages.dev/vex-x3m-2.html)
- [ARCHERY RAGDOLL](https://themindzone.pages.dev/archery-ragdoll.html)
- [INDEX13](https://themindplays.pages.dev/index13.html)
- [BLOCK CUT CLEANER](https://themindzone.pages.dev/block-cut-cleaner.html)
- [TONY ARCHER](https://theskillquest.pages.dev/tony-archer.html)
- [KINGDOM WARS TD](https://learnquester.pages.dev/kingdom-wars-td.html)
- [BUBBLE SHOOTER HAWAII](https://learnquesters.pages.dev/bubble-shooter-hawaii.html)
- [INDEX8](https://learnquester.pages.dev/index8.html)
- [INDEX6](https://themindplaying.web.app/index6.html)
- [MAHJONG CONNECT COOKWARE](https://learnquesters.pages.dev/mahjong-connect-cookware.html)
- [CATEGORY SHOOTER 2](https://learnquester.pages.dev/category-shooter-2.html)
- [CATEGORY ARENA255](https://learnquester.pages.dev/category-arena255.html)
- [MY COTTAGECORE AESTHETIC LOOK](https://learnquesters.pages.dev/my-cottagecore-aesthetic-look.html)
- [NORTHERN LIGHTS THE SECRET OF THE FOREST](https://iskillquest.pages.dev/northern-lights-the-secret-of-the-forest.html)
- [UNSTACK TOWER](https://learnquester.pages.dev/unstack-tower.html)
- [MUSTANG CITY DRIVER](https://themindplay.pages.dev/mustang-city-driver.html)
- [CATEGORY GROW99](https://learnquesters.pages.dev/category-grow99.html)
- [WINTER COSMOFEST](https://learnquesters.pages.dev/winter-cosmofest.html)
- [CATEGORY FASHION105](https://learnquesters.pages.dev/category-fashion105.html)
- [HYPERMARKET 3D STORE CASHIER](https://learnquester.pages.dev/hypermarket-3d-store-cashier.html)
- [VOLLEY BEAN](https://learnquesters.pages.dev/volley-bean.html)
- [MOJICON FRUIT CONNECT](https://themindzone.pages.dev/mojicon-fruit-connect.html)
- [SWIM GOOD](https://theskillquest.pages.dev/swim-good.html)
- [BLUE MUSHROOM CAT RUN](https://themindplay.pages.dev/blue-mushroom-cat-run.html)
- [LAST WAR SURVIVAL](https://iskillquest.pages.dev/last-war-survival.html)
- [DOGGI](https://themindzone.pages.dev/doggi.html)
- [CATEGORY DRESS UP](https://learnquesters.pages.dev/category-dress-up.html)
- [TRIPEAKS SOLITAIRE ESCAPES](https://learnquester.pages.dev/tripeaks-solitaire-escapes.html)
- [SOLITAIRE DELUXE EDITION](https://themindzone.pages.dev/solitaire-deluxe-edition.html)
- [INDEX8](https://theskillquest.pages.dev/index8.html)
- [INDEX6](https://themindplays.pages.dev/index6.html)
- [WOODS OF NEVIA FOREST SURVIVAL](https://theskillquest.pages.dev/woods-of-nevia-forest-survival.html)
- [ARROW LEGEND](https://iskillquest.pages.dev/arrow-legend.html)
- [ARROW PUZZLE](https://learnquester.pages.dev/arrow-puzzle.html)
- [SECRET ROOMS](https://themindplays.pages.dev/secret-rooms.html)
- [MERGE BALLS SHOOTER 2048 CONNECT FRUITS](https://themindplaying.web.app/merge-balls-shooter-2048-connect-fruits.html)
- [COLORWARSIO CONQUEST GAME](https://themindplays.pages.dev/colorwarsio-conquest-game.html)
- [APOCALYPSE SHELTER](https://iskillquest.pages.dev/apocalypse-shelter.html)
- [ANTS EMPIRE EVOLVE SIM](https://thelearnquesters.pages.dev/ants-empire-evolve-sim.html)
- [STICKMAN DOORS AND ISLAND](https://themindzone.pages.dev/stickman-doors-and-island.html)
- [SHEEP VS WOLF](https://iskillquest.pages.dev/sheep-vs-wolf.html)
- [CATEGORY BASKETBALL](https://iskillquest.pages.dev/category-basketball.html)
- [KNOCKOUT DUDES](https://iskillquest.pages.dev/knockout-dudes.html)
- [WOODLAND SLIDE](https://thelearnquesters.pages.dev/woodland-slide.html)
- [PRIVACY](https://learnquesters.pages.dev/privacy.html)
- [TILE GURU](https://learnquester.pages.dev/tile-guru.html)
- [K WEDDING DREAM](https://learnquesters.pages.dev/k-wedding-dream.html)
- [BUTTERFLY KYODAI DELUXE 2](https://iskillquest.pages.dev/butterfly-kyodai-deluxe-2.html)
- [CONTACT](https://skillplay.github.io/contact.html)
- [TEARDOWN DESTRUCTION SANDBOX](https://thelearnquesters.pages.dev/teardown-destruction-sandbox.html)
- [INDEX16](https://theskillquest.pages.dev/index16.html)
- [CATEGORY GROW](https://thelearnquesters.pages.dev/category-grow.html)
- [ITALIAN BRAINROT QUIZ](https://iskillquest.pages.dev/italian-brainrot-quiz.html)
- [ARROWTIX TRAIN YOUR BRAIN](https://learnquester.pages.dev/arrowtix-train-your-brain.html)
- [CATEGORY MEDIEVAL15](https://thelearnquesters.pages.dev/category-medieval15.html)
- [CATEGORY PUZZLE 5](https://themindplaying.web.app/category-puzzle-5.html)
- [EMPIRE CITY](https://thelearnquesters.pages.dev/empire-city.html)
- [BURGER EMPIRE](https://iskillquest.pages.dev/burger-empire.html)
