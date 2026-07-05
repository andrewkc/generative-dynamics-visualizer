# Lab 3: Generative Dynamics Visualizer

El visualizador fue desarrollado como parte del **Laboratorio 3: Visualización de dinámicas generativas en 2D**, implementando todos los algoritmos principales desde cero, sin utilizar librerías especializadas para modelos generativos.

El objetivo principal es comparar diferentes dinámicas generativas mediante visualizaciones animadas, permitiendo estudiar tanto los procesos forward como los procesos generativos inversos utilizados por los modelos modernos de generación continua.

---

# Características

## Diffusion Models

Implementaciones:

- Variance Preserving (VP)
- Variance Exploding (VE)
- sub-VP
- Diffusion Model con **ε-prediction**
- Diffusion Model con **v-prediction**
- Reverse-time SDE
- Probability Flow ODE

---

## Flow Matching

Se implementa completamente:

- Flow Matching
- Interpolación lineal
- Aprendizaje del **velocity field**
- Generación mediante la ODE aprendida

---

## Integradores Numéricos

El laboratorio incluye implementaciones propias de:

- Euler
- Euler–Maruyama
- Heun

Los integradores pueden intercambiarse fácilmente dependiendo del algoritmo generativo utilizado.

---

## Visualizaciones

El laboratorio puede generar automáticamente las siguientes animaciones:

- Comparación de procesos forward (VP, VE y sub-VP)
- Evolución de la densidad
- Trayectorias forward
- Campo de score
- Reverse-time SDE
- Probability Flow ODE
- Flow Matching
- Comparación del número de pasos de integración

Todas las animaciones pueden exportarse como:

- MP4
- GIF
- PNG

---

# Estructura del laboratorio

```text
generative-dynamics-visualizer/

│
├── configs/
├── datasets/
├── diffusion/
├── flow_matching/
├── models/
├── samplers/
├── trainers/
├── visualization/
├── utils/
│
├── checkpoints/
├── outputs/
│
├── train.py
├── generate.py
├── animate.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Arquitectura General

El laboratorio está dividido en varios módulos independientes.

## datasets/

Contiene la implementación de todas las distribuciones sintéticas utilizadas durante el entrenamiento y las visualizaciones.

Actualmente se incluyen:

- Two Moons
- Eight Gaussians
- Checkerboard
- Spirals
- Pinwheel
- Rings
- Gaussian Mixtures

---

## models/

Contiene las redes neuronales utilizadas por los modelos generativos.

Actualmente implementa:

- DiffusionModel
- FlowModel
- Backbone MLP
- Sinusoidal Time Embeddings

---

## diffusion/

Implementa toda la teoría relacionada con **Score-Based Diffusion Models**.

Incluye:

- VP SDE
- VE SDE
- sub-VP SDE
- ScoreModel
- Funciones de pérdida
- Procesos forward
- Probability Flow ODE

---

## flow_matching/

Implementa todos los componentes necesarios para **Flow Matching**.

Incluye:

- Interpolación lineal
- Velocity Targets
- Sampling de trayectorias
- Funciones de pérdida

---

## samplers/

Implementa los algoritmos de integración numérica utilizados durante la generación.

Actualmente incluye:

- ReverseSDESampler
- ProbabilityFlowSampler
- FlowODESampler

junto con los integradores:

- Euler
- Euler–Maruyama
- Heun

---

## trainers/

Implementa el entrenamiento de los distintos modelos.

Actualmente se encuentran disponibles:

- DiffusionTrainer
- FlowTrainer

---

## visualization/

Contiene todas las herramientas necesarias para visualizar las dinámicas aprendidas.

Incluye:

- scatter plots
- density plots
- vector fields
- trayectorias
- animaciones

---

## utils/

Contiene utilidades generales del laboratorio.

Por ejemplo:

- manejo de checkpoints
- configuración
- manejo de dispositivos
- semillas aleatorias
- operaciones sobre tensores

---

# Instalación

Clonar el repositorio

```bash
git clone https://github.com/andrewkc/generative-dynamics-visualizer.git

cd generative-dynamics-visualizer
```

Crear un entorno virtual

```bash
conda create -n genviz python=3.10

conda activate genviz
```

Instalar las dependencias

```bash
pip install -r requirements.txt
```

Verificar la instalación

```bash
python train.py --help
```

Si el menú de ayuda aparece correctamente, la instalación fue realizada con éxito.

# Fundamentos Matemáticos

El laboratorio implementa dos de las familias más importantes de modelos generativos continuos: **Diffusion Models** y **Flow Matching**.

Aunque ambos modelos persiguen el mismo objetivo (transformar una distribución simple en una distribución compleja), utilizan formulaciones matemáticas distintas.

---

## Diffusion Models

Los **Diffusion Models** aprenden el proceso inverso de un proceso de difusión que transforma gradualmente una distribución de datos en ruido gaussiano.

Durante el entrenamiento se utiliza un proceso **forward** conocido y una red neuronal aprende el **score** (o una parametrización equivalente) para invertir dicho proceso.

En este laboratorio se implementan tres tipos de procesos forward:

- Variance Preserving (VP)
- Variance Exploding (VE)
- sub-Variance Preserving (sub-VP)

Además, el visualizador soporta dos parametrizaciones del modelo:

- **ε-prediction**
- **v-prediction**

Una vez entrenado el modelo, es posible generar muestras mediante:

- Reverse-time SDE
- Probability Flow ODE

---

## Flow Matching

**Flow Matching** representa una alternativa reciente a los modelos de difusión.

En lugar de aprender el score de la distribución, el modelo aprende directamente un **velocity field** que transforma una distribución base en la distribución objetivo mediante una ODE.

Durante este laboratorio se implementa:

- interpolación lineal
- construcción de pares de entrenamiento
- entrenamiento del velocity field
- generación mediante la ODE aprendida

---

## Integración Numérica

Todos los procesos continuos requieren resolver una ecuación diferencial.

El laboratorio implementa tres integradores numéricos:

| Integrador | Tipo |
|------------|------|
| Euler | ODE |
| Euler–Maruyama | SDE |
| Heun | ODE (segundo orden) |

Los integradores fueron implementados desde cero y pueden intercambiarse fácilmente entre los distintos samplers.

---

# Distribuciones Soportadas

El visualizador trabaja sobre distribuciones sintéticas bidimensionales.

Actualmente se implementan las siguientes:

| Distribución | Descripción |
|--------------|-------------|
| Two Moons | Dos medias lunas interconectadas. |
| Eight Gaussians | Ocho gaussianas distribuidas en un círculo. |
| Checkerboard | Distribución con regiones desconectadas. |
| Spirals | Espirales bidimensionales. |
| Rings | Anillos concéntricos. |
| Pinwheel | Distribución con geometría rotacional. |

Estas distribuciones permiten analizar el comportamiento de los modelos sobre diferentes geometrías, topologías y niveles de complejidad.

---

# Entrenamiento

Todo el entrenamiento del laboratorio se realiza mediante el programa:

```text
train.py
```

La configuración del entrenamiento se define mediante archivos YAML ubicados en:

```text
configs/
```

Por ejemplo:

```text
configs/

vp.yaml

ve.yaml

subvp.yaml

flow.yaml
```

---

## Entrenar un Diffusion Model

```bash
python train.py --config configs/vp.yaml
```

---

## Entrenar un modelo Flow Matching

```bash
python train.py --config configs/flow.yaml
```

---

Durante el entrenamiento se almacenan automáticamente:

- pesos del modelo
- configuración utilizada
- estado del optimizador
- época actual

Los checkpoints se guardan en:

```text
checkpoints/
```

Por ejemplo:

```text
checkpoints/

vp_epsilon.pt

ve_epsilon.pt

subvp_velocity.pt

flow.pt
```

---

# Configuración mediante YAML

Cada experimento se controla mediante un archivo YAML.

Ejemplo simplificado:

```yaml
dataset: two_moons

model: diffusion

prediction: epsilon

sde: vp

epochs: 500

batch_size: 256

learning_rate: 1e-4

hidden_dim: 256

time_embedding_dim: 64
```

Esto permite modificar completamente el entrenamiento sin cambiar el código fuente.

---

# Checkpoints

Cada checkpoint almacena:

- pesos de la red neuronal
- configuración completa del experimento
- estado del optimizador (opcional)
- época del entrenamiento (opcional)

Gracias a ello es posible reconstruir automáticamente un experimento completo para generar nuevas muestras o producir animaciones posteriormente.

---

# Organización del Flujo de Trabajo

El flujo completo del laboratorio puede resumirse mediante el siguiente diagrama:

```text
                 Dataset
                    │
                    ▼
               train.py
                    │
                    ▼
          Checkpoint (.pt)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   generate.py             animate.py
        │                       │
        ▼                       ▼
   Nuevas muestras      Videos / GIF / PNG
```

Esta estructura permite separar claramente las etapas de entrenamiento, generación y visualización.

# Resultados Esperados

Una vez entrenados los modelos y generadas las animaciones, el laboratorio permite observar visualmente el comportamiento de diferentes modelos generativos continuos sobre distribuciones sintéticas bidimensionales.

Entre los principales resultados se encuentran:

- Comparación entre los procesos forward VP, VE y sub-VP.
- Evolución temporal de la densidad durante la difusión.
- Trayectorias completas de las partículas.
- Campo de score aprendido por el Diffusion Model.
- Generación mediante Reverse-time SDE.
- Generación mediante Probability Flow ODE.
- Transporte aprendido mediante Flow Matching.
- Influencia del número de pasos de integración sobre la calidad de las muestras.

A continuación se muestran algunos ejemplos de resultados obtenidos.

---

## Ejemplos de Resultados

### Forward Comparison Processes

<p align="center">
<img src="outputs\animations\forward_comparison.gif" width="600">
</p>

---

**Scatter Plot**

<p align="center">
<img src="outputs\images\e10.png" width="700">
</p>

---

**Density**

<p align="center">
<img src="outputs\images\e11.png" width="700">
</p>

---

**Trajectories**

<p align="center">
<img src="outputs\images\e12.png" width="700">
</p>

---

**Vector Field**
<p align="center">
<img src="outputs\images\e13.png" width="700">
</p>


# Generación de Muestras

Una vez entrenado un modelo, es posible generar nuevas muestras utilizando el programa:

```text
generate.py
```

Este programa carga automáticamente un checkpoint previamente entrenado, reconstruye el modelo y ejecuta el sampler correspondiente.

---

## Ejemplo: Reverse-time SDE

```bash
python generate.py \
    --checkpoint checkpoints/vp_epsilon.pt \
    --samples 20000 \
    --steps 500
```

---

## Ejemplo: Probability Flow ODE

```bash
python generate.py \
    --checkpoint checkpoints/vp_epsilon.pt \
    --samples 20000 \
    --steps 500 \
    --sampler probability_flow
```

---

## Ejemplo: Flow Matching

```bash
python generate.py \
    --checkpoint checkpoints/flow.pt \
    --samples 20000 \
    --steps 500
```

---

Las muestras generadas se almacenan automáticamente en:

```text
outputs/

samples.pt
```

Las cuales pueden visualizarse posteriormente utilizando los módulos incluidos en `visualization/`.

---

# Generación de Animaciones

Todas las animaciones del laboratorio se generan mediante:

```text
animate.py
```

El programa permite reconstruir automáticamente el modelo entrenado y generar cualquiera de las visualizaciones solicitadas.

---

## Comparación de procesos forward

```bash
python animate.py \
    --animation forward \
    --dataset two_moons \
    --samples 5000 \
    --steps 250
```

---

## Evolución de la densidad

```bash
python animate.py \
    --animation density \
    --dataset checkerboard \
    --samples 5000 \
    --steps 250
```

---

## Trayectorias forward

```bash
python animate.py \
    --animation forward_trajectories \
    --dataset two_moons \
    --samples 5000 \
    --steps 250
```

---

## Reverse-time SDE

```bash
python animate.py \
    --animation reverse_sde \
    --checkpoint checkpoints/vp_epsilon.pt \
    --samples 5000 \
    --steps 250
```

---

## Probability Flow ODE

```bash
python animate.py \
    --animation probability_flow \
    --checkpoint checkpoints/vp_epsilon.pt \
    --samples 5000 \
    --steps 250
```

---

## Flow Matching

```bash
python animate.py \
    --animation flow_matching \
    --checkpoint checkpoints/flow.pt \
    --samples 5000 \
    --steps 250
```

---

## Campo de Score

```bash
python animate.py \
    --animation score_field \
    --checkpoint checkpoints/vp_epsilon.pt \
    --steps 250
```

---

## Comparación del número de pasos

Reverse-time SDE

```bash
python animate.py \
    --animation steps_comparison \
    --sampler reverse_sde \
    --checkpoint checkpoints/vp_epsilon.pt
```

---

Probability Flow ODE

```bash
python animate.py \
    --animation steps_comparison \
    --sampler probability_flow \
    --checkpoint checkpoints/vp_epsilon.pt
```

---

Flow Matching

```bash
python animate.py \
    --animation steps_comparison \
    --sampler flow \
    --checkpoint checkpoints/flow.pt
```

---

# Archivos Generados

Las animaciones se almacenan automáticamente en

```text
outputs/

animations/
```

Cada ejecución genera:

```text
animation.mp4

animation.gif

animation.png
```

dependiendo del tipo de visualización seleccionada.

---

# Parámetros Disponibles

## train.py

| Parámetro | Descripción |
|-----------|-------------|
| `--config` | Archivo YAML del experimento. |

---

## generate.py

| Parámetro | Descripción |
|-----------|-------------|
| `--checkpoint` | Checkpoint previamente entrenado. |
| `--samples` | Número de muestras a generar. |
| `--steps` | Número de pasos de integración. |
| `--sampler` | Sampler utilizado durante la generación. |

---

## animate.py

| Parámetro | Descripción |
|-----------|-------------|
| `--animation` | Tipo de animación. |
| `--checkpoint` | Modelo entrenado. |
| `--dataset` | Distribución sintética utilizada. |
| `--samples` | Número de partículas. |
| `--steps` | Número de pasos temporales. |
| `--fps` | Frames por segundo del video. |
| `--output_dir` | Carpeta de salida. |
| `--sampler` | Sampler utilizado para la comparación de pasos. |

---

# Animaciones Implementadas

| Animación | Estado |
|-----------|:------:|
| Forward VP / VE / sub-VP | OK |
| Evolución de la densidad | OK |
| Trayectorias forward | OK |
| Reverse-time SDE | OK |
| Probability Flow ODE | OK |
| Flow Matching | OK |
| Campo de Score | OK |
| Comparación del número de pasos | OK |

Todas las animaciones utilizan la misma infraestructura de visualización y pueden exportarse directamente en formato MP4, GIF o PNG.

