# HDMI eavesdropping project

**(Development):** Please note, this code-base is in a research state, as it was used for a project. However, I have created a pipeline to simplify a lot of the engine, following SWE principles.

This project will most likely be updated in the future to include the requirements to setup the environment, as I typically use a combination of UV and Conda. Then, also at some point, I imagine I will also streamline some of the experimentation to include it all within the pipeline rather than scratch files.

### Dataset
The dataset should be stored in the following location `_src\hdmi\data\...`

### Environment Setup
The project using both `Conda` and `uv` for the environment, therefore you need to perform the following steps:
```bash
conda create -n dsp python=3.12
conda activate dsp

pip install uv
python -m uv sync
python -m uv sync pip install -e .
```

### Outline
The project is mainly composed of two parts:
1. The `_src\hdmi\` directory contains the majority of the engine behind this codebase.
2. The notebooks then import `_src\hdmi\` and perform the experiments, with the main two provided being: `scratch.py` and `scratch_ac.py`.
