# Diabetes Hackathons

This repository contains code and resources from a single hackathon organized by the Microsoft Tech Club at ISGI University. The project work lives in the Diabetes-hackathon/ folder and includes data, model scripts, and reports produced during that event.

## Repository structure (actual)

- Diabetes-hackathon/  - Primary hackathon project folder containing code, data, and reports
- README.md            - This file (project-level overview)

Files inside Diabetes-hackathon/
- FINAL_PRESENTATION_REPORT.txt  - Final presentation / report for the hackathon project
- diabetes.csv                   - Dataset used for the hackathon (committed in repo)
- model_v0.py, model_v1.py, ...   - Iterative model scripts used during experimentation
- model_v*_report.txt            - Text reports describing model results and notes
- model_v1_output.txt            - Example output from model_v1 (if present)
- sweep_trees.py                 - Script for hyperparameter sweeps or tree-based experiments

> Note: diabetes.csv is currently committed. If this dataset is sensitive or too large, consider removing it from the repo history and hosting it externally (S3, Drive) with a small sample or a download script in the repo.

## Getting started

1. Clone the repository

   git clone https://github.com/dhiatriki/Diabetes-Hackathons.git
   cd Diabetes-Hackathons/Diabetes-hackathon

2. Create a Python virtual environment (recommended)

   python -m venv .venv
   source .venv/bin/activate    # macOS / Linux
   .\.venv\Scripts\activate   # Windows (PowerShell)

3. Install dependencies

   pip install -r requirements.txt

   If `requirements.txt` is not present, typical packages used in these scripts include:

   - numpy
   - pandas
   - scikit-learn
   - matplotlib
   - seaborn

4. Run a model script (example)

   python model_v1.py

   Scripts expect `diabetes.csv` to be in the same directory. Check the top of each model script for any configuration or path variables.

## Reproducing results and reports

- See the model_v*_report.txt files for notes on metrics, configuration, and results for each version.
- The FINAL_PRESENTATION_REPORT.txt contains the final slides/summary submitted for the hackathon.

## Contributing

This repository reflects a one-off hackathon project. If you want to make improvements:

1. Open an issue describing what you'd like to change.
2. Fork the repo and create a branch with your changes.
3. Submit a pull request with a clear description of your modifications.

## License

Add a LICENSE file to make reuse terms explicit. Common choices are MIT or Apache 2.0.

## Contact

This project was completed for a Microsoft Tech Club hackathon at ISGI University. For questions, visit https://github.com/dhiatriki
