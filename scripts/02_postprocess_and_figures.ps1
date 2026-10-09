$ErrorActionPreference = "Stop"
python src/postprocess_formal.py
python src/fixed_feature_control.py
python src/threshold_stability.py
python src/figures.py

