GaiaTonic
AI + Environment | Trash Mapping | Fire Risk | Smart Village AI Visibility

GaiaTonic is an open research platform for applying AI and machine learning to environmental visibility, fire risk, and demographic equity—starting with large-scale street-level trash detection and extending to global sustainability applications.

---

Project Goals:

* Map visible trash from street-level imagery using AI vision models
* Correlate environmental visibility with fire risk and social-demographic factors
* Provide interpretable models and visualizations at tract, county, and national levels
* Enable global adaptation (e.g., Thailand Smart Village initiative)
* Share open tools for reproducible research and community-driven analysis

---

Repository Structure:

gaiatonic/
├── data/                  -> Datasets (safe/public subsets only)
├── scripts/               -> Core Python analysis scripts
├── notebooks/             -> Jupyter demos and explorations
├── models/                -> Saved models (optional)
├── results/               -> Outputs: maps, stats, charts
├── docs/                  -> Paper drafts, writeups
├── README.md              -> This file
├── LICENSE                -> Apache 2.0
└── requirements.txt       -> Python dependencies (to add)

---

Reproducible Analysis:

Coming soon: `repro.sh` to run:

1. pca\_tract.py – tract-level PCA + regression on fire/trash
2. analyze\_trash\_and\_fire\_directions\_tract.py – compute forward/reverse correlations
3. map\_tract.py – visualize scores geographically

---

Dependencies:

To install required packages:
pip install -r requirements.txt

Main libraries include:

* pandas, numpy, scikit-learn
* matplotlib, seaborn, folium, geopandas
* tqdm, pyarrow, joblib

---

Example Use Cases:

* Predict wildfire risk using street-level trash visibility
* Evaluate government performance using residual mapping
* Study demographic drivers of environmental neglect
* Apply AI tools to assist rural sustainability efforts

---

Global Extensions:

* Thailand (Saphung Village Smart AI Lab)
* Mexico (drone-validated trash detection in Tijuana)
* Africa/Asia (planned generalization)

---

License:
Apache 2.0 — Free for research, reuse, and modification with attribution.

---

Acknowledgments:

Led by David K. Tcheng ([https://github.com/DavidTcheng](https://github.com/DavidTcheng)), in collaboration with:

* UC Santa Cruz (Bo Yang, GIS + wildfire risk)
* San Diego Green Infrastructure Consortium
* Tree San Diego
* GaiaTonic (music + Earth innovation)

---

Contact:
Email: [davidtcheng@gmail.com](mailto:davidtcheng@gmail.com)
Web: [www.gaiatonic.com](http://www.gaiatonic.com)
