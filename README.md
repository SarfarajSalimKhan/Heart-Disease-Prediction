Cross-site evaluation of heart disease prediction on the four UCI sites.

Implements the below experiment plan:
  RQ1  How much does AUC drop under leave-one-site-out (LOSO) vs a random split?
  RQ2  Are probabilities still calibrated on an unseen site?
  RQ3  How stable are feature-importance (SHAP) rankings across sites?
  Method check: how much does resampling BEFORE the split inflate scores?

Usage
-----
  pip install numpy pandas scikit-learn scipy matplotlib shap
  python heart_site_study.py                 # downloads UCI data, runs everything
  python heart_site_study.py --quick         # fewer repeats, runs in a few minutes
  python heart_site_study.py --synthetic     # no internet needed (TESTING ONLY, fake data)

If the automatic download fails, download
  https://archive.ics.uci.edu/static/public/45/heart+disease.zip
and put processed.cleveland.data, processed.hungarian.data,
processed.switzerland.data and processed.va.data into the folder given by --data-dir.

Outputs (CSV tables and PNG figures) are written to --out-dir (default: results/).
