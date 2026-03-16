# Getting Started

This repository contains the code used for our analysis. Since the original code heavily uses private student data from New York City, we have created **quickstart.ipynb**, which reproduces the main paper figures using semi-synthetic data. Program-level information (including admissions policies) are generated directly from the high school directory (fall_2022.xlsx), as in our real analysis. Student-level data is generated synthetically. The synthetic data generation has several parameters, which can be easily adjusted in the notebook.

# Description of main analysis files

**pipeline_export.ipynb** takes private student-level data and public program-level data to produce our student- and program-level dataframes for analysis

**results_export.ipynb** contains our main analyses and figures

**paper_tables_reg_export.ipynb** contains code to create supplemental regressions and program-level analyses
