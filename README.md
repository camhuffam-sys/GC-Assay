# Fluorescence assay data analysis
Remade script for generating delta t50 heatmaps using raw lab data from honours but in git.
This program uses data output from a fluorometer/incubator reading a 96 well plate with a fluorescence based assay.
The data in this folder is real fluorescence over time data from one of these assays.
The delta t50 values calculated represent the difference in time taken for the test wells to reach 50% of the controls maximum fluorescence compared to the control wells.
The assay was in quadruplicate with each row other than the ones with the control wells containing a different antibiotic at 3 different concentrations.
The first 4 rows used A.aureus, the last 4 rows used P.aeruginosa.

The controls with no antibiotics were the first 6 columns in the 4th row of each bacteria. So D01-D06 for S.a and H01-H06 for P.a.
For copy and pasting when running the program: 
S.a: D01,D02,D03,D04,D05,D06
P.a: H01,H02,H03,D04,D05,D06

The test wells are all wells in rows A, B, and C for S.a and E, F, and G for P.a.
For copy and pasting when running the program: 
S.a: A01,A02,A03,A04,A05,A06,A07,A08,A09,A10,A11,A12,B01,B02,B03,B04,B05,B06,B07,B08,B09,B10,B11,B12,C01,C02,C03,C04,C05,C06,C07,C08,C09,C10,C11,C12
P.a: E01,E02,E03,E04,E05,E06,E07,E08,E09,E10,E11,E12,F01,F02,F03,F04,F05,F06,F07,F08,F09,F10,F11,F12,G01,G02,G03,G04,G05,G06,G07,G08,G09,G10,G11,G12

## Streamlit application

Install the application dependencies and launch the interactive app from this folder:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Upload an instrument CSV with `Well` and `MeasA:Result` columns. In **Plate analysis**,
choose a bacteria-group count, name the groups, select a control or test role, and click
wells on the visual plate. Run the delta-T50 analysis to view the plate heatmap and
download its CSV and PNG.

The **Fluorescence curves** tab can overlay individual wells or plot a selected replicate
set as a mean with SD or SEM error bars. It also offers optional baseline subtraction
and CSV downloads. The **Data overview** tab previews and downloads cleaned measurements.

Run the existing desktop application with `python main.py`. Run tests with
`python -m pytest -q` after installing the development dependencies using
`python -m pip install -r requirements-dev.txt`.
