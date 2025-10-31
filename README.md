# pt-distance-calculator
Usage: 
1. Download the GTFS dataset für the Public Transport Service you want to use (Viennas Wiener Linien, for example, is here: https://www.data.gv.at/datasets/ab4a73b6-1c2d-42e1-b4d9-049e04889cf0?locale=de) and save the contents in the subfolder "gtfs".
2. save the legs of your journey in CSV format using legs-template.csv as a starting point.
3. Install Python and the following modules:
- pandas
- numpy
with:
pip install pandas numpy
4. Run python wien_offi_kilometer.py --legs legs-template.csv (replace file names with your actual file names)
5. Find the detailed results in kilometer_auswertung.csv