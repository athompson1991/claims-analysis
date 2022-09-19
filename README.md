# Workers' Comp in New York

This project is an analysis of workers' comp claims in the state of New York. The data can be found here: https://data.ny.gov/Government-Finance/Assembled-Workers-Compensation-Claims-Beginning-20/jshw-gkgu

The main presentation is in the aptly titled file `presentation.ipynb`. However, to run the code itself you first clone this repo, create two directories named "plots" and "data", then save the data file into the data repository, renaming it to "full.csv". After installing all the required libraries into your python environment, you can easily run the script with `python script.py`

To run the fully automated code, use the below snippet. However, it should be noted that the download takes a long time _and often fails_ (I repeatedly got 500 server errors, problems on the NY government website side), so I really do recommend downloading the file manually, saving it, and running the code without the download step.

```
git clone git@github.com:athompson1991/claims-analysis.git
cd claims-analysis
pip install -r requirements.txt
python script.py --download --sample
```