## Welcome to ProjectFin

This is a Web User Interface for financial data analysis. Written in python (with Django), it is empowered by AI.


## Set-Up

First, you must create a new virtual enviroment for ProjectFin's website User Interface.
It is recommended to conduct these install through the virtualenv with conda
In terminal enter these command:
(replace x.x by python version)

```
conda create -n yourenvname python=x.x anaconda
source activate yourenvname
```

Once you are inside your virtualenv, you must install all requirements, as shown below.

```
conda install django
conda install pandas
conda install plotly
conda install quandl
conda install -c anaconda scikit-learn
conda install -c conda-forge xgboost
conda install statsmodels
conda install keras
conda install lxml
```

To exit the virtual env: 'source deactivate'
Alternatively, you can use the command below while outside of the virtualenv

```
conda install -n yourenvname [package]
```

## Setting Up Website

Once you are in your virtual env, navigate to the root directory (mysite folder) and enter the following command in the terminal to run django app:

```
python manage.py runserver
```

You can then open your webbrowser, and go to
``` localhost:8000 ```