# Wind farm production analysis

The purpose of this repository is to explore different approaches for modeling time series data on wind farm energy production.

I created 3 main pieces of code: a Jupyter notebook, a Python file for the estimator class and a pytest test file. 

1. The notebook called *Wind farm production forecast* contains the initial data exploration, data preparation, model selection and performance assessment parts.

2. The estimator class *MyLRNNRegression.py* implements an ensemble model based on two models explored in the notebook.

3. In the test file *test_MyLRNNRegression.py* I show an example of a unit test ran with pytest.

Furthermore, the code in the notebook *Get weather data* retrieves weather data from the weather station Hemsby in the UK, through the meteostat API. The weather data is stored in *hemsby_hourly_resampled.csv*.

