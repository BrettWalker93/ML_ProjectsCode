For part 1, all code was implemented in Python 3.10 and run in Visual Studio debugging environment. For part 2, mlp.py was run on Google Colab, and svm.py and knn.py were run in Visual Studio.

Part 1:
libraries:
numpy
numba
pandas
tqdm.auto

First, run correlationmatrix.py to generate the correlation matrix
Second, run predictions.py to generate the predictions from the matrix and the test data
Third, run error.py to calculate the model's error

ATTN: there was sporadically a weird "integer" error when using numba to do for loops. I'm fairly certain I got rid of that error, but it was not always easily reproducible.
