## Basic Python:
``` python

For loop:
# Dictionary:
- for key, value in dictionary.items():
    > print(key + ' -- ' + value)
# Numpy array
- for value in np.nditer(np_array): # to iterate all values in np array
    > print(value)
# Pandas DataFrame:
- for row_label, row_data in dataFrame.iterrows():
    > print(row_label)
    > print(row_data)
    > dataFrame.loc[lab, 'name_length'] = len(row_data['country_name']) #add new column as length of country_name
```
## EDA:
``` python
    - Initial exploration:
        - data.info(): #show information for each column such as datatype and missing values
        - data.value_counts('genre'): find the number of books with each genre
        - data['year'] = data['year'].astype(int): update data type
            - values: int, str, float, dict, list, bool
        - books['genre'].isin(['Fiction', 'Non Fiction']): check whether values in genre column contain those values
    - Missing data:
        - data.isna().sum(): counting missing value for each column
        - DROP MISSING VALUES if they are 5% less than total values
            - threshold = len(data) * 0.05
            - col_to_drop = data.columns[data.isna().sum() <= threshold]
            - data.dropna(subset = col_to_drop, inplace = True)
        - Impute missing values:    
            - cols_with_missing_values = data.columns[data.isna().sum() > 0]
            - for col in cols_with_missing_values[:, -1]:
                - data[col].fillna(data[col].mode()[0)
            - col_dict = data.groupby(<column to groupby>)[<column has missing value>].mean().to_dict()
            - data[<column has missing value>] = data[<column has missing value>].fillna(data[<column to groupby>).map(col_dict)
    - Converting and analyzing categorical data:
        - pd.Series.str.repalce('characters want to remove', character to replace them with')
        - data.select_dtypes('object'): select data with object types only
        - data[<column>].nunique(): count how many unique values in the column
        - data[<column>].str.contains('data analyst'): search for a series whether values contains string
            - data[<column>].str.contains('data analyst|data scientst'): search for multiple values
            - data[<column>].str.contains('^data'): any start with "data"
            - data[<new column>] = np.select(conditions, categories, default = 'Other'): create new column with conditions for values
    - Working with numric data:
        data['median'] = data.groupby(<column1>)[<column2>].transform(lambda x: x.std()): adding summary statistic into DataFrame
- Pattern over time:
    - data = pd.read_csv('link to file", parse_dates = ['date column'])
    - data['date'] = pd.to_datetime(data['date'])
    - data['month'] = data['date'].dt.month
- Cross-tabulation:
    - pd.crostab(data[<column1>], data[<column2>], values = data[<column3>], aggfunc = 'median')
        - column1 will be the index
        - column2 will be the count of combined obervation
        - aggfunc: apply median function for the values argument
- Creating categories:
    - twenty_fifth = data[<column>].quantile(0.25)
    - seventy_fifth = data[<column>].quantile(0.75)
    - median = data[<column>].median()
    - maximum = data[<column>].max()
    - labels = ['economy', 'premium', 'business', 'first']
    - bins = [0, twenty_fifth, median, seventy_fifth, max]
    - data[<new_column>] = pd.cut(data[<column>], labels = labels, bins = bins)
```

## Numpy:
``` python
- Create numpy array: np_array = np.array(list)
- Calculate: bmi = np_weight / np_height ** 2
- Retreive element from array: bmi[bmi > 23] -> return only value over 23
- Random generators:
    - np.random.rand() #Pseudo-random numbers
    - np.random.seed(163): #same seed, same random numbers
    - np.random.randint(0,2): #randomly generate integer between 0 and 1
- Type of Numpy Array: numpy.ndarray
    - array  = np.array([[1,2,3,4],
    -                     5,6,7,8])
    - print(array.shape) -> (2,4) means 2 rows 4 columns
    - array[:, 1:3] -> ([[2,3], [6,7]])
- Basic Statistic:
    - np.mean(np_city[:, 0]) -> mean of first column
    - np.median(np_city[:, 0]) -> median of first column
    - np.corrcoef(np_city[:, 0], np_city[:, 1]) -> correlation between first and second column
    - np.std(np_city[:, 1]) -> standard deviation of the second column
    - np.random.normal(mean, standard deviation, number of samples) -> generate data
    - np_city = np.column_stack((first column data, second column data)) -> stack data into 2 columns
    - np.logical_and(bmi > 21, bmi < 22): can also use np.logical_or(), np.logical_not()
    - np.linspace(start, stop, num)
        - np.quantile(data[<column>], np.linspace(0, 1, 5): split data into quartiles (5 different intervals)

```
## Matplotlib and Seaborn:
``` python

- plt.plot(horizontal, vertical) -> line chart
- plt.scatter(horizontal, vertical, s, c, alpha) -> scatter plot
    - s: size of dots
    - c: color
    - alpha: opacity
- plt.hist(x, bins, range, alpha)
    - bins: number of bins
    - alpha: transparent
- data.plot(kind = 'bar', x, y, title, rot = 45)
    - kind = bar / line / scatter
    - x,y: data for line plot
    - rot: rotate the label for 45 degree
- plt.boxplot(data[<column>])
- plt.legent(['Male', 'Female'])
- plt.clf() -> clean up plot
- plt.axis('equal'): make the x axis is euqal to y axis

- plt.axline(xy1 = (150, 150), slope = 1, linewidth = 2, color = 'green'): draw the line with intercept as xy1, and slope = 1
    - plt.axline(xy1 = (0, intercept), slope))


- Adjust the graph:
    - plt.xscale('log') -> scale horizontal to log
    - plt.xlabel('Label for X')
    - plt.title('Title of the graph')
    - plt.ysticks([0, 2, 4, 6, 8, 10],
    -             ['0B', '2B', '4B', '6B', '8B', '10B'])
    - plt.grid(True)
    - plt.yscale('log'): scale the y axis to log

- tips = sns.load_dataset('tips'): load the dataset name 'tips' as variable "tips"
- sns.histplot(data, x, binwidth)
- sns.scatterplot(x = <column1>, y = <column2>, data = dataFrame, hue = <column3>, hue_order = [<value1>, <value2>])
- sns.lmplot(x = <column1>, y = <column2>, data = dataFrame, ci = None): adding a trendline
    - ci: confident interval
- sns.regplot(x, y, data, ci, line_kws, logistic): also adding a trendline using linear regression
    - line_kws: {'color': 'black'}
    - logistic = True: plot the logistic regression
- sns.countplot(x = <column>, data = data) : countplot for each gender
- sns.heatmap(data.corr(), annot = True)
- sns.pairplot(data, vars = [<column1>, <column2>, <column3>, .etc]): plot all pairwise relationship between numeric variables
- sns.kdeplot(data, x, hue, cut = 0, cumulative = True): Kernel Density Estimate Plot
    - cut: how far pass the minimum and maximum data values the curve should go
    - cumulative: if we are interested in cumulative curve

- fig = plt.figure(): set to plot multiple layers
- Setting HUE:
    - sns.scatterplot(x = <column1>, y = <column2>, data = dataFrame, hue, hue_order, palette = hue_colors)
        - hue = <column3>
        - hue_order = [<value1>, <value2>]
        - hue_colors = {<value1>: 'black', <value2>: 'red'}
- RELATIONAL PLOTS AND SUBPLOTS: used for quatitative variables
    - relplot(x, y, data, kind, col, row, col_wrap, col_order, size, style, alpha, markers, dashes, ci):
        - create subplots in a single figure
        - kind = 'scatter' / 'line'
        - col (~column) = <column_name>
        - row = <column_name>
        - col_wrap = 2: 2 columns per row
        - col_order: order of the fugure
        - size = <column_name>: size of the points
        - style = <column_name>: style of each point base on the assigned column
        - alpha = 0 to 1: transparency of points
        - markers = True: used for lineplot, change datapoint into a marker base on "style" parameter
        - dashes = False: lines in lineplot DONT VARY by subgroup
        - ci = 'sd': confident interval of standard deviation (show us how the spread of distribution)
- COUNTPLOTS AND BARPLOTS: used for categorical variables
    - sns.catplot(x, y, data, kind, order, sym, whis, join, estimator, capsize)
        - kind = 'count' / 'bar' / 'box' / 'point'
        - order = [<list of order>]
        - sym = "" : omit the outliers (used in boxplot)
        - whis (~whiskers): the range of IQR, by default is 1.5
            - whis = 2: set the whiskers = 2 * IQR
            - whis = [5, 95]: show the 5th and 95th percentile
        - join: only used in pointplot (True / False)
        - estimator:
            - from numpy import median
            - estimator = median: calculate median instead of mean by default
        -capsize = 0.2: change the way confident interval is displayed

    -sns.displot(data, x, y , hue, row, col, kind): used to create multiple plot
        - kind = 'hist', 'kde'
- CHANGING PLOT STYLE AND COLOR: have to use "", NOT ''
    - style: include background and axes
        - sns.set_style()
        - preset values: ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']
    - palette: changes the color of the main elements of the plot
        - sns.set_palette()
        - Diverging palette values: ['RdBu', 'PRGn', 'RdBu_r', PRGn_r']
        - Sequential palette values: ['Greys', 'Blue', 'PuRd', 'GnBu']
        - custom palette: ['red', 'green', 'blue', 'yellow', etc]
    - scale: change the scale of the plot elements and label:
        - sns.set_context()
        - values: ['paper', 'notebook', 'talk', 'poster'] with default is 'paper'
- TITLE AND LABELS:
    - 2 types of plots in seaborn:
        - FACETGRID: (relplot, catplot): can create subplots
        - AXESSUBPLOT: (scatterplot, countplot, etc): only create a single plot
    - Set title:
        - g = sns.catplot(...)
        - g.fig.suptitle("New title", y = 1.03): add title for FACETGRID
            - y : set the height of the title, move it up with 1.03
        - g.set_title("This is {col_name}", y = 1.03): set tile for AXESSUBPLOT with variable col_name
    - Axes labels:
        - g.set(xlabel = "New X Label", ylabel = "New Y lable")
        - plt.xticks(rotation = 90): rotate the label of x-axis 90 degree
```
## Pandas:
``` python
- Inspecting DataFrame:
    - data.describe(): #calculate summary statistics for each column
    - data.sort_values([<column_name1>, <column_name2>], ascending = [False, True]): sort data values in specific column
    - data.sort_index(level = [<index1>, <index2>], ascending = [True, False])
    - is_black_or_brown = data['color'].isin(['Black', 'Brown'])
    - data.columns / data.index: for column names, index column
        - data.set_index(<column1>): set column as index
        - data.reset_index(drop = True): reset index
            - drop = True: remove the column as index, reset index
- Change index label: dataFrame.index = <list of variables>
- Use 3rd column as index: pd.read_csv('fileName', index_col = 2)
- apply: dataFrame['name_length'] = dataFrame['country_name'].apply(len)
    > #loop all row and apply len() function to specific column
    > # apply(str.upper) for uppercase
- Summary Statistic:
    - data['height'].mean(): #apply for
        - .median(), .mode(),
        - .var(), .std(),
        - .max(), .min()
        - .sum(), .quantile()
    - data[['weight', 'height']].agg([<function1>, <function2>]) : #apply aggregate function to column
        - data.agg({<column1>: ['mean', 'std], <column2>: ['median']})
        - data.groupby('genre').agg(mean_rating = ('rating', 'mean')): set aggregated name for column calculate mean of rating column
    - data['weight'].cumsum(): cumulative sum of a column
        - cummax(): cumulative max
    - dataFrame.drop_duplicates(subset = [<column1>, <column2>])
    - datFrame[<column>].value_counts(sort = True, normalize = True)
        - nomarlize = True: get the proportions
    - dataFrame.groupby(<column1>)[<column2>].mean()
        - dataFrame.groupby([<column1>, <column2>])[[<column3>, <column4>]].agg([min, max, sum])
    - data.pivot_table(values = 'weight', index = 'color', columns = 'breed', aggfunc = [np.median, np.mean], fill_value = 0, margins = True):
        - #take the mean by default
        - fill_value: replace missing value with 0
        - margins = True: take the means for all columns and rows, not including missing values
        - print(mean_temp_by_year[mean_temp_by_year == mean_temp_by_year.max()]): get the max value in the table
- Slicing and Subsetting:
    - data.loc['Chow Chow': 'Poodle']: #the final value 'Poodle' is included
    - data.loc[('Labrador':'Brown'):('Schnauzer': 'Grey')]: #Slicing the inner index levels correctly
    - data.loc[:, 'name': 'height']: #slicing columns
    - data.iloc[2:5, 1:4]
    - temperatures_bool = temperatures.loc[(temperatures['date'] >= '2010-01-01') & (temperatures['date'] <= '2011-12-31')]
- Access component of date:
    - data['date'].dt.month
- Missing values:
    - data.isna(): detect missing values
        - data.isna().any(): return boolean for each variables with any NaN
        - data.isna().sum(): counting missing values
    - data.dropna(): removing missing values
    - data.fillna(0): fill NaN with 0
- Merging dataframe:
    new_df = df1.merge(df2, how = 'inner', on = [<column>])
```
## Statistic:
``` python
- Definition:
    - Descriptive statistic: describing the data you have
        - measures of center, spread
        - charts to visualize data
    - Inferential statistic: using data to make prediction or estimation
        - hypothesis testing, confident interval
        - regression analysis
    - Spread:
        - Variance: average distance from each data's point to data's mean. The higher, the spreader
            - 1. take distance from mean
            - 2. square each distance
            - 3. sum square distances
            - 4. divide by number of data point - 1

            - or: np.var(data[<column], ddof = 1)
        - Standard deviation: square root of variance
            - np.sqrt(np.var(data[<column>], ddof = 1))
            - or: np.std(data[<column>], ddof = 1)
        - Quantile:
            - np.quantile(data[<column>], 0.5): 50% percent of data ~ exactly the same as median
            - np.quantile(data[<column>], [0, 0.25, 0.5, 0.75, 1])
        - Interquartile Range (IQR): the distance between 25th and 75th percentile
            - import scipy.stats as iqr
            - iqr(data[<column>])
        - Outliers: data point that is substantially different from others
            - outlier < Q1 - 1.5 * IQR or outlier > Q3 + 1.5 * IQR
        - Expected Value: mean of probability distribution
            - value * probability
- Probability and Distribution:
    - data.sample(2, replace = True): get 2 samples from dataframe
        - 2: number of sample
        - replace = True: sample with replacement (rows can be appear more than 1 time)
        - np.random.seed(163): need to set up seed to get same random number
    - Uniform distribution for continuous distribution:
        - from scipy.stats import uniform
        - uniform.cdf(value or "less" want to calculate prob, lower limit, higher limt):
        - 1 - uniform.cdf(value or 'more' want to calculate prob, lower limit, higher limit)
        - uniform.rvs(min, max, number of values): generate random number according to uniform distribution
    - Binominal distribution: only 2 outputs (1 vs 0, win or loss, fail or success)
        - from scipy.stats import binom
        - binom.rvw(# of coins, probability of success, size = # of trials):
            - binon.rvs(1, 0.5, size = 5) : [0, 1, 1, 1, 0]
            - binom.rvs(5, 0.5, size = 1): [3] : the total number of 1 or success
        - Can be describe by n and p:
            - n: total number of trials (second arguments in binom.rvs)
            - p: probability of success (third arguments in binom.rvs)
        - binom.pmf(# of outcome, # of trials, prob of head)
            - binom.pmf(7, 10, 0.5): what's the probability of getting "exactly" 7 heads when flip a coin 10 times
            - binom.cdf(7, 10, 0.5): what's the probaility of getting 7 or "less" heads when flip a coin 10 times
            - 1 - binom(7, 10, 0.5): 7 or "more" heads
        - Expected value = n * p:
            - Expected number of heads out of 10 flips: 10 * 0.5 = 5
        - IMPORTANT: each trial has to be independent
    - Normal Distribution:
        - Properties:
            - Symmetrical in the center by mean
            - NEVER hit 0%
            - Describe by mean and std
        - 68-95-99.7% rule: APPLY ONLY FOR STANDARD NORMAL DISTRIBUTION (mean = 0 and std = 1)
            - 68% fall in std = 1 of the mean
            - 95% fall in std = 2 of the mean
            - 99.7% fall in std = 3 of the mean
        - Python:
            - from scipy.stats import norm
            - norm.cdf(154, 161, 7): the percent of women "shorter" than 154cm with mean = 161cm and std = 7
            - 1 - norm(154, 161. 7): women "taller" than 154cm
            - norm.ppf(0.9, 161. 7): what height are 90% women "shorter" than with mean = 161cm and std = 7?
            - norm.ppf((1 - 0.9), 161. 7): what height are 90% are "taller" than?
            - norm.rvs(161. 7, size = 10): generate random number from the distribution
    - Poisson distribution: the probability of some # of events occurring over a fixed period of time
        - Definition:
            - lambda: average number of events per time interval
            - Distribution's peak is always at lambda value
        - Python:
            - from scipy.stats import poisson
            - poisson.pmf(5, 8): the probability of # adoption per week = 5 if the average number of adoptions is 8 per week
            - poisson.cdf(5, 8): the probability of # adoption per week <= 5 if the average number of adoptions is 8 per week
            - 1 - poisson.cdf(5,8): the probability of # adoption per week >= 5 if the average number of adoptions is 8 per week
            - poisson.rvs(8, size = 10): generate 10 sample from a poisson distribution has lambda = 8
    - Exponential Distribution:
        - Definition:
            - lambda = 0.5: if one customer service ticket is created every 2 mins ~ half ticket is created per minute
            - 1 / lambda: time between events (1 ticket every 2 mins)
        - Python:
            - from scipy.stats import expon
            - expon.cdf(1, scale = 2): probaility of waiting less than 1 min with 1/lambda = 2
            - 1 - expon.cdf(4, scale = 2): probability of waiting time more than 4 mins with 1/lambda = 2
    - Student's T Distribution:
        - Degree of freedom: which affects thickness of the tails
            - Lower df: thicker tails, higher std
            - Higher df: closer to normal distribution
- Correlation:
    - Correlation Coefficient:
        - data[<column1>].corr(data[<column2>]): compute correlation
        - When the data is highly skew, we can apply "log transformation"
        - data[<new_column>] = np.log(data[<column>])
        - Other transformation:
            - square root: (sqrt(x))
            - Reciprocal: 1 / x
- Experiment:
    - Aims to answer: what is  the effect of the treatment on the response
        - Treatment:  explanatory / independent variable
        - Response: response / dependent variable
- EDA:
    - Univariate: involve one variable:
        - Numerics: Histogram, boxplot, summary statistic
        - Categorical: Bar chart, table summary
    - Bivariate: involve 2 variables, also called "covariation"
```
## Regression:
``` python
- Linear Regression: response value = fittedValues + residual
    - Intercept: value of y when x = 0
    - Slope: value of y increases if increase x by 1
    - Equation: y = intercept + slope * x
    - Python:
        - from statsmodels.formula.api import ols (ordinary least squares)
        - mdl_payment_vs_claims = ols('total_payment_sek ~ n_claims', data = data)
            - response ~ exploratory variable
        - mdl_payment_vs_claims = mdl_payment_vs_claims.fit()
        - print(mdl_payment_vs_claims.params)

        - new_variable = ols('response ~ exploratory + 0', data = data).fit(): model without intercept for categories variables
- Making prediction:
    - exploratory_data = pd.DataFrame({'length_cm': np.arange(20:41)}): filter the range of column name 'length_cm' to 20-41
    - predicted_data = exploratory_data.assign(response_var = model.predict(exploratory_data))
    - print(predicted_data)
- Working with model objects:
    - .fittedvalues: prediction on the original dataset
    - .resid: actual response values minus predicted response values
    - .summary(): shows more extended printout of the details of the model
    - .rsquared: show the coefficient of determination
    - .rsquared_adj: the adjust coefficient of determination
    - .mse_resid: mean squared error
    - np.sqrt(mse_resid): residual standard error
    - .pred_table(): show the confusion matrix
- Regression to the mean: ~ extreme cases don't persist over time
- Quantifying model fit:
    - Coefficient of determination: r-squared (single linear regression) or R-Squared (when more than 1 explanatory variables)
        - show how well the linear regression line fits the observed values
        - A larger number is better (from 0 to 1)
        - Adding more explanatory variables will increase R-squared but also cause overfitting
            => adjust coefficient determination penalizes more explanatory variables
            - adj r-squared = 1 - (1-R-squared)* (n_obs - 1) / (n_obs - n_var - 1)
    - Residual standard error (RSE): a typical difference between predicted values and observed values
        - Smaller number is better
    - Mean squared error (MSE) = RSE ** 2
- Visualizing model fit:
    - Residual vs fitted plot:
        - sns.residplot(x = explanatory_variable, y = responsse_values, data, lowess = True)
    - qqplot:
        - from statsmodels.api import qqplot
        - qqplot(data = model.resid, fit = True, line = '45')
    - Scale-location plot:
        - model_norm_resid = model.et_influence().resid_studentized_internal
        - model_norm_resid_sqrt = np.sqrt(np.abs(model_norm_resid))
        - sns.regplot(x = model.fittedvalues, y = model_norm_resid_sqrt, ci = None, lowess = True)
- Leverage and Influence:
    - Leverage: measure how extreme explanatory variable values are
        - High leverage means the explanatory variable has values that are different from other points of the dataset
    - Influence: measure how much the model would change if you left the observation out of the dataset
        - it measures how different the prediction line would look if you would run a linear regression on all data points except that point
    - Cook's distance: the most common measure of influence
    - Python:
        - summary = model.get_influence().summary_frame()
        - data['leverage'] = summary['hat_diag']

- Parallel Slope Regression:
    - model = ols('mass_g ~ length_cm + species + 0', data = fish).fit()
    - coeffs = model.params
    - ic_bream, ic_perch, ic_pike, ic_roach, sl = coeffs
    - plt.axline(xy1 = (0, ic_bream), slope = sl, color = 'blue')
    - Prediction:
        - length_cm = np.arrange(5,61, 5)
        - species = fish['species'].unique()
        - from itertools import product
        - p = product(length_cm, species)
        - expl_data = pd.DataFrame(p, columns = ['length_cm', 'species'])
        - prediction_data = expl_data.assign(mass_g = model.predict(expl_data))
    - Manually calculate value:
        - conditions = [data['species'] == 'Bream', data['species'] == 'Perch', data['species'] == 'Pike', ...]
        - choices = [ic_bream, ic_perch, ic_pike, ...] : choices and conditions have to be same length
        - intercept = np.select(conditions, choices)
        - prediction_data = expl_data.assign(intercept = np.select(conditions, choices),
                                            mass_g = intercept + slope * expl_data['length'])

- Logistic Regression:
    - python:
        - from statsmodels.formula.api import logit
        - mdl = logit(reponse ~ explanatory, data).fit()
        - prediction_data = explanatory_data.assign(has_churned = mdl.predict(explanatory_data))
    - CONFUSION MATRIX:
        - actual_response = churn['has_churned']
        - predicted_response = np.round(model.predict())
        - outcomes = pd.DataFrame({'actual': actual_response, 'predicted' = predicted_response})
        - print(outcomes.value_counts(sort = False))

        - from statsmodels.graphics.mosaicplot import mosaic
        - mosaic(conf_matrix)
    - Odds ratio:
        - odds ratio = probability / (1 - probability)
    - Quantifying: using confusion matrix
        - False Positive: predict YES when it is NO
        - False Negative: predict NO when it is YES
        - Accuracy = (TN + TP) / (TN + TP + FN + FP): proportion of correct predictions
        - Sensitivity = TP / (FN + TP): the proportion of true positive
        - Specificity = TN / (TN + FP): proportion of true negatives
```
## Hypothesis:
``` python
- Workflow of hypothesis testing:
    - Identify population parameter that is hypothesized about
    - Specify the null and alternative hypotheses
    - Determine (standardize) test statistic and corresponding null distribution
    - Conduct hypothesis test in python
    - Measure evidence againt null hypothesis
    - Make a decision comparing evidence to significance level
    - Interpret the result in the context of the original problem

- Calculate sample mean: mean = data[<column>'].mean() : also called summary statistic
    - For categorical column: mean = (data[<column>] == 'value').mean()
- Generate bootstrap distribution: to standardize data we need
        - Sample statistic (point estimate)
        - Hypothesis test statistic
        - standard error (estimated from bootstrap distribution)
    - Step1: resample:
        - new_data = data.sample(frac = 1, replace = True)[<column>]
    - Step2: Calculate point estimate:
        - np.mean(new_data)
    - Step3: repeat step 1 and 2 many times, append to a list:
        - so_boot_dist = []
        - for i in range(5000):
            - so_boot_dist.append()
- Calculate standard error: np.std(so_boot_dist, ddof = 1)
- z_score = (prop_sample - prob_hyp) / std_error or also (sample statistic - population mean) / std_error
- p-values: the probability of obtaining a result, assuming the null hypothesis is true
    - large p-value: large support for H0 means statistic likely "NOT IN" the tail of null hypothesis
        - Fail to reject null hypothesis
    - small p-value: strong evidence again H0 means statistic likely "IN" the tail of null hypothesis
        - Reject null hypothesis
    - Calculate:
        - from scipy.stats import norm
        - 1 - norm.cdf(z_score, loc = 0, scale = 1)
            - loc: default mean value = 0
            - scale: std_error = 1
    - left tail test: use norm.cdf()
    - right tail test: use 1 - norm.cdf()

- Cut off point (significance level alpha)
- Confidence Interval: (1 - alpha)
    - lower = np.quantile(so_boot_dist, 0.025)
    - upper = np.quantile(so_boot_dist, 0.975)
    - IF the hypothesis population parameter is within the Confidence Interval, we should fail to reject H0


- Performing T-test: Compare sample statistics across group of variable
    - z-score is a standardized test statistic
    - t-score = (difference in sample stats - difference in population parameter) / std_error
        - std = sqrt(s1^2 / n1 + s2^2 / 2)
            - s: standard deviation
            - n: sample size
        - if we assume null hypothesis is true -> difference between population parameter = 0
    - degree of freedom (df): n1 + n2 - 2
    - Calculate p-value for t-test:
        - from scipy.stats import t
        - p_values = 1 - t.cdf(t_stat, df) : used for right tail test
        - p_values = t.cdf(t_stat, df) : used for left tail test

- Paired t-test:
    - since variables are not independent, degree of freedom = n - 1
    - Calculate:'
        - mean_diff -> data['diff'].std() -> t_value
        - df = n - 1 -> p_value
    - Easier way:
        - import pingouin
        - pingouin.ttest(x = data['diff'], y = 0, alternative = 'less')
            - x: mean statistic
            - y: mean hypothesis
            - alternative: 'two-sided', 'less', 'greater' for the alternative hypothesis
        - pingouin.ttest(x = data[<column1>], y = data[<column2>], paired = True, alternative = 'less)
            - pass 2 variables instead of passing 1 column called 'diff'

- ANOVA test: a test for difference between groups:
    - pingouin.anova(data, dv = <column1>, between = <column2>)
        - if p-unc < alpha: reject null hypothesis -> at least 2 categories are significant different
    - pingouin.pairwist_test(data, dv, between, padjust = 'none')
```

## A/B Testing:
``` python
- Key performance Indicators (KPIs):A/B test are run to improve KPIs

```
