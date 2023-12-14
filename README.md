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


- Adjust the graph:
    - plt.xscale('log') -> scale horizontal to log
    - plt.xlabel('Label for X')
    - plt.title('Title of the graph')
    - plt.ysticks([0, 2, 4, 6, 8, 10],
    -             ['0B', '2B', '4B', '6B', '8B', '10B'])
    - plt.grid(True)

- tips = sns.load_dataset('tips'): load the dataset name 'tips' as variable "tips"
- sns.histplot(data, x, binwidth)
- sns.scatterplot(x = <column1>, y = <column2>, data = dataFrame, hue = <column3>, hue_order = [<value1>, <value2>])
- sns.lmplot(x = <column1>, y = <column2>, data = dataFrame, ci = None): adding a trendline
    - ci: confident interval
- sns.countplot(x = <column>, data = data) : countplot for each gender
- sns.heatmap(data.corr(), annot = True)
- sns.pairplot(data, vars = [<column1>, <column2>, <column3>, .etc]): plot all pairwise relationship between numeric variables
- sns.kdeplot(data, x, hue, cut = 0, cumulative = True): Kernel Density Estimate Plot
    - cut: how far pass the minimum and maximum data values the curve should go
    - cumulative: if we are interested in cumulative curve
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
        - 
