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

```
## Matplotlib:
``` python

- plt.plot(horizontal, vertical) -> line chart
- plt.scatter(horizontal, vertical, s, c, alpha) -> scatter plot
    - s: size of dots
    - c: color
    - alpha: opacity
- plt.hist(x, bins, range)

- plt.clf() -> clean up plot


- Adjust the graph:
    - plt.xscale('log') -> scale horizontal to log
    - plt.xlabel('Label for X')
    - plt.title('Title of the graph')
    - plt.ysticks([0, 2, 4, 6, 8, 10],
    -             ['0B', '2B', '4B', '6B', '8B', '10B'])
    - plt.grid(True)
```
## Pandas:
``` python
- Inspecting DataFrame:
    - data.info(): #show information for each column such as datatype and missing values
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
- Slicing and Subsetting:
    - data.loc['Chow Chow': 'Poodle']: #the final value 'Poodle' is included
    - data.loc[('Labrador':'Brown'):('Schnauzer': 'Grey')]: #Slicing the inner index levels correctly
    - data.loc[:, 'name': 'height']: #slicing columns
    - data.iloc[2:5, 1:4]
    - temperatures_bool = temperatures.loc[(temperatures['date'] >= '2010-01-01') & (temperatures['date'] <= '2011-12-31')]
        

```


```

