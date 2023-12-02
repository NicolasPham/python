``` python

## Basic Python:
#### For loop:
##### Dictionary:
- for key, value in dictionary.items():
    > print(key + ' -- ' + value)
##### Numpy array
- for value in np.nditer(np_array): # to iterate all values in np array
    > print(value)
##### Pandas DataFrame:
- for row_label, row_data in dataFrame.iterrows():
    > print(row_label)
    > print(row_data)
    > dataFrame.loc[lab, 'name_length'] = len(row_data['country_name']) #add new column as length of country_name


## Numpy:
- Create numpy array: np_array = np.array(list)
- Calculate: bmi = np_weight / np_height ** 2
- Retreive element from array: bmi[bmi > 23] -> return only value over 23
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

## Matplotlib:
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

## Pandas:
- Change index label: dataFrame.index = <list of variables>
- Use 3rd column as index: pd.read_csv('fileName', index_col = 2)
- apply: dataFrame['name_length'] = dataFrame['country_name'].apply(len)
    > #loop all row and apply len() function to specific column
    > # apply(str.upper) for uppercase



```

