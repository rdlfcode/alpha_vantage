# alpha_vantage

## AlgoEvals

We are working on a project called "AlgoEvals" that will be able to predict the future asset price over a given time period, normally 90 days. To make this work for many different asset classes, we will need a way to dynamically encode the dataset based on time series and categorical data. Ideally, this is achieved by having the user denote the different time series (ex. Ticker, ISIN) and the target to predict (ex. return, close price), from there it should be much easier to discern the categoricals from the time series data. Keep in mind that the data has different frequencies--daily prices (OHLCV), monthly economic data such as CPI and non-farm payrolls, and quarterly financial statements. 

One of the most important parts of this project is going to be comparison of different model accuracies and predictions. At a minimum, I would like to compare some simple statistical models, Fast Fourier Transform, LSTMs, vanilla Transformer, Temporal Fusion Transformer, and the Titans architecture with different memory implementations. We are going to need a very thorough data storage solution to keep track of our data sets, model parameters, parameter hypertuning, predictions, training metrics, and so on.

As for how to implement this, it's a complex project so we are starting simple: Python for nearly all of the coding, PyTorch for ML, DuckDB and parquet files for data management and storage, and a very easy frontend framework for Python like Streamlit or Django (not too sure on this bit, if you can think of an easier/more maintainable framework we can use that). 

This project should be broken down into these parts:

1. Data
	1. Data calling (from alpha vantage or another provider)
	2. Processing
	3. Management
	4. Storage
2. ML
	1. Time series and target specification
	2. Encoding/data structure analysis
	3. Model specifications
	4. Training
	5. Parameter Hypertuning
	6. Prediction (n points into the future, not against test data)
3. Frontend
	1. Dashboard for top investment recommendations, based on highest return (could be short positions)
	2. Dashboard for model accuracy (pick an index, ex. SPY, and chart different accuracies and predictions)
	3. A page for drilling down into specific investments

The frontend should be designed in a sense similar to a spreadsheet. I'd like some easy tabs at the top to switch between the different pages

Lastly, I would like all of this to be customizable but uses best settings by default. I think that wherever possible, there should be a json file that contains all the settings. These settings should be able to be used in most places, especially when it comes to model and hypertuning parameters. When hypertuning, the user should be able to specify (in the json file) which parts should be optimized. This can be how the data is normalized, encoded, and ultimately how the models is set up (this goes for model parameters, including the type of memory used for the Titans architecture). 

Now, what I would like you to do is use this rough description and ask me many questions that will help you design this with me. I want to give you the best description possible so that this is done well.

It may be worth doing research online for another solid data provider. To start, I think it would be good to focus on US assets and macro indicators. Ideally, they have other useful data such as insider transactions as well. In the future, it should be more global data. I'm using IBKR as my broker and I know they provide data, it just seems expensive.

It would be good to focus on US equities, but while designing I think it's important to design it in a way where it is easy to implement other classes such as credit in the future.

For now, we are going to trust that the data is correct since we are only using one data source. Suggest ideas for handling missing values.

Sector and industry are two categoricals which come to mind, however the point of this dynamic system is that it will identify the categorical data automatically based on the type of the column and the time series differentiator, in this case the ticker. So if it separating by ticker and there is a string column "sector", it will automatically know that is categorical.

Forward fill the data to synchronize. We don't want future data leaking into the past.

For data transformations, calculate some basic technical indicators to start, such as RSI, MACD, EMA50, EMA200, and I'll add more later. Definitely calculate percentage and log returns.

There should be a setting in the json file for granularity of storage with a default value of every 10 epochs.

Store both raw and processed datasets. I have a lot of storage so it's worth storing both if it avoid recalculating in the future.

Store all possible metrics when hypertuning. Models should be compared using MAPE (but can be configured in settings).

The statistical models you mentioned seem great. I would also include linear and exponential regression and FFT.

Test all memory implementations for the titans architecture.

User configurable prediction horizon, defaulting to 90 days.

All of the main metrics for both investing and model accuracy should be stored. MAPE will be the default for comparing model accuracy, and I would like a combination of alpha (compared to S&P) and Sharpe ratio for portfolio metrics.

I've thought about hyper-tuning ranges, and I think it would be smart to have it intelligently set it. By this I mean if the optimal value is within 10% of the start or end of the range, increase or reduce it by 50%, this way it adapts to other changed parameters which could make the optimal value out of range.

Make it easy to graph model accuracy over training time and I will decide the optimal model complexity.

The system should use the automated detection first, then override with any user specified designations.

Forward fill values to make a uniform sequence.

Let's just use streamlit for simplicity.

The charts should be able to do everything you mentioned. Most important is just selecting date ranges and hovering for details.

For top recommendations, it should have different selections for what the best are. It could be simply highest return, risk adjusted, short/long positions only, etc.

Also use error distributions and metric comparisons in the model accuracy dashboard.

When drilling down, I think it would be good if there were a list of tickers/potential investments recommended by the model that when selected it shows the feature importances and uses an LLM which explains in text why these features are important. I won't be using the news so don't worry about that.

The json configuration should be as extensive as possible. Everything should be in the user-defined file, a separate file will store the best settings.

In the best settings json file, it should store everything that will be dynamically updated. This will include hyperparameter ranges as well. We won't have api keys or db connections stored in the best settings JSON file. If in the user file they set "best": true, then the best settings JSON file will overwrite the user defined. Best will be considered by the user "comparison metric" key, defaulting to "MAPE".

There will be a selection of different data transformations, train/test splits, and so forth. If the value is a tuple, it will assume it is a range and optimize for that. If it is a single value, it will not. For this reason, I think it would be best to have the user defined settings actually be a dictionary in a python file so that some values can be lists that are indexed so that you can see all possible options and change the selection by changing the index. For example,

"data transformations": ["z-scale", "minmax", "log"][0],

...