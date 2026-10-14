# Walmart Weekly Sales Forecasting | Time-Series Analysis with SARIMA

**A Data Science Capstone Project | Retail Analytics | Time-Series Forecasting**

Developed by **Anuj Sandip Jagtap**  
Course: Intellipaat Data Science & AI Certification  
Environment: Google Colab

---

## 1. Project Overview

Retail businesses must balance inventory availability with customer demand. Overstocking increases holding costs and ties up working capital, while understocking can result in lost sales and poor customer experience.

This project analyzes Walmart's historical weekly sales data to identify sales trends, understand the relationship between economic and environmental factors and sales, compare store performance, and forecast future weekly sales using statistical time-series models.

The primary modeling approach is **Seasonal Autoregressive Integrated Moving Average (SARIMA)**, implemented using `SARIMAX` from the Statsmodels library. The fitted model uses historical sales patterns and a 52-week seasonal cycle to generate forecasts for upcoming weeks.

The project demonstrates how historical retail data can be transformed into actionable insights for demand planning, inventory management, and store-level decision-making.

### Key project highlights

- Analyzed **6,435 records across 8 columns** in the Walmart sales dataset.
- Conducted exploratory data analysis (EDA) and statistical analysis of sales trends.
- Investigated unemployment, Consumer Price Index (CPI), temperature, and seasonal sales patterns.
- Identified top- and bottom-performing stores based on historical sales.
- Evaluated AR, ARIMA, and store-specific SARIMA forecasting approaches.
- Generated **12-week sales forecasts** for selected stores.
- Evaluated model predictions using Root Mean Squared Error (RMSE).

---

## 2. Business Problem

Walmart operates multiple stores that experience different demand patterns throughout the year. Retail planners need reliable estimates of upcoming sales to make better operational decisions.

The central business question is:

**How can historical sales patterns and external economic factors be analyzed to forecast weekly store sales and support better inventory and supply-chain planning?**

The project addresses the following business challenges:

1. **Demand uncertainty:** Future weekly sales are unknown, making inventory planning difficult.
2. **Seasonal fluctuations:** Holiday periods can create significant sales spikes that require advance preparation.
3. **Store-level performance variation:** Different stores exhibit different sales levels and demand patterns.
4. **Economic sensitivity:** Changes in unemployment and CPI may be associated with changes in sales.
5. **Resource allocation:** Forecasts can help planners prioritize inventory, distribution, and operational capacity across stores.

### Business objectives

- Understand historical weekly sales behavior.
- Examine seasonal trends and recurring sales peaks.
- Analyze the association between sales and external factors.
- Compare store-level sales performance.
- Develop and evaluate statistical forecasting models.
- Produce future weekly sales estimates that can support planning decisions.

---

## 3. Dataset Description

The project uses the Walmart sales dataset, containing **6,435 rows and 8 columns**.

| Column | Description |
|---|---|
| `Store` | Store identifier |
| `Date` | Date associated with the weekly sales observation |
| `Weekly_Sales` | Total sales recorded for the store during the week |
| `Holiday_Flag` | Indicates whether the observation falls within a designated holiday period |
| `Temperature` | Temperature recorded for the relevant observation |
| `Fuel_Price` | Fuel price indicator |
| `CPI` | Consumer Price Index |
| `Unemployment` | Unemployment rate |

### Why this dataset matters

The dataset combines historical sales with time information and contextual economic or environmental variables. This makes it suitable for retail analytics and time-series forecasting.

The `Date` and `Weekly_Sales` columns form the foundation of the forecasting task, while the remaining fields support exploratory analysis of possible demand drivers.

**Important modeling distinction:** The documented forecasting implementation is a univariate, store-level SARIMAX model using historical weekly sales. Although the dataset includes external variables, their presence in the dataset does not mean they were supplied as exogenous predictors to the fitted forecasting model.

---

## 4. Technology Stack

| Technology | Application |
|---|---|
| Python | Data analysis and model implementation |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data visualization |
| Statsmodels | AR, ARIMA, SARIMA/SARIMAX and statistical diagnostics |
| Scikit-learn | Forecast evaluation metrics |
| Google Colab | Development and experimentation environment |

---

## 5. Project Workflow

The project follows a structured analytics workflow:

1. Business problem definition
2. Data loading and inspection
3. Data cleaning and preprocessing
4. Exploratory data analysis
5. Seasonal pattern identification
6. Store-level time-series preparation
7. Stationarity assessment
8. ACF and PACF analysis
9. Model fitting and forecasting
10. Model evaluation using RMSE
11. Business interpretation and recommendations

This workflow connects analytical findings to a specific operational problem rather than treating model training as an isolated coding exercise.

---

## 6. Data Cleaning and Preprocessing

The initial dataset was inspected to understand its dimensions, data types, and sales behavior.

Key preprocessing activities included:

- Inspecting the dataset's structure and numeric variables.
- Converting the `Date` column to a proper datetime format.
- Organizing observations chronologically for time-series analysis.
- Examining zero or negative sales observations and their potential effect on modeling.
- Filtering records by store to create individual weekly sales series.
- Resetting and setting date indexes where required for time-series operations.
- Separating historical observations into training and testing periods.

### Why chronological ordering is essential

Time-series observations have a temporal dependency. Randomly shuffling the data could allow future observations to influence model development and produce an unrealistic evaluation.

A chronological train-test split instead evaluates how well the model predicts observations that occur after its training period.

**Data-quality consideration:** Zero or negative sales values should be investigated before applying transformations or removing observations. They may reflect data issues or unusual business events. The appropriate treatment should depend on the underlying data, and log transformations require particular care when values are non-positive.

---

## 7. Exploratory Data Analysis (EDA)

EDA was used to understand the behavior of weekly sales, identify recurring patterns, and examine differences across stores.

### 7.1 Weekly sales trends

Time-series plots revealed recurring fluctuations in weekly sales, including pronounced peaks during year-end holiday periods.

These observations indicate that weekly sales are not constant over time. Demand planning should therefore account for both ordinary fluctuations and predictable seasonal periods.

### 7.2 Seasonality and holiday effects

The analysis identified strong seasonal behavior, with major peaks associated with holiday shopping periods such as Thanksgiving and Christmas.

From a retail perspective, these peaks matter because sales increases may require advance inventory replenishment, distribution planning, and staffing adjustments.

A forecasting model that accounts for recurring seasonal patterns is more appropriate for this problem than a model that assumes sales follow only a constant average.

### 7.3 Store-level performance

Historical sales were examined across stores to identify stronger- and weaker-performing locations.

Store-level comparisons help answer questions such as:

- Which stores consistently generate higher weekly sales?
- Which locations may require closer performance monitoring?
- Do stores have substantially different sales scales?
- Should forecasting be performed separately for individual stores?

### 7.4 Economic and environmental factors

The analysis considered unemployment, CPI, and temperature.

The documented findings were:

- **Unemployment:** Sales and unemployment were found to be associated, with some stores appearing more sensitive to economic conditions.
- **CPI:** CPI was identified as a factor associated with differences in sales behavior across stores.
- **Temperature:** Temperature showed minimal impact on weekly sales in the project's analysis.

These findings should be interpreted as observations from the analyzed dataset, not proof that any one variable directly causes sales to increase or decrease.

---

## 8. Forecasting Methodology

Three modeling approaches were explored: Autoregressive (AR), ARIMA, and seasonal ARIMA.

### 8.1 Autoregressive (AR)

An AR model predicts a value using its previous observations.

It provides a useful baseline for assessing how much historical sales information can explain future sales.

### 8.2 ARIMA

ARIMA combines three components:

- **AR — Autoregression:** Uses relationships with previous observations.
- **I — Integration:** Applies differencing when required to achieve stationarity.
- **MA — Moving Average:** Models relationships with past forecast errors.

ARIMA can capture temporal patterns but does not explicitly model a repeating seasonal cycle in its standard form.

### 8.3 SARIMA / SARIMAX

SARIMA extends ARIMA by incorporating seasonal autoregressive, differencing, and moving-average components.

The implementation uses Statsmodels' `SARIMAX` class with the following configuration:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    train,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 52)
)

fitted_model = model.fit()
```

The model was fitted to the training data for each selected store.

**Model configuration**

| Parameter | Value | Interpretation |
|---|---:|---|
| `p` | 1 | One autoregressive term |
| `d` | 0 | No non-seasonal differencing |
| `q` | 1 | One moving-average term |
| `P` | 1 | One seasonal autoregressive term |
| `D` | 0 | No seasonal differencing |
| `Q` | 1 | One seasonal moving-average term |
| `s` | 52 | Annual seasonal period for weekly observations |

The seasonal period of 52 represents approximately one year of weekly observations.

This configuration was selected to model short-term dependencies alongside recurring annual patterns in weekly sales.

### Why SARIMA is appropriate

The EDA identified pronounced seasonal sales peaks. SARIMA provides a statistical framework for learning recurring temporal behavior from historical sales and incorporating it into future estimates.

The model is therefore aligned with the project's central forecasting challenge: anticipating future weekly sales when historical demand contains repeating patterns.

### ACF and PACF diagnostics

Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots were used to inspect temporal dependencies and support model specification.

- ACF measures correlation between observations and their lagged values.
- PACF measures the relationship at a given lag after accounting for shorter lags.

These plots help guide the selection of candidate AR and MA orders. They should be considered alongside model diagnostics and out-of-sample performance rather than treated as automatic proof that a particular model is optimal.

---

## 9. Training, Testing, and Forecast Generation

The forecasting workflow was implemented separately for selected stores.

For each store, the historical weekly sales series was prepared and divided into chronological training and testing periods. The SARIMA model was fitted using the training data, and predictions were generated for the held-out testing observations.

The predictions were compared with the corresponding actual sales to evaluate forecasting performance.

### Example: generating predictions

```python
train_length = len(train)

test_predictions = fitted_model.predict(
    start=train_length,
    end=len(full_series) - 1
)
```

The exact indexes should match the chronological training and testing series used in the notebook.

### Generating the next 12 weeks

After fitting the model, future sales were estimated using:

```python
future_forecast = fitted_model.forecast(steps=12)
```

The forecast was assigned weekly dates to make the predictions interpretable as future sales estimates.

### Forecast output

For each modeled store, the output consists of weekly predicted sales values for the forecast horizon.

These values can help a retail planner estimate upcoming demand and identify possible increases or decreases in expected sales.

**Scope clarification:** The report documents 12-week forecasting for selected stores, including Stores 2, 16, and 21. It does not establish that a production forecasting pipeline has been deployed or that forecasts have been generated and validated for every Walmart store.

---

## 10. Model Evaluation and Results

Forecasts were evaluated using Root Mean Squared Error (RMSE).

### What RMSE measures

RMSE measures the typical magnitude of prediction errors, with larger errors receiving greater weight because errors are squared before averaging.

\[
\mathrm{RMSE} =
\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

Where:

- \(y_i\) represents actual weekly sales.
- \(\hat{y}_i\) represents predicted weekly sales.
- \(n\) represents the number of evaluated observations.

A lower RMSE indicates better predictive performance when models are evaluated on the same target series and test observations.

### Documented RMSE results

The following values are taken from the project report.

| Model | RMSE |
|---|---:|
| AR | 93,805.31 |
| ARIMA | 93,643.70 |
| SARIMA — Store 2 | 71,022.96 |
| SARIMA — Store 21 | 40,489.66 |
| SARIMA — Store 16 | 19,004.62 |

### Interpretation

The reported ARIMA RMSE is slightly lower than the AR RMSE, indicating a modest improvement in the documented baseline comparison.

The report also records lower RMSE values for the store-specific SARIMA models. However, the store-specific results should not be interpreted as a direct ranking of model quality across stores because the stores have different sales levels and may have different test periods.

The meaningful comparison is between models evaluated on the **same store and the same held-out observations**.

### What these results demonstrate

- Historical weekly sales contain information that can be used to forecast future sales.
- Seasonal modeling is a promising approach for the dataset's recurring demand patterns.
- Forecast evaluation can identify differences in predictive performance.
- Store-specific modeling makes it possible to account for differences in individual sales series.

### Limitations of the evaluation

The documented report provides RMSE values but does not provide a complete, comparable set of MAE, MAPE, or benchmark results for every store.

It also does not establish a measured percentage improvement in forecasting accuracy, inventory cost savings, or revenue improvement.

Those claims would require additional validation and, for operational savings, real business outcome data.

---

## 11. How the Fitted Model Solves the Walmart Sales Problem

The model supports the business problem by translating historical weekly sales into estimates of future demand.

### Challenge 1: Uncertain future demand

**Approach:** Fit a seasonal time-series model to historical sales.

**Business value:** Store planners receive a forward-looking sales estimate instead of relying exclusively on historical averages or intuition.

### Challenge 2: Holiday-related demand spikes

**Approach:** Model recurring annual patterns using a seasonal period of 52 weeks.

**Business value:** Forecasts can help identify periods when additional inventory, distribution capacity, or staffing may be needed.

The model does not guarantee that every holiday spike will be predicted accurately, particularly when the underlying event changes from year to year.

### Challenge 3: Differences between stores

**Approach:** Prepare individual sales series and fit models for selected stores.

**Business value:** Forecasts reflect each selected store's historical sales behavior rather than assuming that all stores have identical demand patterns.

### Challenge 4: Forecast reliability

**Approach:** Compare predicted sales with held-out actual sales and calculate RMSE.

**Business value:** Forecast quality becomes measurable, allowing the forecasting approach to be evaluated before it is used for operational decisions.

### Challenge 5: Inventory and supply-chain planning

**Approach:** Generate 12-week future sales estimates.

**Business value:** These forecasts can serve as an input to replenishment planning, inventory allocation, and distribution scheduling.

The current project provides a forecasting foundation. Actual inventory optimization would require additional information such as current inventory, supplier lead times, stockout history, product-level demand, and service-level targets.

---

## 12. Key Business Insights

The project's findings support the following conclusions:

1. **Sales exhibit seasonality.** Year-end holiday periods contain pronounced sales peaks, making seasonal forecasting relevant to retail planning.
2. **Store performance varies.** Historical sales differ across stores, supporting store-level performance analysis.
3. **Economic context matters.** Unemployment and CPI are associated with sales patterns in the analyzed data, although the strength of these relationships can vary by store.
4. **Temperature has limited observed influence.** The analysis found minimal impact on weekly sales.
5. **Forecasting can support planning.** Store-specific sales predictions provide a quantitative basis for anticipating demand.
6. **Evaluation is essential.** RMSE provides a measurable way to compare predicted and actual sales on held-out data.

---

## 13. Recommendations for Retail Decision-Makers

The results suggest several practical ways a retail organization could use this analytical approach.

| Recommendation | Potential application |
|---|---|
| Prepare for seasonal demand | Review forecasts before major holiday periods and plan replenishment early |
| Allocate inventory by store | Use store-level demand estimates as one input to distribution decisions |
| Monitor forecast errors | Track predicted versus actual sales and investigate recurring errors |
| Investigate weaker-performing stores | Combine historical sales comparisons with local operational information |
| Evaluate economic indicators | Explore whether economic changes improve forecasts for particular stores |
| Review forecasts regularly | Retrain or reassess models as new weekly sales observations become available |

These are potential applications of the project, not measured operational outcomes.

A retail organization should validate forecasts against a suitable benchmark before using them to determine inventory or staffing levels.

---

## 14. Project Limitations

This project is a statistical forecasting capstone developed in Google Colab. Several limitations should be considered when interpreting its results.

- **Selected-store scope:** The documented model evaluation covers selected stores rather than demonstrating a fully validated, company-wide forecasting system.
- **Limited model metrics:** RMSE is documented, but additional metrics would provide a more complete view of forecast quality.
- **External predictors:** Although unemployment, CPI, temperature, fuel price, and holiday information are available in the dataset, the documented SARIMAX specification uses the historical sales series without explicit exogenous regressors.
- **Forecast uncertainty:** Point forecasts alone do not quantify the full range of plausible future sales.
- **Changing business conditions:** Unexpected events, promotions, supply disruptions, or structural changes can reduce forecast reliability.
- **Operational validation:** The project does not measure actual inventory savings, reduced stockouts, or additional revenue after implementation.
- **Historical dataset:** Results on the supplied historical data do not automatically guarantee performance on more recent Walmart sales.

These limitations define the boundary between a forecasting prototype and a production-ready retail demand-planning solution.

---

## 15. Future Enhancements

The following enhancements could make the project more robust and useful for real-world deployment.

### 15.1 Improve model validation

- Compare models on identical chronological test windows.
- Introduce rolling-origin or walk-forward validation.
- Compare against seasonal-naive and other simple baselines.
- Evaluate MAE, RMSE, and MAPE where the target values make MAPE appropriate.
- Inspect residuals for remaining autocorrelation and systematic forecasting errors.

### 15.2 Incorporate external variables

Evaluate whether holiday flags, CPI, unemployment, fuel price, and temperature improve forecasts when used as exogenous predictors in SARIMAX.

Any external variables used for forecasting must be available or forecastable at the time the prediction is made. Using future values that would not have been known in practice could create data leakage.

### 15.3 Expand store coverage

Develop a repeatable pipeline that fits, validates, and records forecasting results for all eligible stores.

Store-specific models could be compared with pooled or global forecasting approaches to determine whether sharing information across stores improves reliability.

### 15.4 Quantify forecast uncertainty

Generate prediction intervals so planners can evaluate a range of plausible demand outcomes rather than relying on point forecasts alone.

### 15.5 Build an interactive dashboard

Create a Power BI or Streamlit interface with:

- Historical weekly sales trends.
- Store-level performance comparisons.
- Actual-versus-predicted sales.
- Twelve-week forecasts.
- Model evaluation metrics.
- Forecast uncertainty intervals.

### 15.6 Connect forecasts to inventory decisions

Combine sales forecasts with inventory on hand, replenishment lead times, and service-level requirements to estimate reorder quantities and identify potential stockout risks.

This would extend the project from demand forecasting into inventory decision support.

---

## 16. How to Run the Project

The original project was developed in Google Colab.

### Step 1: Open the notebook

Upload the project notebook to Google Colab or open it from your repository.

### Step 2: Prepare the dataset

Place `Walmart DataSet.csv` in the working environment and update the file path in the notebook as necessary.

### Step 3: Install dependencies if required

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

### Step 4: Execute the notebook

Run the cells in sequence to load and clean the data, explore sales trends, prepare store-specific time series, fit the models, and evaluate the predictions.

### Step 5: Review the outputs

Inspect the sales visualizations, ACF/PACF plots, actual-versus-predicted comparisons, RMSE table, and 12-week forecast plots.

**Repository note:** Update the dataset path and notebook filename to match the files actually committed to your repository. The CSV is not included automatically by this README.

---

## 17. Suggested Repository Structure

```text
walmart-sales-forecasting/
│
├── README.md
├── notebooks/
│   └── walmart_sales_forecasting.ipynb
│
├── data/
│   └── README.md
│
├── outputs/
│   ├── sales_trends.png
│   ├── forecast_comparison.png
│   └── model_evaluation.png
│
└── requirements.txt
```

This is a suggested structure, not a claim about the current contents of the repository.

For a public GitHub repository, add the notebook, requirements, and selected visual outputs. If dataset redistribution is not permitted, provide instructions for obtaining the data rather than committing the raw CSV.

---

## 18. Skills Demonstrated

**Data Analysis**
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Statistical analysis
- Time-series visualization
- Store-level performance analysis

**Machine Learning and Statistical Modeling**
- Time-series forecasting
- AR, ARIMA, and SARIMA/SARIMAX
- Stationarity assessment
- ACF and PACF diagnostics
- Chronological train-test evaluation
- Forecast error analysis

**Business and Analytical Thinking**
- Retail demand forecasting
- Seasonality analysis
- Economic indicator analysis
- Inventory and supply-chain planning
- Translating model outputs into business recommendations

**Tools**
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels
- Scikit-learn
- Google Colab

---

## 19. Conclusion

This project demonstrates an end-to-end analytical approach to a real-world retail problem: forecasting weekly sales to support demand planning.

By examining 6,435 historical records, identifying recurring seasonal patterns, comparing statistical forecasting approaches, and evaluating store-specific SARIMA predictions, the project establishes a practical foundation for data-driven retail planning.

The most important contribution is the connection between time-series modeling and business decision-making. Twelve-week sales forecasts can help planners anticipate demand, while model evaluation provides a basis for assessing forecast reliability before operational adoption.

The project also establishes a clear path for further development through stronger validation, external-variable modeling, uncertainty estimation, broader store coverage, and interactive reporting.

**The ultimate goal is to move from understanding historical sales to making better-informed decisions about future retail demand.**

---

## Author

**Anuj Sandip Jagtap**

Data Science & AI Certification — Intellipaat

LinkedIn: https://www.linkedin.com/in/anuj-jagtap2004

---

*This project was developed for educational and portfolio purposes. The findings and forecasts are based on the analyzed historical dataset and should not be interpreted as official Walmart forecasts or verified financial outcomes.*
