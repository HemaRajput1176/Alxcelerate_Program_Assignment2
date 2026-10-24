# Load Type Prediction --- USEReady ML Assignment

## 1. Project overview

This project builds a machine learning classifier to predict the **load
type of a power system** from historical energy and electrical
measurements.

The target column is `Load_Type`, with three classes:

-   `Light_Load`
-   `Medium_Load`
-   `Maximum_Load`

The goal is not only to achieve a strong score, but to evaluate the
model in a way that resembles a real prediction task: train on earlier
observations and test on the latest month.

## 2. Problem statement

Power-system readings can vary with energy usage, reactive power, power
factor, emissions, and time. The task is to use the available
measurements to predict which load category a reading belongs to.

This is a **multiclass classification** problem because the target has
three categories rather than a continuous numeric value.

## 3. Dataset at a glance

The supplied CSV contains **35,041 rows and 9 columns**. Its timestamps
cover **1 January 2018 to 31 December 2018**.

  Field                                    Meaning
  ---------------------------------------- -------------------------------------------------
  `Date_Time`                              Timestamp of the observation
  `Usage_kWh`                              Energy consumption
  `Lagging_Current_Reactive.Power_kVarh`   Lagging reactive power
  `Leading_Current_Reactive_Power_kVarh`   Leading reactive power
  `CO2(tCO2)`                              CO₂-related measurement as named in the dataset
  `Lagging_Current_Power_Factor`           Lagging power factor
  `Leading_Current_Power_Factor`           Leading power factor
  `NSM`                                    Number of seconds from midnight
  `Load_Type`                              Target class to predict

### Target distribution in the supplied file

  Load type          Rows    Share
  -------------- -------- --------
  Light_Load       18,073   51.58%
  Medium_Load       9,696   27.67%
  Maximum_Load      7,272   20.75%

`Light_Load` is the most common class. This imbalance is one reason to
report macro-F1 and per-class recall in addition to accuracy.

### Initial data-quality observations

The raw file contains missing values in several measurement columns:

  Column                                     Missing values
  ---------------------------------------- ----------------
  `Usage_kWh`                                         1,559
  `Lagging_Current_Reactive.Power_kVarh`                876
  `Leading_Current_Reactive_Power_kVarh`              1,156
  `CO2(tCO2)`                                           455
  `Lagging_Current_Power_Factor`                        350
  `Leading_Current_Power_Factor`                      1,471
  `NSM`                                                 455

There is also one repeated timestamp. The notebook inspects repeated
timestamps and power-factor values before deciding how to handle them.
These checks matter because duplicate or implausible measurements can
distort the analysis.

## 4. Approach

The notebook follows a chronological workflow.

1.  **Load and inspect the data.** Check the columns, data types,
    missing values, duplicates, and date range.
2.  **Explore the target and features.** Review class balance, feature
    distributions, energy usage by class, correlations, and patterns
    over time.
3.  **Clean suspicious records carefully.** Resolve the repeated
    timestamp using a documented quality check and treat power-factor
    values outside the expected 0--100 range as missing.
4.  **Create time features.** Derive cyclical time-of-day features, day
    of week, and month from the timestamp. The raw `NSM` field is
    excluded from model inputs because some values disagree with the
    timestamp.
5.  **Split by time.** Use earlier months for training, November for
    validation/model selection, and December as the final test month.
6.  **Build a baseline and candidate models.** Compare a majority-class
    baseline, Logistic Regression, Random Forest, and Hist Gradient
    Boosting.
7.  **Select using validation results.** Choose the candidate with the
    highest November macro-F1.
8.  **Evaluate once on the final month.** Refit the selected model on
    all data before December, then report December performance and
    inspect misclassifications.

## 5. Why use a chronological split?

A random split can place readings from later dates in training while
earlier readings are used for testing. That can give an overly
optimistic estimate when data changes over time.

This project uses: - **Training:** all observations before November
2018 - **Validation:** November 2018 - **Test:** December 2018

The notebook derives the final month from the data rather than
hard-coding it. December is not used to choose features, tune the model,
or select the winning candidate.

## 6. Features and preprocessing

The model uses the available energy and electrical measurements, plus
timestamp-derived features:

-   Energy consumption and reactive power
-   CO₂-related measurement
-   Lagging and leading power factors
-   `time_sin` and `time_cos`, which represent time of day cyclically
-   Day of week and month

The target `Load_Type` is not included among the input features.

Missing numerical values are handled with median imputation inside
scikit-learn pipelines. For Logistic Regression, numerical features are
also scaled. The imputer and scaler are fitted on the training portion
of each model workflow, rather than on the full dataset.

A key modelling question is whether `Usage_kWh` nearly determines the
target because the original labels may have been assigned using
consumption thresholds. The EDA compares usage ranges across classes. If
the target is rule-derived, a high score may show that the model learned
that rule rather than proving broader real-world forecasting ability.

## 7. Models compared

  -----------------------------------------------------------------------
  Model                               Why it is included
  ----------------------------------- -----------------------------------
  Majority-class baseline             Establishes a simple reference
                                      score

  Logistic Regression                 Provides a straightforward linear
                                      baseline for multiclass
                                      classification

  Random Forest                       Captures nonlinear relationships
                                      and feature interactions

  Hist Gradient Boosting              Provides another tree-based
                                      approach that can model nonlinear
                                      patterns
  -----------------------------------------------------------------------

The models are compared on the same November validation period. The
final candidate is selected using **validation macro-F1**, which gives
each class equal weight. The notebook then evaluates that selected model
on December.

## 8. Evaluation metrics

The notebook reports:

-   **Accuracy:** fraction of all predictions that are correct.
-   **Precision by class:** among readings predicted as a class, how
    many truly belong to it.
-   **Recall by class:** among readings that truly belong to a class,
    how many the model finds.
-   **F1-score by class:** balance between precision and recall.
-   **Macro-F1:** average F1 across the three classes, weighting each
    class equally.
-   **Weighted-F1:** average F1 weighted by each class's support.
-   **Confusion matrix:** shows correct predictions and which classes
    are confused.
-   **Misclassification review:** displays examples where the predicted
    label differs from the recorded label.

Final metric values should be copied from the notebook after running all
cells. They are intentionally not hard-coded here, because they depend
on the executed notebook and environment.

## 9. Analytics and interpretation

The supplied data has a noticeable class imbalance: just over half of
the readings are labelled `Light_Load`, while around one fifth are
`Maximum_Load`. A model could therefore obtain a misleadingly reasonable
accuracy by favouring the most common class, which makes the baseline
and macro-F1 useful.

The dataset also contains missing measurements and a repeated timestamp,
so preprocessing decisions are part of the modelling problem---not just
housekeeping. The time split makes the final evaluation more realistic
for later observations, although one test month cannot prove that
performance will remain stable across other seasons or operating
conditions.

The notebook should be interpreted after execution. In particular,
check: - Whether usage ranges overlap between load classes. - Whether
validation and test scores differ substantially. - Which class has the
lowest recall or F1. - Which class pairs appear most often in the
confusion matrix. - Whether the model's inputs would be available at the
actual time a prediction is needed.



## 10. Limitations

-   The evaluation uses one final month, so results may not represent
    every season or operating condition.
-   The meaning and generation process of `Load_Type` should be
    confirmed with the data owner.
-   Missing and inconsistent measurements can affect both model quality
    and interpretation.
-   Predictive performance on this dataset does not, by itself,
    establish readiness for deployment in a live power system.

## 11. How to run

1.  Open the notebook in Google Colab or Jupyter.
2.  Upload `load_data.csv` when prompted, or place it at the configured
    path.
3.  Run all cells from top to bottom.
4.  Review the generated graphs, validation comparison, final test
    metrics, and error examples.
5.  Update this README with the actual final test results after running
    the notebook.

## Project files

``` text
project/
├── USEReady_Load_Type_Classification.ipynb
├── load_data.csv
└── README.md
```

## Final note

This project focuses on a clear and reproducible modelling process:
understand the data, compare reasonable candidates, select using
validation data, and keep the final test month separate. The final score
matters, but the reasoning behind the score matters just as much.
