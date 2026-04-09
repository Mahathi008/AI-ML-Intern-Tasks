Day 1: Data Cleaning & Preprocessing

📌 Objective

The objective of this task was to understand how to clean and prepare raw data before applying any machine learning algorithms.

---

📂 Dataset Used

Titanic Dataset (CSV format)

---

🔧 Steps Performed

1. **Loaded the Dataset**

   * Imported the dataset using Pandas.
   * Displayed initial rows to understand the structure.

2. **Explored the Data**

   * Checked data types and null values.
   * Understood which columns had missing data.

3. **Handled Missing Values**

   * Filled missing values in `Age` using median.
   * Filled missing values in `Embarked` using mode.
   * Dropped `Cabin` column due to excessive missing data.

4. **Encoded Categorical Features**

   * Converted `Sex` column into numerical values.
   * Applied one-hot encoding on `Embarked` column.

5. **Feature Scaling**

   * Normalized numerical features (`Age`, `Fare`) using MinMaxScaler.

6. **Outlier Detection and Removal**

   * Used boxplot visualization to detect outliers.
   * Removed outliers using IQR method.

7. **Saved Cleaned Dataset**

   * Exported the final processed dataset as `cleaned_titanic.csv`.

---

🛠 Tools & Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

📊 Output

* Successfully cleaned and preprocessed the dataset.
* Generated a new dataset: `cleaned_titanic.csv`.

---

✅ Conclusion

This task helped in understanding the importance of data preprocessing steps like handling missing values, encoding, scaling, and outlier removal. Proper preprocessing improves the performance of machine learning models.

---
