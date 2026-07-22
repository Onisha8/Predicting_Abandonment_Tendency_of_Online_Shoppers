# Predicting Abandonment Tendency of Online Shoppers

## Project Overview  
Cart abandonment is a major challenge for e-commerce businesses, resulting in significant revenue loss. This project uses machine learning techniques to **predict whether an online shopper will abandon their cart**, helping businesses implement proactive strategies to reduce abandonment and increase conversions.

## Objective  
- Build predictive models to classify sessions as “purchase” or “no purchase”  
- Evaluate performance across models: **Logistic Regression, KNN, Decision Tree, Random Forest, and XGBoost**  
- Identify key behavioral features influencing cart abandonment  
- Recommend actionable strategies to enhance customer retention and ROI

## Setup & Usage
```bash
pip install -r requirements.txt
python main.py          # trains all 5 models, saves the best (by macro F1) to models/best_model.joblib
```
The repo has two parallel pieces of work:
- **`notebooks/final_model.ipynb`** — exploratory analysis with per-model `GridSearchCV` hyperparameter tuning.
- **`src/` + `main.py`** — a runnable script version: leak-free preprocessing (scaling/encoding fit only on the training fold, inside a single `sklearn`/`imblearn` `Pipeline`), SMOTE applied only within that pipeline, and the best model persisted with `joblib`.

These use default (untuned) hyperparameters in `src/`, so their numbers won't exactly match the notebook's tuned numbers — both are legitimate, just not directly comparable. Tuning the `src/` pipeline with `GridSearchCV`/`RandomizedSearchCV` is a natural next step (see Future Work).

## Motivation  
With rising competition in e-commerce, predicting user behavior is essential. By identifying high-risk abandonment users early, businesses can tailor engagement tactics like personalized offers or improved UX, ultimately increasing profitability.

## Dataset  
- **Source**: [Online Shoppers Purchasing Intention Dataset]  
- **Size**: 12,330 unique user sessions  
- **Features**: 18 total  
  - 10 numerical (e.g., Bounce Rate, Exit Rate, Page Value)  
  - 8 categorical (e.g., VisitorType, Month)  
- **Target Variable**: Revenue (True = Purchase, False = No Purchase)

## Preprocessing Steps  
- Checked for missing values (none found)  
- Applied **one-hot encoding** for categorical features  
- Used **StandardScaler** for numerical features  
- Addressed **class imbalance** (only 15.5% sessions were purchases) using oversampling

## Models Trained  
| Model               | Notes                                                                 |
|---------------------|-----------------------------------------------------------------------|
| Logistic Regression | Simple, interpretable baseline model                                  |
| K-Nearest Neighbors | Captures local data patterns but is sensitive to noise                |
| Decision Tree       | High interpretability, moderate performance                           |
| Random Forest       | Robust ensemble model, good feature importance insights               |
| **XGBoost**         | **Best performer** in terms of F1-score, precision, recall, and AUC    |

## Evaluation Metrics  
- **F1-Score** (Primary metric)  
- Precision  
- Recall  
- AUC-ROC  
- Accuracy (limited value due to class imbalance)

> **XGBoost Performance (test set, `src/` pipeline with leak-free scaling + SMOTE-in-pipeline)**  
> - Accuracy: 90%
> - Class `Revenue=False` (no purchase): Precision 93%, Recall 95%, F1 94% (support 2,084)
> - Class `Revenue=True` (purchase): Precision 69%, Recall 62%, F1 65% (support 382)
> - Macro F1 (both classes weighted equally): 0.80
> - AUC-ROC: 0.92
>
> **Read this carefully**: the 93%/95%/94% numbers above are for the *majority* class only (no-purchase sessions, 84.5% of the data), which is the easier class to get right. The harder, more decision-relevant class — sessions that end in a purchase — is predicted with 69% precision and 62% recall. Reporting only the majority-class numbers as "the" model performance is misleading, so both are shown here. An earlier version of this README quoted a single AUC-ROC of 0.93 that did not match either pipeline's actual output; the number above is reproduced directly from a fresh run of `main.py` (see `models/metrics.json`).

## Key Findings  
- **XGBoost outperformed** all other models, making it ideal for deployment  
- Important predictors:  
  - Product-Related Duration  
  - Page Value  
  - Special Days (holidays, events)  

## Business Implications  
- **Marketing Optimization**: Focus resources on users predicted to complete purchases  
- **Personalization**: Tailor experience based on session behavior  
- **Revenue Boost**: Reduce abandoned carts → increased conversions

## Challenges  
- Imbalanced dataset required careful balancing of precision and recall  
- Computational cost of tuning ensemble models (especially XGBoost)  
- Session-only data limited long-term behavioral analysis

## Future Work  
- Integrate user- and product-level features (e.g., loyalty, categories, discounts)  
- Explore deep learning models (CNNs, RNNs)  
- Deploy model in real-time scoring engine  
- Conduct A/B testing to measure business impact
- Run `GridSearchCV`/`RandomizedSearchCV` on the leak-free `src/` pipeline (the notebook's tuning was done before the preprocessing fix)
- Tune the decision threshold instead of using the default 0.5 cutoff — since false positives (wrongly flagging a would-be buyer as an abandoner) and false negatives (missing a real abandoner) likely have different business costs
- Add cross-validated metrics (not just a single train/test split) to get a variance estimate on performance

## Final Recommendation  
Use **XGBoost** to power abandonment prediction systems in e-commerce platforms. Combine model insights with personalization and retention strategies to enhance customer experience and maximize ROI.
