# Predicting Abandonment Tendency of Online Shoppers

## Project Overview  
Cart abandonment is a major challenge for e-commerce businesses, resulting in significant revenue loss. This project uses machine learning techniques to **predict whether an online shopper will abandon their cart**, helping businesses implement proactive strategies to reduce abandonment and increase conversions.

## Objective  
- Build predictive models to classify sessions as “purchase” or “no purchase”  
- Evaluate performance across models: **Logistic Regression, KNN, Decision Tree, Random Forest, and XGBoost**  
- Identify key behavioral features influencing cart abandonment  
- Recommend actionable strategies to enhance customer retention and ROI

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

> **XGBoost Performance**  
> - Accuracy: 90%  
> - Precision: 93%  
> - Recall: 96%  
> - F1-Score: 94%  
> - AUC-ROC: 0.93  

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

## Final Recommendation  
Use **XGBoost** to power abandonment prediction systems in e-commerce platforms. Combine model insights with personalization and retention strategies to enhance customer experience and maximize ROI.
