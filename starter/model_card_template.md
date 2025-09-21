# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

**Developed by:** Ken Adachi
**Model Date:** September 2025
**Model version:** 1.0
**Model type:** Random Forest Classifier
**Framework:** scikit-learn

## Intended Use

**Primary intended uses:** This model is designed for educational purposes to demonstrate machine learning model deployment, API creation, and MLOps best practices. It predicts binary income classification (<=50K, >50K) based on demographic features.

## Training Data

**Dataset:** Adult Census Income Dataset from UCI Machine Learning Repository
**Source:** 1994 Census database

## Evaluation Data

**Dataset:** 20% holdout set from Adult Census Income Dataset

The evaluation dataset represents the same population as the training data, randomly split to ensure unbiased performance assessment.


## Metrics
**Model performance on overall test set:**
Based on the trained Random Forest model, the following metrics were achieved:
- **Precision:** 0.7156
- **Recall:** 0.6349 
- **F1-score:** 0.6729
- **Accuracy:** 0.8543

**Slice performance analysis on education categories:**
The model shows significant performance variation across education levels, revealing potential bias issues:

**High-performing groups:**
- Professional school: F1=0.882682(118 samples)
- Doctorate: F1=0.848921(91 samples)
- Masters: F1=0.835821(350 samples)

**Lower-performing groups:**
- High school graduate: F1=0.488330 (2074 samples) -largest groups
- Lower education levels show inconsistent performance with high variance due to small sample sizes and class imbalances

## Ethical Considerations

**Historical bias concerns:** The model is trained on 1994 census data, which may not reflect current socioeconomic conditions and could perpetuate historical inequalities in income distribution across demographic groups.

**Performance disparities:** Significant variation in model performance across education levels could lead to unfair predictions, particularly disadvantaging individuals with lower education levels despite other qualifying factors.

**Protected characteristics:** The dataset contains sensitive attributes including race, sex, and national origin. While these are not direct features in the model, correlated features may indirectly encode these characteristics, potentially leading to discriminatory outcomes.

**Sample size imbalances:** Some education categories have very small sample sizes (e.g., Preschool: 6 samples), making predictions for these groups unreliable and potentially biased.

**Limitations of binary classification:** The $50K threshold may not account for geographic cost-of-living differences or inflation since 1994, potentially misrepresenting current economic realities.

## Caveats and Recommendations

**Major limitations:**
- Model trained on 30-year-old data may not generalize to current population demographics or economic conditions
- Strong performance disparity across education levels indicates systemic bias requiring mitigation
- Small sample sizes for certain categories reduce prediction reliability and statistical significance
- Binary income threshold does not reflect modern economic complexity

**Recommendations for improvement:**
- Implement bias mitigation techniques such as fairness constraints during training
- Use more recent demographic and economic data when available
- Consider ensemble methods that account for demographic fairness
- Apply post-processing techniques to achieve demographic parity across protected groups
- Regular bias auditing using frameworks like Aequitas or AI Fairness 360
