# Assignment 04 Interpretation Memo

**Student Name:** Makenzie Tatum
**Date:** 02/13/2026
**Assignment:** REIT Annual Returns and Predictors (Simple Linear Regression)

---

## 1. Regression Overview

You estimated **three** simple OLS regressions of REIT *annual* returns on different predictors:

| Model | Y Variable | X Variable | Interpretation Focus |
|-------|------------|------------|----------------------|
| 1 | ret (annual) | div12m_me | Dividend yield |
| 2 | ret (annual) | prime_rate | Interest rate sensitivity |
| 3 | ret (annual) | ffo_at_reit | FFO to assets (fundamental performance) |

For each model, summarize the key results in the sections below.

---

## 2. Coefficient Comparison (All Three Regressions)

**Model 1: ret ~ div12m_me**
- Intercept (β₀): [0.1082] (SE: [0.006], p-value: [0.000])
- Slope (β₁): [-0.0687] (SE: [0.032], p-value: [0.035])
- R²: [0.002] | N: [2527]

**Model 2: ret ~ prime_rate**
- Intercept (β₀): [0.1998] (SE: [0.016], p-value: [0.000])
- Slope (β₁): [-0.0194] (SE: [0.003], p-value: [0.000])
- R²: [0.016] | N: [2527]

**Model 3: ret ~ ffo_at_reit**
- Intercept (β₀): [0.0973] (SE: [0.009], p-value: [0.000])
- Slope (β₁): [0.5770] (SE: [0.567], p-value: [0.309])
- R²: [0.000] | N: [2518]

*Note: Model 3 may have fewer observations if ffo_at_reit has missing values; statsmodels drops those rows.*

---

## 3. Slope Interpretation (Economic Units)

**Dividend Yield (div12m_me):**
- A 1 percentage point increase in dividend yield (12-month dividends / market equity) is associated with a [-0.687] change in annual return.
- [Your interpretation: Is higher dividend yield associated with higher or lower returns? Why might this be?]
The higher dividend yield is associated with lower returns because higher investments can cut dividends.
**Prime Loan Rate (prime_rate):**
- A 1 percentage point increase in the year-end prime rate is associated with a [-0.0194] change in annual return.
- [Your interpretation: Does the evidence suggest REIT returns are sensitive to interest rates? In which direction?]
The evidence shows that the REIT returns are sensitive to the interest rates in a negative direction.
**FFO to Assets (ffo_at_reit):**
- A 1 unit increase in FFO/Assets (fundamental performance) is associated with a [0.5770] change in annual return.
- [Your interpretation: Do more profitable REITs (higher FFO/Assets) earn higher returns?]
Yes the more profitable REITs show higher returns when looking at the scatter plot for ffo. The higher it becomes, the less of lower returns there are. 
---

## 4. Statistical Significance

For each slope, at the 5% significance level:
- **div12m_me:** [Significant] — It is significant because the p value is 3%, which is less than the 5% significance level.
- **prime_rate:** [Significant] — It is significant because the p value is 0%, which is less than the 5% significance level.
- **ffo_at_reit:** [Not significant] — It is not signifcant because the p value 30.9%, which is less than the 5% significance level.

**Which predictor has the strongest statistical evidence of a relationship with annual returns?** [Your answer]

--- The prime_rate has the strongest statistical evidence of a relationship with annual returns because it has a p value of 0%, meaning it is extremely significant and provides the strongest evidence.

## 5. Model Fit (R-squared)

Compare R² across the three models:
- [Your interpretation: Which predictor explains the most variation in annual returns? Is R² high or low in general? What does this suggest about other factors driving REIT returns?]
The prime_rate shows the most variation in annual returns because it has the highest R2 at 1.6%, which means that only 1.6% of the variation is explained. This suggests that there are several other factors that are driving REIT returns.
---

## 6. Omitted Variables

By using only one predictor at a time, we might be omitting:
- [Variable 1]: Dividend yields matter because it reflect risk
- [Variable 2]: Prime loan rates matter because it can affects costs.
- [Variable 3]: ffo/assets matter because it can affect returns. 

**Potential bias:** If omitted variables are correlated with both the X variable and ret, our slope estimates may be biased. [Brief discussion of direction if possible]
The slope estimate would be in the direction of the bias.
---

## 7. Summary and Next Steps

**Key Takeaway:**
[2-3 sentences summarizing which predictor(s) show the strongest relationship with REIT annual returns and whether the evidence is consistent with economic theory]
The predictors with the strongest relationship with REIT annual returns would be interest rates and the dividend yields. The evidence is fairly consist with economic theory, however, their r2 indicate that there is not much explained of the variability, meaning their are other factors we need to consider.
**What we would do next:**
- Extend to multiple regression (include two or more predictors)
- Test for heteroskedasticity and other OLS assumption violations
- Examine whether relationships vary by time period or REIT sector

---

## Reproducibility Checklist
- [Yes] Script runs end-to-end without errors
- [Yes] Regression output saved to `Results/regression_div12m_me.txt`, `regression_prime_rate.txt`, `regression_ffo_at_reit.txt`
- [Yes] Scatter plots saved to `Results/scatter_div12m_me.png`, `scatter_prime_rate.png`, `scatter_ffo_at_reit.png`
- [Yes] Report accurately reflects regression results
- [Yes] All interpretations are in economic units (not just statistical jargon)
