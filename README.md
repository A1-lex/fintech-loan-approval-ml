# Automated Loan Approval System

Binary classification model automating loan underwriting decisions, built on the CRISP-DM framework using 20,000 historical loan applications.

## Result
**Gradient Boosting (HistGradientBoostingClassifier)** achieved **0.9733 ROC-AUC** and **F-β (β=0.5) of 0.8601** on held-out test data. A business-cost-optimized decision threshold (0.89) reduced total operational cost to **$256,500**, a **46.3% improvement** over a naive deny-all baseline.

## Important framing
This model predicts `LoanApproved`, what a human underwriter would decide, not whether a borrower will default. The dataset contains no repayment outcomes for denied applicants, so a true default-risk model would require reject inference techniques and repayment data not present here. This distinction and its implications are documented in the notebook (Section 6.4).

## Approach
- **Business framing**: cost-asymmetric evaluation (false approvals cost more than false denials), tunable via F-β
- **Modeling**: Gradient Boosting selected via 5-fold CV across multiple candidate models
- **Threshold optimization**: decision threshold tuned against real operational costs, not just default 0.5
- **Deployment recommendation**: three-tier system, auto-approve, auto-deny, and a human-review band for borderline cases

## Key predictive features
Income-to-loan ratio, credit score and payment history, and debt-to-income ratio were the strongest signals, consistent with standard underwriting logic.

## Limitations
Documented transparently in-notebook: this model automates existing policy rather than predicting true default risk, and inherits any bias present in historical approval decisions.

## Files
- `financial_loan_risk.ipynb`: full analysis notebook
- `financial_loan_data.csv`: dataset

## Requirements
pandas, scikit-learn, matplotlib, seaborn
