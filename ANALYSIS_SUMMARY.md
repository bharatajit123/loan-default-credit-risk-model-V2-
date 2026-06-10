# Loan Default Analysis - Executive Summary

## 🎯 Quick Facts

| Metric | Value |
|--------|-------|
| **Dataset Size** | 255,347 loans |
| **Default Rate** | 11.61% |
| **Best Model** | Gradient Boosting |
| **Best ROC-AUC** | 0.7560 |
| **Recommended Threshold** | 0.20 |
| **Recommended Threshold Recall** | 43.06% |

---

## 📊 Top Risk Factors (Ranked by Importance)

### 1. **Employment Type** ⚠️ HIGHEST RISK
- **Unemployed:** 2.5x higher default probability than employed
- **Self-employed:** 1.8x higher than employed
- **Action:** Extra scrutiny for unemployed/self-employed applicants

### 2. **Age** 📉
- **Younger (<30):** High default risk
- **Older (>55):** Low default risk
- **Sweet spot:** 40-60 years old

### 3. **Income** 💰
- **<$30K:** Very high risk
- **$30K-$80K:** Moderate-high risk
- **>$100K:** Low risk

### 4. **Employment Stability** 🏢
- **0-12 months:** High risk
- **12-24 months:** Moderate risk
- **>24 months:** Low risk

### 5. **Debt-to-Income Ratio** 📈
- **<0.3:** Low risk
- **0.3-0.7:** Moderate risk
- **>0.7:** HIGH RISK ⚠️

---

## 🚨 Red Flags (High-Risk Profile)

**Flag when borrower has 3+ of these:**
- ✗ Age < 30
- ✗ Income < $50K
- ✗ Unemployed or self-employed
- ✗ Months employed < 12
- ✗ DTI ratio > 0.7
- ✗ Credit score < 450
- ✗ Interest rate > 18%
- ✗ No co-signer
- ✗ No mortgage or dependents

---

## 💚 Green Flags (Low-Risk Profile)

**Favorable when borrower has 3+ of these:**
- ✓ Age 40-65
- ✓ Income > $100K
- ✓ Full-time employed
- ✓ Months employed > 50
- ✓ DTI ratio < 0.5
- ✓ Credit score > 650
- ✓ Interest rate < 10%
- ✓ Has co-signer
- ✓ Has mortgage and dependents

---

## 🎯 Model Recommendations by Use Case

### **Conservative Banks** (Minimize Risk)
→ Use **Logistic Regression** with **Threshold 0.50**
- Only flags 0.46% of loans as risky
- 61.86% precision (few false alarms)
- Best for very low-risk portfolios

### **Balanced Approach** (Most Common) ⭐ RECOMMENDED
→ Use **Gradient Boosting** with **Threshold 0.20**
- Flags 16.56% of loans for review
- Catches 43.06% of defaults
- 81.83% accuracy
- Best ROC-AUC (0.7560)

### **Aggressive Lending** (Maximize Approvals)
→ Use **Random Forest** with **Threshold 0.50**
- Highest precision (63.71%)
- 3.76% recall (few false negatives)
- Only flags risky loans

### **Risk-Averse** (Catch Most Defaults)
→ Use **Gradient Boosting** with **Threshold 0.10**
- Catches 76.14% of defaults
- Flags 44.62% for manual review
- Lower precision, higher recall

---

## 📈 Expected Performance

With **Gradient Boosting at 0.20 threshold:**

| Scenario | Value |
|----------|-------|
| **Correctly Approved** | 39,235 loans |
| **Correctly Rejected** | 2,554 loans |
| **False Approvals** | 5,904 loans (extra review needed) |
| **Missed Defaults** | 3,377 loans |

---

## 💡 Business Impact

### Cost-Benefit Example
Assume:
- Average loan loss on default: $50,000
- Cost to manually review: $50/loan
- Cost to approve default: $50,000

**For 51,070 test loans:**
- **False positives cost:** 5,904 × $50 = $295,200
- **Missed defaults cost:** 3,377 × $50,000 = $168,850,000 ❌ HUGE!

✅ **Recommendation:** Accept the false positives; the cost of missed defaults is FAR greater

---

## 🔧 Implementation Checklist

- [ ] Export trained models to production
- [ ] Set up API endpoint for predictions
- [ ] Implement threshold of 0.20
- [ ] Create manual review queue for flagged loans
- [ ] Set up monitoring dashboard
- [ ] Define escalation procedures
- [ ] Plan quarterly model retraining
- [ ] A/B test with control group
- [ ] Track model performance over time

---

## 📊 Key Metrics to Monitor

**Monthly:**
- Actual default rate vs predicted
- Model accuracy on new data
- False positive/negative ratio

**Quarterly:**
- Model drift (performance degradation)
- Feature importance changes
- Need for retraining

**Annually:**
- Compare new models
- Update with new data
- Adjust thresholds if needed

---

## 🚀 Quick Action Items

1. **This Week:** Review this summary with stakeholders
2. **Next Week:** Set up production deployment
3. **Month 1:** Deploy with manual review for flagged loans
4. **Month 3:** Evaluate actual vs predicted performance
5. **Month 6:** Collect feedback, consider model refinements
6. **Year 1:** Retrain model with accumulated data

---

## 📞 Key Numbers

- **Dataset Size:** 255,347 loans
- **Training Set:** 204,277 (80%)
- **Test Set:** 51,070 (20%)
- **Models Evaluated:** 5
- **Visualizations Created:** 10+
- **Recommended Threshold:** 0.20

---

## 🎓 Technical Details

**Best Model:** Gradient Boosting Classifier
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1

**Threshold:** 0.20
- Optimized for F1-score and business impact
- Balances precision/recall tradeoff

**Features:** 16
- 9 numeric + 7 categorical
- All encoded and scaled
- No missing values

---

**Next Steps:** See DEPLOYMENT_GUIDE.md for production setup
