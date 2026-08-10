import requests

# A sample applicant — moderate risk profile: some utilization, a couple late payments
sample_applicant = {
    "RevolvingUtilizationOfUnsecuredLines": 0.35,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    "DebtRatio": 0.4,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2
}

response = requests.post("http://127.0.0.1:5000/predict", json=sample_applicant)
print("Status code:", response.status_code)
print("Response:", response.json())
high_risk_applicant = {
    "RevolvingUtilizationOfUnsecuredLines": 0.95,
    "age": 28,
    "NumberOfTime30-59DaysPastDueNotWorse": 3,
    "DebtRatio": 0.8,
    "MonthlyIncome": 1800,
    "NumberOfOpenCreditLinesAndLoans": 4,
    "NumberOfTimes90DaysLate": 2,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 1,
    "NumberOfDependents": 3
}

response2 = requests.post("http://127.0.0.1:5000/predict", json=high_risk_applicant)
print("\nHigh-risk applicant:")
print("Status code:", response2.status_code)
print("Response:", response2.json())