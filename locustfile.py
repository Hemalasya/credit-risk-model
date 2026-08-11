from locust import HttpUser, task, between
import random

class CreditAPIUser(HttpUser):
    wait_time = between(0.5, 2)  # simulate real users pausing between requests, not hammering nonstop

    @task
    def predict(self):
        applicant = {
            "RevolvingUtilizationOfUnsecuredLines": round(random.uniform(0, 1.2), 2),
            "age": random.randint(21, 75),
            "NumberOfTime30-59DaysPastDueNotWorse": random.randint(0, 4),
            "DebtRatio": round(random.uniform(0, 1.5), 2),
            "MonthlyIncome": random.randint(1500, 12000),
            "NumberOfOpenCreditLinesAndLoans": random.randint(1, 15),
            "NumberOfTimes90DaysLate": random.randint(0, 3),
            "NumberRealEstateLoansOrLines": random.randint(0, 3),
            "NumberOfTime60-89DaysPastDueNotWorse": random.randint(0, 3),
            "NumberOfDependents": random.randint(0, 4)
        }
        self.client.post("/predict", json=applicant)