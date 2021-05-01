from locust import HttpUser, task, between
import json


class APIUser(HttpUser):
    """
    Simulated API user for load testing.

    Only has one action, which is calling the single endpoint.
    """

    #wait_time = between(1.5, 2.5)

    @task
    def predict(self):
        headers = {'content-type': 'application/json','Accept-Encoding':'gzip'}
        self.client.post(
            "/api/predict",
            data= json.dumps(
                {
                    "text": "if you don't load test, you're an absolute buffoon!" # model detects toxicity
                }
            ),
            headers=headers,
            name = "Predict Toxicity"
        )
