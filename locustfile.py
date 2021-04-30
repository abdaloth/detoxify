from locust import FastHttpUser, task
import json

class APIUser(FastHttpUser):
    """
    Simulated API user for load testing.

    Only has one action, which is calling the single endpoint.
    """

    @task
    def predict(self):
        headers = {'content-type': 'application/json','Accept-Encoding':'gzip'}
        self.client.post(
            "/predict",
            data= json.dumps(
                {
                    "text": "if you don't load test, you're an absolute buffoon!" # model detects toxicity
                }
            ),
            headers=headers,
            name = "Predict Toxicity"
        )
