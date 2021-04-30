from locust import FastHttpUser, task
import json


class APIUser(FastHttpUser):
    """
    Simulated API user for load testing
    """

    @task
    def create_post(self):
        headers = {"content-type": "application/json", "Accept-Encoding": "gzip"}
        self.client.post(
            "/secret_add",
            data=json.dumps({"a": 5, "b": 6}),
            headers=headers,
            name="Add Numbers",
        )
