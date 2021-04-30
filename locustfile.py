from locust import FastHttpUser, TaskSet, task
import json

class APIUser(FastHttpUser):

    @task
    def create_post(self):
        headers = {'content-type': 'application/json','Accept-Encoding':'gzip'}
        self.client.post(
            "/secret_add",
            data= json.dumps(
                {
                    "a": 5,
                    "b": 6
                }
            ),
            headers=headers,
            name = "Add Numbers"
        )