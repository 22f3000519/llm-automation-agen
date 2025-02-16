# llm-automation-agent

This is the automation agent build to perform several tasks
### Running the Application

To run the application using `main.py` (default):
```sh
podman run --rm -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 $IMAGE_NAME
```
To override main_B.py
```sh
podman run --rm -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 $IMAGE_NAME uvicorn main_B:app --host 0.0.0.0 --port 8000
```



