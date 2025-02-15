# llm-automation-agent

This is the automation agent build to perform several tasks

## Running Phase A
```bash
docker run -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 your-dockerhub-username/tds-project:latest
docker run -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 your-dockerhub-username/tds-project:latest python main_B.py
to switch between users

