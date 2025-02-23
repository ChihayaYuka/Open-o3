FROM python:3.9-slim-buster
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x chat_launcher.sh
CMD ["./chat_launcher.sh"]
