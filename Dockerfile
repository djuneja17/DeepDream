# Start with conda image
FROM continuumio/anaconda3

# Expose any ports the app is expecting in the environment
ENV PORT 8001
EXPOSE $PORT

# Set up a working folder and install the pre-reqs
WORKDIR /app
RUN pip install --upgrade pip
COPY deep_dream/ /app
RUN pip install -r requirements.txt

# Run the service
CMD [ "python", "main.py"]