from flask import Flask

app = Flask(__name__)
# Add a secret key for session management
app.secret_key = "dev_secret_key" # Replace with a strong, environment-variable-based key in production

from app import routes

