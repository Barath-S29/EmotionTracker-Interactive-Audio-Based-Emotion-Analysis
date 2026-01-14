# Email Configuration Example
# Copy this file to config.py and fill in your actual credentials
# DO NOT commit config.py to git!

# SMTP Server Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Email Credentials
# For Gmail, you need to use an App Password: https://www.getmailbird.com/gmail-app-password/
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"
RECEIVER_EMAIL = "receiver_email@gmail.com"

# Alternative: Use environment variables instead
# Set them in your system or use:
# export SENDER_EMAIL="your_email@gmail.com"
# export SENDER_PASSWORD="your_app_password"
# export RECEIVER_EMAIL="receiver_email@gmail.com"
