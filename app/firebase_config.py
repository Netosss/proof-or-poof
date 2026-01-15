import os
import json
import firebase_admin
from firebase_admin import credentials, firestore, auth

def initialize_firebase():
    """Initializes Firebase Admin SDK using service account from environment."""
    if not firebase_admin._apps:
        service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        if service_account_json:
            try:
                # Load JSON from string
                sa_info = json.loads(service_account_json)
                cred = credentials.Certificate(sa_info)
                firebase_admin.initialize_app(cred)
            except Exception as e:
                print(f"Error initializing Firebase with service account: {e}")
                # Fallback to default credentials (useful for local development if GOOGLE_APPLICATION_CREDENTIALS is set)
                firebase_admin.initialize_app()
        else:
            # Try to initialize with default credentials (ADC)
            firebase_admin.initialize_app()

initialize_firebase()

# Export db and auth for use in other modules
db = firestore.client()

