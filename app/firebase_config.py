import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():
    """Initializes Firebase Admin SDK using service account from environment."""
    if not firebase_admin._apps:
        service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        if service_account_json:
            try:
                sa_info = json.loads(service_account_json)
                cred = credentials.Certificate(sa_info)
                firebase_admin.initialize_app(cred)
            except Exception as e:
                print(f"Error initializing Firebase with service account: {e}")
                firebase_admin.initialize_app()
        else:
            firebase_admin.initialize_app()

initialize_firebase()

db = firestore.client()

