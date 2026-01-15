import os
import time
import logging
import hashlib
import re
from typing import Dict, Callable, Any, Optional
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
import cv2
from firebase_admin import auth, firestore
from app.firebase_config import db

# Set PIL safety limit to prevent decompression bombs (20MP)
Image.MAX_IMAGE_PIXELS = 20_000_000 

logger = logging.getLogger(__name__)

security_scheme = HTTPBearer()

async def get_current_user(cred: HTTPAuthorizationCredentials = Depends(security_scheme)) -> str:
    """Verifies the Firebase ID token and returns the UID."""
    try:
        decoded_token = auth.verify_id_token(cred.credentials)
        return decoded_token['uid']
    except Exception as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def check_and_deduct_credits(uid: str, amount: int = 5):
    """
    Checks if user has enough credits and deducts them atomically.
    Raises 402 if insufficient funds.
    """
    user_ref = db.collection('users').document(uid)

    @firestore.transactional
    def deduct_transaction(transaction, user_ref):
        snapshot = user_ref.get(transaction=transaction)
        if not snapshot.exists:
            raise HTTPException(status_code=404, detail="User not found in Firestore")
        
        user_data = snapshot.to_dict()
        current_credits = user_data.get('credits', 0)
        
        if current_credits < amount:
            raise HTTPException(
                status_code=402, 
                detail=f"Insufficient credits. Required: {amount}, Available: {current_credits}"
            )
        
        transaction.update(user_ref, {
            'credits': current_credits - amount
        })
        return current_credits - amount

    try:
        transaction = db.transaction()
        new_balance = deduct_transaction(transaction, user_ref)
        logger.info(f"Deducted {amount} credits from user {uid}. New balance: {new_balance}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Credit deduction failed for user {uid}: {e}")
        raise HTTPException(status_code=500, detail="Credit deduction failed")

class SecurityManager:
    def __init__(self):
        # Rate limiting storage: {ip: [timestamps]}
        self.rate_limits: Dict[str, list] = {}
        self.MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
        self.MAX_VIDEO_SIZE = 200 * 1024 * 1024 # 200MB
        self.RATE_LIMIT_WINDOW = 60 # seconds
        self.MAX_REQUESTS_PER_WINDOW = 15 # requests per minute

    def check_rate_limit(self, identifier: str):
        """Simple memory-based rate limiting per IP/Identifier with periodic cleanup."""
        now = time.time()
        
        # Periodic cleanup of the entire storage every 1000 requests (to prevent memory leaks)
        if len(self.rate_limits) > 1000:
            self._cleanup_all_limits(now)

        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        # Clean old timestamps for this identifier
        self.rate_limits[identifier] = [t for t in self.rate_limits[identifier] if now - t < self.RATE_LIMIT_WINDOW]
        
        if len(self.rate_limits[identifier]) >= self.MAX_REQUESTS_PER_WINDOW:
            logger.warning(f"Rate limit exceeded for {identifier}")
            raise HTTPException(status_code=429, detail="Too many requests. Please try again in a minute.")
        
        self.rate_limits[identifier].append(now)

    def _cleanup_all_limits(self, now: float):
        """Clear all identifiers that haven't made a request in the last window."""
        expired_keys = [
            k for k, v in self.rate_limits.items() 
            if not v or now - v[-1] > self.RATE_LIMIT_WINDOW
        ]
        for k in expired_keys:
            del self.rate_limits[k]
        logger.info(f"Rate limit cleanup: removed {len(expired_keys)} inactive sessions.")

    def validate_file(self, filename: str, filesize: int, file_path: str = None) -> bool:
        """Check file extension, size, and content integrity using magic bytes."""
        ext = os.path.splitext(filename)[1].lower()
        
        # 1. Size & Extension Check
        if ext in ['.jpg', '.jpeg', '.png', '.webp']:
            if filesize > self.MAX_IMAGE_SIZE:
                raise HTTPException(status_code=413, detail=f"Image too large. Max {self.MAX_IMAGE_SIZE//1024//1024}MB allowed.")
        elif ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            if filesize > self.MAX_VIDEO_SIZE:
                raise HTTPException(status_code=413, detail=f"Video too large. Max {self.MAX_VIDEO_SIZE//1024//1024}MB allowed.")
        else:
            raise HTTPException(status_code=415, detail="Unsupported file format.")

        # 2. Deep Content Validation (Magic Bytes)
        if file_path:
            try:
                if ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    with Image.open(file_path) as img:
                        img.verify() # Verify structure
                        # Re-open to check format consistency
                        with Image.open(file_path) as img2:
                            actual_format = img2.format.lower()
                            if actual_format == 'jpeg': actual_format = 'jpg'
                            # Relaxed check: as long as it's a valid image format we support
                            if actual_format not in ['jpg', 'jpeg', 'png', 'webp']:
                                raise Exception(f"Format mismatch: {actual_format}")
                else:
                    # Video validation: Try to read first frame
                    cap = cv2.VideoCapture(file_path)
                    if not cap.isOpened():
                        raise Exception("Could not open video stream")
                    ret, _ = cap.read()
                    cap.release()
                    if not ret:
                        raise Exception("Could not read video frames")
            except Exception as e:
                logger.error(f"Malicious or corrupted file detected ({filename}): {e}")
                raise HTTPException(status_code=400, detail="Invalid file content or format mismatch.")
                
        return True

    def sanitize_log_message(self, message: str) -> str:
        """Remove potential PII or sensitive paths from logs, masking temp files."""
        # Mask absolute paths and specifically temp file patterns
        # Example: /tmp/tmp_abc123.jpg -> [TEMP_FILE]
        msg = re.sub(r'\/[^\s]+\/tmp[a-zA-Z0-9_]+', '[TEMP_FILE]', message)
        # General path sanitization
        msg = re.sub(r'\/[^\s]+\/([^\/\s]+)', r'.../\1', msg)
        return msg

    def get_safe_hash(self, data: bytes) -> str:
        """Securely hash data for caching to prevent collisions/poisoning."""
        return hashlib.sha256(data).hexdigest()

    async def secure_execute(self, request: Request, filename: str, filesize: int, temp_path: str, func: Callable, uid: Optional[str] = None, *args, **kwargs) -> Any:
        """
        A secure wrapper for media processing functions.
        Handles Rate Limiting, Validation, and Log Sanitization.
        """
        # 1. Rate Limit (Prefer UID over IP)
        self.check_rate_limit(uid or request.client.host)
        
        # 2. Deep Validation
        self.validate_file(filename, filesize, temp_path)
        
        try:
            # 3. Execution
            start_time = time.time()
            result = await func(temp_path, *args, **kwargs)
            duration = time.time() - start_time
            
            # 4. Safe Logging
            safe_msg = self.sanitize_log_message(f"Successfully processed {filename} in {duration:.2f}s")
            logger.info(safe_msg)
            
            return result
        except Exception as e:
            err_msg = self.sanitize_log_message(f"Error processing {filename}: {str(e)}")
            logger.error(err_msg)
            raise HTTPException(status_code=500, detail="Internal processing error.")

security_manager = SecurityManager()
