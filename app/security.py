import os
import time
import logging
import hashlib
import re
from typing import Dict, Callable, Any, Optional
import aiohttp
import json
from fastapi import HTTPException, Request, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
import cv2
from firebase_admin import auth, firestore
from app.firebase_config import db
from upstash_redis import Redis

# Set PIL safety limit to prevent decompression bombs (20MP)
Image.MAX_IMAGE_PIXELS = 20_000_000 

logger = logging.getLogger(__name__)

# Initialize Redis using Upstash REST API
redis_url = os.getenv("UPSTASH_REDIS_HOST") # User said "url is our host env var"
redis_token = os.getenv("UPSTASH_REDIS_PASSWORD") # User said "token is the password var"

redis_client = None
if redis_url and redis_token:
    try:
        redis_client = Redis(url=redis_url, token=redis_token)
        logger.info("Upstash Redis client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Upstash Redis client: {e}")
else:
    logger.warning("Redis credentials not found. Rate limiting will fallback to memory.")

# Legacy Auth Scheme (Might be removed later)
security_scheme = HTTPBearer(auto_error=False)

async def get_current_user(cred: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)) -> Optional[str]:
    """Verifies the Firebase ID token and returns the UID. Optional for migration."""
    if not cred:
        return None
    try:
        decoded_token = auth.verify_id_token(cred.credentials)
        return decoded_token['uid']
    except Exception as e:
        logger.warning(f"Invalid token: {e}")
        return None

async def verify_turnstile(token: str) -> bool:
    """Verifies Cloudflare Turnstile token."""
    secret = os.getenv("TURNSTILE_SECRET_KEY")
    if not secret:
        logger.warning("TURNSTILE_SECRET_KEY not set. Skipping validation (DEV MODE).")
        return True
        
    url = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    payload = {"secret": secret, "response": token}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload) as response:
                result = await response.json()
                if not result.get("success"):
                    logger.warning(f"Turnstile validation failed: {result}")
                    return False
                return True
    except Exception as e:
        logger.error(f"Turnstile connection error: {e}")
        # Fail open or closed? Usually closed for security, but open for reliability if CF is down.
        # Let's fail closed for now as it's a security feature.
        return False

def get_client_ip(request: Request) -> str:
    """Extracts the real client IP from headers, falling back to host."""
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip
    
    x_forwarded = request.headers.get("x-forwarded-for")
    if x_forwarded:
        return x_forwarded.split(",")[0].strip()
        
    return request.client.host if request.client else "127.0.0.1"

# Constants for the New Device Limit
MAX_NEW_DEVICES_PER_IP = 3
IP_DEVICE_WINDOW = 86400  # 24 hours in seconds

async def check_ip_device_limit(ip_address: str, device_id: str, turnstile_token: Optional[str] = None):
    """
    Checks if this IP has created too many unique device IDs.
    If limit exceeded, requires a valid Turnstile token.
    """
    if not redis_client:
        return # Fail open if Redis is down
        
    ip_key = f"ip_devices:{ip_address}"
    
    # 1 & 2. Check known device and current count in one network trip
    pipeline = redis_client.pipeline()
    pipeline.sismember(ip_key, device_id)
    pipeline.scard(ip_key)
    results = pipeline.exec()
    
    is_known = results[0]
    current_count = results[1]
    
    if is_known:
        return # Known device, let them through
    
    # 3. If the fingerprint is a "fallback" one, we are more suspicious
    is_fallback = device_id.startswith("mobile-fallback")
    limit = 1 if is_fallback else MAX_NEW_DEVICES_PER_IP

    if current_count >= limit:
        # Limit reached! Is there a CAPTCHA token?
        if not turnstile_token:
            logger.warning(f"IP {ip_address} reached device limit. CAPTCHA required.")
            raise HTTPException(
                status_code=403, 
                detail={"code": "CAPTCHA_REQUIRED", "message": "Verification needed"}
            )
        
        # 4. Verify the CAPTCHA
        is_human = await verify_turnstile(turnstile_token)
        if not is_human:
            raise HTTPException(status_code=403, detail="Invalid CAPTCHA")
            
    # 5. Register the new device to this IP using a pipeline
    write_pipeline = redis_client.pipeline()
    write_pipeline.sadd(ip_key, device_id)
    write_pipeline.expire(ip_key, IP_DEVICE_WINDOW)
    write_pipeline.exec()

# ---- Guest Wallet Logic ----

def get_guest_wallet(device_id: str) -> dict:
    """Retrieves or creates a guest wallet for the device."""
    doc_ref = db.collection('guest_wallets').document(device_id)
    doc = doc_ref.get()
    
    if doc.exists:
        return doc.to_dict()
    else:
        # Create new wallet with 10 free credits
        new_wallet = {
            "credits": 10, 
            "last_active": firestore.SERVER_TIMESTAMP,
            "is_banned": False
        }
        doc_ref.set(new_wallet)
        # Return with integer timestamp for immediate use if needed, or just dict
        # Firestore timestamp might need conversion if used immediately, but for now just returning dict is fine.
        return new_wallet

def check_ban_status(device_id: str) -> bool:
    """Checks if the device is banned."""
    wallet = get_guest_wallet(device_id)
    return wallet.get("is_banned", False)

def deduct_guest_credits(device_id: str, cost: int = 5) -> int:
    """
    Deducts credits from guest wallet atomically.
    Raises HTTPException(402) if insufficient funds.
    Returns new balance.
    """
    doc_ref = db.collection('guest_wallets').document(device_id)
    
    @firestore.transactional
    def update_in_transaction(transaction, ref):
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            # Should have been created by get_guest_wallet or logic check
            # Create it now if missing (edge case)
            transaction.set(ref, {"credits": 10, "last_active": firestore.SERVER_TIMESTAMP, "is_banned": False})
            current_credits = 10
        else:
            current_credits = snapshot.get("credits")
            if current_credits is None:
                current_credits = 0
        
        if current_credits < cost:
            raise HTTPException(
                status_code=402, 
                detail="Insufficient credits"
            )
            
        transaction.update(ref, {"credits": current_credits - cost, "last_active": firestore.SERVER_TIMESTAMP})
        return current_credits - cost

    transaction = db.transaction()
    try:
        new_balance = update_in_transaction(transaction, doc_ref)
        logger.info(f"Deducted {cost} credits from device {device_id}. New balance: {new_balance}")
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Guest credit deduction failed for {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Wallet transaction failed")

async def check_and_deduct_credits(uid: str, amount: int = 5):
    """
    Legacy: Checks if user has enough credits and deducts them atomically.
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
        # Memory fallback (optional, if Redis fails)
        self.rate_limits: Dict[str, list] = {}
        self.MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
        self.MAX_VIDEO_SIZE = 200 * 1024 * 1024 # 200MB
        self.RATE_LIMIT_WINDOW = 60 # seconds
        self.MAX_REQUESTS_PER_WINDOW = 10 # 10 requests per minute (Guest Policy)

    def check_rate_limit(self, identifier: str):
        """Rate limiting using Redis (preferred) or Memory (fallback)."""
        if redis_client:
            self._check_rate_limit_redis(identifier)
        else:
            self._check_rate_limit_memory(identifier)

    def _check_rate_limit_redis(self, identifier: str):
        key = f"rate_limit:{identifier}"
        try:
            # Atomic increment
            current_count = redis_client.incr(key)
            if current_count == 1:
                redis_client.expire(key, self.RATE_LIMIT_WINDOW)
            
            if current_count > self.MAX_REQUESTS_PER_WINDOW:
                logger.warning(f"Redis Rate limit exceeded for {identifier}")
                raise HTTPException(status_code=429, detail="Too many requests. Please try again in a minute.")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Redis rate limit error: {e}. Falling back to memory.")
            self._check_rate_limit_memory(identifier)

    def _check_rate_limit_memory(self, identifier: str):
        """Simple memory-based rate limiting per IP/Identifier with periodic cleanup."""
        now = time.time()
        
        if len(self.rate_limits) > 1000:
            self._cleanup_all_limits(now)

        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        self.rate_limits[identifier] = [t for t in self.rate_limits[identifier] if now - t < self.RATE_LIMIT_WINDOW]
        
        if len(self.rate_limits[identifier]) >= self.MAX_REQUESTS_PER_WINDOW:
            logger.warning(f"Memory Rate limit exceeded for {identifier}")
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
        if ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.heic', '.heif', '.tiff', '.tif', '.bmp']:
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
                if ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.heic', '.heif', '.tiff', '.tif', '.bmp']:
                    with Image.open(file_path) as img:
                        img.verify() # Verify structure
                        # Re-open to check format consistency
                        with Image.open(file_path) as img2:
                            actual_format = img2.format.lower()
                            if actual_format == 'jpeg': actual_format = 'jpg'
                            # Relaxed check: as long as it's a valid image format we support
                            # HEIC might register as 'heic' or 'heif' depending on the library
                            # TIFF as 'tiff'
                            if actual_format not in ['jpg', 'jpeg', 'png', 'webp', 'gif', 'heic', 'heif', 'tiff', 'tif', 'bmp']:
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
        # 1. Rate Limit (Prefer UID over IP, or Device ID if passed in kwargs or inferred)
        # For guest flow, uid might be the Device ID or None.
        # If uid is None, use IP.
        identifier = uid or request.client.host
        self.check_rate_limit(identifier)
        
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
