import os
import time
import logging
import hashlib
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Callable, Any, Optional
import aiohttp
import json
from fastapi import HTTPException, Request, Header
from PIL import Image
import cv2
from firebase_admin import firestore
from app.firebase_config import db
from app.config import settings
from upstash_redis import Redis

# Prevent decompression-bomb attacks
Image.MAX_IMAGE_PIXELS = settings.pil_max_image_pixels

logger = logging.getLogger(__name__)

redis_url = os.getenv("UPSTASH_REDIS_HOST")
redis_token = os.getenv("UPSTASH_REDIS_PASSWORD")

redis_client = None
if redis_url and redis_token:
    try:
        redis_client = Redis(url=redis_url, token=redis_token)
        logger.info("Upstash Redis client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Upstash Redis client: {e}")
else:
    logger.warning("Redis credentials not found. Rate limiting will fallback to memory.")

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

IP_DEVICE_WINDOW = settings.rate_limit_window_sec

async def check_ip_device_limit(ip_address: str, device_id: str, turnstile_token: Optional[str] = None, token_already_verified: bool = False):
    """
    Checks if this IP has created too many unique device IDs.
    If limit exceeded, requires a valid Turnstile token.
    """
    if not redis_client:
        return # Fail open if Redis is down
        
    ip_key = f"ip_devices:{ip_address}"

    pipeline = redis_client.pipeline()
    pipeline.sismember(ip_key, device_id)
    pipeline.scard(ip_key)
    results = pipeline.exec()
    
    is_known = results[0]
    current_count = results[1]

    if is_known:
        return

    is_fallback = device_id.startswith("mobile-fallback")
    limit = 1 if is_fallback else settings.max_new_devices_per_ip

    if current_count >= limit:
        if not token_already_verified:
            if not turnstile_token:
                logger.warning(f"IP {ip_address} reached device limit. STRICT CAPTCHA required.")
                raise HTTPException(
                    status_code=403,
                    detail={"code": "STRICT_CAPTCHA_REQUIRED", "message": "High activity detected. Strict verification needed."}
                )
            is_human = await verify_turnstile(turnstile_token)
            if not is_human:
                raise HTTPException(status_code=403, detail="Invalid CAPTCHA")

    write_pipeline = redis_client.pipeline()
    write_pipeline.sadd(ip_key, device_id)
    write_pipeline.expire(ip_key, IP_DEVICE_WINDOW)
    write_pipeline.exec()

def get_guest_wallet(device_id: str) -> dict:
    """Retrieves or creates a guest wallet for the device."""
    doc_ref = db.collection('guest_wallets').document(device_id)
    doc = doc_ref.get()
    
    if doc.exists:
        return doc.to_dict()
    else:
        new_wallet = {
            "credits": settings.welcome_credits,
            "last_active": firestore.SERVER_TIMESTAMP,
            "is_banned": False,
            "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.wallet_ttl_days),
        }
        doc_ref.set(new_wallet)
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
            transaction.set(ref, {
                "credits": settings.welcome_credits,
                "last_active": firestore.SERVER_TIMESTAMP,
                "is_banned": False,
                "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.wallet_ttl_days),
            })
            current_credits = settings.welcome_credits
        else:
            current_credits = snapshot.get("credits")
            if current_credits is None:
                current_credits = 0
        
        if current_credits < cost:
            raise HTTPException(
                status_code=402, 
                detail="Insufficient credits"
            )
            
        transaction.update(ref, {
            "credits": current_credits - cost,
            "last_active": firestore.SERVER_TIMESTAMP,
            "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.wallet_ttl_days),
        })
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

class SecurityManager:
    def __init__(self):
        self.rate_limits: Dict[str, list] = {}
        self.MAX_IMAGE_SIZE = settings.max_image_upload_bytes
        self.MAX_VIDEO_SIZE = settings.max_video_upload_bytes
        self.RATE_LIMIT_WINDOW = settings.rate_limit_request_window_sec
        self.MAX_REQUESTS_PER_WINDOW = settings.rate_limit_max_requests

    def check_rate_limit(self, identifier: str):
        """Rate limiting using Redis (preferred) or Memory (fallback)."""
        if redis_client:
            self._check_rate_limit_redis(identifier)
        else:
            self._check_rate_limit_memory(identifier)

    def _check_rate_limit_redis(self, identifier: str):
        key = f"rate_limit:{identifier}"
        try:
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
        
        if len(self.rate_limits) > settings.rate_limit_memory_limit:
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

        if ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.heic', '.heif', '.tiff', '.tif', '.bmp']:
            if filesize > self.MAX_IMAGE_SIZE:
                raise HTTPException(status_code=413, detail=f"Image too large. Max {self.MAX_IMAGE_SIZE//1024//1024}MB allowed.")
        elif ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            if filesize > self.MAX_VIDEO_SIZE:
                raise HTTPException(status_code=413, detail=f"Video too large. Max {self.MAX_VIDEO_SIZE//1024//1024}MB allowed.")
        else:
            raise HTTPException(status_code=415, detail="Unsupported file format.")

        if file_path:
            try:
                if ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.heic', '.heif', '.tiff', '.tif', '.bmp']:
                    with Image.open(file_path) as img:
                        img.verify()
                        with Image.open(file_path) as img2:
                            actual_format = img2.format.lower()
                            if actual_format == 'jpeg': actual_format = 'jpg'
                            if actual_format not in ['jpg', 'jpeg', 'png', 'webp', 'gif', 'heic', 'heif', 'tiff', 'tif', 'bmp']:
                                raise Exception(f"Format mismatch: {actual_format}")
                else:
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
        """Strip sensitive file paths from log messages."""
        msg = re.sub(r'\/[^\s]+\/tmp[a-zA-Z0-9_]+', '[TEMP_FILE]', message)
        msg = re.sub(r'\/[^\s]+\/([^\/\s]+)', r'.../\1', msg)
        return msg

    def get_safe_hash(self, data: bytes) -> str:
        """Securely hash data for caching to prevent collisions/poisoning."""
        return hashlib.sha256(data).hexdigest()

    async def secure_execute(self, request: Request, filename: str, filesize: int, temp_path: str, func: Callable, uid: Optional[str] = None, *args, **kwargs) -> Any:
        """Rate-limits, validates, executes, and sanitizes logs for a media processing call."""
        identifier = uid or request.client.host
        self.check_rate_limit(identifier)
        self.validate_file(filename, filesize, temp_path)

        try:
            start_time = time.time()
            result = await func(temp_path, *args, **kwargs)
            duration = time.time() - start_time
            safe_msg = self.sanitize_log_message(f"Successfully processed {filename} in {duration:.2f}s")
            logger.info(safe_msg)
            return result
        except Exception as e:
            err_msg = self.sanitize_log_message(f"Error processing {filename}: {str(e)}")
            logger.error(err_msg)
            raise HTTPException(status_code=500, detail="Internal processing error.")

security_manager = SecurityManager()
