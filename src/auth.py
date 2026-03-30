import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

# Configuration (In production, use environment variables)
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-for-dev-only-change-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 week

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

USERS_FILE = 'data/users.csv'

class User(BaseModel):
    username: str
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def load_users():
    if not os.path.exists(USERS_FILE):
        return pd.DataFrame(columns=['username', 'hashed_password'])
    return pd.read_csv(USERS_FILE)

def save_user(username, hashed_password):
    users_df = load_users()
    if username in users_df['username'].values:
        return False
    
    new_user = pd.DataFrame([{
        'username': username,
        'hashed_password': hashed_password
    }])
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    os.makedirs('data', exist_ok=True)
    users_df.to_csv(USERS_FILE, index=False)
    return True

def get_user(username: str):
    users_df = load_users()
    user_row = users_df[users_df['username'] == username]
    if user_row.empty:
        return None
    return UserInDB(
        username=user_row.iloc[0]['username'],
        hashed_password=user_row.iloc[0]['hashed_password']
    )

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(username)
    if user is None:
        raise credentials_exception
    return user
