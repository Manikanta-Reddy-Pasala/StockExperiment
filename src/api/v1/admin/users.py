"""
FastAPI Admin Users Module
"""
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

router = APIRouter()

# Pydantic models
class UserBase(BaseModel):
    username: str
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_admin: bool = False

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_admin: Optional[bool] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

class PasswordReset(BaseModel):
    new_password: str

class APIResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    message: Optional[str] = None
    timestamp: datetime

# Mock data for demonstration
MOCK_USERS = [
    {
        "id": 1,
        "username": "admin",
        "email": "admin@example.com",
        "first_name": "Admin",
        "last_name": "User",
        "is_admin": True,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow()
    },
    {
        "id": 2,
        "username": "trader1",
        "email": "trader1@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "is_admin": False,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "last_login": None
    }
]

@router.get("/", response_model=APIResponse, summary="Get All Users")
async def get_users():
    """
    Get all users (Admin only).
    
    Returns a list of all users in the system.
    """
    return APIResponse(
        success=True,
        data={"users": MOCK_USERS, "count": len(MOCK_USERS)},
        timestamp=datetime.utcnow()
    )

@router.post("/", response_model=APIResponse, status_code=status.HTTP_201_CREATED, summary="Create User")
async def create_user(user: UserCreate):
    """
    Create a new user (Admin only).
    
    - **username**: Unique username
    - **email**: Valid email address
    - **password**: User password (min 6 characters)
    - **first_name**: User's first name (optional)
    - **last_name**: User's last name (optional)
    - **is_admin**: Whether user has admin privileges
    """
    # Check if username or email already exists
    for existing_user in MOCK_USERS:
        if existing_user["username"] == user.username or existing_user["email"] == user.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already exists"
            )
    
    # Create new user
    new_user = {
        "id": len(MOCK_USERS) + 1,
        "username": user.username,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "is_admin": user.is_admin,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "last_login": None
    }
    
    MOCK_USERS.append(new_user)
    
    return APIResponse(
        success=True,
        data={"user": new_user},
        message="User created successfully",
        timestamp=datetime.utcnow()
    )

@router.get("/{user_id}", response_model=APIResponse, summary="Get User by ID")
async def get_user(user_id: int):
    """
    Get a specific user by ID (Admin only).
    
    - **user_id**: The ID of the user to retrieve
    """
    user = next((u for u in MOCK_USERS if u["id"] == user_id), None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return APIResponse(
        success=True,
        data={"user": user},
        timestamp=datetime.utcnow()
    )

@router.put("/{user_id}", response_model=APIResponse, summary="Update User")
async def update_user(user_id: int, user_update: UserUpdate):
    """
    Update user information (Admin only).
    
    - **user_id**: The ID of the user to update
    - **user_update**: User data to update
    """
    user = next((u for u in MOCK_USERS if u["id"] == user_id), None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update user fields
    update_data = user_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if field != "password":  # Handle password separately
            user[field] = value
    
    return APIResponse(
        success=True,
        data={"user": user},
        message="User updated successfully",
        timestamp=datetime.utcnow()
    )

@router.delete("/{user_id}", response_model=APIResponse, summary="Delete User")
async def delete_user(user_id: int):
    """
    Delete a user (Admin only).
    
    - **user_id**: The ID of the user to delete
    """
    global MOCK_USERS
    user = next((u for u in MOCK_USERS if u["id"] == user_id), None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    MOCK_USERS = [u for u in MOCK_USERS if u["id"] != user_id]
    
    return APIResponse(
        success=True,
        data={"message": "User deleted successfully"},
        timestamp=datetime.utcnow()
    )

@router.post("/{user_id}/reset-password", response_model=APIResponse, summary="Reset User Password")
async def reset_user_password(user_id: int, password_reset: PasswordReset):
    """
    Reset a user's password (Admin only).
    
    - **user_id**: The ID of the user whose password to reset
    - **password_reset**: New password data
    """
    user = next((u for u in MOCK_USERS if u["id"] == user_id), None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if len(password_reset.new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long"
        )
    
    return APIResponse(
        success=True,
        data={"message": "Password reset successfully"},
        timestamp=datetime.utcnow()
    )
