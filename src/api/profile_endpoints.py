import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.profile_workflow import UserProfileWorkflow
from src.core.user_segmentation import UserSegmentationEngine
from src.data.loader import load_user_profile, load_all_users
from src.models.user_profile import UserProfile

logger = logging.getLogger(__name__)

router = APIRouter()

# 初始化核心组件
workflow_engine = UserProfileWorkflow()
segmentation_engine = UserSegmentationEngine()


class ProfileGenerationRequest(BaseModel):
    """画像生成请求"""
    user_id: str = Field(..., description="用户ID")
    user_data: Dict[str, Any] = Field(..., description="用户数据")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "user_data": {
                    "registration_data": {
                        "age": 28,
                        "gender": "女",
                        "occupation": "软件工程师",
                        "location": "北京"
                    },
                    "text_data": [
                        "我喜欢看科技类的文章",
                        "最近在学习人工智能",
                        "周末喜欢去健身房运动"
                    ],
                    "behavior_data": {
                        "activity_data": {"daily_actions": 25}
                    }
                }
            }
        }


class ProfileUpdateRequest(BaseModel):
    """画像更新请求"""
    user_id: str = Field(..., description="用户ID")
    new_data: Dict[str, Any] = Field(..., description="新数据")


class ProfileResponse(BaseModel):
    """画像响应"""
    success: bool = Field(..., description="是否成功")
    profile: Optional[UserProfile] = Field(None, description="用户画像")
    message: str = Field(description="响应消息")


@router.post("/generate", response_model=ProfileResponse)
async def generate_profile(request: ProfileGenerationRequest):
    """
    根据提交的信息生成用户画像
    """
    try:
        user_profile = workflow_engine.generate_profile(
            request.user_id,
            request.user_data
        )
        return ProfileResponse(
            success=True,
            profile=user_profile,
            message="用户画像生成成功"
        )
    except Exception as e:
        logger.exception(f"生成用户画像失败: {e}")
        raise HTTPException(status_code=500, detail=f"画像生成失败: {str(e)}")


@router.get("/profile/{user_id}", response_model=ProfileResponse)
async def get_user_profile(user_id: str):
    user_profile = load_user_profile(user_id)
    return ProfileResponse(
        success=True,
        profile=user_profile,
        message="用户画像获取成功"
    )


@router.post("/profile/create_segments")
async def create_segments():
    users = load_all_users()
    return segmentation_engine.create_intelligent_segments(users)
