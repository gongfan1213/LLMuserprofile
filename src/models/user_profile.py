from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class TagType(str, Enum):
    """标签类型枚举"""
    EXPLICIT = "explicit"  # 显式标签
    IMPLICIT = "implicit"  # 隐式标签


class TagCategory(str, Enum):
    """标签分类枚举"""
    BASIC_INFO = "basic_info"  # 基础信息
    DEMOGRAPHICS = "demographics"  # 人口统计
    INTERESTS = "interests"  # 兴趣爱好
    BEHAVIOR = "behavior"  # 行为特征
    PREFERENCES = "preferences"  # 偏好
    INTENT = "intent"  # 意图
    SENTIMENT = "sentiment"  # 情绪
    VALUES = "values"  # 价值观
    CONSUMPTION = "consumption"  # 消费习惯
    LIFECYCLE = "lifecycle"  # 生命周期


class EmotionType(str, Enum):
    """情绪类型枚举"""
    JOY = "joy"  # 喜悦
    ANGER = "anger"  # 愤怒
    SADNESS = "sadness"  # 悲伤
    SURPRISE = "surprise"  # 惊讶
    FEAR = "fear"  # 恐惧
    DISGUST = "disgust"  # 厌恶
    NEUTRAL = "neutral"  # 中立


class UserTag(BaseModel):
    """用户标签模型"""
    tag_id: str = Field(..., description="标签ID")
    name: str = Field(..., description="标签名称")
    value: Any = Field(..., description="标签值")
    category: TagCategory = Field(..., description="标签分类")
    tag_type: TagType = Field(..., description="标签类型")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="置信度")
    source: str = Field(..., description="标签来源")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class UserEmotion(BaseModel):
    """用户情绪模型"""
    emotion_type: EmotionType = Field(..., description="情绪类型")
    intensity: float = Field(..., ge=0.0, le=1.0, description="情绪强度")
    context: str = Field(..., description="情绪上下文")
    source_text: str = Field(..., description="源文本")
    timestamp: datetime = Field(default_factory=datetime.now)


class UserIntent(BaseModel):
    """用户意图模型"""
    intent_name: str = Field(..., description="意图名称")
    intent_category: str = Field(..., description="意图分类")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="实体信息")
    context: str = Field(..., description="意图上下文")
    priority: int = Field(default=1, description="优先级")
    status: str = Field(default="active", description="状态")
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = Field(None, description="过期时间")


class UserSegment(BaseModel):
    """用户分群模型"""
    segment_id: str = Field(..., description="分群ID")
    segment_name: str = Field(..., description="分群名称")
    description: str = Field(..., description="分群描述")
    member_count: int = Field(default=0, description="成员数量")
    user_ids: List[str] = Field(default_factory=list, description="用户ID列表")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class UserProfile(BaseModel):
    """用户画像主模型"""
    user_id: str = Field(..., description="用户ID")
    tags: List[UserTag] = Field(default_factory=list, description="用户标签")
    emotions: List[UserEmotion] = Field(default_factory=list, description="情绪记录")
    intents: List[UserIntent] = Field(default_factory=list, description="意图记录")
    segments: List[str] = Field(default_factory=list, description="所属分群ID列表")

    # 画像统计信息
    profile_completeness: float = Field(default=0.0, ge=0.0, le=1.0, description="画像完整度")
    last_activity: Optional[datetime] = Field(None, description="最后活跃时间")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展元数据")

    def get_tags_by_category(self, category: TagCategory) -> List[UserTag]:
        """根据分类获取标签"""
        return [tag for tag in self.tags if tag.category == category]

    def get_active_intents(self) -> List[UserIntent]:
        """获取活跃意图"""
        now = datetime.now()
        return [
            intent for intent in self.intents
            if intent.status == "active" and (
                intent.expires_at is None or intent.expires_at > now
            )
        ]

    def get_recent_emotions(self, days: int = 7) -> List[UserEmotion]:
        """获取最近7天的情绪记录"""
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
        return [
            emotion for emotion in self.emotions
            if emotion.timestamp >= cutoff_date
        ]
