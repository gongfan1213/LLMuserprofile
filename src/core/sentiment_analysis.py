import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.llm_client import create_llm
from src.models.extract_models import ExtractEmotionModel, ExtractValueModel, ExtractMbitModel, ExtractProductSentimentModal
from src.models.user_profile import UserEmotion, EmotionType, UserTag, TagCategory, TagType

logger = logging.getLogger(__name__)


class EmotionAnalysisResult(BaseModel):
    """情绪分析结果"""
    primary_emotion: EmotionType = Field(..., description="主要情绪")
    emotion_intensity: float = Field(..., description="情绪强度")
    secondary_emotions: List[Tuple[EmotionType, float]] = Field(default_factory=list, description="次要情绪")
    emotion_triggers: List[str] = Field(default_factory=list, description="情绪触发因素")
    context_analysis: str = Field(..., description="情境分析")


class ValueAnalysisResult(BaseModel):
    """价值观分析结果"""
    core_values: List[str] = Field(..., description="核心价值观")
    personality_traits: List[str] = Field(..., description="性格特征")
    worldview_indicators: List[str] = Field(..., description="世界观指标")
    behavioral_patterns: List[str] = Field(..., description="行为模式")
    confidence_score: float = Field(..., description="分析可信度")


class SentimentAnalyzer:
    """情绪分析器"""

    def __init__(self):
        self.llm = create_llm()

    def analyze_user_emotions(self, data: Dict[str, Any]) -> List[UserEmotion]:
        """
        分析用户情绪
        
        Args:
            data: 用户文本数据，包括评论、反馈、对话等
            
        Returns:
            List[UserEmotion]: 情绪分析结果列表
        """
        emotions = []

        # 分析评论情绪
        if "comments" in data:
            comment_emotions = self._analyze_comment_emotions(data["comments"])
            emotions.extend(comment_emotions)

        # 分析产品评价情绪
        if "product_reviews" in data:
            review_emotions = self._analyze_review_emotions(data["product_reviews"])
            emotions.extend(review_emotions)

        # 可以添加更多情绪分析方法，如社交媒体内容、对话等
        return emotions

    def analyze_user_values(self, data: Dict[str, Any]) -> List[UserTag]:
        """
        分析用户价值观和性格特征
        
        Args:
            data: 用户文本数据和行为数据
            
        Returns:
            List[UserTag]: 价值观相关的标签列表
        """
        value_tags = []

        # 分析文本中的价值观
        if "text_content" in data:
            text_values = self._analyze_text_values(data["text_content"])
            value_tags.extend(text_values)

        # MBIT性格分析
        if "personality_data" in data:
            personality_tags = self._analyze_personality_type(data["personality_data"])
            value_tags.extend(personality_tags)

        # 还可以定义更多的维度进行分析
        return value_tags

    def analyze_multi_dimensional_sentiment(self, text: str,
        dimensions: List[str]) -> Dict[str, UserEmotion]:
        """
        多维度情绪分析
        
        Args:
            text: 待分析文本
            dimensions: 分析维度列表，如["产品功能", "价格", "服务"]
            
        Returns:
            Dict[str, UserEmotion]: 各维度的情绪分析结果
        """
        dimension_emotions = {}
        for dimension in dimensions:
            emotion = self._analyze_dimension_sentiment(text, dimension)
            if emotion:
                dimension_emotions[dimension] = emotion

        return dimension_emotions

    def _analyze_comment_emotions(self, comments: List[Dict[str, Any]]) -> List[UserEmotion]:
        """分析评论情绪"""
        emotions = []

        for comment in comments:
            comment_text = comment.get("content", "")
            if not comment_text:
                continue

            emotion_result = self._extract_emotion_from_text(comment_text)

            if emotion_result:
                emotion = UserEmotion(
                    emotion_type=emotion_result["emotion_type"],
                    intensity=emotion_result["intensity"],
                    context=comment.get("context", "用户评论"),
                    source_text=comment_text,
                    timestamp=comment.get("timestamp", datetime.now())
                )
                emotions.append(emotion)

        return emotions

    def _analyze_review_emotions(self, reviews: List[Dict[str, Any]]) -> List[UserEmotion]:
        """分析产品评价情绪"""
        emotions = []

        for review in reviews:
            review_text = review.get("content", "")
            rating = review.get("rating", 0)

            if not review_text:
                continue

            # 多维度分析产品评价
            dimensions = ["产品质量", "价格", "服务", "物流"]
            dimension_emotions = self.analyze_multi_dimensional_sentiment(
                review_text, dimensions
            )

            emotions.extend(dimension_emotions.values())

            # 基于评分推断整体情绪
            if rating > 0:
                if rating >= 4:
                    overall_emotion = EmotionType.JOY
                    intensity = min(rating / 5.0, 1.0)
                elif rating <= 2:
                    overall_emotion = EmotionType.ANGER
                    intensity = max((3 - rating) / 3.0, 0.3)
                else:
                    overall_emotion = EmotionType.NEUTRAL
                    intensity = 0.5

                emotion = UserEmotion(
                    emotion_type=overall_emotion,
                    intensity=intensity,
                    context="产品评价整体情绪",
                    source_text=review_text,
                    timestamp=review.get("timestamp", datetime.now())
                )
                emotions.append(emotion)

        return emotions

    def _analyze_text_values(self, text_content: List[str]) -> List[UserTag]:
        """分析文本中的价值观"""
        tags = []

        if not text_content:
            return tags

        combined_text = " ".join(text_content[:200])  # 限制文本长度
        values_prompt = ChatPromptTemplate.from_template("""
        请分析以下文本内容，识别用户体现出的价值观和性格特征：
        
        文本内容：
        {text}
        """)

        chain = values_prompt | self.llm.with_structured_output(ExtractValueModel, method="function_calling")

        response: ExtractValueModel = chain.invoke(input={"text": combined_text})
        return [self._create_value_tag(
            "core_value", response.value,
            response.confidence,
            response.evidence
        )]

    def _analyze_personality_type(self, personality_data: Dict[str, Any]) -> List[UserTag]:
        """MBTI性格类型分析"""
        tags = []

        # 如果有文本数据，进行MBTI分析
        if "text_samples" in personality_data:
            text_samples = personality_data["text_samples"]
            combined_text = " ".join(text_samples[:200])

            mbti_prompt = ChatPromptTemplate.from_template("""
            请根据以下文本分析用户的MBTI性格倾向：
            
            文本样本：
            {text}
            
            """)

            chain = mbti_prompt | self.llm.with_structured_output(ExtractMbitModel, method="function_calling")
            response: ExtractMbitModel = chain.invoke(input={"text": combined_text})
            # 创建MBTI标签
            tags.append(self._create_value_tag(
                "mbti_type", response.mbti_type,
                response.confidence, "基于文本分析的MBTI推断"
            ))
            tags.append(self._create_value_tag(
                "personality_trait", response.key_traits,
                response.confidence, "MBTI分析关键特征"
            ))
        return tags

    def _extract_emotion_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """从文本提取情绪"""
        if not text:
            return None

        emotion_prompt = ChatPromptTemplate.from_template("""
        请分析以下文本的情绪：
        (对于情绪类型，请使用以下枚举值：joy/anger/sadness/surprise/fear/disgust/neutral)
        文本：
        {text}
        """)

        chain = emotion_prompt | self.llm.with_structured_output(ExtractEmotionModel, method="function_calling")
        result: ExtractEmotionModel = chain.invoke(input={"text": text})
        emotion_type_str = result.emotion_type
        try:
            emotion_type = EmotionType(emotion_type_str)
        except ValueError:
            emotion_type = EmotionType.NEUTRAL

        return {
            "emotion_type": emotion_type,
            "intensity": result.intensity,
            "emotion_words": result.emotion_words
        }

    def _analyze_dimension_sentiment(self, text: str, dimension: str) -> Optional[UserEmotion]:
        """分析特定维度的情绪"""
        dimension_prompt = ChatPromptTemplate.from_template("""
        请分析以下文本中关于"{dimension}"的情绪态度：
        对于emotion_type，只能使用以下枚举值：positive/negative/neutral
        文本：{text}
       
   
        """)

        chain = dimension_prompt | self.llm.with_structured_output(ExtractProductSentimentModal, method="function_calling")
        response: ExtractProductSentimentModal = chain.invoke(input={"text": text, "dimension": dimension})
        if not response.has_dimension_content:
            return None

        emotion_type_str = response.emotion_type
        try:
            emotion_type = EmotionType(emotion_type_str)
        except ValueError:
            emotion_type = EmotionType.NEUTRAL

        return UserEmotion(
            emotion_type=emotion_type,
            intensity=response.intensity,
            context=f"对{dimension}的情绪",
            source_text=text,
            timestamp=datetime.now()
        )

    @staticmethod
    def _create_value_tag(name: str, value: Any,
        confidence: float, evidence: str) -> UserTag:
        """创建价值观标签"""
        return UserTag(
            tag_id=str(uuid.uuid4()),
            name=name,
            value=value,
            category=TagCategory.VALUES,
            tag_type=TagType.IMPLICIT,
            confidence=confidence,
            source="value_analysis",
            metadata={"evidence": evidence}
        )
