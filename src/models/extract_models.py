from typing import Literal

from pydantic import Field, BaseModel


class ExtractEntityModel(BaseModel):
    name: str = Field(..., description="实体名称")
    type: Literal["brand", "product", "person", "location", "organization", "other"] \
        = Field(..., description="实体类型，可选值：brand/product/person/location/organization/other")
    value: str = Field(..., description="实体值")
    confidence: float = Field(..., description="置信度，范围0-1")
    context: str = Field(..., description="实体出现的上下文")


class EntityCollection(BaseModel):
    entities: list[ExtractEntityModel] = Field(..., description="所有实体的集合")


class ExtractKeywordModel(BaseModel):
    keyword: str = Field(..., description="关键词")
    frequency: int = Field(..., description="关键词出现频率")
    context: str = Field(..., description="关键词出现的上下文")
    relevance: float = Field(..., description="相关性，范围0-1")


class KeywordCollection(BaseModel):
    keywords: list[ExtractKeywordModel] = Field(..., description="所有关键词的集合")


class ExtractInterestModel(BaseModel):
    interest: str = Field(..., description="兴趣点")
    category: Literal["sports", "technology", "entertainment", "lifestyle", "education", "travel", "food", "finance", "other"] = \
        Field(..., description="兴趣分类，可选值只能是 ： sports/technology/entertainment/lifestyle/education/travel/food/other)")
    intensity: float = Field(..., description="相关性，范围0-1")
    evidence: str = Field(..., description="支撑证据")


class InterestCollection(BaseModel):
    interests: list[ExtractInterestModel] = Field(..., description="所有兴趣点的集合")


class ExtractEmotionModel(BaseModel):
    emotion_type: Literal["joy", "anger", "sadness", "surprise", "fear", "disgust", "neutral"] = \
        Field(..., description="情绪类型，可选值只能是：joy/anger/sadness/surprise/fear/disgust/neutral")
    intensity: float = Field(..., description="情绪强度，范围0-1")
    emotion_words: list[str] = Field(..., description="与情绪相关的词汇列表")


class ExtractValueModel(BaseModel):
    value: str = Field(..., description="价值观标题")
    evidence: str = Field(..., description="支撑证据")
    confidence: float = Field(..., description="置信度，范围0-1")


class ExtractMbitModel(BaseModel):
    mbti_type: Literal["ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP", "INFJ", "INFP", "ENFJ", "ENFP", "INTJ", "INTP", "ENTJ", "ENTP"] = Field(
        ..., description="MBTI类型")
    confidence: float = Field(..., description="整体置信度，范围0-1")
    key_traits: list[str] = Field(..., description="关键特征列表")


class ExtractProductSentimentModal(BaseModel):
    has_dimension_content: bool = Field(
        ..., description="是否包含维度内容")
    emotion_type: Literal["positive", "negative", "neutral"] = Field(
        ..., description="情感类型(positive/negative/neutral)")
    intensity: float = Field(..., description="情感强度，范围0-1")
    relevant_snippets: list[str] = Field(..., description="与情感相关的文本片段列表")
