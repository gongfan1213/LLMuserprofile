import uuid
from abc import ABC
from typing import Dict, List, Any, Optional

from langchain.prompts import ChatPromptTemplate

from src.llm_client import create_llm
from src.models.extract_models import KeywordCollection, InterestCollection, EntityCollection
from src.models.user_profile import TagType, UserTag, TagCategory


class TagExtractor(ABC):
    """标签提取器基类"""

    def __init__(self):
        self.llm = create_llm()

    def extract_tags(self, data: Dict[str, Any], tag_type: TagType) -> List[UserTag]:

        if tag_type == TagType.EXPLICIT:
            return self._extract_explicit_tags(data)
        elif tag_type == TagType.IMPLICIT:
            return self._extract_implicit_tags(data)
        else:
            raise ValueError(f"Unsupported tag type: {tag_type}")

    def _extract_explicit_tags(self, data: Dict[str, Any]) -> List[UserTag]:
        """提取显式标签，显式标签置信度直接给1.0"""
        explicit_tags = []

        # 处理注册信息
        if "registration_data" in data:
            reg_data = data["registration_data"]

            # 基础人口统计信息
            if "age" in reg_data:
                age_range = self._classify_age_range(reg_data["age"])
                explicit_tags.append(self._create_tag("age_range", age_range, TagCategory.DEMOGRAPHICS,
                                                      TagType.EXPLICIT, 1.0, "registration", {"raw_age": reg_data["age"]}
                                                      ))

            if "gender" in reg_data:
                explicit_tags.append(self._create_tag("gender", reg_data["gender"], TagCategory.DEMOGRAPHICS,
                                                      TagType.EXPLICIT, 1.0, "registration"
                                                      ))

            if "occupation" in reg_data:
                explicit_tags.append(self._create_tag("occupation", reg_data["occupation"], TagCategory.BASIC_INFO,
                                                      TagType.EXPLICIT, 1.0, "registration"
                                                      ))

            if "education" in reg_data:
                explicit_tags.append(self._create_tag("education", reg_data["education"], TagCategory.DEMOGRAPHICS,
                                                      TagType.EXPLICIT, 1.0, "registration"
                                                      ))

            if "location" in reg_data:
                explicit_tags.append(self._create_tag("location", reg_data["location"], TagCategory.DEMOGRAPHICS,
                                                      TagType.EXPLICIT, 1.0, "registration"
                                                      ))

        # 处理问卷数据
        if "survey_data" in data:
            survey_data = data["survey_data"]
            for question, answer in survey_data.items():
                category = self._classify_survey_category(question)
                # 对于问卷问题，只给0.9的置信度
                explicit_tags.append(self._create_tag(f"survey_{question}", answer, category,
                                                      TagType.EXPLICIT, 0.9, "survey", {"question": question}
                                                      ))
        return explicit_tags

    def _extract_implicit_tags(self, data: Dict[str, Any]) -> List[UserTag]:
        """提取隐式标签"""
        implicit_tags = []

        # 处理文本数据
        if "text_data" in data:
            text_data = data["text_data"]

            # NER实体识别
            ner_tags = self._extract_entities(text_data)
            implicit_tags.extend(ner_tags)

            # 关键词提取
            keyword_tags = self._extract_keywords(text_data)
            implicit_tags.extend(keyword_tags)

            # 兴趣偏好分析
            interest_tags = self._extract_interests(text_data)
            implicit_tags.extend(interest_tags)

        # 处理行为数据
        if "behavior_data" in data:
            behavior_data = data["behavior_data"]
            behavior_tags = self._extract_behavior_tags(behavior_data)
            implicit_tags.extend(behavior_tags)

        return implicit_tags

    def _extract_entities(self, text_data: List[str]) -> List[UserTag]:
        """实体识别"""
        if not text_data:
            return []

        # 合并文本数据，实际开发中可以根据长度进行一定的限制，比如压缩或者截断
        combined_text = " ".join(text_data)

        # 创建NER提示模板
        ner_prompt = ChatPromptTemplate.from_template("""
        请从以下文本中提取实体信息，包括品牌、产品、人物、地点、组织等：
        (entity_type只能使用以下值：brand/product/person/location/organization/other)
        
        需要提取的文本: 
        
        {text}
        
        """)
        chain = ner_prompt | self.llm.with_structured_output(EntityCollection, method="function_calling")
        response: EntityCollection = chain.invoke(input={"text": combined_text})
        tags = []
        for entity in response.entities:
            # 根据实体类型映射到标签分类
            category = self._map_entity_type_to_category(entity.type)
            tags.append(self._create_tag(
                f"entity_{entity.type}",
                entity.value,
                category,
                TagType.IMPLICIT,
                entity.confidence,
                "ner",  # 命名实体识别
                {"entity_type": entity.type, "context": entity.context}
            ))
        return tags

    def _extract_keywords(self, text_data: List[str]) -> List[UserTag]:
        """关键词提取"""
        if not text_data:
            return []

        combined_text = " ".join(text_data[:10])

        keyword_prompt = ChatPromptTemplate.from_template("""
        请从以下文本中提取高频讨论的关键词，这些关键词应该反映用户的兴趣和关注点：
        
        需要提取的文本:
        
        {text}
        
        请返回前10个最重要的关键词
        """)

        chain = keyword_prompt | self.llm.with_structured_output(KeywordCollection, method="function_calling")
        response: KeywordCollection = chain.invoke(input={"text": combined_text})
        tags = []
        for kw in response.keywords:
            tags.append(self._create_tag(
                "keyword",
                kw.keyword,
                TagCategory.INTERESTS,
                TagType.IMPLICIT,
                kw.relevance,
                "keyword_extraction",
                {"frequency": kw.frequency, "context": kw.context}
            ))
        return tags

    def _extract_interests(self, text_data: List[str]) -> List[UserTag]:
        """兴趣偏好提取"""
        if not text_data:
            return []

        combined_text = " ".join(text_data[:15])

        interest_prompt = ChatPromptTemplate.from_template("""
        请分析以下文本，识别用户的兴趣爱好和偏好：
        (对于兴趣分类，只能使用以下值： sports/technology/entertainment/lifestyle/education/travel/food/finance/other)
        需要提取的文本: 
        
        {text}
       
        """)
        chain = interest_prompt | self.llm.with_structured_output(InterestCollection, method="function_calling")
        response: InterestCollection = chain.invoke(input={"text": combined_text})
        tags = []
        for interest in response.interests:
            tags.append(self._create_tag(
                "interest",
                interest.interest,
                TagCategory.INTERESTS,
                TagType.IMPLICIT,
                interest.intensity,
                "interest_analysis",
                {
                    "interest_category": interest.category,
                    "evidence": interest.evidence
                }
            ))
        return tags

    def _extract_behavior_tags(self, behavior_data: Dict[str, Any]) -> List[UserTag]:
        """行为标签提取"""
        tags = []

        # 活跃度分析
        if "activity_data" in behavior_data:
            activity = behavior_data["activity_data"]
            # 活跃度等级分类
            activity_level = self._classify_activity_level(activity)
            # 置信度我们给1.0，可以根据你的实际业务进行调整
            tags.append(self._create_tag(
                "activity_level", activity_level, TagCategory.BEHAVIOR,
                TagType.IMPLICIT, 1.0, "behavior_analysis", activity
            ))

        # 使用时间分析
        if "usage_time" in behavior_data:
            usage_patterns = self._analyze_usage_patterns(behavior_data["usage_time"])
            for pattern_name, pattern_value in usage_patterns.items():
                # 置信度我们给0.7，因为这里的用户分类不能百分之百确定是准确的，可以根据你的实际业务进行调整
                tags.append(self._create_tag(
                    pattern_name, pattern_value, TagCategory.BEHAVIOR,
                    TagType.IMPLICIT, 0.7, "usage_analysis"
                ))
        return tags

    @staticmethod
    def _create_tag(name: str, value: Any, category: TagCategory,
        tag_type: TagType, confidence: float, source: str,
        metadata: Optional[Dict[str, Any]] = None) -> UserTag:
        """创建标签对象"""
        return UserTag(
            tag_id=str(uuid.uuid4()),
            name=name,
            value=value,
            category=category,
            tag_type=tag_type,
            confidence=confidence,
            source=source,
            metadata=metadata or {}
        )

    @staticmethod
    def _classify_age_range(age: int) -> str:
        """年龄分段分类"""
        if age < 18:
            return "未成年"
        elif age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        elif age < 65:
            return "55-64"
        else:
            return "65+"

    @staticmethod
    def _classify_survey_category(question: str) -> TagCategory:
        """问卷问题分类，根据不同的一些问题的关键词进行分类"""
        question_lower = question.lower()
        if any(word in question_lower for word in ["hobby", "interest", "like", "prefer"]):
            return TagCategory.INTERESTS
        elif any(word in question_lower for word in ["buy", "purchase", "shopping"]):
            return TagCategory.CONSUMPTION
        elif any(word in question_lower for word in ["value", "belief", "opinion"]):
            return TagCategory.VALUES
        else:
            return TagCategory.BASIC_INFO

    @staticmethod
    def _map_entity_type_to_category(entity_type: str) -> TagCategory:
        """实体类型映射到标签分类"""
        entity_mapping = {
            "brand": TagCategory.PREFERENCES,
            "product": TagCategory.INTERESTS,
            "location": TagCategory.BEHAVIOR,
            "person": TagCategory.INTERESTS,
            "organization": TagCategory.INTERESTS
        }
        return entity_mapping.get(entity_type, TagCategory.BASIC_INFO)

    @staticmethod
    def _classify_activity_level(activity_data: Dict[str, Any]) -> str:
        """活跃度分类"""
        # 简单的活跃度分类逻辑
        daily_actions = activity_data.get("daily_actions", 0)
        if daily_actions > 50:
            return "高活跃"
        elif daily_actions > 20:
            return "中活跃"
        elif daily_actions > 5:
            return "低活跃"
        else:
            return "非活跃"

    @staticmethod
    def _analyze_usage_patterns(usage_time: Dict[str, Any]) -> Dict[str, str]:
        """使用模式分析"""
        patterns = {}

        # 分析使用时段
        peak_hours = usage_time.get("peak_hours", [])
        if peak_hours:
            if any(hour in peak_hours for hour in range(6, 12)):
                patterns["morning_user"] = "是"
            if any(hour in peak_hours for hour in range(12, 18)):
                patterns["afternoon_user"] = "是"
            if any(hour in peak_hours for hour in range(18, 24)):
                patterns["evening_user"] = "是"
            if any(hour in peak_hours for hour in list(range(0, 6)) + list(range(22, 24))):
                patterns["night_user"] = "是"

        return patterns
