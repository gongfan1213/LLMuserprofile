import json
import uuid
from collections import defaultdict
from typing import Dict, List, Any, Optional

import numpy as np
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.llm_client import create_llm
from src.models.user_profile import UserProfile, UserSegment, TagCategory


class SegmentationRule(BaseModel):
    """分群规则"""
    rule_id: str = Field(..., description="规则ID")
    condition_type: str = Field(..., description="条件类型")
    field: str = Field(..., description="字段名")
    operator: str = Field(..., description="操作符")
    value: Any = Field(..., description="比较值")
    weight: float = Field(default=1.0, description="权重")


class SegmentProfile(BaseModel):
    """分群画像"""
    segment_id: str = Field(..., description="分群ID")
    common_tags: List[Dict[str, Any]] = Field(..., description="共同标签")
    behavior_patterns: List[str] = Field(..., description="行为模式")
    demographics: Dict[str, Any] = Field(..., description="人口统计特征")
    interests: List[str] = Field(..., description="共同兴趣")


class LifecyclePrediction(BaseModel):
    """生命周期预测"""
    user_id: str = Field(..., description="用户ID")
    current_stage: str = Field(..., description="当前阶段")
    predicted_stage: str = Field(..., description="预测阶段")
    transition_probability: float = Field(..., description="转换概率")
    key_indicators: List[str] = Field(..., description="关键指标")
    time_to_transition: Optional[int] = Field(None, description="预计转换时间(天)")


class UserSegmentationEngine:
    """用户分群引擎"""

    def __init__(self):
        self.llm = create_llm()
        self.segments = {}  # 存储分群信息

    def create_intelligent_segments(self, users: List[UserProfile],
        segment_count: Optional[int] = None) -> List[UserSegment]:
        if not users:
            return []

        # 特征提取
        features = self._extract_user_features(users)
        # 使用LLM分析最佳分群策略
        # segmentation_strategy = self._analyze_segmentation_strategy(users)
        segmentation_strategy = {
            "method": "ml_based",
        }

        # 执行分群
        if segmentation_strategy.get("method") == "llm_based":
            segments = self._llm_based_segmentation(users, segmentation_strategy)
        else:
            segments = self._ml_based_segmentation(users, features, segment_count)

        return segments

    @staticmethod
    def _extract_user_features(users: List[UserProfile]) -> np.ndarray:
        """提取用户特征向量"""
        features = []

        for user in users:
            feature_vector = []

            # 标签特征
            tag_categories = {}
            for tag in user.tags:
                category = tag.category.value
                tag_categories[category] = tag_categories.get(category, 0) + 1

            # 标准化各类别标签数量
            for category in TagCategory:
                feature_vector.append(tag_categories.get(category.value, 0))

            # 活跃度特征
            active_intents = len(user.get_active_intents())
            recent_emotions = len(user.get_recent_emotions())
            feature_vector.extend([active_intents, recent_emotions])

            # 画像完整度
            feature_vector.append(user.profile_completeness)
            features.append(feature_vector)

        return np.array(features)


    def _llm_based_segmentation(self, users: List[UserProfile],
        strategy: Dict[str, Any]) -> List[UserSegment]:
        """基于LLM的分群"""
        segments = []

        # 将用户数据转换为文本描述
        user_descriptions = []
        for user in users:
            description = self._generate_user_description(user)
            user_descriptions.append({"user_id": user.user_id, "description": description})

        # 使用LLM进行分群
        segmentation_prompt = ChatPromptTemplate.from_template("""
        请根据以下用户描述进行智能分群：
        
        用户描述：
        {user_descriptions}
        
        推荐分群：
        {recommended_segments}
        
        请将用户分配到合适的分群中，以JSON格式返回：
        {{
            "segments": [
                {{
                    "segment_name": "分群名称",
                    "segment_description": "分群描述",
                    "members": ["user_id1", "user_id2"],
                    "key_characteristics": ["特征1", "特征2"],
                   
                }}
            ]
        }}
        """)

        try:
            user_desc_text = json.dumps(user_descriptions[:50], ensure_ascii=False)  # 限制数量
            rec_segments_text = json.dumps(strategy.get("recommended_segments", []), ensure_ascii=False)

            response = self.llm.invoke(segmentation_prompt.format(
                user_descriptions=user_desc_text,
                recommended_segments=rec_segments_text
            ))
            result = json.loads(response.content)

            # 创建分群对象
            for seg_data in result.get("segments", []):
                segment = UserSegment(
                    segment_id=str(uuid.uuid4()),
                    segment_name=seg_data["segment_name"],
                    description=seg_data["segment_description"],
                    member_count=len(seg_data.get("members", []))
                )
                segments.append(segment)
                self.segments[segment.segment_id] = segment

        except Exception as e:
            print(f"LLM segmentation error: {e}")

        return segments

    def _ml_based_segmentation(self, users: List[UserProfile], features: np.ndarray,
        segment_count: Optional[int]) -> List[UserSegment]:
        """基于机器学习的分群"""
        if len(users) < 3:
            return []

        # 特征标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 确定分群数量,最大分群先定义为5
        if segment_count is None:
            segment_count = min(5, max(2, len(users) // 10))

        # K-means聚类
        kmeans = KMeans(n_clusters=segment_count, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # 创建分群
        segments = []
        user_clusters = defaultdict(list)

        for i, user in enumerate(users):
            cluster_id = cluster_labels[i]
            user_clusters[cluster_id].append(user)

        for cluster_id, cluster_users in user_clusters.items():
            # 生成分群名称和描述
            segment_name = f"用户群体_{cluster_id + 1}"
            segment_description = self._generate_segment_description(cluster_users)

            segment = UserSegment(
                segment_id=str(uuid.uuid4()),
                segment_name=segment_name,
                description=segment_description,
                user_ids=[user.user_id for user in cluster_users],
                member_count=len(cluster_users)
            )

            segments.append(segment)
            self.segments[segment.segment_id] = segment

        return segments



    @staticmethod
    def _generate_user_description(user: UserProfile) -> str:
        """生成用户描述"""
        description_parts = [f"用户ID: {user.user_id}", f"画像完整度: {user.profile_completeness:.2f}"]
        # 标签信息
        if user.tags:
            tag_summaries = []
            for category in TagCategory:
                category_tags = user.get_tags_by_category(category)
                if category_tags:
                    tag_values = [str(tag.value) for tag in category_tags[:3]]
                    tag_summaries.append(f"{category.value}: {', '.join(tag_values)}")

            if tag_summaries:
                description_parts.append("标签: " + "; ".join(tag_summaries))

        # 意图信息
        active_intents = user.get_active_intents()
        if active_intents:
            intent_names = [intent.intent_name for intent in active_intents[:3]]
            description_parts.append(f"活跃意图: {', '.join(intent_names)}")

        return "; ".join(description_parts)



    def _llm_segment_matching(self, user_profile: UserProfile,
        segments: List[UserSegment]) -> List[str]:
        """使用LLM进行分群匹配"""
        if not segments:
            return []

        # 生成分群描述
        segment_descriptions = []
        for segment in segments:
            description = f"分群ID: {segment.segment_id}, 名称: {segment.segment_name}, 描述: {segment.description}"
            segment_descriptions.append(description)

        user_description = self._generate_user_description(user_profile)

        matching_prompt = ChatPromptTemplate.from_template("""
        请判断用户应该属于哪些分群：
        
        用户描述：{user_description}
        
        可选分群：
        {segment_descriptions}
        
        请返回匹配的分群ID，以JSON格式：
        {{
            "matching_segments": ["segment_id1", "segment_id2"],
            "matching_reasons": ["原因1", "原因2"]
        }}
        """)

        try:
            response = self.llm.invoke(matching_prompt.format(
                user_description=user_description,
                segment_descriptions="\n".join(segment_descriptions)
            ))

            result = json.loads(response.content)
            return result.get("matching_segments", [])

        except Exception as e:
            print(f"LLM segment matching error: {e}")
            return []

    @staticmethod
    def _generate_segment_description(users: List[UserProfile]) -> str:
        """生成分群描述"""
        if not users:
            return "空分群"

        # 统计分群特征
        avg_completeness = sum(u.profile_completeness for u in users) / len(users)
        total_tags = sum(len(u.tags) for u in users)
        avg_tags = total_tags / len(users)

        # 统计主要标签类别
        category_counts = defaultdict(int)
        for user in users:
            for tag in user.tags:
                category_counts[tag.category.value] += 1

        main_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        category_names = [cat[0] for cat in main_categories]

        description = f"包含{len(users)}个用户，平均画像完整度{avg_completeness:.2f}，平均标签数{avg_tags:.1f}，主要特征类别：{', '.join(category_names)}"

        return description

