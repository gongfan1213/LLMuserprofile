import json
import logging
from datetime import datetime
from typing import Dict, List, Any, TypedDict, Annotated

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from src.core.intent_analysis import IntentAnalyzer
from src.core.sentiment_analysis import SentimentAnalyzer
from src.core.tag_extraction import TagExtractor
from src.core.user_segmentation import UserSegmentationEngine
from src.data.loader import save_user_profile
from src.models.user_profile import UserTag, UserIntent, UserEmotion, UserProfile, TagType, TagCategory

logger = logging.getLogger(__name__)


class ProfileWorkflowState(TypedDict):
    """工作流状态定义"""
    user_id: str
    input_data: Dict[str, Any]
    extracted_tags: List[UserTag]
    analyzed_intents: List[UserIntent]
    analyzed_emotions: List[UserEmotion]
    user_profile: UserProfile | None
    error_messages: List[str]
    workflow_status: str
    messages: Annotated[List, add_messages]


class UserProfileWorkflow:
    """用户画像生成工作流"""

    def __init__(self):
        self.tag_extractor = TagExtractor()
        self.intent_analyzer = IntentAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.segmentation_engine = UserSegmentationEngine()

        # 构建工作流图
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> CompiledStateGraph:
        """构建工作流图"""
        workflow = StateGraph(ProfileWorkflowState)

        # 添加节点
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("extract_explicit_tags", self._extract_explicit_tags)
        workflow.add_node("extract_implicit_tags", self._extract_implicit_tags)
        workflow.add_node("analyze_intents", self._analyze_intents)
        workflow.add_node("analyze_emotions", self._analyze_emotions)
        workflow.add_node("analyze_values", self._analyze_values)
        workflow.add_node("generate_profile", self._generate_profile)
        workflow.add_node("validate_profile", self._validate_profile)
        workflow.add_node("error_handler", self._error_handler)

        # 设置入口点
        workflow.set_entry_point("validate_input")

        # 添加边（定义流程）
        workflow.add_edge("validate_input", "extract_explicit_tags")
        workflow.add_edge("extract_explicit_tags", "extract_implicit_tags")
        workflow.add_edge("extract_implicit_tags", "analyze_intents")
        workflow.add_edge("analyze_intents", "analyze_emotions")
        workflow.add_edge("analyze_emotions", "analyze_values")
        workflow.add_edge("analyze_values", "generate_profile")
        workflow.add_edge("generate_profile", "validate_profile")

        # 添加条件边
        workflow.add_conditional_edges(
            "validate_profile",
            self._should_end_workflow,
            {
                "end": END,
                "error": "error_handler"
            }
        )
        workflow.add_edge("error_handler", END)
        return workflow.compile()

    def generate_profile(self, user_id: str, input_data: Dict[str, Any]) -> UserProfile:
        # 初始化状态
        initial_state = ProfileWorkflowState(
            user_id=user_id,
            input_data=input_data,
            extracted_tags=[],
            analyzed_intents=[],
            analyzed_emotions=[],
            user_profile=None,
            error_messages=[],
            workflow_status="running",
            messages=[]
        )
        # 执行工作流
        final_state = self.workflow.invoke(initial_state)
        logger.info(f"工作流执行完成，执行结果：\n {json.dumps(final_state, indent=4, ensure_ascii=False, default=str)}")
        if final_state["workflow_status"] == "completed":
            user_profile: UserProfile = final_state["user_profile"]
            save_user_profile(user_profile, user_id)
            return final_state["user_profile"]
        else:
            raise Exception(f"Profile generation failed: {final_state['error_messages']}")

    @staticmethod
    def _validate_input(state: ProfileWorkflowState) -> ProfileWorkflowState:
        """验证输入数据"""
        user_id = state["user_id"]
        input_data = state["input_data"]

        state["messages"].append(HumanMessage(content=f"开始为用户 {user_id} 生成画像"))

        if not user_id:
            state["error_messages"].append("用户ID不能为空")
            state["workflow_status"] = "error"
            return state

        if not input_data:
            state["error_messages"].append("输入数据不能为空")
            state["workflow_status"] = "error"
            return state

        state["messages"].append(AIMessage(content="输入数据验证通过"))
        return state

    def _extract_explicit_tags(self, state: ProfileWorkflowState) -> ProfileWorkflowState:
        """提取显式标签"""
        logger.info("开始提取显式标签")
        try:
            input_data = state["input_data"]

            state["messages"].append(AIMessage(content="开始提取显式标签"))

            # 提取显式标签
            explicit_tags = self.tag_extractor.extract_tags(
                input_data, TagType.EXPLICIT
            )

            state["extracted_tags"].extend(explicit_tags)
            state["messages"].append(AIMessage(
                content=f"成功提取 {len(explicit_tags)} 个显式标签"
            ))
        except Exception as e:
            logger.exception(f"显式标签提取失败：{str(e)}")
            state["error_messages"].append(f"显式标签提取失败: {str(e)}")
            state["messages"].append(AIMessage(content=f"显式标签提取出错: {str(e)}"))

        return state

    def _extract_implicit_tags(self, state: ProfileWorkflowState) -> ProfileWorkflowState:
        """提取隐式标签"""
        logging.info("开始提取隐式标签")
        try:
            input_data = state["input_data"]

            state["messages"].append(AIMessage(content="开始提取隐式标签"))
            # 提取隐式标签
            implicit_tags = self.tag_extractor.extract_tags(input_data, TagType.IMPLICIT)
            state["extracted_tags"].extend(implicit_tags)
            state["messages"].append(AIMessage(
                content=f"成功提取 {len(implicit_tags)} 个隐式标签"
            ))

        except Exception as e:
            logger.exception(f"隐式标签提取失败：{str(e)}")
            state["error_messages"].append(f"隐式标签提取失败: {str(e)}")
            state["messages"].append(AIMessage(content=f"隐式标签提取出错: {str(e)}"))

        return state

    def _analyze_intents(self, state: ProfileWorkflowState) -> ProfileWorkflowState:
        """分析用户意图"""
        logging.info("开始分析用户意图")
        try:
            input_data = state["input_data"]
            state["messages"].append(AIMessage(content="开始分析用户意图"))

            # 分析用户意图
            intents = self.intent_analyzer.analyze_user_intent(input_data)
            state["analyzed_intents"].extend(intents)

            state["messages"].append(AIMessage(
                content=f"成功分析 {len(intents)} 个用户意图和相关偏好标签"
            ))

        except Exception as e:
            logger.exception(f"用户意图分析失败：{str(e)}")
            state["error_messages"].append(f"意图分析失败: {str(e)}")
            state["messages"].append(AIMessage(content=f"意图分析出错: {str(e)}"))

        return state

    def _analyze_emotions(self, state: ProfileWorkflowState) -> ProfileWorkflowState:
        """分析用户情绪"""
        logging.info("开始分析用户情绪")
        try:
            input_data = state["input_data"]

            state["messages"].append(AIMessage(content="开始分析用户情绪"))

            # 分析用户情绪
            emotions = self.sentiment_analyzer.analyze_user_emotions(input_data)
            state["analyzed_emotions"].extend(emotions)
            state["messages"].append(AIMessage(
                content=f"成功分析 {len(emotions)} 条情绪记录"
            ))

        except Exception as e:
            logger.exception(f"用户情绪分析失败: {str(e)}")
            state["error_messages"].append(f"情绪分析失败: {str(e)}")
            state["messages"].append(AIMessage(content=f"情绪分析出错: {str(e)}"))

        return state

    def _analyze_values(self, state: ProfileWorkflowState) -> ProfileWorkflowState:
        """分析价值观"""
        try:
            input_data = state["input_data"]

            state["messages"].append(AIMessage(content="开始分析用户价值观"))

            # 分析价值观
            value_tags = self.sentiment_analyzer.analyze_user_values(input_data)
            state["extracted_tags"].extend(value_tags)

            state["messages"].append(AIMessage(
                content=f"成功分析价值观，生成 {len(value_tags)} 个价值观标签"
            ))

        except Exception as e:
            state["error_messages"].append(f"价值观分析失败: {str(e)}")
            state["messages"].append(AIMessage(content=f"价值观分析出错: {str(e)}"))

        return state

    def _generate_profile(self, state: ProfileWorkflowState) -> ProfileWorkflowState:
        """生成用户画像"""
        try:
            user_id = state["user_id"]
            extracted_tags = state["extracted_tags"]
            analyzed_intents = state["analyzed_intents"]
            analyzed_emotions = state["analyzed_emotions"]

            state["messages"].append(AIMessage(content="开始生成用户画像"))

            # 计算画像完整度
            completeness = self._calculate_completeness(extracted_tags)

            # 创建用户画像
            user_profile = UserProfile(
                user_id=user_id,
                tags=extracted_tags,
                emotions=analyzed_emotions,
                intents=analyzed_intents,
                segments=[],
                profile_completeness=completeness,
                last_activity=datetime.now(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={"workflow_version": "1.0", "generation_time": datetime.now().isoformat()}
            )

            state["user_profile"] = user_profile
            state["messages"].append(AIMessage(
                content=f"成功生成用户画像，包含 {len(extracted_tags)} 个标签，完整度 {completeness:.2f}"
            ))

        except Exception as e:
            logger.exception(f"画像生成失败：{str(e)}")
            state["error_messages"].append(f"画像生成失败: {str(e)}")
            state["messages"].append(AIMessage(content=f"画像生成出错: {str(e)}"))

        return state



    def _validate_profile(self, state: ProfileWorkflowState) -> ProfileWorkflowState:
        """验证用户画像"""
        try:
            user_profile = state["user_profile"]

            if not user_profile:
                state["error_messages"].append("用户画像生成失败")
                state["workflow_status"] = "error"
                return state

            state["messages"].append(AIMessage(content="开始验证用户画像"))

            # 验证画像质量
            validation_results = self._validate_profile_quality(user_profile)

            if validation_results["is_valid"]:
                state["workflow_status"] = "completed"
                state["messages"].append(AIMessage(content="用户画像验证通过，生成完成"))
            else:
                state["error_messages"].extend(validation_results["errors"])
                state["workflow_status"] = "error"
                state["messages"].append(AIMessage(
                    content=f"用户画像验证失败: {validation_results['errors']}"
                ))

        except Exception as e:
            state["error_messages"].append(f"画像验证失败: {str(e)}")
            state["workflow_status"] = "error"
            state["messages"].append(AIMessage(content=f"画像验证出错: {str(e)}"))

        return state

    @staticmethod
    def _error_handler(state: ProfileWorkflowState) -> ProfileWorkflowState:
        """错误处理"""
        state["workflow_status"] = "error"
        error_summary = "; ".join(state["error_messages"])
        state["messages"].append(AIMessage(content=f"工作流执行失败: {error_summary}"))
        return state

    @staticmethod
    def _should_end_workflow(state: ProfileWorkflowState) -> str:
        """判断是否结束工作流"""
        if state["workflow_status"] == "completed":
            return "end"
        else:
            return "error"

    @staticmethod
    def _calculate_completeness(tags: List[UserTag]) -> float:
        """计算画像完整度"""
        if not tags:
            return 0.0

        # 按类别统计标签
        covered_categories = set()
        for tag in tags:
            covered_categories.add(tag.category)

        # 计算覆盖率，收集的标签/总计的标签
        total_categories = len(TagCategory)
        coverage = len(covered_categories) / total_categories

        # 考虑标签置信度均值
        avg_confidence = sum(tag.confidence for tag in tags) / len(tags)

        # 综合计算完整度
        completeness = (coverage * 0.7 + avg_confidence * 0.3)
        return min(completeness, 1.0)

    @staticmethod
    def _validate_profile_quality(user_profile: UserProfile) -> Dict[str, Any]:
        """验证画像质量"""
        errors = []

        # 检查基本字段
        if not user_profile.user_id:
            errors.append("用户ID为空")

        # 检查标签数量
        if len(user_profile.tags) < 3:
            errors.append("标签数量过少，需要至少3个标签")

        # 检查画像完整度
        if user_profile.profile_completeness < 0.3:
            errors.append("画像完整度过低")

        # 检查标签置信度
        low_confidence_tags = [tag for tag in user_profile.tags if tag.confidence < 0.4]
        if len(low_confidence_tags) > len(user_profile.tags) * 0.5:
            errors.append("过多低置信度标签")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": []
        }
