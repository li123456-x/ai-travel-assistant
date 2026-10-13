from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship
from config.database import Base


# ==================================================
# Travel Session
# ==================================================

class TravelSession(Base):

    __tablename__ = "travel_sessions"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    thread_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True
    )

    user_request = Column(
        Text,
        nullable=False
    )

    origin = Column(
        String(100),
        nullable=False
    )

    destination = Column(
        String(100),
        nullable=False
    )

    departure_date = Column(
        String(20),
        nullable=False
    )

    return_date = Column(
        String(20),
        nullable=False
    )

    days = Column(
        Integer,
        nullable=False
    )

    people = Column(
        Integer,
        nullable=False
    )

    budget = Column(
        Float,
        nullable=False
    )

    preferences = Column(
        Text
    )

    constraints = Column(
        Text
    )

    status = Column(
        String(50),
        default="planning"
    )

    created_at = Column(
        DateTime,
        default=datetime.now
    )

    updated_at = Column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now
    )

    plans = relationship(
        "TravelPlan",
        back_populates="session",
        cascade="all, delete-orphan"
    )

    # 新增：
    # 一个旅游任务可以有很多条 Agent 运行日志
    agent_runs = relationship(
        "AgentRun",
        back_populates="session",
        cascade="all, delete-orphan"
    )


# ==================================================
# Travel Plan
# ==================================================

class TravelPlan(Base):

    __tablename__ = "travel_plans"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    session_id = Column(
        Integer,
        ForeignKey(
            "travel_sessions.id",
            ondelete="CASCADE"
        ),
        nullable=False
    )

    version = Column(
        Integer,
        nullable=False
    )

    attractions_result = Column(
        Text
    )

    hotel_result = Column(
        Text
    )

    transport_result = Column(
        Text
    )

    planner_result = Column(
        Text,
        nullable=False
    )

    is_final = Column(
        Boolean,
        default=False
    )

    created_at = Column(
        DateTime,
        default=datetime.now
    )

    session = relationship(
        "TravelSession",
        back_populates="plans"
    )

    ai_reviews = relationship(
        "AIReview",
        back_populates="plan",
        cascade="all, delete-orphan"
    )

    human_reviews = relationship(
        "HumanReview",
        back_populates="plan",
        cascade="all, delete-orphan"
    )
    memory = relationship(
        "TravelPlanMemory", back_populates="plan", uselist=False,
        cascade="all, delete-orphan",
    )


# ==================================================
# AI Review
# ==================================================

class AIReview(Base):

    __tablename__ = "ai_reviews"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    plan_id = Column(
        Integer,
        ForeignKey(
            "travel_plans.id",
            ondelete="CASCADE"
        ),
        nullable=False
    )

    review_passed = Column(
        Boolean,
        nullable=False
    )

    review_result = Column(
        Text
    )

    created_at = Column(
        DateTime,
        default=datetime.now
    )

    plan = relationship(
        "TravelPlan",
        back_populates="ai_reviews"
    )


# ==================================================
# Human Review
# ==================================================

class HumanReview(Base):

    __tablename__ = "human_reviews"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    plan_id = Column(
        Integer,
        ForeignKey(
            "travel_plans.id",
            ondelete="CASCADE"
        ),
        nullable=False
    )

    approved = Column(
        Boolean,
        nullable=False
    )

    comment = Column(
        Text
    )

    created_at = Column(
        DateTime,
        default=datetime.now
    )

    plan = relationship(
        "TravelPlan",
        back_populates="human_reviews"
    )


# ==================================================
# Agent Harness Run
# ==================================================

class AgentRun(Base):

    __tablename__ = "agent_runs"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    # 属于哪一次旅游任务
    session_id = Column(
        Integer,
        ForeignKey(
            "travel_sessions.id",
            ondelete="CASCADE"
        ),
        nullable=True
    )

    # LangGraph thread_id
    thread_id = Column(
        String(100),
        nullable=True,
        index=True
    )

    # attraction / hotel / transport / planner / reviewer
    agent_name = Column(
        String(50),
        nullable=False,
        index=True
    )
    trigger_reason = Column(
        String(50),
        nullable=True
    )
    # 开始时间
    started_at = Column(
        DateTime,
        default=datetime.now,
        nullable=False
    )

    # 结束时间
    finished_at = Column(
        DateTime,
        nullable=True
    )

    # 总耗时，单位秒
    duration = Column(
        Float,
        nullable=True
    )

    # 是否执行成功
    success = Column(
        Boolean,
        default=False,
        nullable=False
    )

    # 重试了几次
    retry_count = Column(
        Integer,
        default=0,
        nullable=False
    )

    # 最终错误信息
    error_message = Column(
        Text,
        nullable=True
    )

    session = relationship(
        "TravelSession",
        back_populates="agent_runs"
    )
    execution_steps = relationship(
        "ExecutionStep",
        back_populates="agent_run",
        cascade="all, delete-orphan"
    )
class ExecutionStep(Base):

    __tablename__ = "execution_steps"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    agent_run_id = Column(
        Integer,
        ForeignKey(
            "agent_runs.id",
            ondelete="CASCADE"
        ),
        nullable=False
    )

    # tool / llm
    step_type = Column(
        String(30),
        nullable=False
    )

    # 比如 intercity_tavily / amap_city_route / transport_llm
    step_name = Column(
        String(100),
        nullable=False
    )

    duration = Column(
        Float,
        nullable=True
    )

    success = Column(
        Boolean,
        default=False,
        nullable=False
    )

    error_message = Column(
        Text,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.now
    )

    agent_run = relationship(
        "AgentRun",
        back_populates="execution_steps"
    )


class TravelPlanMemory(Base):
    __tablename__ = "travel_plan_memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(
        Integer, ForeignKey("travel_plans.id", ondelete="CASCADE"),
        nullable=False, unique=True,
    )
    request_signature = Column(String(64), nullable=False, index=True)
    request_payload = Column(JSON, nullable=False)
    result_snapshot = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    plan = relationship("TravelPlan", back_populates="memory")
