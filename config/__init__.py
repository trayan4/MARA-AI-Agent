"""
Configuration loader for MARA system.
Loads settings from config/settings.yaml and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4-turbo-preview"
    vision_model: str = "gpt-4-vision-preview"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 120
    max_retries: int = 3
    retry_delay: int = 2


class EmbeddingsConfig(BaseModel):
    model: str = "BAAI/bge-base-en-v1.5"
    dimension: int = 768
    batch_size: int = 32
    max_length: int = 512


class VectorStoreConfig(BaseModel):
    type: str = "sqlite"
    path: str = "data/vector_store.db"
    top_k: int = 5
    similarity_threshold: float = 0.7
    use_hybrid: bool = True
    hybrid_alpha: float = 0.7


class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = ["\n\n", "\n", ". ", " ", ""]


class AgentConfig(BaseModel):
    max_retries: int = 3
    timeout: int = 60


class PlannerConfig(AgentConfig):
    use_meta_planning: bool = False


class RAGConfig(AgentConfig):
    max_chunks: int = 10
    rerank: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class VisionConfig(AgentConfig):
    max_image_size: int = 4096
    supported_formats: list[str] = ["png", "jpg", "jpeg", "webp"]
    detail_level: str = "high"


class DataConfig(AgentConfig):
    max_rows: int = 100000
    safe_mode: bool = True


class CriticConfig(AgentConfig):
    confidence_threshold: float = 0.8
    hallucination_check: bool = True
    consistency_check: bool = True
    citation_validation: bool = True
    trigger_replan_threshold: float = 0.6


class ReportConfig(BaseModel):
    format: str = "json"
    include_evidence: bool = True
    include_metadata: bool = True


class AgentsConfig(BaseModel):
    planner: PlannerConfig = PlannerConfig()
    rag: RAGConfig = RAGConfig()
    vision: VisionConfig = VisionConfig()
    data: DataConfig = DataConfig()
    critic: CriticConfig = CriticConfig()
    report: ReportConfig = ReportConfig()


class OrchestrationConfig(BaseModel):
    max_iterations: int = 10
    parallel_execution: bool = True
    timeout: int = 300


class ShortTermMemoryConfig(BaseModel):
    enabled: bool = True
    path: str = "data/logs/agent_traces.jsonl"


class LongTermMemoryConfig(BaseModel):
    enabled: bool = False
    path: str = "data/knowledge_store.db"


class CacheConfig(BaseModel):
    enabled: bool = True
    type: str = "sqlite"
    path: str = "data/cache.db"
    ttl: int = 3600


class MemoryConfig(BaseModel):
    short_term: ShortTermMemoryConfig = ShortTermMemoryConfig()
    long_term: LongTermMemoryConfig = LongTermMemoryConfig()
    cache: CacheConfig = CacheConfig()


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    cors_enabled: bool = True
    cors_origins: list[str] = ["*"]
    rate_limit: int = 100
    auth_enabled: bool = False


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    path: str = "data/logs/mara.log"


class TracingConfig(BaseModel):
    enabled: bool = False
    endpoint: str = "http://localhost:4318"


class MetricsConfig(BaseModel):
    enabled: bool = False
    port: int = 9090


class ObservabilityConfig(BaseModel):
    logging: LoggingConfig = LoggingConfig()
    tracing: TracingConfig = TracingConfig()
    metrics: MetricsConfig = MetricsConfig()


class PathsConfig(BaseModel):
    data_dir: str = "data"
    uploads_dir: str = "data/uploads"
    outputs_dir: str = "data/outputs"
    logs_dir: str = "data/logs"


class Settings(BaseSettings):
    """Main settings class that loads from YAML and environment variables."""
    
    # OpenAI API Key (from environment)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    
    # Component configurations
    llm: LLMConfig = LLMConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    agents: AgentsConfig = AgentsConfig()
    orchestration: OrchestrationConfig = OrchestrationConfig()
    memory: MemoryConfig = MemoryConfig()
    api: APIConfig = APIConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    paths: PathsConfig = PathsConfig()
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "settings.yaml"
    
    if not config_path.exists():
        return {}
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_settings() -> Settings:
    """
    Get settings instance with values from YAML and environment variables.
    Environment variables take precedence over YAML.
    """
    # Load YAML config
    yaml_config = load_yaml_config()
    
    # Create settings instance with YAML data
    settings = Settings(**yaml_config)
    
    # Ensure data directories exist
    for path_attr in ["data_dir", "uploads_dir", "outputs_dir", "logs_dir"]:
        path = Path(getattr(settings.paths, path_attr))
        path.mkdir(parents=True, exist_ok=True)
    
    return settings


# Global settings instance
settings = get_settings()