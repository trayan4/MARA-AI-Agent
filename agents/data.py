"""
Data Agent - Performs statistical analysis and data manipulation.
Uses Pandas, NumPy, and safe Python execution for data operations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import json

import pandas as pd
import numpy as np
from loguru import logger

from config import settings
from tools.python_executor import PythonExecutor
from tools.openai_client import get_openai_client


@dataclass
class DataAnalysisResult:
    """Represents a data analysis result."""
    
    operation: str
    data_path: Optional[str]
    result: Any
    summary: str
    insights: List[str]
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        # Handle non-serializable types
        serialized_result = self._serialize_result(self.result)
        
        return {
            'operation': self.operation,
            'data_path': self.data_path,
            'result': serialized_result,
            'summary': self.summary,
            'insights': self.insights,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    def _serialize_result(self, obj: Any) -> Any:
        """Convert result to JSON-serializable format."""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_result(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_result(item) for item in obj]
        else:
            return obj


class DataAgent:
    """
    Data analysis agent using Pandas and statistical methods.
    Performs descriptive statistics, trend analysis, anomaly detection, etc.
    """
    
    def __init__(self):
        """Initialize Data Agent."""
        self.executor = PythonExecutor()
        self.client = get_openai_client()
        
        self.max_rows = settings.agents.data.max_rows
        self.safe_mode = settings.agents.data.safe_mode
        self.max_retries = settings.agents.data.max_retries
        self.timeout = settings.agents.data.timeout
        
        logger.info(
            f"Data Agent initialized (max_rows={self.max_rows}, "
            f"safe_mode={self.safe_mode})"
        )
    
    def _load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            data_path: Path to CSV, Excel, or other data file
        
        Returns:
            DataFrame
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Determine file type and load
        extension = data_path.suffix.lower()
        
        if extension == '.csv':
            df = pd.read_csv(data_path)
        elif extension in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path)
        elif extension == '.json':
            df = pd.read_json(data_path)
        elif extension == '.parquet':
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        # Truncate if too large
        if len(df) > self.max_rows:
            logger.warning(f"DataFrame has {len(df)} rows, truncating to {self.max_rows}")
            df = df.head(self.max_rows)
        
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def _find_column(self, df: pd.DataFrame, column_name: str) -> Optional[str]:
        """
        Find column in dataframe (case-insensitive).
        
        Args:
            df: DataFrame to search
            column_name: Column name to find
        
        Returns:
            Actual column name or None
        """
        # Exact match first
        if column_name in df.columns:
            return column_name
        
        # Case-insensitive match
        column_lower = column_name.lower()
        for col in df.columns:
            if col.lower() == column_lower:
                return col
        
        # Partial match (for things like "sales" matching "Sales" or "SalesAmount")
        for col in df.columns:
            if column_lower in col.lower():
                return col
        
        return None
    
    def analyze_dataframe(
        self,
        data_path: Union[str, Path],
        analysis_type: str = "describe",
        columns: Optional[List[str]] = None
    ) -> DataAnalysisResult:
        """
        Perform statistical analysis on DataFrame.
        
        Args:
            data_path: Path to data file
            analysis_type: Type of analysis (describe, correlations, missing_values, value_counts)
            columns: Specific columns to analyze
        
        Returns:
            DataAnalysisResult with analysis
        """
        logger.info(f"Analyzing data: {data_path} ({analysis_type})")
        
        df = self._load_data(data_path)
        
        # Filter columns if specified
        if columns:
            df = df[columns]
        
        # Perform analysis based on type
        if analysis_type == "describe":
            result = self._analyze_describe(df)
        elif analysis_type == "correlations":
            result = self._analyze_correlations(df)
        elif analysis_type == "missing_values":
            result = self._analyze_missing_values(df)
        elif analysis_type == "value_counts":
            result = self._analyze_value_counts(df)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        return result
    
    def _analyze_describe(self, df: pd.DataFrame) -> DataAnalysisResult:
        """Generate descriptive statistics."""
        stats = df.describe(include='all').to_dict()
        
        # Generate insights
        insights = []
        
        for col in df.select_dtypes(include=[np.number]).columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            insights.append(f"{col}: mean={mean_val:.2f}, std={std_val:.2f}")
        
        summary = f"Analyzed {df.shape[0]} rows and {df.shape[1]} columns. Generated descriptive statistics."
        
        return DataAnalysisResult(
            operation="describe",
            data_path=None,
            result=stats,
            summary=summary,
            insights=insights,
            confidence=0.95,
            metadata={'shape': df.shape, 'dtypes': df.dtypes.astype(str).to_dict()}
        )
    
    def _analyze_correlations(self, df: pd.DataFrame) -> DataAnalysisResult:
        """Analyze correlations between numerical columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return DataAnalysisResult(
                operation="correlations",
                data_path=None,
                result={},
                summary="No numeric columns found for correlation analysis",
                insights=[],
                confidence=0.0,
                metadata={}
            )
        
        corr_matrix = numeric_df.corr()
        
        # Find strong correlations
        insights = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    insights.append(
                        f"Strong correlation between {col1} and {col2}: {corr_val:.3f}"
                    )
        
        summary = f"Calculated correlations for {len(numeric_df.columns)} numeric columns"
        
        return DataAnalysisResult(
            operation="correlations",
            data_path=None,
            result=corr_matrix.to_dict(),
            summary=summary,
            insights=insights,
            confidence=0.9,
            metadata={'numeric_columns': list(numeric_df.columns)}
        )
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> DataAnalysisResult:
        """Analyze missing values."""
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100).round(2)
        
        result = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_pct
        })
        
        insights = []
        for col in result.index:
            if result.loc[col, 'missing_percentage'] > 0:
                insights.append(
                    f"{col}: {result.loc[col, 'missing_count']} missing "
                    f"({result.loc[col, 'missing_percentage']:.1f}%)"
                )
        
        summary = f"Found missing values in {len(insights)} columns"
        
        return DataAnalysisResult(
            operation="missing_values",
            data_path=None,
            result=result.to_dict(),
            summary=summary,
            insights=insights,
            confidence=1.0,
            metadata={'total_rows': len(df)}
        )
    
    def _analyze_value_counts(self, df: pd.DataFrame) -> DataAnalysisResult:
        """Analyze value distributions for categorical columns."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        result = {}
        insights = []
        
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(10).to_dict()
            result[col] = value_counts
            
            top_value = df[col].value_counts().index[0]
            top_count = df[col].value_counts().iloc[0]
            insights.append(
                f"{col}: Most common value is '{top_value}' ({top_count} occurrences)"
            )
        
        summary = f"Analyzed value distributions for {len(categorical_cols)} categorical columns"
        
        return DataAnalysisResult(
            operation="value_counts",
            data_path=None,
            result=result,
            summary=summary,
            insights=insights,
            confidence=0.95,
            metadata={'categorical_columns': list(categorical_cols)}
        )
    
    def execute_pandas_code(
        self,
        code: str,
        data_path: Optional[Union[str, Path]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> DataAnalysisResult:
        """
        Execute custom Pandas code.
        
        Args:
            code: Python code to execute
            data_path: Optional data file to load as 'df'
            context: Additional context variables
        
        Returns:
            DataAnalysisResult with execution results
        """
        logger.info("Executing custom Pandas code")
        
        # Prepare context
        exec_context = context or {}
        
        if data_path:
            df = self._load_data(data_path)
            exec_context['df'] = df
        
        # Execute code
        execution_result = self.executor.execute(code, context=exec_context)
        
        if not execution_result['success']:
            return DataAnalysisResult(
                operation="execute_code",
                data_path=str(data_path) if data_path else None,
                result=None,
                summary=f"Code execution failed: {execution_result['error']}",
                insights=[],
                confidence=0.0,
                metadata={'error': execution_result['error']}
            )
        
        # Extract insights from output
        insights = []
        if execution_result.get('stdout'):
            insights.append(f"Output: {execution_result['stdout'][:200]}")
        
        return DataAnalysisResult(
            operation="execute_code",
            data_path=str(data_path) if data_path else None,
            result=execution_result.get('result'),
            summary="Code executed successfully",
            insights=insights,
            confidence=0.85,
            metadata={
                'variables': list(execution_result.get('variables', {}).keys()),
                'stdout': execution_result.get('stdout', '')
            }
        )
    
    def detect_anomalies(
        self,
        data_path: Union[str, Path],
        column: str,
        method: str = "zscore",
        threshold: float = 3.0
    ) -> DataAnalysisResult:
        """
        Detect anomalies in numerical data.
        
        Args:
            data_path: Path to data file
            column: Column to analyze
            method: Detection method (zscore, iqr, isolation_forest)
            threshold: Threshold for anomaly detection
        
        Returns:
            DataAnalysisResult with anomalies
        """
        logger.info(f"Detecting anomalies in {column} using {method}")
        
        df = self._load_data(data_path)
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        data = df[column].dropna()
        
        if method == "zscore":
            z_scores = np.abs((data - data.mean()) / data.std())
            anomalies = df[z_scores > threshold]
        elif method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            anomalies = df[(data < lower_bound) | (data > upper_bound)]
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        insights = [
            f"Found {len(anomalies)} anomalies out of {len(df)} records",
            f"Anomaly percentage: {len(anomalies) / len(df) * 100:.2f}%"
        ]
        
        if len(anomalies) > 0:
            insights.append(f"Min anomaly value: {anomalies[column].min():.2f}")
            insights.append(f"Max anomaly value: {anomalies[column].max():.2f}")
        
        return DataAnalysisResult(
            operation="detect_anomalies",
            data_path=str(data_path),
            result=anomalies.to_dict(orient='records'),
            summary=f"Detected {len(anomalies)} anomalies using {method} method",
            insights=insights,
            confidence=0.8,
            metadata={
                'method': method,
                'threshold': threshold,
                'column': column,
                'anomaly_count': len(anomalies)
            }
        )
    
    def calculate_trends(
        self,
        data_path: Union[str, Path],
        value_column: str,
        time_column: Optional[str] = None,
        trend_type: str = "growth_rate",
        window: int = 7
    ) -> DataAnalysisResult:
        """
        Calculate trends in time-series or sequential data.
        
        Args:
            data_path: Path to data file
            value_column: Column with values to analyze
            time_column: Column with time/date values (optional)
            trend_type: Type of trend (growth_rate, moving_average, linear_trend)
            window: Window size for moving averages
        
        Returns:
            DataAnalysisResult with trend analysis
        """
        logger.info(f"Calculating {trend_type} trends for {value_column}")
        
        df = self._load_data(data_path)
        
        # if value_column not in df.columns:
        #     raise ValueError(f"Column '{value_column}' not found")

        actual_column = self._find_column(df, value_column)
        if not actual_column:
            available = ', '.join(df.columns[:10])
            raise ValueError(f"Column '{value_column}' not found. Available columns: {available}")
        value_column = actual_column
        
        # Sort by time if specified
        if time_column:
            if time_column not in df.columns:
                raise ValueError(f"Time column '{time_column}' not found")
            df = df.sort_values(time_column)
        
        data = df[value_column]
        
        if trend_type == "growth_rate":
            growth = data.pct_change() * 100
            result = {
                'growth_rates': growth.tolist(),
                'avg_growth': growth.mean(),
                'total_growth': ((data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100)
            }
            insights = [
                f"Average growth rate: {growth.mean():.2f}%",
                f"Total growth: {result['total_growth']:.2f}%"
            ]
        
        elif trend_type == "moving_average":
            ma = data.rolling(window=window).mean()
            result = {
                'moving_average': ma.tolist(),
                'window': window
            }
            insights = [
                f"Calculated {window}-period moving average",
                f"Latest MA value: {ma.iloc[-1]:.2f}"
            ]
        
        elif trend_type == "linear_trend":
            from scipy import stats
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            
            result = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value
            }
            insights = [
                f"Trend slope: {slope:.4f}",
                f"R-squared: {r_value**2:.4f}",
                f"Trend is {'significant' if p_value < 0.05 else 'not significant'} (p={p_value:.4f})"
            ]
        
        else:
            raise ValueError(f"Unknown trend type: {trend_type}")
        
        return DataAnalysisResult(
            operation="calculate_trends",
            data_path=str(data_path),
            result=result,
            summary=f"Calculated {trend_type} trends for {value_column}",
            insights=insights,
            confidence=0.85,
            metadata={
                'trend_type': trend_type,
                'value_column': value_column,
                'time_column': time_column
            }
        )
    
    def group_and_aggregate(
        self,
        data_path: Union[str, Path],
        group_by: List[str],
        aggregations: Dict[str, str]
    ) -> DataAnalysisResult:
        """
        Group data and calculate aggregations.
        
        Args:
            data_path: Path to data file
            group_by: Columns to group by
            aggregations: Dict of {column: aggregation_function}
        
        Returns:
            DataAnalysisResult with grouped results
        """
        logger.info(f"Grouping by {group_by} with aggregations {aggregations}")
        
        df = self._load_data(data_path)
        
        # Validate columns
        # for col in group_by:
        #     if col not in df.columns:
        #         raise ValueError(f"Group column '{col}' not found")
        fixed_group_by = []
        for col in group_by:
            actual_col = self._find_column(df, col)
            if not actual_col:
                available = ', '.join(df.columns[:10])
                raise ValueError(f"Group column '{col}' not found. Available columns: {available}")
            fixed_group_by.append(actual_col)
        group_by = fixed_group_by

        # Fix aggregation columns
        fixed_aggregations = {}
        for col, agg in aggregations.items():
            actual_col = self._find_column(df, col)
            if actual_col:
                fixed_aggregations[actual_col] = agg
        aggregations = fixed_aggregations
        
        for col in aggregations.keys():
            if col not in df.columns:
                raise ValueError(f"Aggregation column '{col}' not found")
        
        # Perform groupby
        grouped = df.groupby(group_by).agg(aggregations)
        
        insights = [
            f"Created {len(grouped)} groups",
            f"Grouped by: {', '.join(group_by)}"
        ]
        
        # Add top groups
        if len(grouped) > 0:
            first_agg_col = list(aggregations.keys())[0]
            top_groups = grouped.nlargest(3, first_agg_col)
            insights.append(f"Top 3 groups by {first_agg_col}:")
            for idx, row in top_groups.iterrows():
                insights.append(f"  {idx}: {row[first_agg_col]:.2f}")
        
        return DataAnalysisResult(
            operation="group_and_aggregate",
            data_path=str(data_path),
            # result=grouped.reset_index().to_dict(orient='records'),
            result=(
                grouped.reset_index().to_dict(orient='records') 
                if grouped.index.name not in grouped.columns 
                else grouped.to_dict(orient='records')
            ),
            summary=f"Grouped data by {group_by} and calculated aggregations",
            insights=insights,
            confidence=0.95,
            metadata={
                'group_by': group_by,
                'aggregations': aggregations,
                'group_count': len(grouped)
            }
        )


# Singleton instance
_data_agent: Optional[DataAgent] = None


def get_data_agent() -> DataAgent:
    """Get or create singleton Data Agent instance."""
    global _data_agent
    if _data_agent is None:
        _data_agent = DataAgent()
    return _data_agent