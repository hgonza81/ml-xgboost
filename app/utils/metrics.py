"""
Metrics collection and monitoring utilities.
"""

import time
from datetime import datetime
from typing import Dict, Any
from collections import defaultdict, deque
import threading
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Thread-safe metrics collection for monitoring and observability."""
    
    def __init__(self, max_history: int = 1000):
        self._lock = threading.Lock()
        self._max_history = max_history
        
        # Counters
        self._prediction_count = 0
        self._error_count = 0
        self._request_count = 0
        
        # Timing metrics
        self._response_times = deque(maxlen=max_history)
        self._prediction_times = deque(maxlen=max_history)
        
        # Error tracking
        self._error_types = defaultdict(int)
        
        # Model usage tracking
        self._model_usage = defaultdict(int)
        
        # Start time for uptime calculation
        self._start_time = time.time()
        
        logger.info(f"Metrics collector initialized with max history: {max_history}")
    
    def record_request(self, duration_ms: float):
        """Record a request with its duration."""
        with self._lock:
            self._request_count += 1
            self._response_times.append(duration_ms)
    
    def record_prediction(self, sample_count: int, duration_ms: float = None, model_version: str = None):
        """Record a prediction request."""
        with self._lock:
            self._prediction_count += sample_count
            
            if duration_ms is not None:
                self._prediction_times.append(duration_ms)
            
            if model_version:
                self._model_usage[model_version] += 1
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        with self._lock:
            self._error_count += 1
            self._error_types[error_type] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            current_time = time.time()
            uptime_seconds = current_time - self._start_time
            
            # Calculate response time statistics
            response_times = list(self._response_times)
            response_stats = self._calculate_stats(response_times) if response_times else {}
            
            # Calculate prediction time statistics
            prediction_times = list(self._prediction_times)
            prediction_stats = self._calculate_stats(prediction_times) if prediction_times else {}
            
            return {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "uptime_seconds": round(uptime_seconds, 2),
                "counters": {
                    "total_requests": self._request_count,
                    "total_predictions": self._prediction_count,
                    "total_errors": self._error_count
                },
                "response_time_ms": response_stats,
                "prediction_time_ms": prediction_stats,
                "error_breakdown": dict(self._error_types),
                "model_usage": dict(self._model_usage),
                "rates": {
                    "requests_per_second": round(self._request_count / uptime_seconds, 2) if uptime_seconds > 0 else 0,
                    "predictions_per_second": round(self._prediction_count / uptime_seconds, 2) if uptime_seconds > 0 else 0,
                    "error_rate": round(self._error_count / max(self._request_count, 1), 4)
                }
            }
    
    def _calculate_stats(self, values: list) -> Dict[str, float]:
        """Calculate statistical metrics for a list of values."""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "min": round(min(sorted_values), 2),
            "max": round(max(sorted_values), 2),
            "mean": round(sum(sorted_values) / count, 2),
            "p50": round(sorted_values[count // 2], 2),
            "p95": round(sorted_values[int(count * 0.95)], 2),
            "p99": round(sorted_values[int(count * 0.99)], 2)
        }
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._prediction_count = 0
            self._error_count = 0
            self._request_count = 0
            self._response_times.clear()
            self._prediction_times.clear()
            self._error_types.clear()
            self._model_usage.clear()
            self._start_time = time.time()
        
        logger.info("Metrics reset")
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health-specific metrics."""
        with self._lock:
            current_time = time.time()
            uptime_seconds = current_time - self._start_time
            
            # Recent error rate (last 100 requests)
            recent_requests = min(self._request_count, 100)
            recent_error_rate = self._error_count / max(recent_requests, 1) if recent_requests > 0 else 0
            
            return {
                "uptime_seconds": round(uptime_seconds, 2),
                "total_requests": self._request_count,
                "total_errors": self._error_count,
                "error_rate": round(recent_error_rate, 4),
                "is_healthy": recent_error_rate < 0.1  # Less than 10% error rate
            }


# Global metrics instance
_metrics: MetricsCollector = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance (singleton pattern).
    
    Returns:
        MetricsCollector: The global metrics collector
    """
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
        logger.info("Global metrics collector created")
    return _metrics


# Convenience instance for easy access
metrics = get_metrics_collector()