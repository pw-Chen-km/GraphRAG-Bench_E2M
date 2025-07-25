"""
Time Statistics Module - Performance timing and statistics
Provides comprehensive timing and performance measurement capabilities
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class TimingStats:
    """Data class for timing statistics"""
    total_time: float
    count: int
    average_time: float
    min_time: Optional[float] = None
    max_time: Optional[float] = None


class TimeStatistic:
    """
    Comprehensive timing and performance measurement system.
    Tracks individual timers and stage-based timing with detailed statistics.
    """

    def __init__(self):
        """Initialize timing system"""
        self._start_times: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._total_times: Dict[str, float] = {}
        self._min_times: Dict[str, float] = {}
        self._max_times: Dict[str, float] = {}
        self._stage_times: List[float] = []
        self._stage_names: List[str] = []

    def start_stage(self, stage_name: Optional[str] = None) -> None:
        """
        Start timing a new stage.
        
        Args:
            stage_name: Optional name for the stage
        """
        self._stage_times.append(time.time())
        if stage_name:
            self._stage_names.append(stage_name)
        else:
            self._stage_names.append(f"stage_{len(self._stage_times)}")

    def stop_last_stage(self) -> float:
        """
        Stop the last stage and return elapsed time.
        
        Returns:
            Time elapsed for the last stage in seconds
            
        Raises:
            RuntimeError: If no stage is currently running
        """
        if not self._stage_times:
            raise RuntimeError("No stage is currently running")
            
        self._stage_times.append(time.time())
        elapsed_time = self._stage_times[-1] - self._stage_times[-2]
        return elapsed_time

    def end_stage(self, stage_name: str) -> float:
        """
        End timing for a specific stage.
        
        Args:
            stage_name: Name of the stage to end
            
        Returns:
            Time elapsed for the stage in seconds
            
        Raises:
            RuntimeError: If no stage is currently running
        """
        return self.stop_last_stage()

    def start(self, name: str) -> None:
        """
        Start timing for a named operation.
        
        Args:
            name: Name of the operation to time
        """
        self._start_times[name] = time.time()

    def end(self, name: str) -> float:
        """
        End timing for a named operation and return elapsed time.
        
        Args:
            name: Name of the operation to end timing for
            
        Returns:
            Elapsed time in seconds
            
        Raises:
            RuntimeError: If operation was not started
        """
        if name not in self._start_times:
            raise RuntimeError(f"TimeStatistic: {name} not started")
            
        elapsed_time = time.time() - self._start_times[name]
        self._add_time(name, elapsed_time)
        del self._start_times[name]
        return elapsed_time

    def _add_time(self, name: str, elapsed_time: float) -> None:
        """
        Add timing data for an operation.
        
        Args:
            name: Operation name
            elapsed_time: Time elapsed for the operation
        """
        if name not in self._total_times:
            self._total_times[name] = 0
            self._counts[name] = 0
            self._min_times[name] = elapsed_time
            self._max_times[name] = elapsed_time
        else:
            self._min_times[name] = min(self._min_times[name], elapsed_time)
            self._max_times[name] = max(self._max_times[name], elapsed_time)
            
        self._total_times[name] += elapsed_time
        self._counts[name] += 1

    def get_statistics(self, name: str) -> TimingStats:
        """
        Get comprehensive statistics for a named operation.
        
        Args:
            name: Name of the operation
            
        Returns:
            TimingStats object with detailed statistics
            
        Raises:
            RuntimeError: If no data exists for the operation
        """
        if name not in self._total_times:
            raise RuntimeError(f"TimeStatistic: {name} has no statistics")
            
        total_time = self._total_times[name]
        count = self._counts[name]
        average_time = total_time / count
        min_time = self._min_times.get(name)
        max_time = self._max_times.get(name)
        
        return TimingStats(
            total_time=total_time,
            count=count,
            average_time=average_time,
            min_time=min_time,
            max_time=max_time
        )

    def get_all_statistics(self) -> Dict[str, TimingStats]:
        """
        Get statistics for all tracked operations.
        
        Returns:
            Dictionary mapping operation names to their statistics
        """
        return {
            name: self.get_statistics(name)
            for name in self._total_times.keys()
        }

    def get_stage_statistics(self) -> Dict[str, float]:
        """
        Get statistics for all completed stages.
        
        Returns:
            Dictionary mapping stage names to their durations
        """
        if len(self._stage_times) < 2:
            return {}
            
        stage_stats = {}
        for i in range(0, len(self._stage_times) - 1, 2):
            if i + 1 < len(self._stage_times):
                stage_name = self._stage_names[i // 2] if i // 2 < len(self._stage_names) else f"stage_{i // 2}"
                duration = self._stage_times[i + 1] - self._stage_times[i]
                stage_stats[stage_name] = duration
                
        return stage_stats

    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset timing data for specific operation or all operations.
        
        Args:
            name: Name of operation to reset. If None, resets all operations
        """
        if name is None:
            self._start_times.clear()
            self._counts.clear()
            self._total_times.clear()
            self._min_times.clear()
            self._max_times.clear()
            self._stage_times.clear()
            self._stage_names.clear()
        else:
            self._start_times.pop(name, None)
            self._counts.pop(name, None)
            self._total_times.pop(name, None)
            self._min_times.pop(name, None)
            self._max_times.pop(name, None)

    def is_running(self, name: str) -> bool:
        """
        Check if an operation is currently being timed.
        
        Args:
            name: Name of the operation
            
        Returns:
            True if operation is running, False otherwise
        """
        return name in self._start_times

    def get_running_operations(self) -> List[str]:
        """
        Get list of currently running operations.
        
        Returns:
            List of operation names that are currently being timed
        """
        return list(self._start_times.keys())

    def format_statistics(self, name: str) -> str:
        """
        Format statistics for a named operation as a string.
        
        Args:
            name: Name of the operation
            
        Returns:
            Formatted string with statistics
        """
        stats = self.get_statistics(name)
        return (
            f"{name}: Total={stats.total_time:.3f}s, "
            f"Count={stats.count}, "
            f"Avg={stats.average_time:.3f}s"
            + (f", Min={stats.min_time:.3f}s" if stats.min_time else "")
            + (f", Max={stats.max_time:.3f}s" if stats.max_time else "")
        )

    def print_summary(self) -> None:
        """Print summary of all timing statistics"""
        print("=== Timing Statistics Summary ===")
        for name in self._total_times.keys():
            print(self.format_statistics(name))
        
        stage_stats = self.get_stage_statistics()
        if stage_stats:
            print("\n=== Stage Statistics ===")
            for stage_name, duration in stage_stats.items():
                print(f"{stage_name}: {duration:.3f}s")