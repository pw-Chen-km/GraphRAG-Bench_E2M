"""
Enhanced Utilities Module - Refactored utility functions
Provides comprehensive utility functions for GraphRAG system
"""
import re
import hashlib
import json
import os
import pickle
from typing import List, Dict, Any, Union, Optional, Tuple
from collections import defaultdict
import numpy as np
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import wraps
import time


class DataProcessor:
    """Data processing utilities"""
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\-.,!?;:()\[\]{}"\']', '', text)
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text using frequency analysis"""
        if not text:
            return []
        
        # Tokenize and count
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = defaultdict(int)
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] += 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    @staticmethod
    def split_text_by_sentences(text: str) -> List[str]:
        """Split text into sentences using advanced regex"""
        if not text:
            return []
        
        # More sophisticated sentence splitting
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences


class TextAnalyzer:
    """Text analysis utilities"""
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Jaccard similarity"""
        if not text1 or not text2:
            return 0.0
        
        # Tokenize texts
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def extract_named_entities(text: str) -> List[str]:
        """Extract potential named entities from text"""
        if not text:
            return []
        
        # Simple named entity extraction using capitalization patterns
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter out common words that are capitalized
        common_words = {'The', 'This', 'That', 'These', 'Those', 'I', 'You', 'He', 'She', 'It', 'We', 'They'}
        entities = [entity for entity in entities if entity not in common_words]
        
        return list(set(entities))
    
    @staticmethod
    def analyze_text_complexity(text: str) -> Dict[str, Any]:
        """Analyze text complexity metrics"""
        if not text:
            return {}
        
        sentences = DataProcessor.split_text_by_sentences(text)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Calculate metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / len(words) if words else 0
        
        return {
            'sentence_count': len(sentences),
            'word_count': len(words),
            'unique_word_count': unique_words,
            'avg_sentence_length': avg_sentence_length,
            'vocabulary_diversity': vocabulary_diversity,
            'complexity_score': avg_sentence_length * vocabulary_diversity
        }


class FileManager:
    """File management utilities"""
    
    @staticmethod
    def ensure_directory(path: str) -> None:
        """Ensure directory exists, create if necessary"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def safe_save_json(data: Any, filepath: str, indent: int = 2) -> bool:
        """Safely save data as JSON with error handling"""
        try:
            FileManager.ensure_directory(os.path.dirname(filepath))
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            return True
        except Exception as e:
            logging.error(f"Failed to save JSON to {filepath}: {e}")
            return False
    
    @staticmethod
    def safe_load_json(filepath: str) -> Optional[Any]:
        """Safely load JSON data with error handling"""
        try:
            if not os.path.exists(filepath):
                return None
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load JSON from {filepath}: {e}")
            return None
    
    @staticmethod
    def safe_save_pickle(data: Any, filepath: str) -> bool:
        """Safely save data as pickle with error handling"""
        try:
            FileManager.ensure_directory(os.path.dirname(filepath))
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logging.error(f"Failed to save pickle to {filepath}: {e}")
            return False
    
    @staticmethod
    def safe_load_pickle(filepath: str) -> Optional[Any]:
        """Safely load pickle data with error handling"""
        try:
            if not os.path.exists(filepath):
                return None
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Failed to load pickle from {filepath}: {e}")
            return None


class PerformanceMonitor:
    """Performance monitoring utilities"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
    
    def start_timer(self, name: str) -> None:
        """Start timing an operation"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing an operation and return duration"""
        if name not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[name]
        self.metrics[name].append(duration)
        del self.start_times[name]
        return duration
    
    def get_average_time(self, name: str) -> float:
        """Get average time for an operation"""
        times = self.metrics.get(name, [])
        return sum(times) / len(times) if times else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        for name, times in self.metrics.items():
            if times:
                summary[name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        return summary


class AsyncExecutor:
    """Asynchronous execution utilities"""
    
    @staticmethod
    async def execute_with_timeout(func, timeout: float, *args, **kwargs):
        """Execute function with timeout"""
        try:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        except asyncio.TimeoutError:
            logging.warning(f"Function execution timed out after {timeout} seconds")
            return None
    
    @staticmethod
    async def batch_process(items: List[Any], processor_func, max_workers: int = 4, 
                           batch_size: int = 10) -> List[Any]:
        """Process items in batches asynchronously"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Process batch with limited concurrency
            semaphore = asyncio.Semaphore(max_workers)
            
            async def process_item(item):
                async with semaphore:
                    return await processor_func(item)
            
            batch_results = await asyncio.gather(
                *[process_item(item) for item in batch],
                return_exceptions=True
            )
            
            # Filter out exceptions
            valid_results = [r for r in batch_results if not isinstance(r, Exception)]
            results.extend(valid_results)
        
        return results
    
    @staticmethod
    def run_in_threadpool(func, *args, **kwargs):
        """Run function in thread pool"""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, func, *args, **kwargs)


class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_text_content(text: str, min_length: int = 1, max_length: int = 10000) -> bool:
        """Validate text content"""
        if not isinstance(text, str):
            return False
        
        length = len(text.strip())
        return min_length <= length <= max_length
    
    @staticmethod
    def validate_list_content(items: List[Any], min_items: int = 0, max_items: int = 10000) -> bool:
        """Validate list content"""
        if not isinstance(items, list):
            return False
        
        return min_items <= len(items) <= max_items
    
    @staticmethod
    def validate_dict_structure(data: Dict[str, Any], required_keys: List[str] = None) -> bool:
        """Validate dictionary structure"""
        if not isinstance(data, dict):
            return False
        
        if required_keys:
            return all(key in data for key in required_keys)
        
        return True


class CacheManager:
    """Simple cache management"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = defaultdict(int)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache"""
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_accessed]
            del self.access_count[least_accessed]
        
        self.cache[key] = value
        self.access_count[key] = 1
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.access_count.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': sum(self.access_count.values()) / len(self.cache) if self.cache else 0
        }


# Global instances
data_processor = DataProcessor()
text_analyzer = TextAnalyzer()
file_manager = FileManager()
performance_monitor = PerformanceMonitor()
data_validator = DataValidator()
cache_manager = CacheManager()


# Convenience functions for backward compatibility
def sanitize_text(text: str) -> str:
    """Clean and normalize text content"""
    return data_processor.sanitize_text(text)


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text"""
    return data_processor.extract_keywords(text, max_keywords)


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity"""
    return text_analyzer.calculate_text_similarity(text1, text2)


def save_json(data: Any, filepath: str) -> bool:
    """Save data as JSON"""
    return file_manager.safe_save_json(data, filepath)


def load_json(filepath: str) -> Optional[Any]:
    """Load JSON data"""
    return file_manager.safe_load_json(filepath)


def validate_text(text: str) -> bool:
    """Validate text content"""
    return data_validator.validate_text_content(text) 