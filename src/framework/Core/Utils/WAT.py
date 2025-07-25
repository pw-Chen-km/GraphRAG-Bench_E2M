"""
WAT (Wikipedia Annotation Tool) utilities for GraphRAG system.
Provides functionality for handling Wikipedia entity annotations and related data structures.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
from enum import Enum


class AnnotationConfidence(Enum):
    """Confidence levels for WAT annotations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class WikiEntityInfo:
    """
    Wikipedia entity information container.
    
    Stores metadata about Wikipedia entities including
    their ID, title, and other relevant information.
    """
    wiki_id: str
    wiki_title: str
    description: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class PriorExplanation:
    """
    Prior explanation information for entity mentions.
    
    Contains probability and confidence information
    for entity mention detection.
    """
    entity_mention_probability: float
    confidence_score: float
    detection_method: str = "wat"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class WATAnnotation:
    """
    WAT (Wikipedia Annotation Tool) annotation container.
    
    Represents an entity annotation with position information,
    confidence scores, and Wikipedia entity details.
    """
    
    def __init__(
        self, 
        start: int, 
        end: int, 
        rho: float, 
        explanation: Optional[Dict[str, Any]], 
        spot: str, 
        wiki_id: str, 
        wiki_title: str
    ):
        """
        Initialize WAT annotation.
        
        Args:
            start: Character offset start (inclusive)
            end: Character offset end (exclusive)
            rho: Annotation accuracy/confidence score
            explanation: Explanation dictionary containing prior information
            spot: Annotated text span
            wiki_id: Wikipedia entity ID
            wiki_title: Wikipedia entity title
        """
        self.start = start
        self.end = end
        self.rho = rho
        self.explanation = explanation
        self.spot = spot
        self.wiki_id = wiki_id
        self.wiki_title = wiki_title
        
        # Extract prior probability if explanation is available
        self.prior_prob = None
        if self.explanation is not None:
            prior_info = self.explanation.get('prior_explanation', {})
            self.prior_prob = prior_info.get('entity_mention_probability', 0.0)
    
    @property
    def span_length(self) -> int:
        """Get the length of the annotated span."""
        return self.end - self.start
    
    @property
    def confidence_level(self) -> AnnotationConfidence:
        """Get confidence level based on rho score."""
        if self.rho >= 0.9:
            return AnnotationConfidence.VERY_HIGH
        elif self.rho >= 0.7:
            return AnnotationConfidence.HIGH
        elif self.rho >= 0.5:
            return AnnotationConfidence.MEDIUM
        else:
            return AnnotationConfidence.LOW
    
    @property
    def as_dict(self) -> Dict[str, Any]:
        """Convert annotation to dictionary representation."""
        return {
            "start": self.start,
            "end": self.end,
            "rho": self.rho,
            "explanation": self.explanation,
            "spot": self.spot,
            "wiki_id": self.wiki_id,
            "wiki_title": self.wiki_title,
            "prior_prob": self.prior_prob,
            "span_length": self.span_length,
            "confidence_level": self.confidence_level.value
        }
    
    def overlaps_with(self, other: 'WATAnnotation') -> bool:
        """
        Check if this annotation overlaps with another.
        
        Args:
            other: Another WAT annotation
            
        Returns:
            True if annotations overlap, False otherwise
        """
        return not (self.end <= other.start or other.end <= self.start)
    
    def contains(self, other: 'WATAnnotation') -> bool:
        """
        Check if this annotation contains another.
        
        Args:
            other: Another WAT annotation
            
        Returns:
            True if this annotation contains the other, False otherwise
        """
        return self.start <= other.start and self.end >= other.end
    
    def get_overlap_ratio(self, other: 'WATAnnotation') -> float:
        """
        Calculate overlap ratio with another annotation.
        
        Args:
            other: Another WAT annotation
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        if not self.overlaps_with(other):
            return 0.0
        
        overlap_start = max(self.start, other.start)
        overlap_end = min(self.end, other.end)
        overlap_length = overlap_end - overlap_start
        
        union_start = min(self.start, other.start)
        union_end = max(self.end, other.end)
        union_length = union_end - union_start
        
        return overlap_length / union_length if union_length > 0 else 0.0
    
    def __str__(self) -> str:
        """String representation of the annotation."""
        return f"WATAnnotation('{self.spot}' [{self.start}:{self.end}], rho={self.rho:.3f}, wiki_id={self.wiki_id})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the annotation."""
        return (f"WATAnnotation(start={self.start}, end={self.end}, rho={self.rho}, "
                f"spot='{self.spot}', wiki_id='{self.wiki_id}', wiki_title='{self.wiki_title}')")


class WATAnnotationProcessor:
    """
    Processor for WAT annotations with filtering and validation capabilities.
    """
    
    def __init__(self, min_confidence: float = 0.5, min_span_length: int = 1):
        """
        Initialize the WAT annotation processor.
        
        Args:
            min_confidence: Minimum confidence threshold for annotations
            min_span_length: Minimum span length for annotations
        """
        self.min_confidence = min_confidence
        self.min_span_length = min_span_length
    
    def filter_annotations(self, annotations: List[WATAnnotation]) -> List[WATAnnotation]:
        """
        Filter annotations based on confidence and span length criteria.
        
        Args:
            annotations: List of WAT annotations to filter
            
        Returns:
            Filtered list of annotations
        """
        filtered = []
        
        for annotation in annotations:
            if (annotation.rho >= self.min_confidence and 
                annotation.span_length >= self.min_span_length):
                filtered.append(annotation)
        
        return filtered
    
    def remove_overlapping_annotations(
        self, 
        annotations: List[WATAnnotation], 
        overlap_threshold: float = 0.5
    ) -> List[WATAnnotation]:
        """
        Remove overlapping annotations, keeping the highest confidence ones.
        
        Args:
            annotations: List of WAT annotations
            overlap_threshold: Threshold for considering annotations as overlapping
            
        Returns:
            List of annotations with overlaps removed
        """
        if not annotations:
            return []
        
        # Sort by confidence (rho) in descending order
        sorted_annotations = sorted(annotations, key=lambda x: x.rho, reverse=True)
        non_overlapping = []
        
        for annotation in sorted_annotations:
            is_overlapping = False
            
            for existing in non_overlapping:
                if annotation.get_overlap_ratio(existing) > overlap_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                non_overlapping.append(annotation)
        
        return non_overlapping
    
    def get_entity_coverage(self, annotations: List[WATAnnotation], text_length: int) -> float:
        """
        Calculate entity coverage ratio in the text.
        
        Args:
            annotations: List of WAT annotations
            text_length: Total length of the text
            
        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        if text_length == 0:
            return 0.0
        
        covered_chars = set()
        for annotation in annotations:
            covered_chars.update(range(annotation.start, annotation.end))
        
        return len(covered_chars) / text_length
    
    def get_annotation_statistics(self, annotations: List[WATAnnotation]) -> Dict[str, Any]:
        """
        Get statistics about the annotations.
        
        Args:
            annotations: List of WAT annotations
            
        Returns:
            Dictionary containing annotation statistics
        """
        if not annotations:
            return {
                "total_annotations": 0,
                "avg_confidence": 0.0,
                "avg_span_length": 0.0,
                "confidence_distribution": {},
                "unique_entities": 0
            }
        
        confidences = [ann.rho for ann in annotations]
        span_lengths = [ann.span_length for ann in annotations]
        unique_entities = len(set(ann.wiki_id for ann in annotations))
        
        # Calculate confidence distribution
        confidence_dist = {}
        for ann in annotations:
            level = ann.confidence_level.value
            confidence_dist[level] = confidence_dist.get(level, 0) + 1
        
        return {
            "total_annotations": len(annotations),
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_span_length": sum(span_lengths) / len(span_lengths),
            "confidence_distribution": confidence_dist,
            "unique_entities": unique_entities,
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "min_span_length": min(span_lengths),
            "max_span_length": max(span_lengths)
        }


class WATAnnotationValidator:
    """
    Validator for WAT annotations to ensure data quality and consistency.
    """
    
    @staticmethod
    def validate_annotation(annotation: WATAnnotation) -> List[str]:
        """
        Validate a single WAT annotation.
        
        Args:
            annotation: WAT annotation to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check position validity
        if annotation.start < 0:
            errors.append("Start position cannot be negative")
        
        if annotation.end < 0:
            errors.append("End position cannot be negative")
        
        if annotation.start >= annotation.end:
            errors.append("Start position must be less than end position")
        
        # Check confidence score
        if not 0.0 <= annotation.rho <= 1.0:
            errors.append("Confidence score (rho) must be between 0.0 and 1.0")
        
        # Check text span
        if not annotation.spot or annotation.spot.strip() == "":
            errors.append("Annotated text span cannot be empty")
        
        # Check Wikipedia entity information
        if not annotation.wiki_id or annotation.wiki_id.strip() == "":
            errors.append("Wikipedia entity ID cannot be empty")
        
        if not annotation.wiki_title or annotation.wiki_title.strip() == "":
            errors.append("Wikipedia entity title cannot be empty")
        
        return errors
    
    @staticmethod
    def validate_annotations(annotations: List[WATAnnotation]) -> Dict[str, List[str]]:
        """
        Validate a list of WAT annotations.
        
        Args:
            annotations: List of WAT annotations to validate
            
        Returns:
            Dictionary mapping annotation index to list of error messages
        """
        validation_results = {}
        
        for i, annotation in enumerate(annotations):
            errors = WATAnnotationValidator.validate_annotation(annotation)
            if errors:
                validation_results[str(i)] = errors
        
        return validation_results
    
    @staticmethod
    def check_for_overlaps(annotations: List[WATAnnotation]) -> List[Dict[str, Any]]:
        """
        Check for overlapping annotations in the list.
        
        Args:
            annotations: List of WAT annotations to check
            
        Returns:
            List of overlap information dictionaries
        """
        overlaps = []
        
        for i, ann1 in enumerate(annotations):
            for j, ann2 in enumerate(annotations[i+1:], i+1):
                if ann1.overlaps_with(ann2):
                    overlap_info = {
                        "annotation1_index": i,
                        "annotation2_index": j,
                        "annotation1": str(ann1),
                        "annotation2": str(ann2),
                        "overlap_ratio": ann1.get_overlap_ratio(ann2),
                        "overlap_span": (max(ann1.start, ann2.start), min(ann1.end, ann2.end))
                    }
                    overlaps.append(overlap_info)
        
        return overlaps
