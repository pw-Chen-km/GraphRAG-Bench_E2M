"""
Component Registry - Manage all GraphRAG components
Using Registry Pattern for unified component management and lifecycle control.
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import weakref
from collections import defaultdict
from Core.Common.Logger import logger


@dataclass
class ComponentInfo:
    """
    Component information container.
    
    Stores metadata about registered components including their type,
    description, dependencies, and additional metadata.
    """
    name: str
    component: Any
    type: str
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    version: Optional[str] = None


class ComponentLifecycleManager:
    """
    Manages component lifecycle operations.
    
    Handles component initialization, validation, and cleanup operations.
    """
    
    @staticmethod
    def validate_component(component: Any, component_type: str) -> bool:
        """
        Validate if a component meets the requirements for its type.
        
        Args:
            component: Component instance to validate
            component_type: Type of the component
            
        Returns:
            True if component is valid, False otherwise
        """
        if component is None:
            return False
        
        # Add type-specific validation logic here
        if component_type == "llm":
            return hasattr(component, 'aask')
        elif component_type == "embedding":
            return hasattr(component, 'embed')
        elif component_type == "retriever":
            return hasattr(component, 'retrieve')
        
        return True
    
    @staticmethod
    def initialize_component(component: Any, component_type: str) -> bool:
        """
        Initialize a component if it has an initialization method.
        
        Args:
            component: Component instance to initialize
            component_type: Type of the component
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if hasattr(component, 'initialize'):
                component.initialize()
            elif hasattr(component, '__init__'):
                # Component already initialized
                pass
            return True
        except Exception as e:
            logger.error(f"Failed to initialize component: {e}")
            return False
    
    @staticmethod
    def cleanup_component(component: Any) -> bool:
        """
        Clean up a component if it has a cleanup method.
        
        Args:
            component: Component instance to cleanup
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            if hasattr(component, 'cleanup'):
                component.cleanup()
            elif hasattr(component, 'close'):
                component.close()
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup component: {e}")
            return False


class DependencyResolver:
    """
    Resolves component dependencies and manages dependency graphs.
    """
    
    def __init__(self):
        """Initialize the dependency resolver."""
        self._dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self._reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
    
    def add_dependency(self, component_name: str, dependencies: List[str]) -> None:
        """
        Add dependencies for a component.
        
        Args:
            component_name: Name of the component
            dependencies: List of dependency component names
        """
        self._dependency_graph[component_name] = dependencies
        for dep in dependencies:
            self._reverse_dependencies[dep].add(component_name)
    
    def remove_dependency(self, component_name: str) -> None:
        """
        Remove dependencies for a component.
        
        Args:
            component_name: Name of the component to remove
        """
        if component_name in self._dependency_graph:
            # Remove from reverse dependencies
            for dep in self._dependency_graph[component_name]:
                self._reverse_dependencies[dep].discard(component_name)
            del self._dependency_graph[component_name]
    
    def get_dependencies(self, component_name: str) -> List[str]:
        """
        Get dependencies for a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            List of dependency component names
        """
        return self._dependency_graph.get(component_name, [])
    
    def get_dependents(self, component_name: str) -> Set[str]:
        """
        Get components that depend on the specified component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Set of component names that depend on this component
        """
        return self._reverse_dependencies.get(component_name, set())
    
    def check_circular_dependencies(self) -> List[List[str]]:
        """
        Check for circular dependencies in the dependency graph.
        
        Returns:
            List of circular dependency chains
        """
        visited = set()
        rec_stack = set()
        circular_chains = []
        
        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                # Found circular dependency
                cycle_start = path.index(node)
                circular_chains.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for dep in self._dependency_graph.get(node, []):
                dfs(dep, path.copy())
            
            rec_stack.discard(node)
        
        for component in self._dependency_graph:
            if component not in visited:
                dfs(component, [])
        
        return circular_chains


class ComponentRegistry:
    """
    Component Registry for managing GraphRAG system components.
    
    Provides registration, retrieval, and lifecycle management for all
    system components with dependency tracking and validation.
    """
    
    def __init__(self):
        """Initialize the component registry."""
        self._components: Dict[str, ComponentInfo] = {}
        self._component_types: Dict[str, List[str]] = defaultdict(list)
        self._component_refs: Dict[str, weakref.ref] = {}
        self._lifecycle_manager = ComponentLifecycleManager()
        self._dependency_resolver = DependencyResolver()
    
    def register_component(
        self, 
        name: str, 
        component: Any, 
        component_type: str = "general", 
        description: str = "",
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Register a component in the registry.
        
        Args:
            name: Component name
            component: Component instance
            component_type: Type of the component
            description: Component description
            dependencies: List of dependency component names
            metadata: Additional metadata
            
        Raises:
            ValueError: If component name already exists
        """
        if name in self._components:
            raise ValueError(f"Component {name} already exists")
        
        # Validate component
        if not self._lifecycle_manager.validate_component(component, component_type):
            raise ValueError(f"Component {name} failed validation for type {component_type}")
        
        # Initialize component
        if not self._lifecycle_manager.initialize_component(component, component_type):
            raise ValueError(f"Component {name} failed initialization")
        
        component_info = ComponentInfo(
            name=name,
            component=component,
            type=component_type,
            description=description,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self._components[name] = component_info
        self._component_types[component_type].append(name)
        
        # Register dependencies
        self._dependency_resolver.add_dependency(name, component_info.dependencies)
        
        # Create weak reference for garbage collection
        self._component_refs[name] = weakref.ref(component)
        
        logger.info(f"✅ Registered component: {name} ({component_type})")
    
    def get_component(self, name: str) -> Optional[Any]:
        """
        Get a component by name.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None if not found
        """
        if name not in self._components:
            logger.warning(f"Component {name} not found")
            return None
        
        component_info = self._components[name]
        component = component_info.component
        
        # Check if component is still valid
        if component is None:
            logger.warning(f"Component {name} has been destroyed")
            return None
        
        return component
    
    def get_components_by_type(self, component_type: str) -> List[Any]:
        """
        Get all components of a specific type.
        
        Args:
            component_type: Type of components to retrieve
            
        Returns:
            List of component instances
        """
        component_names = self._component_types.get(component_type, [])
        components = []
        
        for name in component_names:
            component = self.get_component(name)
            if component is not None:
                components.append(component)
        
        return components
    
    def unregister_component(self, name: str) -> bool:
        """
        Unregister a component from the registry.
        
        Args:
            name: Component name to unregister
            
        Returns:
            True if successfully unregistered, False otherwise
        """
        if name not in self._components:
            return False
        
        component_info = self._components[name]
        
        # Cleanup component
        self._lifecycle_manager.cleanup_component(component_info.component)
        
        # Remove from dependency tracking
        self._dependency_resolver.remove_dependency(name)
        
        # Remove from type index
        component_type = component_info.type
        if component_type in self._component_types:
            self._component_types[component_type] = [
                x for x in self._component_types[component_type] if x != name
            ]
        
        # Clean up references
        if name in self._component_refs:
            del self._component_refs[name]
        
        # Remove component
        del self._components[name]
        
        logger.info(f"🗑️ Unregistered component: {name}")
        return True
    
    def has_component(self, name: str) -> bool:
        """
        Check if a component exists in the registry.
        
        Args:
            name: Component name to check
            
        Returns:
            True if component exists, False otherwise
        """
        return name in self._components
    
    def get_component_info(self, name: str = None) -> Dict[str, Any]:
        """
        Get component information.
        
        Args:
            name: Component name, if None returns all component info
            
        Returns:
            Component information dictionary
        """
        if name:
            if name not in self._components:
                return {}
            
            component_info = self._components[name]
            return {
                "name": component_info.name,
                "type": component_info.type,
                "description": component_info.description,
                "dependencies": component_info.dependencies,
                "metadata": component_info.metadata,
                "exists": component_info.component is not None,
                "created_at": component_info.created_at,
                "version": component_info.version
            }
        else:
            return {
                name: {
                    "type": info.type,
                    "description": info.description,
                    "dependencies": info.dependencies,
                    "metadata": info.metadata,
                    "exists": info.component is not None,
                    "created_at": info.created_at,
                    "version": info.version
                }
                for name, info in self._components.items()
            }
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get the complete dependency graph.
        
        Returns:
            Dependency graph dictionary
        """
        return dict(self._dependency_resolver._dependency_graph)
    
    def get_component_types(self) -> Dict[str, List[str]]:
        """
        Get component type mappings.
        
        Returns:
            Dictionary mapping component types to component names
        """
        return dict(self._component_types)
    
    def list_components(self) -> List[str]:
        """
        List all registered component names.
        
        Returns:
            List of component names
        """
        return list(self._components.keys())
    
    def clear(self) -> None:
        """Clear all components from the registry."""
        # Cleanup all components
        for component_info in self._components.values():
            self._lifecycle_manager.cleanup_component(component_info.component)
        
        self._components.clear()
        self._component_types.clear()
        self._component_refs.clear()
        self._dependency_resolver = DependencyResolver()
        
        logger.info("🧹 Cleared all components")
    
    def validate_dependencies(self) -> List[str]:
        """
        Validate that all dependencies are satisfied.
        
        Returns:
            List of missing dependencies
        """
        missing_deps = []
        
        for name, component_info in self._components.items():
            for dep in component_info.dependencies:
                if not self.has_component(dep):
                    missing_deps.append(f"{name} -> {dep}")
        
        return missing_deps
    
    def check_circular_dependencies(self) -> List[List[str]]:
        """
        Check for circular dependencies in the registry.
        
        Returns:
            List of circular dependency chains
        """
        return self._dependency_resolver.check_circular_dependencies()
    
    def get_component_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the registry.
        
        Returns:
            Statistics dictionary
        """
        total_components = len(self._components)
        type_counts = {t: len(names) for t, names in self._component_types.items()}
        active_components = sum(1 for info in self._components.values() 
                              if info.component is not None)
        
        # Check for issues
        missing_deps = self.validate_dependencies()
        circular_deps = self.check_circular_dependencies()
        
        return {
            "total_components": total_components,
            "active_components": active_components,
            "component_types": type_counts,
            "dependency_relations": len(self._dependency_resolver._dependency_graph),
            "missing_dependencies": len(missing_deps),
            "circular_dependencies": len(circular_deps),
            "health_status": "healthy" if not missing_deps and not circular_deps else "issues_detected"
        }


# Global component registry instance
_global_registry: Optional[ComponentRegistry] = None


def get_global_registry() -> ComponentRegistry:
    """
    Get the global component registry instance.
    
    Returns:
        Global ComponentRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ComponentRegistry()
    return _global_registry


def register_global_component(name: str, component: Any, **kwargs) -> None:
    """
    Register a component in the global registry.
    
    Args:
        name: Component name
        component: Component instance
        **kwargs: Additional registration parameters
    """
    registry = get_global_registry()
    registry.register_component(name, component, **kwargs)


def get_global_component(name: str) -> Optional[Any]:
    """
    Get a component from the global registry.
    
    Args:
        name: Component name
        
    Returns:
        Component instance or None if not found
    """
    registry = get_global_registry()
    return registry.get_component(name) 