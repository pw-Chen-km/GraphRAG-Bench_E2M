"""
Namespace management for storage organization.

Provides workspace and namespace abstractions for organizing storage resources
in the GraphRAG system. This module handles path management and resource isolation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict


@dataclass
class Workspace:
    """
    Workspace manager for organizing storage resources.
    
    A workspace provides a top-level organization unit for storage resources,
    managing base directories and experiment namespaces.
    
    Attributes:
        working_dir: Base working directory for all storage operations
        exp_name: Optional experiment name for namespace isolation
    """
    
    working_dir: str = ""
    exp_name: Optional[str] = None
    
    def __post_init__(self):
        """Initialize workspace and ensure directory exists."""
        self._ensure_working_directory()
    
    @staticmethod
    def new(working_dir: str, exp_name: Optional[str] = None) -> "Workspace":
        """
        Create a new workspace instance.
        
        Args:
            working_dir: Base working directory
            exp_name: Optional experiment name
            
        Returns:
            New workspace instance
        """
        return Workspace(working_dir, exp_name)
    
    @staticmethod
    def get_path(working_dir: str, exp_name: Optional[str] = None) -> Optional[str]:
        """
        Get the full path for a workspace.
        
        Args:
            working_dir: Base working directory
            exp_name: Optional experiment name
            
        Returns:
            Full workspace path or None if invalid
        """
        if exp_name is None:
            return working_dir
        return os.path.join(working_dir, exp_name)
    
    def _ensure_working_directory(self):
        """Ensure the working directory exists."""
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)
    
    def make_for(self, namespace: str) -> "Namespace":
        """
        Create a namespace within this workspace.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            New namespace instance
        """
        return Namespace(self, namespace)
    
    def get_load_path(self) -> Optional[str]:
        """
        Get the path for loading data from this workspace.
        
        Returns:
            Load path or None if workspace is empty
        """
        load_path = self.get_path(self.working_dir, self.exp_name)
        
        # If no experiment name and directory is empty, return None
        if load_path == self.working_dir:
            try:
                files = [x for x in os.scandir(load_path) if x.is_file()]
                if len(files) == 0:
                    return None
            except OSError:
                return None
        
        return load_path
    
    def get_save_path(self) -> str:
        """
        Get the path for saving data to this workspace.
        
        Returns:
            Save path (directory will be created if it doesn't exist)
        """
        save_path = self.get_path(self.working_dir, self.exp_name)
        
        if save_path is None:
            raise ValueError("Save path cannot be None")
        
        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)
        return save_path
    
    def get_workspace_info(self) -> Dict[str, str]:
        """Get information about the workspace configuration."""
        return {
            "working_dir": self.working_dir,
            "exp_name": self.exp_name or "default",
            "load_path": self.get_load_path() or "none",
            "save_path": self.get_save_path()
        }


@dataclass
class Namespace:
    """
    Namespace for organizing resources within a workspace.
    
    A namespace provides isolation and organization for different types of
    storage resources within a workspace.
    
    Attributes:
        workspace: Parent workspace instance
        namespace: Namespace identifier
    """
    
    workspace: Workspace
    namespace: Optional[str] = None
    
    def get_load_path(self, resource_name: Optional[str] = None) -> Optional[str]:
        """
        Get the path for loading a resource from this namespace.
        
        Args:
            resource_name: Optional specific resource name
            
        Returns:
            Load path or None if workspace has no data
        """
        if self.namespace is None:
            raise ValueError("Namespace must be set to get resource load path")
        
        load_path = self.workspace.get_load_path()
        if load_path is None:
            return None
        
        if resource_name:
            return os.path.join(load_path, f"{self.namespace}_{resource_name}")
        return os.path.join(load_path, self.namespace)
    
    def get_save_path(self, resource_name: Optional[str] = None) -> str:
        """
        Get the path for saving a resource to this namespace.
        
        Args:
            resource_name: Optional specific resource name
            
        Returns:
            Save path (directory will be created if it doesn't exist)
        """
        if self.namespace is None:
            raise ValueError("Namespace must be set to get resource save path")
        
        save_path = self.workspace.get_save_path()
        
        if resource_name:
            return os.path.join(save_path, f"{self.namespace}_{resource_name}")
        return os.path.join(save_path, self.namespace)
    
    def get_namespace_info(self) -> Dict[str, str]:
        """Get information about the namespace configuration."""
        return {
            "workspace_dir": self.workspace.working_dir,
            "exp_name": self.workspace.exp_name or "default",
            "namespace": self.namespace or "none",
            "load_path": self.get_load_path() or "none",
            "save_path": self.get_save_path()
        }
    
    def exists(self) -> bool:
        """
        Check if this namespace has any existing data.
        
        Returns:
            True if namespace has data, False otherwise
        """
        load_path = self.get_load_path()
        if load_path is None:
            return False
        
        return os.path.exists(load_path) and any(os.scandir(load_path))
    
    def clear(self) -> bool:
        """
        Clear all data in this namespace.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            save_path = self.get_save_path()
            if os.path.exists(save_path):
                for item in os.listdir(save_path):
                    item_path = os.path.join(save_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        import shutil
                        shutil.rmtree(item_path)
            return True
        except Exception:
            return False
