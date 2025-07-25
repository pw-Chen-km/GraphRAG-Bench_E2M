
from __future__ import annotations

import json
import os.path
import uuid
from abc import ABC
from json import JSONDecodeError
from typing import Any, Optional, Type, TypeVar, Union, Dict, List
from dataclasses import dataclass

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from Core.Common.Constants import (
    MESSAGE_ROUTE_CAUSE_BY,
    MESSAGE_ROUTE_FROM,
    MESSAGE_ROUTE_TO,
    MESSAGE_ROUTE_TO_ALL
)
from Core.Common.Logger import logger
from Core.Common.Utils import any_to_str, any_to_str_set
from Core.Utils.Exceptions import handle_exception


class SerializationMixin(BaseModel, extra="forbid"):
    """
    Polymorphic serialization/deserialization mixin for subclasses.
    
    This mixin enables proper polymorphic serialization by adding class type
    information to serialized objects and reconstructing the correct type
    during deserialization.
    
    Note: Pydantic is not designed for polymorphism by default. This mixin
    provides a workaround by adding class name information to serialized objects.
    """

    __is_polymorphic_base = False
    __subclasses_map__ = {}

    @model_serializer(mode="wrap")
    def __serialize_with_class_type__(self, default_serializer) -> Any:
        """Serialize with class type information."""
        result = default_serializer(self)
        result["__module_class_name"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        return result

    @model_validator(mode="wrap")
    @classmethod
    def __convert_to_real_type__(cls, value: Any, handler):
        """Convert serialized data to the correct polymorphic type."""
        if not isinstance(value, dict):
            return handler(value)

        # Remove class name from dict to avoid extra field errors
        class_full_name = value.pop("__module_class_name", None)

        # Handle non-polymorphic base classes
        if not cls.__is_polymorphic_base:
            if class_full_name is None:
                return handler(value)
            elif str(cls) == f"<class '{class_full_name}'>":
                return handler(value)
            else:
                # Trying to instantiate wrong type
                pass

        # Handle polymorphic base classes
        if class_full_name is None:
            raise ValueError("Missing __module_class_name field")

        class_type = cls.__subclasses_map__.get(class_full_name)
        if class_type is None:
            raise TypeError(f"Class {class_full_name} has not been defined yet!")

        return class_type(**value)

    def __init_subclass__(cls, is_polymorphic_base: bool = False, **kwargs):
        """Register subclass for polymorphic serialization."""
        cls.__is_polymorphic_base = is_polymorphic_base
        cls.__subclasses_map__[f"{cls.__module__}.{cls.__qualname__}"] = cls
        super().__init_subclass__(**kwargs)


@dataclass
class SimpleMessage(BaseModel):
    """
    Simple message structure for basic communication.
    
    Attributes:
        content: Message content
        role: Role of the message sender
    """
    content: str
    role: str


@dataclass
class Document(BaseModel):
    """
    Represents a document with metadata and content.
    
    Attributes:
        root_path: Root directory path
        filename: Document filename
        content: Document content
    """
    root_path: str = ""
    filename: str = ""
    content: str = ""

    def get_meta(self) -> Document:
        """Get metadata-only version of the document."""
        return Document(root_path=self.root_path, filename=self.filename)

    @property
    def root_relative_path(self) -> str:
        """Get relative path from root directory."""
        return os.path.join(self.root_path, self.filename)

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return self.content


class Message(BaseModel):
    """
    Message class for communication between components.
    
    Attributes:
        id: Unique message identifier
        content: Message content
        instruct_content: Optional instruction content
        role: Message role (system/user/assistant)
        sent_from: Source identifier
        send_to: Set of destination identifiers
    """
    id: str = Field(default="", validate_default=True)
    content: str
    instruct_content: Optional[BaseModel] = Field(default=None, validate_default=True)
    role: str = "user"
    sent_from: str = Field(default="", validate_default=True)
    send_to: set[str] = Field(default={MESSAGE_ROUTE_TO_ALL}, validate_default=True)

    @field_validator("id", mode="before")
    @classmethod
    def check_id(cls, id: str) -> str:
        """Generate UUID if no ID provided."""
        return id if id else uuid.uuid4().hex

    @field_validator("sent_from", mode="before")
    @classmethod
    def check_sent_from(cls, sent_from: Any) -> str:
        """Convert sent_from to string."""
        return any_to_str(sent_from if sent_from else "")

    @field_validator("send_to", mode="before")
    @classmethod
    def check_send_to(cls, send_to: Any) -> set:
        """Convert send_to to set of strings."""
        return any_to_str_set(send_to if send_to else {MESSAGE_ROUTE_TO_ALL})

    @field_serializer("send_to", mode="plain")
    def ser_send_to(self, send_to: set) -> List[str]:
        """Serialize send_to set to list."""
        return list(send_to)

    def __init__(self, content: str = "", **data: Any):
        """Initialize message with content and optional data."""
        data["content"] = data.get("content", content)
        super().__init__(**data)

    def __setattr__(self, key: str, val: Any) -> None:
        """Override setattr to handle special route attributes."""
        if key == MESSAGE_ROUTE_CAUSE_BY:
            new_val = any_to_str(val)
        elif key == MESSAGE_ROUTE_FROM:
            new_val = any_to_str(val)
        elif key == MESSAGE_ROUTE_TO:
            new_val = any_to_str_set(val)
        else:
            new_val = val
        super().__setattr__(key, new_val)

    def __str__(self) -> str:
        """String representation of the message."""
        if self.instruct_content:
            return f"{self.role}: {self.instruct_content.model_dump()}"
        return f"{self.role}: {self.content}"

    def __repr__(self) -> str:
        return self.__str__()

    def rag_key(self) -> str:
        """Get key for RAG search operations."""
        return self.content

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for LLM calls."""
        return {"role": self.role, "content": self.content}

    def dump(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(exclude_none=True, warnings=False)

    @staticmethod
    @handle_exception(exception_type=JSONDecodeError, default_return=None)
    def load(val: str) -> Optional[Message]:
        """Load message from JSON string."""
        try:
            data = json.loads(val)
            msg_id = data.pop("id", None)
            msg = Message(**data)
            if msg_id:
                msg.id = msg_id
            return msg
        except JSONDecodeError as err:
            logger.error(f"Failed to parse JSON: {val}, error: {err}")
        return None


class UserMessage(Message):
    """User message for OpenAI compatibility."""
    
    def __init__(self, content: str):
        super().__init__(content=content, role="user")


class SystemMessage(Message):
    """System message for OpenAI compatibility."""
    
    def __init__(self, content: str):
        super().__init__(content=content, role="system")


class AIMessage(Message):
    """AI assistant message for OpenAI compatibility."""
    
    def __init__(self, content: str):
        super().__init__(content=content, role="assistant")


# Generic type variable for context classes
T = TypeVar("T", bound="BaseContext")


class BaseContext(BaseModel, ABC):
    """Base class for context objects."""
    
    @classmethod
    @handle_exception
    def loads(cls: Type[T], val: str) -> Optional[T]:
        """Load context from JSON string."""
        data = json.loads(val)
        return cls(**data)


class CodingContext(BaseContext):
    """
    Context for coding-related operations.
    
    Attributes:
        filename: Target filename
        design_doc: Design document
        task_doc: Task description document
        code_doc: Code document
        code_plan_and_change_doc: Code planning and change document
    """
    filename: str
    design_doc: Optional[Document] = None
    task_doc: Optional[Document] = None
    code_doc: Optional[Document] = None
    code_plan_and_change_doc: Optional[Document] = None



