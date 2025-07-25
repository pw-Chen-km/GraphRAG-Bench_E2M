"""
Asynchronous HTTP client utilities for GraphRAG system.
Provides pure async HTTP client functionality with support for various request types.
"""

from typing import Any, Mapping, Optional, Union, AsyncGenerator, Dict
import aiohttp
from aiohttp.client import DEFAULT_TIMEOUT
from aiohttp import ClientSession, ClientResponse


class AsyncHttpClient:
    """
    Asynchronous HTTP client with support for various request types.
    
    This class provides a clean interface for making asynchronous HTTP requests
    with proper session management and error handling.
    """
    
    def __init__(self, timeout: int = DEFAULT_TIMEOUT.total):
        """
        Initialize the async HTTP client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
    
    async def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Mapping[str, str]] = None,
        json_data: Any = None,
        data: Any = None,
        headers: Optional[Dict] = None,
        encoding: str = "utf-8"
    ) -> ClientResponse:
        """
        Make an HTTP request with the specified method.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL
            params: Query parameters
            json_data: JSON data to send
            data: Form data to send
            headers: Request headers
            encoding: Response encoding
            
        Returns:
            ClientResponse object
        """
        async with ClientSession() as session:
            async with session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                data=data,
                headers=headers,
                timeout=self.timeout
            ) as response:
                return response
    
    async def post(
        self,
        url: str,
        params: Optional[Mapping[str, str]] = None,
        json_data: Any = None,
        data: Any = None,
        headers: Optional[Dict] = None,
        as_json: bool = False,
        encoding: str = "utf-8"
    ) -> Union[str, Dict]:
        """
        Make an asynchronous POST request.
        
        Args:
            url: Target URL
            params: Query parameters
            json_data: JSON data to send
            data: Form data to send
            headers: Request headers
            as_json: Whether to return response as JSON
            encoding: Response encoding
            
        Returns:
            Response data as string or dictionary
        """
        response = await self._make_request(
            method="POST",
            url=url,
            params=params,
            json_data=json_data,
            data=data,
            headers=headers,
            encoding=encoding
        )
        
        if as_json:
            return await response.json()
        else:
            content = await response.read()
            return content.decode(encoding)
    
    async def get(
        self,
        url: str,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Dict] = None,
        as_json: bool = False,
        encoding: str = "utf-8"
    ) -> Union[str, Dict]:
        """
        Make an asynchronous GET request.
        
        Args:
            url: Target URL
            params: Query parameters
            headers: Request headers
            as_json: Whether to return response as JSON
            encoding: Response encoding
            
        Returns:
            Response data as string or dictionary
        """
        response = await self._make_request(
            method="GET",
            url=url,
            params=params,
            headers=headers,
            encoding=encoding
        )
        
        if as_json:
            return await response.json()
        else:
            content = await response.read()
            return content.decode(encoding)
    
    async def post_stream(
        self,
        url: str,
        params: Optional[Mapping[str, str]] = None,
        json_data: Any = None,
        data: Any = None,
        headers: Optional[Dict] = None,
        encoding: str = "utf-8"
    ) -> AsyncGenerator[str, None]:
        """
        Make an asynchronous streaming POST request.
        
        Args:
            url: Target URL
            params: Query parameters
            json_data: JSON data to send
            data: Form data to send
            headers: Request headers
            encoding: Response encoding
            
        Yields:
            Response content line by line
        """
        response = await self._make_request(
            method="POST",
            url=url,
            params=params,
            json_data=json_data,
            data=data,
            headers=headers,
            encoding=encoding
        )
        
        async for line in response.content:
            yield line.decode(encoding)


# Global client instance for convenience
_global_client: Optional[AsyncHttpClient] = None


def get_global_client() -> AsyncHttpClient:
    """
    Get the global async HTTP client instance.
    
    Returns:
        Global AsyncHttpClient instance
    """
    global _global_client
    if _global_client is None:
        _global_client = AsyncHttpClient()
    return _global_client


# Legacy function aliases for backward compatibility
async def apost(
    url: str,
    params: Optional[Mapping[str, str]] = None,
    json: Any = None,
    data: Any = None,
    headers: Optional[Dict] = None,
    as_json: bool = False,
    encoding: str = "utf-8",
    timeout: int = DEFAULT_TIMEOUT.total,
) -> Union[str, Dict]:
    """
    Legacy async POST function for backward compatibility.
    
    Args:
        url: Target URL
        params: Query parameters
        json: JSON data to send
        data: Form data to send
        headers: Request headers
        as_json: Whether to return response as JSON
        encoding: Response encoding
        timeout: Request timeout
        
    Returns:
        Response data as string or dictionary
    """
    client = AsyncHttpClient(timeout=timeout)
    return await client.post(
        url=url,
        params=params,
        json_data=json,
        data=data,
        headers=headers,
        as_json=as_json,
        encoding=encoding
    )


async def apost_stream(
    url: str,
    params: Optional[Mapping[str, str]] = None,
    json: Any = None,
    data: Any = None,
    headers: Optional[Dict] = None,
    encoding: str = "utf-8",
    timeout: int = DEFAULT_TIMEOUT.total,
) -> AsyncGenerator[str, None]:
    """
    Legacy async streaming POST function for backward compatibility.
    
    Args:
        url: Target URL
        params: Query parameters
        json: JSON data to send
        data: Form data to send
        headers: Request headers
        encoding: Response encoding
        timeout: Request timeout
        
    Yields:
        Response content line by line
    """
    client = AsyncHttpClient(timeout=timeout)
    async for line in client.post_stream(
        url=url,
        params=params,
        json_data=json,
        data=data,
        headers=headers,
        encoding=encoding
    ):
        yield line