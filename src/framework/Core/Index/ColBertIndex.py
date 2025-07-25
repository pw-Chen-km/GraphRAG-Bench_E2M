"""
ColBERT Index Module

This module provides a ColBERT-based index implementation for GraphRAG.
ColBERT is a neural retrieval method that uses token-level encodings for
zero-shot retrieval on out-of-domain datasets.
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, Run, RunConfig
from colbert.data import Queries

from Core.Common.Logger import logger
from Core.Index.BaseIndex import BaseIndex, ColbertNodeResult, ColbertEdgeResult


class ColBertIndex(BaseIndex):
    """
    ColBERT-based index implementation for neural retrieval.
    
    This class provides a neural retrieval method that tends to work well
    in zero-shot settings on out-of-domain datasets, due to its use of
    token-level encodings rather than sentence or chunk level encodings.
    """

    def __init__(self, config):
        """
        Initialize the ColBERT index with configuration.
        
        Args:
            config: Configuration object containing ColBERT parameters
        """
        super().__init__(config)
        self.index_config = ColBERTConfig(
            root=os.path.dirname(self.config.persist_path),
            experiment=os.path.basename(self.config.persist_path),
            doc_maxlen=self.config.doc_maxlen,
            query_maxlen=self.config.query_maxlen,
            nbits=self.config.nbits,
            kmeans_niters=self.config.kmeans_niters,
        )

    async def _update_index(self, elements: List[dict], meta_data: List[str]) -> None:
        """
        Update the ColBERT index with new elements.
        
        Args:
            elements: List of data elements to index
            meta_data: List of metadata keys (not used in ColBERT)
        """
        with Run().context(
            RunConfig(
                nranks=self.config.ranks, 
                experiment=self.index_config.experiment,
                root=self.index_config.root
            )
        ):
            indexer = Indexer(checkpoint=self.config.model_name, config=self.index_config)
            
            # Extract content from elements and build index
            elements_content = [element["content"] for element in elements]
            indexer.index(name=self.config.index_name, collection=elements_content, overwrite=True)
            
            self._index = Searcher(
                index=self.config.index_name, 
                collection=elements_content, 
                checkpoint=self.config.model_name
            )

    async def _load_index(self) -> bool:
        """
        Load the ColBERT index from storage.
        
        Returns:
            True if loading successful, False otherwise
        """
        try:
            colbert_config = ColBERTConfig.load_from_index(
                Path(self.config.persist_path) / "indexes" / self.config.index_name
            )
            searcher = Searcher(
                index=self.config.index_name, 
                index_root=(Path(self.config.persist_path) / "indexes"),
                config=colbert_config
            )
            self._index = searcher
            return True
        except Exception as e:
            logger.error("Loading ColBERT index failed", exc_info=e)
            return False

    async def retrieval(self, query: str, top_k: Optional[int] = None) -> Tuple:
        """
        Retrieve documents using ColBERT search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            Tuple of search results
        """
        if top_k is None:
            top_k = self._get_retrieve_top_k()

        results = tuple(self._index.search(query, k=top_k))
        return results

    async def retrieval_nodes(self, query: str, top_k: Optional[int], graph: Any, 
                            need_score: bool = False, tree_node: bool = False) -> Tuple[List[dict], Optional[List[float]]]:
        """
        Retrieve nodes from the graph using ColBERT search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            graph: Graph instance to search in
            need_score: Whether to return similarity scores
            tree_node: Whether to return tree node format
            
        Returns:
            Tuple of (nodes, scores) where scores is optional
        """
        result = ColbertNodeResult(*(await self.retrieval(query, top_k)))
        
        if tree_node:
            return await result.get_tree_node_data(graph, need_score)
        else:
            return await result.get_node_data(graph, need_score)

    async def retrieval_edges(self, query: str, top_k: Optional[int], graph: Any, 
                            need_score: bool = False) -> Tuple[List[dict], Optional[List[float]]]:
        """
        Retrieve edges from the graph using ColBERT search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            graph: Graph instance to search in
            need_score: Whether to return similarity scores
            
        Returns:
            Tuple of (edges, scores) where scores is optional
        """
        results = await self.retrieval(query, top_k)
        result = ColbertEdgeResult(*results)
        
        return await result.get_edge_data(graph, need_score)

    async def retrieval_batch(self, queries: List[str], top_k: Optional[int] = None) -> List[Any]:
        """
        Retrieve documents for multiple queries using ColBERT.
        
        Args:
            queries: List of search queries
            top_k: Number of top results to return per query
            
        Returns:
            List of search results for each query
        """
        if top_k is None:
            top_k = self._get_retrieve_top_k()
            
        try:
            if isinstance(queries, str):
                queries = Queries(path=None, data={0: queries})
            elif not isinstance(queries, Queries):
                queries = Queries(data=queries)

            return self._index.search_all(queries, k=top_k).data
        except Exception as e:
            logger.exception(f"Failed to search queries {queries}: {e}")
            return []

    def _get_retrieve_top_k(self) -> int:
        """
        Get the default number of top results to retrieve.
        
        Returns:
            Default top_k value
        """
        return getattr(self.config, 'retrieve_top_k', 10)

    def _storage_index(self) -> None:
        """
        Store the ColBERT index.
        
        Note: ColBERT index is stored automatically during creation.
        """
        # ColBERT index is stored automatically during creation
        pass

    def _get_index(self) -> 'ColBertIndex':
        """
        Get a new ColBERT index instance.
        
        Returns:
            New ColBERT index instance
        """
        return ColBertIndex(self.config)

    async def _similarity_score(self, object_q: str, object_d: str) -> float:
        """
        Calculate similarity score between query and document using ColBERT.
        
        Args:
            object_q: Query text
            object_d: Document text
            
        Returns:
            Similarity score
        """
        encoded_q = self._index.encode(object_q, full_length_search=False)
        encoded_d = self._index.checkpoint.docFromText(object_d).float()
        real_score = encoded_q[0].matmul(encoded_d[0].T).max(dim=1).values.sum().detach().cpu().numpy()
        return real_score

    async def get_max_score(self, queries: List[str]) -> float:
        """
        Get the maximum possible score for a list of queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            Maximum score
        """
        assert isinstance(queries, list)
        encoded_query = self._index.encode(queries, full_length_search=False)
        encoded_doc = self._index.checkpoint.docFromText(queries).float()
        max_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()
        return max_score

    async def upsert(self, data: List[Any]) -> None:
        """
        Upsert documents into the ColBERT index.
        
        Args:
            data: List of document data to upsert
        """
        # ColBERT doesn't support incremental updates
        pass
