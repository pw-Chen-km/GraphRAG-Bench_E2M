import asyncio
import json
import random
from dataclasses import replace
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from Core.Common.Logger import logger
from Core.Prompt import GraphPrompt
from Core.Schema.ChunkSchema import TextChunk
from Core.Utils.Display import StatusDisplay

from .GraphBuilderFactory import (
    GraphBuilder,
    GraphBuilderConfig,
    GraphBuilderFactory,
    GraphType,
    RichKnowledgeGraphBuilder,
)


class EvolvingGraphBuilder(GraphBuilder):
    """Graph builder that performs iterative self-evolution on top of a seed graph."""

    async def execute(self, chunks: List[TextChunk], force_rebuild: bool = False) -> nx.Graph:
        """Execute the evolving graph construction pipeline."""
        StatusDisplay.show_processing_status("Graph Building", details="Evolving Graph")

        if not force_rebuild and self.graph is not None:
            StatusDisplay.show_info("Using existing evolved graph")
            return self.graph

        self.graph = await self._build_graph(chunks)
        StatusDisplay.show_success(
            "Evolving graph construction completed, nodes: %s, edges: %s"
            % (self.graph.number_of_nodes(), self.graph.number_of_edges())
        )
        return self.graph

    async def _build_graph(self, chunks: List[TextChunk]) -> nx.Graph:
        """Build the graph by seeding and then iteratively evolving it."""
        initial_graph = await self._seed_initial_graph(chunks)
        evolved_graph = await self._run_evolution_loop(initial_graph, chunks)
        return evolved_graph

    async def _seed_initial_graph(self, chunks: List[TextChunk]) -> nx.Graph:
        """Use a baseline graph builder to generate the initial graph."""
        seed_builder_type = self._get_evolution_setting(
            "seed_graph_type", GraphType.RICH_KNOWLEDGE.value
        )
        try:
            seed_graph_type = GraphType(seed_builder_type)
        except ValueError:
            logger.warning(
                "Unsupported seed_graph_type '%s', falling back to rich knowledge graph.",
                seed_builder_type,
            )
            seed_graph_type = GraphType.RICH_KNOWLEDGE

        base_config: GraphBuilderConfig = replace(self.config, graph_type=seed_graph_type)

        if seed_graph_type is GraphType.RICH_KNOWLEDGE:
            base_builder = RichKnowledgeGraphBuilder(base_config, self.context)
        else:
            # Fall back to factory lookup for any future supported seed builders
            base_builder = GraphBuilderFactory._builders.get(seed_graph_type)
            if base_builder is None:
                base_builder = RichKnowledgeGraphBuilder
            base_builder = base_builder(base_config, self.context)

        logger.info("Seeding evolving graph using %s", seed_graph_type.value)
        return await base_builder._build_graph(chunks)  # pylint: disable=protected-access

    async def _run_evolution_loop(
        self, initial_graph: nx.Graph, chunks: List[TextChunk]
    ) -> nx.Graph:
        """Iteratively refine the graph based on discovered anomalies."""
        graph = initial_graph
        max_steps = int(self._get_evolution_setting("max_steps", 3))

        for step in range(max_steps):
            logger.info("Evolution step %d/%d", step + 1, max_steps)
            anomalies = await self._discover_anomalies(graph, chunks)
            if not anomalies:
                logger.info("Evolution converged: no anomalies detected.")
                break

            hypothesis = await self._generate_hypothesis(anomalies)
            if not hypothesis:
                logger.warning("Unable to generate hypothesis for detected anomalies.")
                continue

            graph = self._refactor_graph(graph, hypothesis)

        return graph

    async def _discover_anomalies(
        self, graph: nx.Graph, chunks: List[TextChunk]
    ) -> List[Dict[str, Any]]:
        """Identify structural or ontological anomalies in the graph."""
        anomalies: List[Dict[str, Any]] = []
        chunk_map = {chunk.chunk_id: chunk.content for chunk in chunks}

        relation_names = sorted(
            {
                data.get("relation_name")
                for _, _, data in graph.edges(data=True)
                if data.get("relation_name")
            }
        )

        if relation_names:
            critic_prompt = GraphPrompt.ONTOLOGY_CRITIC.format(
                relation_list=json.dumps(relation_names, ensure_ascii=False)
            )
            try:
                response = await self.llm.aask(critic_prompt, format="json")
                anomalies.extend(response.get("anomalies", []))
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Ontology critic prompt failed: %s", exc)

        candidate_count = int(
            self._get_evolution_setting("implicit_relation_candidate_pairs", 50)
        )
        hop_distance = int(self._get_evolution_setting("implicit_relation_hop", 1))

        candidate_pairs = self._generate_candidate_pairs(graph, candidate_count)

        if candidate_pairs:
            tasks = [
                self._find_implicit_relation_with_llm(
                    pair, graph, chunk_map, hop_distance
                )
                for pair in candidate_pairs
            ]
            for result in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(result, Exception):
                    logger.warning(
                        "Implicit relation inference task failed: %s", result
                    )
                    continue
                if result:
                    anomalies.append({"type": "implicit_relation", "hypothesis": result})

        return anomalies

    async def _generate_hypothesis(
        self,
        anomalies: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a refactoring hypothesis for the detected anomalies."""

        for anomaly in anomalies:
            anomaly_type = anomaly.get("type")

            if anomaly_type in {"redundancy", "vagueness"}:
                prompt = GraphPrompt.ONTOLOGY_REFACTOR_HYPOTHESIS.format(
                    anomaly=json.dumps(anomaly, ensure_ascii=False)
                )
                try:
                    response = await self.llm.aask(prompt, format="json")
                    if response.get("operation"):
                        return response
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Ontology refactor hypothesis failed: %s", exc)

            if anomaly_type == "implicit_relation":
                hypothesis = anomaly.get("hypothesis")
                if hypothesis and hypothesis.get("operation"):
                    return hypothesis

        return {}

    def _refactor_graph(self, graph: nx.Graph, hypothesis: Dict[str, Any]) -> nx.Graph:
        """Apply the hypothesis operations to refactor the graph."""
        operation = hypothesis.get("operation")
        if not operation:
            return graph

        new_graph = graph.copy()

        if operation == "MERGE_RELATIONS":
            canonical_name = hypothesis.get("canonical_name")
            relations = set(hypothesis.get("relations", []))
            if canonical_name and relations:
                for _, _, data in new_graph.edges(data=True):
                    if data.get("relation_name") in relations:
                        data["relation_name"] = canonical_name
        elif operation == "SPECIALIZE_RELATION":
            specializations = hypothesis.get("specializations", [])
            for specialization in specializations:
                src = specialization.get("from")
                tgt = specialization.get("to")
                relation_name = specialization.get("relation_name")
                if not (src and tgt and relation_name):
                    continue
                if new_graph.has_edge(src, tgt):
                    new_graph.edges[(src, tgt)]["relation_name"] = relation_name
                else:
                    new_graph.add_edge(src, tgt, relation_name=relation_name, weight=1.0)
        elif operation == "CREATE_RELATION":
            src = hypothesis.get("from")
            tgt = hypothesis.get("to")
            relation_name = hypothesis.get("relation_name")
            attributes = {k: v for k, v in hypothesis.items() if k not in {"operation", "from", "to"}}
            if src and tgt and relation_name:
                if not new_graph.has_node(src):
                    new_graph.add_node(src, entity_name=src)
                if not new_graph.has_node(tgt):
                    new_graph.add_node(tgt, entity_name=tgt)
                attributes.setdefault("relation_name", relation_name)
                attributes.setdefault("weight", 1.0)
                new_graph.add_edge(src, tgt, **attributes)
        else:
            logger.warning("Unsupported evolution operation: %s", operation)
            return graph

        return new_graph

    def _generate_candidate_pairs(
        self, graph: nx.Graph, num_candidates: int
    ) -> List[Tuple[str, str]]:
        if graph.number_of_nodes() < 2 or num_candidates <= 0:
            return []

        nodes = list(graph.nodes)
        unique_pairs: Set[Tuple[str, str]] = set()

        random_target = max(1, num_candidates // 2)
        attempts = 0
        max_attempts = random_target * 5
        while len(unique_pairs) < random_target and attempts < max_attempts:
            attempts += 1
            node_a, node_b = random.sample(nodes, 2)
            if graph.has_edge(node_a, node_b):
                continue
            pair = tuple(sorted((node_a, node_b)))
            unique_pairs.add(pair)

        type_buckets: Dict[str, List[str]] = {}
        for node_id, data in graph.nodes(data=True):
            node_type = data.get("entity_type") or data.get("type")
            if node_type:
                type_buckets.setdefault(str(node_type), []).append(node_id)

        remaining = num_candidates - len(unique_pairs)
        for bucket_nodes in type_buckets.values():
            if remaining <= 0 or len(bucket_nodes) < 2:
                continue
            random.shuffle(bucket_nodes)
            for node_a, node_b in islice(
                self._iter_unconnected_pairs(bucket_nodes, graph), remaining
            ):
                pair = tuple(sorted((node_a, node_b)))
                if pair not in unique_pairs:
                    unique_pairs.add(pair)
                    remaining -= 1
                    if remaining <= 0:
                        break

        return [tuple(pair) for pair in unique_pairs]

    def _iter_unconnected_pairs(
        self, nodes: List[str], graph: nx.Graph
    ) -> Iterable[Tuple[str, str]]:
        seen: Set[Tuple[str, str]] = set()
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1 :]:
                pair = tuple(sorted((node_a, node_b)))
                if pair in seen:
                    continue
                seen.add(pair)
                if graph.has_edge(node_a, node_b):
                    continue
                yield node_a, node_b

    async def _find_implicit_relation_with_llm(
        self,
        pair: Tuple[str, str],
        graph: nx.Graph,
        chunk_map: Dict[str, str],
        hop_distance: int,
    ) -> Optional[Dict[str, Any]]:
        node_a, node_b = pair
        subgraph_nodes = self._collect_k_hop_nodes(graph, pair, hop_distance)
        subgraph = graph.subgraph(subgraph_nodes).copy()

        node_a_summary = self._summarize_node(node_a, graph, chunk_map)
        node_b_summary = self._summarize_node(node_b, graph, chunk_map)

        supporting_passages = self._collect_supporting_passages(
            [node_a_summary, node_b_summary]
        )

        subgraph_context = self._serialize_subgraph(subgraph)
        text_context_sections = [
            "Node A Summary:",
            json.dumps(node_a_summary, ensure_ascii=False, indent=2),
            "Node B Summary:",
            json.dumps(node_b_summary, ensure_ascii=False, indent=2),
        ]
        if supporting_passages:
            text_context_sections.append("Supporting Passages:")
            text_context_sections.append(
                json.dumps(supporting_passages, ensure_ascii=False, indent=2)
            )
        text_context = "\n".join(text_context_sections)

        prompt = GraphPrompt.GRAPH_STRUCTURE_REASONING_PROMPT.format(
            entity1_name=node_a,
            entity2_name=node_b,
            subgraph_context=subgraph_context,
            text_context=text_context,
        )

        try:
            response = await self.llm.aask(prompt, format="json")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Graph structure reasoning prompt failed: %s", exc)
            return None

        if not response or not response.get("is_related"):
            return None

        triple = response.get("triple") or {}
        subject = triple.get("subject")
        predicate = triple.get("predicate")
        obj = triple.get("object")

        if not (subject and predicate and obj):
            return None

        subject = str(subject)
        obj = str(obj)
        predicate = str(predicate)

        if graph.has_edge(subject, obj):
            return None

        if {subject, obj} != {node_a, node_b}:
            # Ensure the inferred relation maps back to the evaluated pair.
            return None

        source_ids = [
            passage.get("chunk_id")
            for passage in supporting_passages
            if passage.get("chunk_id")
        ]

        hypothesis: Dict[str, Any] = {
            "operation": "CREATE_RELATION",
            "from": subject,
            "to": obj,
            "relation_name": predicate,
        }
        if source_ids:
            hypothesis["source_id"] = source_ids
        if response.get("reasoning"):
            hypothesis["description"] = response["reasoning"]

        return hypothesis

    def _collect_k_hop_nodes(
        self, graph: nx.Graph, seeds: Iterable[str], hop_distance: int
    ) -> Set[str]:
        if hop_distance <= 0:
            return set(seeds)

        visited: Set[str] = set(seeds)
        frontier: Set[str] = set(seeds)

        for _ in range(hop_distance):
            next_frontier: Set[str] = set()
            for node in frontier:
                next_frontier.update(graph.neighbors(node))
            next_frontier.difference_update(visited)
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier

        return visited

    def _summarize_node(
        self, node_id: str, graph: nx.Graph, chunk_map: Dict[str, str]
    ) -> Dict[str, Any]:
        node_data = dict(graph.nodes[node_id])
        node_data.setdefault("entity_name", node_id)
        node_data.setdefault("source_id", [])

        source_ids = self._normalize_source_ids(node_data.get("source_id"))
        node_data["source_id"] = source_ids
        node_data["source_passages"] = [
            {"chunk_id": chunk_id, "content": chunk_map.get(chunk_id, "")}
            for chunk_id in source_ids
            if chunk_id in chunk_map
        ]
        return node_data

    def _collect_supporting_passages(
        self, node_summaries: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        passages: List[Dict[str, Any]] = []
        seen_chunks: set[str] = set()
        for summary in node_summaries:
            for passage in summary.get("source_passages", []):
                chunk_id = passage.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    passages.append(passage)
        return passages

    def _normalize_source_ids(self, source_field: Any) -> List[str]:
        if not source_field:
            return []
        if isinstance(source_field, list):
            return [str(item).strip() for item in source_field if str(item).strip()]
        if isinstance(source_field, str):
            parts = [part.strip() for part in source_field.split(",")]
            return [part for part in parts if part]
        return [str(source_field).strip()]

    def _serialize_subgraph(self, subgraph: nx.Graph) -> str:
        triples: List[str] = ["Triples:"]
        for src, tgt, data in subgraph.edges(data=True):
            relation = data.get("relation_name") or data.get("type") or "related_to"
            triples.append(f"({src}, {relation}, {tgt})")

        if len(triples) == 1:
            triples.append("(No edges in subgraph)")

        node_lines = ["Nodes:"]
        for node, data in subgraph.nodes(data=True):
            node_lines.append(f"{node}: {json.dumps(data, ensure_ascii=False)}")

        return "\n".join(triples + ["", *node_lines])

    def _get_evolution_setting(self, key: str, default: Any) -> Any:
        if not self.config.evolution:
            return default
        return self.config.evolution.get(key, default)

    async def _extract_entities_relations(
        self, chunk: TextChunk
    ) -> tuple[List[Any], List[Any]]:
        """Not used. Evolving builder delegates extraction to the seed builder."""
        raise NotImplementedError(
            "EvolvingGraphBuilder delegates extraction to the seed graph builder."
        )


GraphBuilderFactory.register_builder(GraphType.EVOLVING, EvolvingGraphBuilder)

