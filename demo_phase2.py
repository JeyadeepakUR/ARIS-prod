"""Demo script to test Phase 2 ML modules with document uploads.

Usage:
    python demo_phase2.py sample.txt

This will:
1. Extract scientific roles (tasks, methods, datasets, metrics, findings)
2. Cluster concepts into domain ontologies
3. Extract semantic relations between entities
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from argparse import ArgumentParser
from typing import Any, cast
import uuid
from datetime import UTC, datetime

# Optional: only run if ML dependencies are installed
try:
    from aris.ml import (
        SciBertRoleTool,
        OntologyClusteringTool,
        SemanticRelationTool,
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML dependencies not installed. Run: pip install aris[ml]")
    sys.exit(1)

# Phase 1 ingestion (optional PDF/text support)
from aris.graph.document_ingestion import DocumentIngestor
from aris.graph import (
    HypothesisInductionEngine,
    KnowledgeGraph,
    Node,
    Edge,
)

# Try to load spaCy for entity extraction
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    nlp = None


def extract_key_terms(text: str) -> list[str]:
    """Extract key technical terms from text (noun phrases, technical terms)."""
    if not SPACY_AVAILABLE or not nlp:
        # Fallback: simple regex-based extraction
        # Extract capitalized multi-word terms and technical-looking phrases
        terms = []
        # Multi-word capitalized terms
        terms.extend(re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b', text))
        # Acronyms
        terms.extend(re.findall(r'\b[A-Z]{2,}\b', text))
        # Hyphenated technical terms
        terms.extend(re.findall(r'\b[a-z]+-[a-z]+(?:-[a-z]+)*\b', text))
        return list(set(terms))[:50]  # Limit to 50 unique terms
    
    doc = nlp(text)
    terms = set()
    
    # Extract noun chunks (technical phrases)
    for chunk in doc.noun_chunks:
        # Skip if too short or contains only stopwords
        if len(chunk.text.split()) > 1 and len(chunk.text) > 4:
            # Skip person names, locations, dates, etc.
            if not any(ent.label_ in ("PERSON", "GPE", "DATE", "TIME", "MONEY", "PERCENT", "ORDINAL", "CARDINAL") 
                      for ent in doc.ents if ent.start <= chunk.start < ent.end or ent.start < chunk.end <= ent.end):
                terms.add(chunk.text.lower().strip())
    
    # Extract named entities that are concepts (not persons, places, etc.)
    for ent in doc.ents:
        if ent.label_ in ("PRODUCT", "WORK_OF_ART", "LAW", "LANGUAGE", "EVENT", "FAC", "ORG"):
            if len(ent.text) > 3:
                terms.add(ent.text.lower().strip())
    
    return list(terms)[:100]  # Limit to 100 unique terms


def extract_entities_from_text(text: str) -> list[dict]:
    """Extract short entity mentions with positions for relation extraction."""
    if not SPACY_AVAILABLE or not nlp:
        print("⚠️  spaCy not available. Skipping entity extraction.")
        return []
    
    doc = nlp(text)
    entities = []
    seen_positions = set()
    
    # Extract noun chunks as potential entities
    for chunk in doc.noun_chunks:
        # Skip generic pronouns and single letters
        if chunk.text.lower() in ("it", "this", "that", "these", "those", "he", "she", "they"):
            continue
        if len(chunk.text) < 2:
            continue
            
        # Skip person names, locations, dates
        is_excluded = False
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "GPE", "DATE", "TIME", "MONEY", "PERCENT", "ORDINAL", "CARDINAL"):
                if (ent.start <= chunk.start < ent.end) or (ent.start < chunk.end <= ent.end):
                    is_excluded = True
                    break
        
        if not is_excluded:
            start = chunk.start_char
            end = chunk.end_char
            if (start, end) not in seen_positions:
                entities.append({
                    "id": f"e_{len(entities)}",
                    "text": chunk.text,
                    "start": start,
                    "end": end
                })
                seen_positions.add((start, end))
    
    # Also include named entities that are concepts
    for ent in doc.ents:
        if ent.label_ in ("PRODUCT", "WORK_OF_ART", "LAW", "LANGUAGE", "EVENT", "FAC", "ORG", "NORP"):
            start = ent.start_char
            end = ent.end_char
            if (start, end) not in seen_positions and len(ent.text) > 2:
                entities.append({
                    "id": f"e_{len(entities)}",
                    "text": ent.text,
                    "start": start,
                    "end": end
                })
                seen_positions.add((start, end))
    
    return entities[:50]  # Limit to first 50 entities for performance


def demo_role_induction(text: str) -> list[dict]:
    """Extract scientific roles from text and return conceptual candidates."""
    print("\n" + "="*80)
    print("MODULE 10: SciBERT Role Induction")
    print("="*80)
    
    tool = SciBertRoleTool()
    result = tool.execute(text)
    data = json.loads(result)
    
    if "error" in data:
        print(f"❌ Error: {data['error']}")
        print(f"   Missing: {', '.join(data.get('required', []))}")
        print(f"   Hint: {data.get('install_hint', 'pip install aris[ml]')}")
        return []
    
    # Filter to only conceptual roles (exclude background, generic spans)
    CONCEPTUAL_ROLES = {"method", "result", "finding", "objective", "conclusion"}
    conceptual_candidates = [
        c for c in data 
        if c.get("role") in CONCEPTUAL_ROLES and c.get("confidence", 0) > 0.3
    ]
    
    print(f"\n✅ Found {len(data)} role candidates ({len(conceptual_candidates)} conceptual):")
    for candidate in conceptual_candidates[:10]:  # Show first 10 conceptual
        print(f"\n  Role: {candidate['role']}")
        print(f"  Span: '{candidate['span_text'][:80]}...'" if len(candidate['span_text']) > 80 else f"  Span: '{candidate['span_text']}'")
        print(f"  Confidence: {candidate['confidence']:.3f}")
        start = candidate.get('start')
        end = candidate.get('end')
        if start is not None and end is not None:
            print(f"  Position: [{start}, {end})")
    
    return conceptual_candidates


def demo_ontology_clustering(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster concepts into domain ontologies."""
    print("\n" + "="*80)
    print("MODULE 11: Domain Ontology Clustering")
    print("="*80)
    
    tool = OntologyClusteringTool()
    input_json = json.dumps(concepts)
    result = tool.execute(input_json)
    data = json.loads(result)
    
    if "error" in data:
        print(f"❌ Error: {data['error']}")
        print(f"   Missing: {', '.join(data.get('required', []))}")
        return []
    
    print(f"\n✅ Found {len(data)} clusters:")
    for cluster in data[:5]:  # Show first 5 clusters
        print(f"\n  Cluster {cluster['cluster_id']}:")
        print(f"  Size: {cluster['size']}")
        print(f"  Stability: {cluster['stability']:.3f}")
        print(f"  Members: {', '.join(cluster['member_ids'][:5])}" + 
              (f" ... (+{len(cluster['member_ids'])-5} more)" if len(cluster['member_ids']) > 5 else ""))
    return data


def build_concept_graph(concepts: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> KnowledgeGraph:
    """Build a deterministic concept graph from clustered terms.

    - Nodes: each concept term
    - Edges: within-cluster deterministic connections
      * For clusters with >=3 members: connect first three pairwise (triangle)
      * For clusters with >=4: also add chain edges among first 4
      * For clusters with 2: connect the pair
    """
    # Deterministic UUIDs based on concept ids
    concept_id_to_node_id: dict[str, uuid.UUID] = {}
    nodes: list[Node] = []

    for c in concepts:
        cid = c["id"]
        label = c["text"]
        node_id = uuid.uuid5(uuid.NAMESPACE_URL, f"concept:{cid}")
        concept_id_to_node_id[cid] = node_id
        # Use a fixed document_id for determinism
        doc_id = uuid.uuid5(uuid.NAMESPACE_URL, "concept_graph")
        nodes.append(Node(node_id=node_id, document_id=doc_id, label=label, node_type="concept", metadata={"kind": "concept"}))

    edges: list[Edge] = []
    for cl in clusters:
        members = cl.get("member_ids", [])
        if len(members) == 2:
            a, b = members
            na_opt = concept_id_to_node_id.get(a)
            nb_opt = concept_id_to_node_id.get(b)
            if na_opt is None or nb_opt is None:
                continue
            na = cast(uuid.UUID, na_opt)
            nb = cast(uuid.UUID, nb_opt)
            eid = uuid.uuid5(uuid.NAMESPACE_URL, f"edge:{cl['cluster_id']}:{str(na)}:{str(nb)}")
            edges.append(
                Edge(
                    edge_id=eid,
                    source_id=na,
                    target_id=nb,
                    edge_type="cluster_link",
                    evidence="cluster_membership",
                    reasoning_trace_id=uuid.uuid5(uuid.NAMESPACE_URL, "concept_graph_trace"),
                    confidence=float(cl.get("stability", 0.5)),
                    metadata={"cluster_id": str(cl["cluster_id"])},
                    created_at=datetime.now(UTC),
                )
            )
        elif len(members) >= 3:
            first_three = members[:3]
            node_ids_opt = [concept_id_to_node_id.get(m) for m in first_three]
            if all(node_ids_opt):
                node_ids = [cast(uuid.UUID, nid) for nid in node_ids_opt]
                # Triangle edges
                pairs = [(0,1),(0,2),(1,2)]
                for i,j in pairs:
                    na = node_ids[i]
                    nb = node_ids[j]
                    eid = uuid.uuid5(uuid.NAMESPACE_URL, f"edge:{cl['cluster_id']}:{str(na)}:{str(nb)}")
                    edges.append(
                        Edge(
                            edge_id=eid,
                            source_id=na,
                            target_id=nb,
                            edge_type="cluster_link",
                            evidence="cluster_membership_triangle",
                            reasoning_trace_id=uuid.uuid5(uuid.NAMESPACE_URL, "concept_graph_trace"),
                            confidence=float(cl.get("stability", 0.75)),
                            metadata={"cluster_id": str(cl["cluster_id"])},
                            created_at=datetime.now(UTC),
                        )
                    )
            # Optional chain among first 4
            if len(members) >= 4:
                chain_members = members[:4]
                chain_nodes_opt = [concept_id_to_node_id.get(m) for m in chain_members]
                if all(chain_nodes_opt):
                    chain_nodes = [cast(uuid.UUID, nid) for nid in chain_nodes_opt]
                    for i in range(len(chain_nodes)-1):
                        na = chain_nodes[i]
                        nb = chain_nodes[i+1]
                        eid = uuid.uuid5(uuid.NAMESPACE_URL, f"edge:{cl['cluster_id']}:chain:{str(na)}:{str(nb)}")
                        edges.append(
                            Edge(
                                edge_id=eid,
                                source_id=na,
                                target_id=nb,
                                edge_type="cluster_chain",
                                evidence="cluster_sequential",
                                reasoning_trace_id=uuid.uuid5(uuid.NAMESPACE_URL, "concept_graph_trace"),
                                confidence=min(1.0, float(cl.get("stability", 0.6)) + 0.1),
                                metadata={"cluster_id": str(cl["cluster_id"])},
                                created_at=datetime.now(UTC),
                            )
                        )

    return KnowledgeGraph(graph_id=uuid.uuid5(uuid.NAMESPACE_URL, "concept_graph"), nodes=nodes, edges=edges, created_at=datetime.now(UTC))


def demo_hypothesis_induction_from_clusters(concepts: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> None:
    """Run Module 13: build concept graph and induce hypotheses."""
    print("\n" + "="*80)
    print("MODULE 13: Descriptive Hypothesis Induction")
    print("="*80)

    if not clusters:
        print("\n⚠️  No clusters available. Skipping hypothesis induction.")
        return

    graph = build_concept_graph(concepts, clusters)
    engine = HypothesisInductionEngine(min_confidence=0.5)
    hypotheses = engine.induce_hypotheses(graph)

    print(f"\n✅ Found {len(hypotheses)} hypothesis candidates:")
    # Summarize by type
    type_counts: dict[str, int] = {}
    for h in hypotheses:
        type_counts[h.hypothesis_type] = type_counts.get(h.hypothesis_type, 0) + 1
    print("  Breakdown:")
    for t, count in sorted(type_counts.items()):
        print(f"    - {t}: {count}")

    # Show up to 10 samples across types
    shown = 0
    for h in hypotheses:
        if shown >= 10:
            break
        print(f"\n  Type: {h.hypothesis_type}")
        print(f"  Description: {h.description}")
        print(f"  Confidence: {h.confidence:.3f}")
        shown += 1


def demo_relation_extraction(text: str, entities: list[dict]) -> None:
    """Extract semantic relations between entities."""
    print("\n" + "="*80)
    print("MODULE 12: Semantic Relation Extraction")
    print("="*80)
    
    tool = SemanticRelationTool()
    input_data = {
        "text": text,
        "entities": entities
    }
    input_json = json.dumps(input_data)
    result = tool.execute(input_json)
    data = json.loads(result)
    
    if "error" in data:
        print(f"❌ Error: {data['error']}")
        print(f"   Missing: {', '.join(data.get('required', []))}")
        return
    
    print(f"\n✅ Found {len(data)} relation candidates:")
    for relation in data[:15]:  # Show first 15
        subj = relation.get("subject_span", "?")
        obj = relation.get("object_span", "?")
        rel_type = relation.get("relation_type", "?")
        print(f"\n  {subj} --[{rel_type}]--> {obj}")
        print(f"  Predicate: '{relation.get('predicate', '?')}'")
        print(f"  Confidence: {relation.get('confidence', 0):.3f}")


def main() -> None:
    """Run Phase 2 demo on provided document (text or PDF)."""
    parser = ArgumentParser(description="Phase 2 demo runner (roles, ontology, relations)")
    parser.add_argument("document", type=str, help="Path to input document (text or PDF)")
    parser.add_argument("--format", choices=["text", "pdf"], default="text", help="Input format")
    args = parser.parse_args()

    doc_path = Path(args.document)
    if not doc_path.exists():
        print(f"❌ File not found: {doc_path}")
        sys.exit(1)

    # Use DocumentIngestor for explicit format handling
    ingestor = DocumentIngestor()
    try:
        doc = ingestor.ingest(doc_path, args.format)
    except ImportError as e:
        # Likely missing pypdf for PDF support
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to ingest document: {e}")
        sys.exit(1)

    text = doc.content
    print(f"\n📄 Processing document: {doc_path}")
    print(f"   Format: {doc.format}")
    print(f"   Length: {len(text)} characters")
    if doc.format == "pdf":
        page_count = doc.metadata.get("page_count", "?")
        extracted_pages = doc.metadata.get("extracted_pages", "?")
        print(f"   Pages: {page_count} (extracted {extracted_pages})")
    
    # Module 10: Role Induction - extract conceptual candidates
    role_candidates = demo_role_induction(text)
    
    if not role_candidates:
        print("\n⚠️  No conceptual candidates found. Skipping clustering and relation extraction.")
    else:
        # Module 11: Ontology Clustering - extract key terms from role spans
        print(f"\n🔍 Extracting key terms from {len(role_candidates)} conceptual spans...")
        all_terms = set()
        for candidate in role_candidates:
            terms = extract_key_terms(candidate["span_text"])
            all_terms.update(terms)
        
        unique_terms = list(all_terms)[:200]  # Limit to 200 most unique terms
        print(f"   Found {len(unique_terms)} unique technical terms")
        
        concepts = [
            {"id": f"term_{i}", "text": term}
            for i, term in enumerate(unique_terms)
        ]
        clusters = demo_ontology_clustering(concepts)
        
        # Module 13: Hypothesis induction from concept clusters
        demo_hypothesis_induction_from_clusters(concepts, clusters)
        
        # Module 12: Relation Extraction - extract entities from full text
        print(f"\n🔍 Extracting entities from document for relation analysis...")
        entities = extract_entities_from_text(text)
        print(f"   Found {len(entities)} entity mentions")
        
        if entities and len(entities) >= 2:
            # Debug: show sample entities
            print(f"   Sample entities: {', '.join([e['text'] for e in entities[:5]])}")
            demo_relation_extraction(text, entities)
        else:
            print("\n⚠️  Insufficient entities extracted. Skipping relation extraction.")
    
    print("\n" + "="*80)
    print("✅ Phase 2 Demo Complete")
    print("="*80)


if __name__ == "__main__":
    main()