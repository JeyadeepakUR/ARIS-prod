"""Quick debug script to check entity extraction."""
import spacy
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

# Load PDF text
from aris.graph.document_ingestion import DocumentIngestor
ingestor = DocumentIngestor()
doc = ingestor.ingest(Path("paper.pdf"), "pdf")
text = doc.content[:5000]  # First 5000 chars

print(f"Text length: {len(text)}")
print("\n" + "="*80)
print("SAMPLE TEXT:")
print(text[:500])
print("\n" + "="*80)

doc_nlp = nlp(text)

print("\nNOUN CHUNKS:")
for i, chunk in enumerate(list(doc_nlp.noun_chunks)[:20]):
    print(f"{i}: '{chunk.text}' [{chunk.start_char}, {chunk.end_char})")

print("\n" + "="*80)
print("\nNAMED ENTITIES:")
for i, ent in enumerate(list(doc_nlp.ents)[:20]):
    print(f"{i}: '{ent.text}' - {ent.label_} [{ent.start_char}, {ent.end_char})")

print("\n" + "="*80)
print("\nVERB-BASED PATTERNS:")
for token in doc_nlp:
    if token.pos_ == "VERB":
        subjects = [child.text for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
        objects = [child.text for child in token.children if child.dep_ in ("dobj", "pobj", "attr")]
        if subjects and objects:
            print(f"{subjects[0]} --[{token.lemma_}]--> {objects[0]}")
