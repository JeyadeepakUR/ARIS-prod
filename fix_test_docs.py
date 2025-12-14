import re

# Read file
with open('tests/test_knowledge_graph.py', 'r') as f:
    content = f.read()

# Replace pattern: Document("...", "source_X", "txt", {...})
# to make_doc("...", "source_X", X) 
# This pattern handles both numeric and string first arguments
pattern = r'Document\("(?:doc_)?(\d+|[^"]*)",\s*"([^"]*source_)(\d+)",\s*"txt",\s*(\{[^}]*\})\)'
def replace_doc(match):
    text_or_num = match.group(1)
    source_prefix = match.group(2)
    num = match.group(3)
    metadata = match.group(4)
    
    # If first arg is just a number, use it as both text and id
    if text_or_num.isdigit():
        text = f"Content {text_or_num}"
    else:
        text = text_or_num
    
    if metadata == '{}':
        return f'make_doc("{text}", "source_{num}", {num})'
    else:
        return f'make_doc("{text}", "source_{num}", {num}, {metadata})'

content = re.sub(pattern, replace_doc, content)

# Also fix remaining Document calls that have text in first position
# Pattern: Document("text here", "source_N", "txt", {...})
pattern2 = r'Document\("([^"]+)",\s*"source_(\d+)",\s*"txt",\s*(\{[^}]*\})\)'
def replace_doc2(match):
    text = match.group(1)
    num = match.group(2)
    metadata = match.group(3)
    if metadata == '{}':
        return f'make_doc("{text}", "source_{num}", {num})'
    else:
        return f'make_doc("{text}", "source_{num}", {num}, {metadata})'

content = re.sub(pattern2, replace_doc2, content)

# Also fix DocumentCorpus created_at parameter 
pattern3 = r'created_at=datetime\.now\(\),'
content = re.sub(pattern3, 'created_at=datetime.now(UTC),', content)

# Write back
with open('tests/test_knowledge_graph.py', 'w') as f:
    f.write(content)

print('Replaced Document() calls with make_doc()')
