import re

file_path = r"C:\Users\gemin\.gemini\antigravity\brain\3f18fa18-8f01-4178-9a44-9227baffb1fa\.system_generated\steps\354\content.md"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Pattern to extract paper titles from hugging face trending papers
matches = re.findall(r'/papers/(\d{4}\.\d{4,5})', text)
unique_ids = []
for m in matches:
    if m not in unique_ids:
        unique_ids.append(m)

print(f"Total unique trending paper IDs found: {len(unique_ids)}")

# Look for titles in h3 or anchor tags
titles = re.findall(r'<h3[^>]*>.*?<a[^>]*>(.*?)</a>.*?</h3>', text, re.DOTALL)
if not titles:
    # try broader pattern
    titles = re.findall(r'href="/papers/\d{4}\.\d{4,5}"[^>]*>(.*?)</a>', text, re.DOTALL)

clean_titles = []
for t in titles:
    clean = re.sub(r'<[^>]+>', '', t).strip()
    if clean and clean not in clean_titles and len(clean) > 5 and not clean.startswith('http'):
        clean_titles.append(clean)

for i, t in enumerate(clean_titles[:20]):
    pid = unique_ids[i] if i < len(unique_ids) else "N/A"
    print(f"{i+1:02d}. [{pid}] {t}")
