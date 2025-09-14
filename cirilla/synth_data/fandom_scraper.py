import asyncio
import json
import os
import re
from pathlib import Path
import fandom
from span_marker import SpanMarkerModel

span_model_name = "tomaarsen/span-marker-bert-base-fewnerd-fine-super"
sm_model = SpanMarkerModel.from_pretrained(span_model_name).cuda()

def clean_text(text: str) -> str:
    """Clean text until the first unwanted section appears, then stop."""
    cleaned_lines = []
    unwanted_sections = {"Footnotes", "References", "Videos", "Gallery", "External links", "See also", "Other"}

    for line in text.split("\n"):
        line = line.strip()

        # hard stop if line starts with any unwanted section
        if any(line.startswith(sec) for sec in unwanted_sections):
            break

        # remove JSON-LD / schema.org lines
        if line.startswith("{\"@context\":") or line.startswith("{ \"@context\":"):
            continue

        # remove footnotes like ↑ 6.0 6.1 Time of Contempt
        if re.match(r"^↑\s?\d+(\.\d+)*.*", line):
            continue

        # remove empty lines
        if not line:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def classify_entities(text: str):
    ents = sm_model.predict(text)
    suggestions = set()
    for e in ents:
        if e["score"] < 0.9:
            continue
        suggestions.add(e["span"].strip())
    return suggestions

def extract_instructions(text: str):
    """Extract instructions from a text and remove them from the main text."""
    lines = text.split('\n')
    capture = False
    extracted_lines = []
    remaining_lines = []

    for i, line in enumerate(lines):
        if i > 0 and lines[i-1].startswith('Quick Answers'):
            capture = True
        elif line.startswith('{'):
            capture = False

        if capture:
            extracted_lines.append(line)
        else:
            remaining_lines.append(line)

    # parse questions/answers
    questions, answers = [], []
    capture = False
    for l in extracted_lines:
        if l.startswith('						Provided by:'):
            capture = False
        if capture:
            answers[-1].append(l)
        if l.endswith('?'):
            questions.append(l)
            answers.append([])
            capture = True

    ans_clean = []
    for a in answers:
        a = ''.join(a)
        a = a.replace('\n', ' ').replace('\t', '').replace("'", '')
        ans_clean.append(a)

    q_a_dict = {q: a for q, a in zip(questions, ans_clean)}
    if len(questions) > 0:
        return q_a_dict, '\n'.join(remaining_lines)
    else:
        return None, text


async def fetch_page(title: str, out_path: Path, instruct_path: Path, search_counter: dict, q: asyncio.Queue, queued: set, visited: set):
    """Download all pages returned by search for a title."""
    try:
        results = await asyncio.to_thread(fandom.search, title)
        search_counter["count"] += 1
        search_counter["queue_size"] = q.qsize()
        print(f"[SEARCH #{search_counter['count']}] {title} -> queue size: {search_counter['queue_size']}")

        if not results:
            print(f"[MISS] {title}")
            return None

        for (page_title, _) in results:
            if page_title in visited:
                continue
            try:
                page = await asyncio.to_thread(fandom.page, page_title)
                text = getattr(page, "plain_text", "")
                if not text:
                    continue

                instructions, cleaned_text = extract_instructions(text)
                cleaned_text = clean_text(cleaned_text)

                fname = page_title.replace("/", "_") + ".txt"
                fpath = out_path / fname
                if not fpath.exists():
                    await asyncio.to_thread(fpath.write_text, cleaned_text, encoding="utf-8")

                if instructions:
                    instruct_file = instruct_path / f"{page_title}.json"
                    def write_json():
                        if not instruct_file.exists():
                            with open(instruct_file, "w", encoding="utf-8") as f:
                                json.dump(instructions, f, ensure_ascii=False, indent=2)
                    await asyncio.to_thread(write_json)
                    print(f"[INSTRUCT] {len(instructions)} instructions saved for {page_title}")

                print(f"[FETCH] {page_title}")
                visited.add(page_title)

            except Exception as inner_e:
                print(f"[ERROR] Failed page {page_title}: {inner_e}")

        return True

    except Exception as e:
        print(f"[ERROR] {title}: {e}")
        return None


async def fetch_worker(q: asyncio.Queue, out_path: Path, instruct_path: Path, visited: set, search_counter: dict, queued: set):
    while True:
        title = await q.get()
        queued.discard(title)
        if title in visited:
            q.task_done()
            continue
        res = await fetch_page(title, out_path, instruct_path, search_counter, q, queued, visited)
        if res:
            visited.add(res)
        q.task_done()


async def classify_worker(out_path: Path, visited: set, q: asyncio.Queue, done: set, queued: set):
    while True:
        for fname in os.listdir(out_path):
            if not fname.endswith(".txt") or fname in done:
                continue
            fpath = out_path / fname
            text = fpath.read_text(encoding="utf-8")
            suggestions = classify_entities(text)
            print(f"[NER] {fname}: {len(suggestions)} suggestions")
            for s in suggestions:
                if s not in visited and s not in queued:
                    queued.add(s)
                    await q.put(s)
                    print(f"[QUEUE] Added '{s}' -> queue size: {q.qsize()}")
            done.add(fname)
        await asyncio.sleep(5)


async def _main(in_path: Path, out_path: Path, instruct_path: Path, n_workers: int = 50):

    out_path.mkdir(parents=True, exist_ok=True)
    instruct_path.mkdir(parents=True, exist_ok=True)

    seeds = []
    for file in os.listdir(in_path):
        if not file.endswith(".json"):
            continue
        with open(in_path / file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                seeds.extend([str(x) for x in data])
    print(f"[SEEDS] {len(seeds)}")

    q = asyncio.Queue()
    visited, done = set(), set()
    queued = set()

    for s in seeds:
        if s not in visited and s not in queued:
            queued.add(s)
            await q.put(s)

    search_counter = {'count': 0, 'queue_size': 0}
    fetchers = [asyncio.create_task(fetch_worker(q, out_path, instruct_path, visited, search_counter, queued)) for _ in range(n_workers)]
    classifier = asyncio.create_task(classify_worker(out_path, visited, q, done, queued))

    await asyncio.gather(*fetchers, classifier)


def scrape_fandom(in_path: Path,
                out_path: Path,
                instruct_path: Path,
                n_workers: int = 50,
                wiki: str = "Witcher",
                lang: str = "en"):

    fandom.set_wiki(wiki)
    fandom.set_lang(lang)

    asyncio.run(_main(in_path, out_path, instruct_path, n_workers))
