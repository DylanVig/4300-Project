interface Props {
  summary: string;
  items: { name: string }[];
  onPick: (index: number) => void;
}

type Seg =
  | { kind: 'text'; text: string }
  | { kind: 'link'; text: string; index: number };

const TRAILING_NOISE = /[\s!?.,;:↗"'`~\-—–]+$|[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2300}-\u{23FF}]+$/u;

function trimTrailing(s: string): string {
  let prev = s;
  let next = s.replace(TRAILING_NOISE, '');
  while (next !== prev) {
    prev = next;
    next = next.replace(TRAILING_NOISE, '');
  }
  return next.trim();
}

function escapeRegex(s: string): string {
  return s.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
}

function applyMatches(segs: Seg[], pattern: string, itemIndex: number): { segs: Seg[]; hit: boolean } {
  const re = new RegExp(`(?<![A-Za-z0-9])${pattern}(?![A-Za-z0-9])`, 'gi');
  const next: Seg[] = [];
  let hit = false;
  for (const seg of segs) {
    if (seg.kind !== 'text') {
      next.push(seg);
      continue;
    }
    let last = 0;
    const txt = seg.text;
    let m: RegExpExecArray | null;
    re.lastIndex = 0;
    while ((m = re.exec(txt)) !== null) {
      hit = true;
      if (m.index > last) next.push({ kind: 'text', text: txt.slice(last, m.index) });
      next.push({ kind: 'link', text: m[0], index: itemIndex });
      last = m.index + m[0].length;
    }
    if (last < txt.length) next.push({ kind: 'text', text: txt.slice(last) });
  }
  return { segs: next, hit };
}

export default function RagSummary({ summary, items, onPick }: Props) {
  if (!summary) return null;

  const sorted = items
    .map((it, i) => ({ name: it.name, i }))
    .filter((it) => it.name && it.name.trim().length > 0)
    .sort((a, b) => b.name.length - a.name.length);

  let segs: Seg[] = [{ kind: 'text', text: summary }];
  for (const { name, i } of sorted) {
    const candidates: string[] = [];
    candidates.push(name);
    const trimmed = trimTrailing(name);
    if (trimmed && trimmed !== name && trimmed.length >= 8) {
      candidates.push(trimmed);
    }
    let anyHit = false;
    for (const c of candidates) {
      const res = applyMatches(segs, escapeRegex(c), i);
      segs = res.segs;
      if (res.hit) anyHit = true;
    }
    if (!anyHit && name.length >= 30) {
      const prefix = name.slice(0, 25).trimEnd();
      if (prefix.length >= 12) {
        const res = applyMatches(segs, escapeRegex(prefix), i);
        segs = res.segs;
      }
    }
  }

  return (
    <p className="rag-summary__text">
      {segs.map((s, k) =>
        s.kind === 'text' ? (
          <span key={k}>{s.text}</span>
        ) : (
          <button
            key={k}
            type="button"
            className="rag-summary__link"
            onClick={() => onPick(s.index)}
          >
            {s.text}
          </button>
        ),
      )}
    </p>
  );
}
