interface Props {
  summary: string;
  items: { name: string }[];
  onPick: (index: number) => void;
}

type Seg =
  | { kind: 'text'; text: string }
  | { kind: 'link'; text: string; index: number };

export default function RagSummary({ summary, items, onPick }: Props) {
  if (!summary) return null;

  const sorted = items
    .map((it, i) => ({ name: it.name, i }))
    .filter((it) => it.name && it.name.trim().length > 0)
    .sort((a, b) => b.name.length - a.name.length);

  let segs: Seg[] = [{ kind: 'text', text: summary }];
  for (const { name, i } of sorted) {
    const escaped = name.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
    const re = new RegExp(`(?<![A-Za-z0-9])${escaped}(?![A-Za-z0-9])`, 'gi');
    const next: Seg[] = [];
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
        if (m.index > last) {
          next.push({ kind: 'text', text: txt.slice(last, m.index) });
        }
        next.push({ kind: 'link', text: m[0], index: i });
        last = m.index + m[0].length;
      }
      if (last < txt.length) next.push({ kind: 'text', text: txt.slice(last) });
    }
    segs = next;
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
