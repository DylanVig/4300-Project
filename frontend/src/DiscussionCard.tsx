import { useState } from 'react';
import type { RedditPost } from './types';

const QUALITY_COPY: Record<string, string> = {
  strong: 'STRONG MATCH',
  moderate: 'MODERATE MATCH',
  weak: 'WEAK MATCH',
};

function TagDots({ score }: { score: number }) {
  const filled = score === 0 ? 0 : score <= 25 ? 1 : score <= 50 ? 2 : score <= 75 ? 3 : 4;
  return (
    <span className="tag-dots" aria-label={`${filled} of 4`}>
      {Array.from({ length: 4 }, (_, i) => (
        <span key={i} className={`tag-dot${i < filled ? ' tag-dot--on' : ''}`} />
      ))}
    </span>
  );
}

function RankNumeral({ n }: { n: number }) {
  return <span className="rank-numeral">{String(n).padStart(2, '0')}</span>;
}

function MatchBar({ score }: { score: number }) {
  const pct = Math.min(100, Math.max(0, Math.round(score * 100)));
  return (
    <div className="matchbar" aria-label={`Match score ${pct}%`}>
      <div className="matchbar__fill" style={{ width: `${pct}%` }} />
      <span className="matchbar__val">{pct}</span>
    </div>
  );
}

function decodeEntities(s: string): string {
  return s
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x27;/g, "'");
}

function formatCount(n: number): string {
  if (n >= 1000) {
    const k = n / 1000;
    return `${k >= 10 ? Math.round(k) : k.toFixed(1)}k`;
  }
  return String(n);
}

export interface DiscussionCardProps {
  post: RedditPost;
  rank: number;
  isTop: boolean;
}

export default function DiscussionCard({ post, rank, isTop }: DiscussionCardProps) {
  const [commentsOpen, setCommentsOpen] = useState<boolean>(false);
  const sportKey = (post.sport || '').toLowerCase();
  const hasComments = (post.top_comments?.length ?? 0) > 0;

  return (
    <article id={`disc-card-${rank}`} className={`disc-card ${isTop ? 'disc-card--top' : ''}`}>
      <header className="disc-card__head">
        <RankNumeral n={rank} />
        <div className="disc-card__title-wrap">
          <h3 className="disc-card__title">
            <a
              href={post.permalink}
              target="_blank"
              rel="noopener noreferrer"
              className="disc-card__title-link"
            >
              {decodeEntities(post.title)}
              <span className="disc-card__title-arrow"> ↗</span>
            </a>
          </h3>
        </div>
        <div className="disc-card__score">
          <MatchBar score={post.retrieval_score} />
          {post.match_quality && (
            <span className={`match-badge match-badge--${post.match_quality}`}>
              {QUALITY_COPY[post.match_quality] ?? post.match_quality}
            </span>
          )}
        </div>
      </header>

      <div className="disc-card__chips">
        {post.sport && (
          <span className={`disc-chip disc-chip--sport disc-chip--${sportKey}`}>
            {post.sport}
          </span>
        )}
        {post.subreddit && (
          <span className="disc-chip disc-chip--sub">r/{post.subreddit}</span>
        )}
        <span className="disc-chip disc-chip--stat" title="Reddit upvotes">
          ↑ {formatCount(post.score ?? 0)}
        </span>
        <span className="disc-chip disc-chip--stat" title="Comments on Reddit">
          💬 {formatCount(post.num_comments ?? 0)}
        </span>
      </div>

      {post.selftext_excerpt && (
        <div className="disc-card__body">
          <p className="disc-card__excerpt">{decodeEntities(post.selftext_excerpt)}</p>
        </div>
      )}

      {post.tags && post.tags.length > 0 && (
        <div className="disc-card__tags">
          <span className="ex-card__tags-label">WHY IT MATCHED</span>
          {post.tags.map((t, i) => (
            <span key={i} className="tag">
              {t.label}
              <TagDots score={t.score} />
            </span>
          ))}
        </div>
      )}

      {hasComments && (
        <div className="ex-card__actions">
          <button
            type="button"
            className="btn-ghost"
            onClick={() => setCommentsOpen((v) => !v)}
          >
            {commentsOpen
              ? 'Hide top discussions'
              : `View top ${post.top_comments.length} discussion${post.top_comments.length === 1 ? '' : 's'}`}
            <span className={`chev ${commentsOpen ? 'chev--up' : ''}`}>↓</span>
          </button>
        </div>
      )}

      {commentsOpen && hasComments && (
        <ul className="disc-comments">
          {post.top_comments.map((c, i) => (
            <li key={i} className="disc-comment">
              <span className="disc-comment__score">↑ {formatCount(c.score ?? 0)}</span>
              <p className="disc-comment__body">{decodeEntities(c.body)}</p>
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}
