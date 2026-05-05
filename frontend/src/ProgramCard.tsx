// ProgramCard — extracted from App.tsx.
// Handles rendering a single program with its week-by-week schedule and the
// optional form-cue tooltips on exercise names (when useLlm is true).

import { useState } from 'react';
import type { Program, ScheduleEntry, FormCue } from './types';

const QUALITY_COPY: Record<string, string> = {
  strong: 'STRONG MATCH',
  moderate: 'MODERATE',
  weak: 'WEAK',
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

export interface ProgramCardProps {
  program: Program;
  rank: number;
  isTop: boolean;
  useLlm: boolean;
  openCueKey: string | null;
  setOpenCueKey: (k: string | null) => void;
  /** Normalised map of exerciseNameLower → FormCue. Only populated for top card. */
  cues: Record<string, FormCue>;
}

export default function ProgramCard({
  program, rank, isTop, useLlm, openCueKey, setOpenCueKey, cues,
}: ProgramCardProps) {
  const [schedOpen, setSchedOpen] = useState<boolean>(false);
  const grouped = groupScheduleByWeekDay(program.schedule);
  const totalSessions = new Set(
    program.schedule.map((s) => `${s.week}-${s.day}`),
  ).size;

  return (
    <article className={`pg-card ${isTop ? 'pg-card--top' : ''}`}>
      <header className="pg-card__head">
        <RankNumeral n={rank} />
        <div className="pg-card__title-wrap">
          <h3 className="pg-card__title">{program.title}</h3>
          {program.description && (
            <p className="pg-card__desc">{program.description}</p>
          )}
        </div>
        <div className="pg-card__score">
          <MatchBar score={program.score} />
          {program.match_quality && (
            <span className={`match-badge match-badge--${program.match_quality}`}>
              {QUALITY_COPY[program.match_quality] ?? program.match_quality}
            </span>
          )}
        </div>
      </header>

      <div className="pg-card__stats">
        <div className="stat">
          <span className="stat__label">LENGTH</span>
          <span className="stat__value">
            {program.program_length_weeks ?? '—'}
            <small>wk</small>
          </span>
        </div>
        <div className="stat">
          <span className="stat__label">LEVEL</span>
          <span className="stat__value stat__value--text">
            {program.level ?? '—'}
          </span>
        </div>
        <div className="stat">
          <span className="stat__label">SESSIONS</span>
          <span className="stat__value">{totalSessions}</span>
        </div>
        <div className="stat stat--wide">
          <span className="stat__label">GOALS</span>
          <div className="stat__chips">
            {program.goal.map((g, i) => (
              <span key={i} className="tag">{g}</span>
            ))}
          </div>
        </div>
      </div>

      {program.tags && program.tags.length > 0 && (
        <div className="pg-card__tags">
          <span className="ex-card__tags-label">WHY IT MATCHED</span>
          {program.tags.map((t, i) => (
            <span key={i} className="tag">
              {t.label}
              <TagDots score={t.score} />
            </span>
          ))}
        </div>
      )}

      <div className="ex-card__actions">
        <button
          type="button"
          className="btn-ghost"
          onClick={() => setSchedOpen((v) => !v)}
        >
          {schedOpen ? 'Hide schedule' : 'View week-by-week schedule'}
          <span className={`chev ${schedOpen ? 'chev--up' : ''}`}>↓</span>
        </button>
      </div>

      {schedOpen && (
        <div className="schedule">
          {grouped.map((wg, wi) => (
            <div key={wi} className="sched-week">
              <div className="sched-week__label">
                <span className="sched-week__num">W{wg.week ?? '?'}</span>
                <span className="sched-week__line" />
              </div>
              <div className="sched-week__days">
                {wg.days.map((dg, di) => (
                  <div key={di} className="sched-day">
                    <h4 className="sched-day__title">Day {dg.day ?? '?'}</h4>
                    <ul className="sched-day__list">
                      {dg.entries.map((entry, ei) => {
                        const cueKey = entry.exercise_name.trim().toLowerCase();
                        const cue = isTop ? cues[cueKey] : undefined;
                        const openKey = `${wi}-${di}-${ei}`;
                        const isOpen = openCueKey === openKey;
                        return (
                          <li key={ei} className="sched-entry">
                            <span className="sched-entry__name">
                              {entry.exercise_name}
                            </span>
                            {entry.sets != null && entry.reps != null && (
                              <span className="sched-entry__reps">
                                {entry.sets}×{entry.reps}
                                {entry.rep_type === 'seconds' ? 's' : ''}
                              </span>
                            )}
                            {cue && useLlm && (
                              <button
                                type="button"
                                className={`cue-btn ${isOpen ? 'cue-btn--open' : ''}`}
                                onClick={() =>
                                  setOpenCueKey(isOpen ? null : openKey)
                                }
                                aria-label="Form cues"
                              >
                                <span>FORM</span>
                              </button>
                            )}
                            {cue && isOpen && (
                              <div className="cue-panel">
                                <div className="cue-panel__head">FORM CUES</div>
                                <ul>
                                  {cue.form_cues.map((c, ci) => (
                                    <li key={ci}>{c}</li>
                                  ))}
                                </ul>
                                {cue.safety && (
                                  <div className="cue-panel__safety">
                                    <span className="cue-panel__safety-label">
                                      ⚠ SAFETY
                                    </span>
                                    <p>{cue.safety}</p>
                                  </div>
                                )}
                              </div>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}

function groupScheduleByWeekDay(schedule: ScheduleEntry[]) {
  const weekMap = new Map<number | null | undefined, Map<number | null | undefined, ScheduleEntry[]>>();
  for (const entry of schedule) {
    if (!weekMap.has(entry.week)) weekMap.set(entry.week, new Map());
    const dayMap = weekMap.get(entry.week)!;
    if (!dayMap.has(entry.day)) dayMap.set(entry.day, []);
    dayMap.get(entry.day)!.push(entry);
  }
  const sortKey = (v: number | null | undefined) =>
    v ?? Number.POSITIVE_INFINITY;
  return [...weekMap.entries()]
    .sort((a, b) => sortKey(a[0]) - sortKey(b[0]))
    .map(([week, dayMap]) => ({
      week,
      days: [...dayMap.entries()]
        .sort((a, b) => sortKey(a[0]) - sortKey(b[0]))
        .map(([day, entries]) => ({ day, entries })),
    }));
}
