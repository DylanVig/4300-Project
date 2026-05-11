import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import type { Exercise, FormCue, Program, RedditPost } from './types';
import {
  EQUIPMENT_OPTIONS,
  DIFFICULTY_OPTIONS,
  MUSCLE_OPTIONS,
  SPORT_OPTIONS,
} from './types';
import ExerciseCard, { type PlanState } from './ExerciseCard';
import ProgramCard from './ProgramCard';
import DiscussionCard from './DiscussionCard';
import RagSummary from './RagSummary';
import MuscleGraph from './MuscleGraph';
import MuscleRadar from './MuscleRadar';
import MuscleMap from './MuscleMap';
import './App.css';

type Tab = 'exercises' | 'programs' | 'sports';
type Method = 'tfidf' | 'svd';
type ResultsView = 'ir' | 'rag';

type RagState<T> = {
  results: T[];
  refinedQuery: string;
  summary: string;
  loading: boolean;
  error: string | null;
};

const emptyExerciseRag: RagState<Exercise> = {
  results: [],
  refinedQuery: '',
  summary: '',
  loading: false,
  error: null,
};
const emptyProgramRag: RagState<Program> = {
  results: [],
  refinedQuery: '',
  summary: '',
  loading: false,
  error: null,
};
const emptySportsRag: RagState<RedditPost> = {
  results: [],
  refinedQuery: '',
  summary: '',
  loading: false,
  error: null,
};

export default function App() {
  const [useLlm, setUseLlm] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<Tab>('exercises');

  const [searchTerm, setSearchTerm] = useState<string>('');
  const [exercises, setExercises] = useState<Exercise[]>([]);
  const [queryMuscles, setQueryMuscles] = useState<string[]>([]);
  const [exerciseMethod, setExerciseMethod] = useState<Method>('tfidf');
  const [exerciseView, setExerciseView] = useState<ResultsView>('ir');
  const [exerciseRag, setExerciseRag] = useState<RagState<Exercise>>(emptyExerciseRag);
  const [selectedEquipment, setSelectedEquipment] = useState<string[]>([]);
  const [difficulty, setDifficulty] = useState<string>('');
  const [injuries, setInjuries] = useState<string[]>([]);
  const [showInjuries, setShowInjuries] = useState<boolean>(false);
  const [showEquipment, setShowEquipment] = useState<boolean>(false);
  const [expandedCards, setExpandedCards] = useState<Record<number, boolean>>({ 0: true });
  const [selectedExerciseIndex, setSelectedExerciseIndex] = useState<number>(0);
  const [planState, setPlanState] = useState<PlanState>({ loading: false, text: '', error: null });
  const [vizModal, setVizModal] = useState<'map' | 'network' | 'radar' | null>(null);
  const [planModalOpen, setPlanModalOpen] = useState<boolean>(false);

  const [programSearchTerm, setProgramSearchTerm] = useState<string>('');
  const [programs, setPrograms] = useState<Program[]>([]);
  const [programMethod, setProgramMethod] = useState<Method>('tfidf');
  const [programsLoading, setProgramsLoading] = useState<boolean>(false);
  const [programView, setProgramView] = useState<ResultsView>('ir');
  const [programRag, setProgramRag] = useState<RagState<Program>>(emptyProgramRag);
  const [openCueKey, setOpenCueKey] = useState<string | null>(null);
  const [topCues, setTopCues] = useState<Record<string, FormCue>>({});

  const [sportsSearchTerm, setSportsSearchTerm] = useState<string>('');
  const [sportsResults, setSportsResults] = useState<RedditPost[]>([]);
  const [sportsMethod, setSportsMethod] = useState<Method>('tfidf');
  const [sportsLoading, setSportsLoading] = useState<boolean>(false);
  const [sportsView, setSportsView] = useState<ResultsView>('ir');
  const [sportsRag, setSportsRag] = useState<RagState<RedditPost>>(emptySportsRag);
  const [sportsFilter, setSportsFilter] = useState<string[]>([]);

  useEffect(() => {
    fetch('/api/config')
      .then((r) => r.json())
      .then((cfg) => setUseLlm(!!cfg.use_llm))
      .catch(() => setUseLlm(false));
  }, []);

  type SearchOverrides = {
    equipment?: string[];
    difficulty?: string;
    injuries?: string[];
    method?: Method;
  };

  const runSearch = async (query: string, overrides: SearchOverrides = {}) => {
    if (!query.trim()) {
      setExercises([]);
      setQueryMuscles([]);
      setPlanState({ loading: false, text: '', error: null });
      return;
    }
    const eq = overrides.equipment ?? selectedEquipment;
    const diff = overrides.difficulty ?? difficulty;
    const inj = overrides.injuries ?? injuries;
    const method = overrides.method ?? exerciseMethod;
    const body: Record<string, unknown> = { query, method };
    if (eq.length > 0) body.equipment = eq;
    if (diff) body.difficulty = diff;
    if (inj.length > 0) body.injuries = inj;
    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      setExercises(data.results ?? []);
      setQueryMuscles(Array.isArray(data.query_muscles) ? data.query_muscles : []);
      setPlanState({ loading: false, text: '', error: null });
      setExpandedCards({ 0: true });
      setSelectedExerciseIndex(0);
    } catch (err) {
      console.error('search failed', err);
      setExercises([]);
      setQueryMuscles([]);
    }
  };

  const runProgramSearch = async (query: string, methodOverride?: Method) => {
    if (!query.trim()) {
      setPrograms([]);
      setTopCues({});
      setOpenCueKey(null);
      return;
    }
    const method = methodOverride ?? programMethod;
    setProgramsLoading(true);
    try {
      const res = await fetch('/api/search_programs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, method }),
      });
      const data = await res.json();
      const results: Program[] = data.results ?? [];
      setPrograms(results);
      setTopCues({});
      setOpenCueKey(null);

      const top = results[0];
      if (useLlm && top && top.schedule && top.schedule.length > 0) {
        const seen = new Set<string>();
        const names: string[] = [];
        for (const entry of top.schedule) {
          const nm = entry.exercise_name?.trim();
          if (!nm) continue;
          const key = nm.toLowerCase();
          if (seen.has(key)) continue;
          seen.add(key);
          names.push(nm);
        }
        if (names.length > 0) {
          fetch('/api/enrich_program', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ exercises: names }),
          })
            .then((r) => (r.ok ? r.json() : null))
            .then((payload) => {
              if (!payload || typeof payload !== 'object') return;
              const cuesRaw = (payload as { cues?: Record<string, FormCue> }).cues;
              if (!cuesRaw) return;
              const normalized: Record<string, FormCue> = {};
              for (const [k, v] of Object.entries(cuesRaw)) {
                if (v && typeof v === 'object' && Array.isArray((v as FormCue).form_cues)) {
                  normalized[k.toLowerCase()] = v as FormCue;
                }
              }
              setTopCues(normalized);
            })
            .catch(() => { /* fail silently */ });
        }
      }
    } catch (err) {
      console.error('program search failed', err);
      setPrograms([]);
    } finally {
      setProgramsLoading(false);
    }
  };

  const runRagSearch = async (query: string, overrides: SearchOverrides = {}) => {
    if (!query.trim()) {
      setExerciseRag(emptyExerciseRag);
      return;
    }
    const eq = overrides.equipment ?? selectedEquipment;
    const diff = overrides.difficulty ?? difficulty;
    const inj = overrides.injuries ?? injuries;
    const method = overrides.method ?? exerciseMethod;
    const body: Record<string, unknown> = { query, method };
    if (eq.length > 0) body.equipment = eq;
    if (diff) body.difficulty = diff;
    if (inj.length > 0) body.injuries = inj;
    setExerciseRag((s) => ({ ...s, loading: true, error: null }));
    try {
      const res = await fetch('/api/rag_search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        setExerciseRag({
          results: [], refinedQuery: '', summary: '', loading: false,
          error: res.status === 503 ? 'LLM not configured' : `Request failed (${res.status})`,
        });
        return;
      }
      const data = await res.json();
      setExerciseRag({
        results: data.results ?? [],
        refinedQuery: data.refined_query ?? '',
        summary: data.summary ?? '',
        loading: false,
        error: null,
      });
      setSelectedExerciseIndex(0);
    } catch (err) {
      console.error('rag search failed', err);
      setExerciseRag({
        results: [], refinedQuery: '', summary: '', loading: false,
        error: err instanceof Error ? err.message : 'RAG search failed',
      });
    }
  };

  const runRagProgramSearch = async (query: string, methodOverride?: Method) => {
    if (!query.trim()) {
      setProgramRag(emptyProgramRag);
      return;
    }
    const method = methodOverride ?? programMethod;
    setProgramRag((s) => ({ ...s, loading: true, error: null }));
    try {
      const res = await fetch('/api/rag_search_programs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, method }),
      });
      if (!res.ok) {
        setProgramRag({
          results: [], refinedQuery: '', summary: '', loading: false,
          error: res.status === 503 ? 'LLM not configured' : `Request failed (${res.status})`,
        });
        return;
      }
      const data = await res.json();
      setProgramRag({
        results: data.results ?? [],
        refinedQuery: data.refined_query ?? '',
        summary: data.summary ?? '',
        loading: false,
        error: null,
      });
    } catch (err) {
      console.error('rag program search failed', err);
      setProgramRag({
        results: [], refinedQuery: '', summary: '', loading: false,
        error: err instanceof Error ? err.message : 'RAG search failed',
      });
    }
  };

  const runSportsSearch = async (
    query: string,
    methodOverride?: Method,
    sportsOverride?: string[],
  ) => {
    if (!query.trim()) {
      setSportsResults([]);
      return;
    }
    const method = methodOverride ?? sportsMethod;
    const sports = sportsOverride ?? sportsFilter;
    const body: Record<string, unknown> = { query, method };
    if (sports.length > 0) body.sports = sports;
    setSportsLoading(true);
    try {
      const res = await fetch('/api/search_reddit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      setSportsResults(data.results ?? []);
    } catch (err) {
      console.error('sports search failed', err);
      setSportsResults([]);
    } finally {
      setSportsLoading(false);
    }
  };

  const runRagSportsSearch = async (
    query: string,
    methodOverride?: Method,
    sportsOverride?: string[],
  ) => {
    if (!query.trim()) {
      setSportsRag(emptySportsRag);
      return;
    }
    const method = methodOverride ?? sportsMethod;
    const sports = sportsOverride ?? sportsFilter;
    const body: Record<string, unknown> = { query, method };
    if (sports.length > 0) body.sports = sports;
    setSportsRag((s) => ({ ...s, loading: true, error: null }));
    try {
      const res = await fetch('/api/rag_search_reddit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        setSportsRag({
          results: [], refinedQuery: '', summary: '', loading: false,
          error: res.status === 503 ? 'LLM not configured' : `Request failed (${res.status})`,
        });
        return;
      }
      const data = await res.json();
      setSportsRag({
        results: data.results ?? [],
        refinedQuery: data.refined_query ?? '',
        summary: data.summary ?? '',
        loading: false,
        error: null,
      });
    } catch (err) {
      console.error('rag sports search failed', err);
      setSportsRag({
        results: [], refinedQuery: '', summary: '', loading: false,
        error: err instanceof Error ? err.message : 'RAG search failed',
      });
    }
  };

  const handleGeneratePlan = async (exercise: Exercise) => {
    setPlanState({ loading: true, text: '', error: null });
    setPlanModalOpen(true);
    const pool = displayedExercises.filter(e => e.name !== exercise.name);
    try {
      const res = await fetch('/api/enrich_exercise', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: exercise.name,
          primaryMuscles: exercise.primaryMuscles,
          secondaryMuscles: exercise.secondaryMuscles,
          equipment: exercise.equipment,
          instructions: exercise.instructions,
          pool: pool.map(e => ({
            name: e.name,
            primaryMuscles: e.primaryMuscles,
            equipment: e.equipment,
          })),
        }),
      });
      if (!res.ok || !res.body) {
        setPlanState({ loading: false, text: '', error: 'Could not generate plan.' });
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let acc = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const payload = JSON.parse(line.slice(6));
            if (payload.error) {
              setPlanState({ loading: false, text: acc, error: payload.error });
              return;
            }
            if (typeof payload.content === 'string') {
              acc += payload.content;
              setPlanState({ loading: true, text: acc, error: null });
            }
          } catch { /* ignore malformed line */ }
        }
      }
      setPlanState({ loading: false, text: acc, error: null });
    } catch (err) {
      setPlanState({
        loading: false,
        text: '',
        error: err instanceof Error ? err.message : 'Plan generation failed',
      });
    }
  };

  const triggerExerciseSearch = (query: string, overrides: SearchOverrides = {}) => {
    runSearch(query, overrides);
    if (useLlm && exerciseView === 'rag' && query.trim()) {
      runRagSearch(query, overrides);
    }
  };

  const triggerProgramSearch = (query: string, methodOverride?: Method) => {
    runProgramSearch(query, methodOverride);
    if (useLlm && programView === 'rag' && query.trim()) {
      runRagProgramSearch(query, methodOverride);
    }
  };

  const triggerSportsSearch = (
    query: string,
    methodOverride?: Method,
    sportsOverride?: string[],
  ) => {
    runSportsSearch(query, methodOverride, sportsOverride);
    if (useLlm && sportsView === 'rag' && query.trim()) {
      runRagSportsSearch(query, methodOverride, sportsOverride);
    }
  };

  const toggleSportFilter = (s: string) => {
    const next = sportsFilter.includes(s)
      ? sportsFilter.filter((x) => x !== s)
      : [...sportsFilter, s];
    setSportsFilter(next);
    if (sportsSearchTerm.trim()) triggerSportsSearch(sportsSearchTerm, undefined, next);
  };

  const toggleEquipment = (v: string) => {
    const next = selectedEquipment.includes(v)
      ? selectedEquipment.filter((x) => x !== v)
      : [...selectedEquipment, v];
    setSelectedEquipment(next);
    if (searchTerm.trim()) triggerExerciseSearch(searchTerm, { equipment: next });
  };

  const changeDifficulty = (v: string) => {
    setDifficulty(v);
    if (searchTerm.trim()) triggerExerciseSearch(searchTerm, { difficulty: v });
  };

  const toggleInjury = (v: string) => {
    const next = injuries.includes(v)
      ? injuries.filter((x) => x !== v)
      : [...injuries, v];
    setInjuries(next);
    if (searchTerm.trim()) triggerExerciseSearch(searchTerm, { injuries: next });
  };

  const changeMethod = (m: Method) => {
    if (activeTab === 'exercises') {
      setExerciseMethod(m);
      if (searchTerm.trim()) triggerExerciseSearch(searchTerm, { method: m });
    } else if (activeTab === 'programs') {
      setProgramMethod(m);
      if (programSearchTerm.trim()) triggerProgramSearch(programSearchTerm, m);
    } else {
      setSportsMethod(m);
      if (sportsSearchTerm.trim()) triggerSportsSearch(sportsSearchTerm, m);
    }
  };

  const changeView = (v: ResultsView) => {
    if (activeTab === 'exercises') {
      setExerciseView(v);
      setSelectedExerciseIndex(0);
      if (v === 'rag' && useLlm && searchTerm.trim()) {
        runRagSearch(searchTerm);
      }
    } else if (activeTab === 'programs') {
      setProgramView(v);
      if (v === 'rag' && useLlm && programSearchTerm.trim()) {
        runRagProgramSearch(programSearchTerm);
      }
    } else {
      setSportsView(v);
      if (v === 'rag' && useLlm && sportsSearchTerm.trim()) {
        runRagSportsSearch(sportsSearchTerm);
      }
    }
  };

  const toggleCard = (idx: number) => {
    setExpandedCards((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const scrollToCard = (cssId: string, index: number, isExercise: boolean) => {
    if (isExercise) setSelectedExerciseIndex(index);
    requestAnimationFrame(() => {
      document.getElementById(cssId)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  };

  const currentMethod =
    activeTab === 'exercises' ? exerciseMethod
    : activeTab === 'programs' ? programMethod
    : sportsMethod;

  const displayedExercises = exerciseView === 'rag' ? exerciseRag.results : exercises;
  const selectedExercise = displayedExercises[selectedExerciseIndex] ?? null;

  const welcomeCard = (
    <div className="welcome-card">
      <div className="welcome-card__title">ATHLETIC TRAINING FINDER</div>
      <div className="welcome-card__rule" />
      <p className="welcome-card__body">
        Describe a fitness goal and we&apos;ll return the top{' '}
        <strong>
          {activeTab === 'exercises'
            ? 'exercises'
            : activeTab === 'programs'
              ? 'workout programs'
              : 'coaching discussions'}
        </strong>{' '}
        ranked by relevance to your query.
      </p>
      <ul className="welcome-card__hints">
        <li>Switch <strong>KEYWORD ↔ SEMANTIC</strong> to compare retrieval methods</li>
        {useLlm && <li>Use <strong>IR + RAG</strong> for LLM-assisted reranking</li>}
        {activeTab === 'exercises' && <li>Filter by equipment, difficulty, or injuries</li>}
        {activeTab === 'sports' && <li>Filter by <strong>sport</strong> (soccer · basketball)</li>}
      </ul>
      <div className="welcome-card__try">
        Try: &ldquo;
        {activeTab === 'exercises'
          ? 'improve vertical jump'
          : activeTab === 'programs'
            ? '8 week hypertrophy'
            : 'youth soccer pressing triggers'}
        &rdquo;
      </div>
    </div>
  );

  return (
    <div className="app">
      {/* ─── LEFT RAIL ──────────────────────────────────────────────── */}
      <aside className="rail">
        <div className="rail__brand">
          <div className="brand-name">
            ATHLETIC<br />TRAINING<br /><span>FINDER</span>
          </div>
          <div className="brand-sub">CORNELL · CS 4300</div>
        </div>

        <div className="rail__tabs">
          <button
            type="button"
            className={`railtab ${activeTab === 'exercises' ? 'is-active' : ''}`}
            onClick={() => setActiveTab('exercises')}
          >
            <span className="railtab__idx">01</span>
            <span className="railtab__name">EXERCISES</span>
          </button>
          <button
            type="button"
            className={`railtab ${activeTab === 'programs' ? 'is-active' : ''}`}
            onClick={() => setActiveTab('programs')}
          >
            <span className="railtab__idx">02</span>
            <span className="railtab__name">PROGRAMS</span>
          </button>
          <button
            type="button"
            className={`railtab ${activeTab === 'sports' ? 'is-active' : ''}`}
            onClick={() => setActiveTab('sports')}
          >
            <span className="railtab__idx">03</span>
            <span className="railtab__name">SPORTS</span>
          </button>
        </div>

        <div className="rail__section">
          <div className="rail__section-label">RETRIEVAL MODE</div>
          <div className="method-toggle">
            <button
              type="button"
              className={`method ${currentMethod === 'tfidf' ? 'is-active' : ''}`}
              onClick={() => changeMethod('tfidf')}
            >
              <span className="method__name">KEYWORD</span>
              <span className="method__tech">tf-idf</span>
            </button>
            <button
              type="button"
              className={`method ${currentMethod === 'svd' ? 'is-active' : ''}`}
              onClick={() => changeMethod('svd')}
            >
              <span className="method__name">SEMANTIC</span>
              <span className="method__tech">svd</span>
            </button>
          </div>
        </div>

        <div className="rail__section">
          <div className="rail__section-label">QUERY</div>
          <div className="search-box">
            <svg width="16" height="16" viewBox="0 0 16 16" className="search-box__icon" aria-hidden>
              <circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" strokeWidth="1.75" />
              <path d="M11 11 L15 15" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" />
            </svg>
            <textarea
              rows={2}
              value={
                activeTab === 'exercises'
                  ? searchTerm
                  : activeTab === 'programs'
                    ? programSearchTerm
                    : sportsSearchTerm
              }
              placeholder={
                activeTab === 'exercises'
                  ? 'e.g. improve vertical jump'
                  : activeTab === 'programs'
                    ? 'e.g. 8 week hypertrophy'
                    : 'e.g. youth soccer pressing triggers'
              }
              onChange={(e) => {
                if (activeTab === 'exercises') setSearchTerm(e.target.value);
                else if (activeTab === 'programs') setProgramSearchTerm(e.target.value);
                else setSportsSearchTerm(e.target.value);
              }}
              onInput={(e) => {
                const el = e.currentTarget;
                el.style.height = 'auto';
                el.style.height = el.scrollHeight + 'px';
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (activeTab === 'exercises') triggerExerciseSearch(searchTerm);
                  else if (activeTab === 'programs') triggerProgramSearch(programSearchTerm);
                  else triggerSportsSearch(sportsSearchTerm);
                }
              }}
            />
          </div>
          <button
            type="button"
            className="run-btn"
            onClick={() =>
              activeTab === 'exercises'
                ? triggerExerciseSearch(searchTerm)
                : activeTab === 'programs'
                  ? triggerProgramSearch(programSearchTerm)
                  : triggerSportsSearch(sportsSearchTerm)
            }
          >
            <span>RUN SEARCH</span>
            <span className="run-btn__arrow">→</span>
          </button>
        </div>

        {activeTab === 'exercises' && (
          <>
            <div className="rail__section">
              <div className="rail__section-label">FILTERS</div>
              <button
                type="button"
                className="filter-toggle"
                onClick={() => setShowEquipment((s) => !s)}
              >
                <span className="filter-toggle__label">EQUIPMENT AND DIFFICULTY</span>
                {(selectedEquipment.length + (difficulty ? 1 : 0)) > 0 && (
                  <span className="filter-toggle__count">
                    {selectedEquipment.length + (difficulty ? 1 : 0)}
                  </span>
                )}
                <span className={`chev ${showEquipment ? 'chev--up' : ''}`}>↓</span>
              </button>
              {showEquipment && (
                <div className="filter-panel">
                  <div className="filters__sub-label">DIFFICULTY</div>
                  <div className="diff-toggle">
                    <button
                      type="button"
                      className={`diff ${difficulty === '' ? 'is-active' : ''}`}
                      onClick={() => changeDifficulty('')}
                    >
                      ANY
                    </button>
                    {DIFFICULTY_OPTIONS.map((opt) => (
                      <button
                        key={opt}
                        type="button"
                        className={`diff ${difficulty === opt ? 'is-active' : ''}`}
                        onClick={() => changeDifficulty(opt)}
                      >
                        {opt.toUpperCase()}
                      </button>
                    ))}
                  </div>
                  <div className="filters__sub-label filters__sub-label--gap">EQUIPMENT</div>
                  <div className="chip-grid">
                    {EQUIPMENT_OPTIONS.map((opt) => (
                      <button
                        key={opt}
                        type="button"
                        className={`chip ${selectedEquipment.includes(opt) ? 'is-active' : ''}`}
                        onClick={() => toggleEquipment(opt)}
                      >
                        {opt}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="rail__section">
              <button
                type="button"
                className="injury-toggle"
                onClick={() => setShowInjuries((s) => !s)}
              >
                <span className="injury-toggle__warn">⚠</span>
                <span className="injury-toggle__label">INJURED MUSCLES TO AVOID</span>
                {injuries.length > 0 && (
                  <span className="injury-toggle__count">{injuries.length}</span>
                )}
                <span className={`chev ${showInjuries ? 'chev--up' : ''}`}>↓</span>
              </button>
              {showInjuries && (
                <>
                  <p className="injury-note">
                    Exercises that primarily target these muscles will be filtered
                    out of your results.
                  </p>
                  <div className="chip-grid">
                    {MUSCLE_OPTIONS.map((opt) => (
                      <button
                        key={opt}
                        type="button"
                        className={`chip chip--injury ${injuries.includes(opt) ? 'is-active' : ''}`}
                        onClick={() => toggleInjury(opt)}
                      >
                        {opt}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          </>
        )}

        {activeTab === 'sports' && (
          <div className="rail__section">
            <div className="rail__section-label">SPORT</div>
            <div className="chip-grid">
              {SPORT_OPTIONS.map((s) => (
                <button
                  key={s}
                  type="button"
                  className={`chip ${sportsFilter.includes(s) ? 'is-active' : ''}`}
                  onClick={() => toggleSportFilter(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {useLlm && activeTab === 'exercises' && (
          <div className="rail__section">
            <div className="rail__section-label">SESSION PLAN</div>
            <button
              type="button"
              className="run-btn"
              onClick={() => selectedExercise && handleGeneratePlan(selectedExercise)}
              disabled={!selectedExercise || planState.loading}
            >
              <span>
                {planState.loading
                  ? 'Building session…'
                  : planState.text
                    ? 'Regenerate session'
                    : "Generate today's session"}
              </span>
              <span className="run-btn__arrow">→</span>
            </button>
            {planState.text && !planState.loading && (
              <button
                type="button"
                className="view-plan-btn"
                onClick={() => setPlanModalOpen(true)}
              >
                View plan ↗
              </button>
            )}
          </div>
        )}

        <div className="rail__foot">
          <span>v1.0</span>
          <span>CS 4300</span>
        </div>
      </aside>

      {/* ─── MAIN ───────────────────────────────────────────────────── */}
      <main className="main">
        {activeTab === 'exercises' ? (
          <div className="workspace workspace--exercises" data-tab="exercises">
            <section className="results">
              <header className="results__head">
                <div className="results__head-row">
                  <div>
                    <div className="results__label">RESULTS</div>
                    <h2 className="results__title">
                      <span className="results__count">
                        {exerciseView === 'rag' ? exerciseRag.results.length : exercises.length}
                      </span>
                      <span>exercises ranked by</span>
                      <span className="results__method">
                        {exerciseMethod === 'tfidf' ? 'KEYWORD RETRIEVAL' : 'SEMANTIC RETRIEVAL'}
                      </span>
                    </h2>
                  </div>
                  <div className="results__query">
                    <span className="results__query-label">QUERY</span>
                    <span className="results__query-text">"{searchTerm}"</span>
                  </div>
                </div>
                <div className="viewtabs">
                  <button
                    type="button"
                    className={`viewtab ${exerciseView === 'ir' ? 'is-active' : ''}`}
                    onClick={() => changeView('ir')}
                  >
                    IR
                  </button>
                  <button
                    type="button"
                    className={`viewtab ${exerciseView === 'rag' ? 'is-active' : ''}`}
                    onClick={() => changeView('rag')}
                  >
                    IR + RAG
                  </button>
                </div>
              </header>

              {exerciseView === 'rag' && useLlm && (exerciseRag.refinedQuery || exerciseRag.loading || exerciseRag.error) && (
                <div className="refined-query">
                  <div className="refined-query__row">
                    <span className="refined-query__label">ORIGINAL</span>
                    <span className="refined-query__text">{searchTerm || '—'}</span>
                  </div>
                  <div className="refined-query__row">
                    <span className="refined-query__label">REFINED</span>
                    <span className="refined-query__text refined-query__text--accent">
                      {exerciseRag.loading
                        ? 'refining query…'
                        : exerciseRag.error
                          ? `(error: ${exerciseRag.error})`
                          : exerciseRag.refinedQuery || '—'}
                    </span>
                  </div>
                </div>
              )}

              {exerciseView === 'rag' && useLlm && !exerciseRag.loading && exerciseRag.summary && (
                <div className="rag-summary">
                  <span className="rag-summary__label">AI SUMMARY</span>
                  <RagSummary
                    summary={exerciseRag.summary}
                    items={exerciseRag.results.map((e) => ({ name: e.name }))}
                    onPick={(i) => scrollToCard(`ex-card-${i + 1}`, i, true)}
                  />
                </div>
              )}

              {exerciseView === 'ir' ? (
                <div className="results__list">
                  {exercises.map((ex, i) => (
                    <ExerciseCard
                      key={`${ex.name}-${i}`}
                      exercise={ex}
                      rank={i + 1}
                      expanded={!!expandedCards[i]}
                      isSelected={selectedExerciseIndex === i}
                      onToggleExpand={() => toggleCard(i)}
                      onSelectCard={() => setSelectedExerciseIndex(i)}
                    />
                  ))}
                  {exercises.length === 0 && (
                    searchTerm.trim() ? (
                      <div className="empty">
                        <div className="empty__icon">◎</div>
                        <p>No results. Try broadening your filters or query.</p>
                      </div>
                    ) : welcomeCard
                  )}
                </div>
              ) : exerciseRag.loading ? (
                <div className="loading">
                  <div className="loading__spinner" />
                  <p>Refining query &amp; reranking with LLM…</p>
                </div>
              ) : (
                <div className="results__list">
                  {exerciseRag.results.map((ex, i) => (
                    <ExerciseCard
                      key={`rag-${ex.name}-${i}`}
                      exercise={ex}
                      rank={i + 1}
                      expanded={!!expandedCards[i]}
                      isSelected={selectedExerciseIndex === i}
                      onToggleExpand={() => toggleCard(i)}
                      onSelectCard={() => setSelectedExerciseIndex(i)}
                    />
                  ))}
                  {exerciseRag.results.length === 0 && !exerciseRag.error && (
                    searchTerm.trim() ? (
                      <div className="empty">
                        <div className="empty__icon">◎</div>
                        <p>Run a search to see IR + RAG results.</p>
                      </div>
                    ) : welcomeCard
                  )}
                </div>
              )}
            </section>

            <div className="vizcol">
              <aside className="bodypanel bodypanel--clickable" onClick={() => setVizModal('map')}>
                <div className="bodypanel__head">
                  <div>
                    <div className="bodypanel__label">MUSCLE MAP</div>
                    <div className="bodypanel__sub">
                      {selectedExercise ? selectedExercise.name : 'Select an exercise'}
                    </div>
                    <div className="bodypanel__desc">Primary and secondary muscles worked by the selected exercise. Click any result to update.</div>
                  </div>
                  <span className="expand-hint">expand ↗</span>
                </div>
                <MuscleMap
                  primaryMuscles={selectedExercise?.primaryMuscles ?? []}
                  secondaryMuscles={selectedExercise?.secondaryMuscles ?? []}
                />
              </aside>

              <section className="netpanel netpanel--clickable" onClick={() => setVizModal('network')}>
                <header className="netpanel__head">
                  <div>
                    <div className="netpanel__label">MUSCLE NETWORK</div>
                    <h2 className="netpanel__title">
                      <span>how results connect to muscles</span>
                      {queryMuscles.length > 0 && (
                        <span className="netpanel__count">
                          · {queryMuscles.length} query muscle
                          {queryMuscles.length === 1 ? '' : 's'}
                        </span>
                      )}
                    </h2>
                    <div className="netpanel__desc">Hover an exercise or muscle to see its connections across all results.</div>
                  </div>
                  <span className="expand-hint">expand ↗</span>
                </header>
                <MuscleGraph
                  queryMuscles={queryMuscles}
                  exercises={
                    exerciseView === 'rag' && exerciseRag.results.length > 0
                      ? exerciseRag.results
                      : exercises
                  }
                />
              </section>

              <section className="netpanel netpanel--clickable" onClick={() => setVizModal('radar')}>
                <header className="netpanel__head">
                  <div>
                    <div className="netpanel__label">MUSCLE COVERAGE</div>
                    <h2 className="netpanel__title">
                      <span>muscle coverage across results</span>
                    </h2>
                    <div className="netpanel__desc">Outer = better coverage in the top results.</div>
                  </div>
                  <span className="expand-hint">expand ↗</span>
                </header>
                <MuscleRadar
                  queryMuscles={queryMuscles}
                  exercises={
                    exerciseView === 'rag' && exerciseRag.results.length > 0
                      ? exerciseRag.results
                      : exercises
                  }
                />
              </section>
            </div>
          </div>
        ) : activeTab === 'programs' ? (
          <div className="workspace workspace--programs">
            <section className="results results--full">
              <header className="results__head">
                <div className="results__head-row">
                  <div>
                    <div className="results__label">PROGRAMS</div>
                    <h2 className="results__title">
                      <span className="results__count">
                        {programView === 'rag' ? programRag.results.length : programs.length}
                      </span>
                      <span>programs ranked by</span>
                      <span className="results__method">
                        {programMethod === 'tfidf' ? 'KEYWORD RETRIEVAL' : 'SEMANTIC RETRIEVAL'}
                      </span>
                    </h2>
                  </div>
                  <div className="results__query">
                    <span className="results__query-label">QUERY</span>
                    <span className="results__query-text">"{programSearchTerm}"</span>
                  </div>
                </div>
                <div className="viewtabs">
                  <button
                    type="button"
                    className={`viewtab ${programView === 'ir' ? 'is-active' : ''}`}
                    onClick={() => changeView('ir')}
                  >
                    IR
                  </button>
                  <button
                    type="button"
                    className={`viewtab ${programView === 'rag' ? 'is-active' : ''}`}
                    onClick={() => changeView('rag')}
                  >
                    IR + RAG
                  </button>
                </div>
              </header>

              {programView === 'rag' && useLlm && (programRag.refinedQuery || programRag.loading || programRag.error) && (
                <div className="refined-query">
                  <div className="refined-query__row">
                    <span className="refined-query__label">ORIGINAL</span>
                    <span className="refined-query__text">{programSearchTerm || '—'}</span>
                  </div>
                  <div className="refined-query__row">
                    <span className="refined-query__label">REFINED</span>
                    <span className="refined-query__text refined-query__text--accent">
                      {programRag.loading
                        ? 'refining query…'
                        : programRag.error
                          ? `(error: ${programRag.error})`
                          : programRag.refinedQuery || '—'}
                    </span>
                  </div>
                </div>
              )}

              {programView === 'rag' && useLlm && !programRag.loading && programRag.summary && (
                <div className="rag-summary">
                  <span className="rag-summary__label">AI SUMMARY</span>
                  <RagSummary
                    summary={programRag.summary}
                    items={programRag.results.map((p) => ({ name: p.title }))}
                    onPick={(i) => scrollToCard(`pg-card-${i + 1}`, i, false)}
                  />
                </div>
              )}

              {programView === 'ir' && programsLoading ? (
                <div className="loading">
                  <div className="loading__spinner" />
                  <p>
                    Searching programs index…
                    <br />
                    <small>first run may take ~25s</small>
                  </p>
                </div>
              ) : programView === 'rag' && programRag.loading ? (
                <div className="loading">
                  <div className="loading__spinner" />
                  <p>Refining query &amp; reranking with LLM…</p>
                </div>
              ) : programView === 'ir' ? (
                <div className="results__list">
                  {programs.map((pg, i) => (
                    <ProgramCard
                      key={`${pg.title}-${i}`}
                      program={pg}
                      rank={i + 1}
                      isTop={i === 0}
                      useLlm={useLlm}
                      openCueKey={openCueKey}
                      setOpenCueKey={setOpenCueKey}
                      cues={topCues}
                    />
                  ))}
                  {programs.length === 0 && (
                    programSearchTerm.trim() ? (
                      <div className="empty">
                        <div className="empty__icon">◎</div>
                        <p>No results. Try a different query.</p>
                      </div>
                    ) : welcomeCard
                  )}
                </div>
              ) : (
                <div className="results__list">
                  {programRag.results.map((pg, i) => (
                    <ProgramCard
                      key={`rag-${pg.title}-${i}`}
                      program={pg}
                      rank={i + 1}
                      isTop={i === 0}
                      useLlm={useLlm}
                      openCueKey={openCueKey}
                      setOpenCueKey={setOpenCueKey}
                      cues={topCues}
                    />
                  ))}
                  {programRag.results.length === 0 && !programRag.error && (
                    programSearchTerm.trim() ? (
                      <div className="empty">
                        <div className="empty__icon">◎</div>
                        <p>Run a search to see IR + RAG results.</p>
                      </div>
                    ) : welcomeCard
                  )}
                </div>
              )}
            </section>
          </div>
        ) : (
          <div className="workspace workspace--sports">
            <section className="results results--full">
              <header className="results__head">
                <div className="results__head-row">
                  <div>
                    <div className="results__label">SPORTS</div>
                    <h2 className="results__title">
                      <span className="results__count">
                        {sportsView === 'rag' ? sportsRag.results.length : sportsResults.length}
                      </span>
                      <span>discussions ranked by</span>
                      <span className="results__method">
                        {sportsMethod === 'tfidf' ? 'KEYWORD RETRIEVAL' : 'SEMANTIC RETRIEVAL'}
                      </span>
                    </h2>
                  </div>
                  <div className="results__query">
                    <span className="results__query-label">QUERY</span>
                    <span className="results__query-text">"{sportsSearchTerm}"</span>
                  </div>
                </div>
                <div className="viewtabs">
                  <button
                    type="button"
                    className={`viewtab ${sportsView === 'ir' ? 'is-active' : ''}`}
                    onClick={() => changeView('ir')}
                  >
                    IR
                  </button>
                  <button
                    type="button"
                    className={`viewtab ${sportsView === 'rag' ? 'is-active' : ''}`}
                    onClick={() => changeView('rag')}
                  >
                    IR + RAG
                  </button>
                </div>
              </header>

              {sportsView === 'rag' && useLlm && (sportsRag.refinedQuery || sportsRag.loading || sportsRag.error) && (
                <div className="refined-query">
                  <div className="refined-query__row">
                    <span className="refined-query__label">ORIGINAL</span>
                    <span className="refined-query__text">{sportsSearchTerm || '—'}</span>
                  </div>
                  <div className="refined-query__row">
                    <span className="refined-query__label">REFINED</span>
                    <span className="refined-query__text refined-query__text--accent">
                      {sportsRag.loading
                        ? 'refining query…'
                        : sportsRag.error
                          ? `(error: ${sportsRag.error})`
                          : sportsRag.refinedQuery || '—'}
                    </span>
                  </div>
                </div>
              )}

              {sportsView === 'rag' && useLlm && !sportsRag.loading && sportsRag.summary && (
                <div className="rag-summary">
                  <span className="rag-summary__label">AI SUMMARY</span>
                  <RagSummary
                    summary={sportsRag.summary}
                    items={sportsRag.results.map((p) => ({ name: p.title }))}
                    onPick={(i) => scrollToCard(`disc-card-${i + 1}`, i, false)}
                  />
                </div>
              )}

              {sportsView === 'ir' && sportsLoading ? (
                <div className="loading">
                  <div className="loading__spinner" />
                  <p>
                    Searching discussions index…
                    <br />
                    <small>first run may take a few seconds</small>
                  </p>
                </div>
              ) : sportsView === 'rag' && sportsRag.loading ? (
                <div className="loading">
                  <div className="loading__spinner" />
                  <p>Refining query &amp; reranking with LLM…</p>
                </div>
              ) : sportsView === 'ir' ? (
                <div className="results__list">
                  {sportsResults.map((p, i) => (
                    <DiscussionCard
                      key={`${p.id}-${i}`}
                      post={p}
                      rank={i + 1}
                      isTop={i === 0}
                    />
                  ))}
                  {sportsResults.length === 0 && (
                    sportsSearchTerm.trim() ? (
                      <div className="empty">
                        <div className="empty__icon">◎</div>
                        <p>No results. Try a different query or sport filter.</p>
                      </div>
                    ) : welcomeCard
                  )}
                </div>
              ) : (
                <div className="results__list">
                  {sportsRag.results.map((p, i) => (
                    <DiscussionCard
                      key={`rag-${p.id}-${i}`}
                      post={p}
                      rank={i + 1}
                      isTop={i === 0}
                    />
                  ))}
                  {sportsRag.results.length === 0 && !sportsRag.error && (
                    sportsSearchTerm.trim() ? (
                      <div className="empty">
                        <div className="empty__icon">◎</div>
                        <p>Run a search to see IR + RAG results.</p>
                      </div>
                    ) : welcomeCard
                  )}
                </div>
              )}
            </section>
          </div>
        )}
      </main>

      {planModalOpen && (
        <div className="viz-modal-overlay" onClick={() => setPlanModalOpen(false)}>
          <div className="viz-modal viz-modal--plan" onClick={(e) => e.stopPropagation()}>
            <div className="viz-modal__head">
              <span className="viz-modal__label">
                SESSION PLAN — {selectedExercise?.name ?? 'EXERCISE'}
              </span>
              <button className="viz-modal__close" onClick={() => setPlanModalOpen(false)}>✕</button>
            </div>
            <div className="viz-modal__body">
              {planState.error && <p className="plan-panel__error">{planState.error}</p>}
              {planState.text && (
                <div className="plan-panel__text">
                  <ReactMarkdown>{planState.text}</ReactMarkdown>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {vizModal && (
        <div className="viz-modal-overlay" onClick={() => setVizModal(null)}>
          <div className={`viz-modal viz-modal--${vizModal}`} onClick={(e) => e.stopPropagation()}>
            <div className="viz-modal__head">
              <span className="viz-modal__label">
                {vizModal === 'map' ? 'MUSCLE MAP' : vizModal === 'network' ? 'MUSCLE NETWORK' : 'MUSCLE COVERAGE'}
              </span>
              <button className="viz-modal__close" onClick={() => setVizModal(null)}>✕</button>
            </div>
            <div className="viz-modal__body">
              {vizModal === 'map' && (
                <MuscleMap
                  primaryMuscles={selectedExercise?.primaryMuscles ?? []}
                  secondaryMuscles={selectedExercise?.secondaryMuscles ?? []}
                />
              )}
              {vizModal === 'network' && (
                <MuscleGraph
                  queryMuscles={queryMuscles}
                  exercises={exerciseView === 'rag' && exerciseRag.results.length > 0 ? exerciseRag.results : exercises}
                />
              )}
              {vizModal === 'radar' && (
                <MuscleRadar
                  queryMuscles={queryMuscles}
                  exercises={exerciseView === 'rag' && exerciseRag.results.length > 0 ? exerciseRag.results : exercises}
                />
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
