// export interface Episode {
//   title: string;
//   descr: string;
//   imdb_rating: number;
// }

export type MatchQuality = 'strong' | 'moderate' | 'weak';

export interface Tag { label: string; score: number }

export interface Exercise {
  name: string;
  score: number;
  match_quality?: MatchQuality;
  tags?: Tag[];
  primaryMuscles: string[];
  secondaryMuscles: string[];
  level: string | null;
  equipment: string | null;
  category: string | null;
  instructions: string[];
}

export type SearchMethod = 'tfidf' | 'svd';

export interface SearchRequest {
  query: string;
  equipment?: string[];
  difficulty?: string;
  injuries?: string[];
  method?: SearchMethod;
}

export interface ProgramScheduleEntry {
  week: number | null;
  day: number | null;
  exercise_name: string;
  sets: number | null;
  reps: number | null;
  rep_type: string | null;
}

export type ScheduleEntry = ProgramScheduleEntry;

export interface Program {
  title: string;
  description: string;
  goal: string[];
  level: string | null;
  program_length_weeks: number | null;
  score: number;
  match_quality?: MatchQuality;
  tags?: Tag[];
  schedule: ProgramScheduleEntry[];
}

export const EQUIPMENT_OPTIONS: string[] = [
  'barbell',
  'dumbbell',
  'body only',
  'cable',
  'machine',
  'kettlebells',
  'bands',
  'medicine ball',
  'exercise ball',
  'foam roll',
  'e-z curl bar',
  'other',
];

export const DIFFICULTY_OPTIONS: ['beginner', 'intermediate', 'expert'] = [
  'beginner',
  'intermediate',
  'expert',
];

export interface FormCue {
  form_cues: string[];
  safety: string;
}

export interface MuscleGraphNode {
  id: string;
  label: string;
  type: 'muscle' | 'exercise';
  isQueryRelevant?: boolean;
}

export interface MuscleGraphEdge {
  source: string;
  target: string;
  type: 'exercise-muscle' | 'muscle-cooccurrence';
}

export interface MuscleGraphData {
  nodes: MuscleGraphNode[];
  edges: MuscleGraphEdge[];
}

export const MUSCLE_OPTIONS: string[] = [
  'abdominals',
  'abductors',
  'adductors',
  'biceps',
  'calves',
  'chest',
  'forearms',
  'glutes',
  'hamstrings',
  'lats',
  'lower back',
  'middle back',
  'neck',
  'quadriceps',
  'shoulders',
  'traps',
  'triceps',
];
