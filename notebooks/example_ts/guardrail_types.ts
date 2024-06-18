
export interface GuardrailApiResponse {
    metrics: Metric[];
    validation_results: ValidationResults;
    perf_info: any;
    action: Action;
    score_perf_info: any;
    scores: Score[];
    metadata: Metadata;
}

export interface Action {
    block_message: string;
    action_type: string;
    is_action_block: boolean;
}

export interface Metric {
    [key: string]: any;
}

export interface Score {
    [key: string]: number | null;
}

export interface ValidationResults {
    report: ValidationReportEntry[]
}

export interface ValidationReportEntry {
  id: string;
  metric: string;
  details: string;
  value: number;
  upper_threshold: number | null;
  lower_threshold: number | null;
  allowed_values: any[] | null;
  disallowed_values: any[] | null;
  must_be_none: any | null;
  must_be_non_none: any | null;
  failure_level: string;
}

export interface Metadata {
    policy_id: string;
    policy_version: number;
    container_version: string;
    policy_last_updated_ms: number;
}
