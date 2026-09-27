'read completed batch reports and write an advisory comparison artifact'
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import NormalDist

ARMS = ('rescue', 'fusion', 'prior', 'strength')
METRICS = ('macro_f05', 'pair_precision', 'pair_recall')


def adjusted_interval(gain, standard_error, family_size):
    if family_size < 1:
        raise ValueError('Family size must be positive')
    if gain is None or standard_error is None:
        return None
    if not math.isfinite(gain) or not math.isfinite(standard_error) or standard_error < 0:
        return None
    critical = NormalDist().inv_cdf(1-.05/(2*family_size))
    return [gain-critical*standard_error, gain+critical*standard_error]


def metrics(value):
    if not isinstance(value, dict):
        return None
    value = value.get('overall', value)
    return {key: value.get(key) for key in METRICS} if any(key in value for key in METRICS) else None


def failed_guards(value, prefix=''):
    'retain existing reported failures, without reinterpreting their thresholds'
    failures = []
    if not isinstance(value, dict):
        return failures
    for key, item in value.items():
        path = f'{prefix}.{key}' if prefix else key
        if key == 'checks' and isinstance(item, dict):
            failures.extend(f'{path}.{name}' for name, passed in item.items() if passed is False)
        elif isinstance(item, dict):
            failures.extend(failed_guards(item, path))
    return failures


def review_arm(root, arm, family_size):
    path = root/arm/'report.json'
    if not path.exists():
        return {'arm': arm, 'status': 'pending', 'report_path': str(path), 'review_eligible': False}
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return {'arm': arm, 'status': 'unreadable', 'report_path': str(path),
                'error': str(error), 'review_eligible': False}
    check = report.get('check', {})
    paired = check.get('paired', check)
    before = metrics(paired.get('incumbent', check.get('before')))
    after = metrics(paired.get('candidate', check.get('after')))
    delta = {key: after[key]-before[key] if after[key] is not None and before[key] is not None else None
             for key in METRICS} if before is not None and after is not None else None
    interval = adjusted_interval(paired.get('paired_gain'), paired.get('paired_standard_error'), family_size)
    locked = report.get('locked_candidate', report.get('selected'))
    policy = locked.get('policy') if isinstance(locked, dict) else None
    policy_path = path.with_name('locked_policy.json')
    if policy is None and policy_path.exists():
        policy = json.loads(policy_path.read_text())
    promoted = report.get('promoted') is True
    changed = report.get('submission_changed') is True
    res = {'arm': arm, 'status': 'completed', 'report_path': str(path),
        'promoted': promoted, 'submission_changed': changed,
        'matching_sha256': report.get('matching_sha256'), 'changes': report.get('changes'),
        'baseline_tune': metrics(report.get('incumbent_tune_replay', report.get('baseline_tune_replay'))),
        'baseline_check': before, 'candidate_check': after, 'check_deltas': delta,
        'candidate_all_tune_descriptive': metrics(report.get('candidate_all_previous_tune_descriptive',
                                                             report.get('candidate_previous_tune_descriptive'))),
        'selected_model': locked, 'selected_policy': policy,
        'existing_failed_guards': failed_guards({key: report[key] for key in
            ('check', 'calibration_check', 'density_replay_check', 'family_selection') if key in report}),
        'paired_gain': paired.get('paired_gain'), 'paired_standard_error': paired.get('paired_standard_error'),
        'saved_family_size': paired.get('family_size'), 'review_family_size': family_size,
        'review_adjusted_interval95': interval,
        'review_positive_lower_bound': bool(interval is not None and interval[0] > 0),
        'review_eligible': bool(promoted and changed and interval is not None and interval[0] > 0),
        'submission_recommendation_from_job': report.get('submission_recommendation'),
        'reported_limitations': report.get('limitations', [])}
    if 'trials' in report:
        res['search_trials'] = [{'name': row.get('name'), 'eligible': row.get('eligible'),
            'search': metrics(row.get('search')), 'gain': row.get('gain'),
            'failed_guards': [name for name, passed in row.get('search_checks', {}).items() if passed is False]}
            for row in report['trials']]
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('artifacts/competition_runs'))
    parser.add_argument('--family-size', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('artifacts/competition_runs/comparison.json'))
    args = parser.parse_args()
    if args.family_size < 1:
        parser.error('--family-size must be positive')
    comparison = {'reviewed_at_utc': datetime.now(timezone.utc).isoformat(),
        'family_size': args.family_size, 'confidence_alpha': .05,
        'scope': 'Advisory review only; existing results, thresholds and promotion flags are never changed.',
        'limitations': ['Historical upstream training/model selection exposed evaluation owners; these are not pristine pipeline holdouts.',
            'Shared target competition and repeated reference evidence create owner dependence; normal intervals are approximate.',
            'Bonferroni adjustment covers the declared batch arms, not all historical adaptive experimentation.',
            'All-tune candidate metrics are descriptive. No hidden-test or leaderboard score is estimated.'],
        'arms': [review_arm(args.results_dir, arm, args.family_size) for arm in ARMS]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'arms': [{'arm': item['arm'], 'status': item['status'],
        'review_eligible': item['review_eligible']} for item in comparison['arms']]}))


if __name__ == '__main__':
    main()
