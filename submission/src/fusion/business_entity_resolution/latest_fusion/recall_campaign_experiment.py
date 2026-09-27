'shared additions-only evaluation for eight predeclared recall hypotheses'
import importlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from . import additions_experiment as actions
from . import contrastive_experiment as validation
from .pipeline import log
from .structured_protocol import WARNING

METHOD_MODULES = {
    **dict.fromkeys(('honest_leaf', 'crossview_consensus', 'hard_negative'), 'recall_campaign_a'),
    **dict.fromkeys(('sibling_bridge', 'orphan_factor', 'source_corroboration'), 'recall_campaign_b'),
    **dict.fromkeys(('monotone_verifier', 'country_consensus'), 'recall_campaign_c'),
}


def _module(method):
    if method not in METHOD_MODULES:
        raise ValueError('Unknown recall campaign method: '+str(method))
    return importlib.import_module('latest_fusion.'+METHOD_MODULES[method])


def score_candidates(module, model, frame, baseline, **kwargs):


    rows = frame.join(baseline.select('tid'), on='tid', how='anti', maintain_order='left')
    prob = np.asarray(module.predict(model, rows, baseline=baseline, **kwargs), dtype=np.float32)
    if prob.shape != (rows.height,) or not np.isfinite(prob).all() or \
            (prob < 0).any() or (prob > 1).any():
        raise ValueError('Recall verifier returned invalid probability array')
    columns = list(dict.fromkeys(['qid', 'tid'] + [c for c in ('co', 'y', 'own', 'baseline_p') if c in rows]))
    scored = rows.select(columns).with_columns(pl.Series('p_alias', prob),
        pl.Series('p_decoy', 1.-prob), pl.lit(0.).alias('p_other'),
        pl.Series('p', prob))
    if 'baseline_raw' in rows:
        scored = scored.with_columns(rows['baseline_raw'].alias('raw'))
    if set(baseline.columns)-set(scored.columns):
        raise ValueError('Scored candidates cannot preserve baseline schema: '+str(baseline.columns))
    return scored


def run(train_frame, refs, targets, fit_frame, cal_refs, check_refs, baseline_train,
        core_features, output_dir, fit_refs=None, s2_count=None, method=None, **unused):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    module = _module(method)
    validation._validate_fit(train_frame, fit_frame, cal_refs, check_refs, fit_refs)
    ff = validation._features(core_features)
    fit = fit_frame.filter(pl.col('co').is_in(actions.COUNTRIES))
    log(f'recall campaign {method}: fitting new verifier on {fit.height:,} isolated pairs')
    model = module.fit(method, fit, ff, output/'models', baseline=baseline_train,
                       s2_count=s2_count, refs=refs, targets=targets)
    scored = score_candidates(module, model, train_frame, baseline_train,
                              s2_count=s2_count, refs=refs, targets=targets)
    ranked = actions.rank_candidates(scored, baseline_train)
    cal, check = actions._country(cal_refs), actions._country(check_refs)
    rule, calibration = actions.select_policy(ranked, baseline_train, cal)

    (output/'locked_rule.json').write_text(json.dumps({'method': method, 'rule': rule,
        'calibration': calibration}, indent=2, allow_nan=False))
    evidence = None
    promoted = False
    if rule['enabled']:
        adds = actions.propose_additions(ranked, baseline_train, check, rule, ranked=True)
        evidence = actions._guard(actions.evaluate_additions(baseline_train, adds, check), 25, check=True)
        adds.write_parquet(output/'check_additions.parquet')
        promoted = evidence['passed']
        log(f'recall campaign {method}: check TPadded={evidence["TPadded"]}, FPadded={evidence["FPadded"]}, '
            f'F0.5 {evidence["before"]["macro_f05"]:.9f} -> {evidence["after"]["macro_f05"]:.9f}; promoted={promoted}')
    else:
        log(f'recall campaign {method}: no eligible calibration rule; preserve incumbent')
    report = {'method': method, 'promoted': promoted, 'model': model['report'], 'rule': rule,
        'calibration': calibration, 'check': evidence, 'no_French_labels': True,
        'policy': 'Only globally unoccupied targets; all incumbent matches preserved; France frozen.',
        'promotion': 'At least 25 check true additions, zero false additions, F0.5 gain >1e-5, overall/country P/R/F nonregression.',
        'campaign_methods': list(METHOD_MODULES), 'campaign_size': len(METHOD_MODULES),
        'multiple_comparisons': 'Per-method checks are exploratory across eight predeclared methods; no check-driven retuning or automatic union of candidates.',
        'historical_upstream_exposure_warning': WARNING,
        'candidate_pool': 'Existing full fusion union. These methods recover rejected links; absent true candidates remain a separate ceiling.',
        'scored_unoccupied_pairs': scored.height}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return {'method': method, 'model': model, 'rule': rule, 'promoted': promoted,
            'report': report, 'output_dir': str(output)}


def infer(bundle, test_frame, test_refs, test_targets, baseline_test, s2_count=None, **unused):
    report = {'method': bundle['method'], 'promoted': bundle['promoted'], 'additions': 0,
              'france_policy': 'Exact incumbent decisions', 'accepted_targets': baseline_test.height}
    if not bundle['promoted']:
        return baseline_test.clone(), report
    rows = score_candidates(_module(bundle['method']), bundle['model'], test_frame, baseline_test,
                            s2_count=s2_count, refs=test_refs, targets=test_targets)
    additions = actions.propose_additions(rows, baseline_test, actions._country(test_refs), bundle['rule'])
    accepted = actions.apply_additions(baseline_test, additions)
    additions.write_parquet(Path(bundle['output_dir'])/'test_additions.parquet')
    report.update(additions=additions.height, accepted_targets=accepted.height, rule=bundle['rule'])
    return accepted, report
