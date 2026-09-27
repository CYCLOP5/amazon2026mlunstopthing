'frozen-owner, bounded label-shift calibration experiment with fail-closed export'
import gc
import json
from pathlib import Path
import shutil
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from er.stack import decode
from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import export
from final_repair.pipeline import changes
from latest_fusion.pipeline import log, logit, partition, sha256, sigmoid, write_json
from latest_fusion.structured_pipeline import INCUMBENT_SHA, METHOD, replay, assert_same_pool, assert_same_pairs
from latest_fusion.structured_protocol import split_references, WARNING
from latest_fusion.tune_pipeline import anchor_metrics
from latest_fusion.tuning import fit_population, split_search_check

KEYS = ['qid', 'tid']
WEIGHTS = (.25, .5, 1.)


def prepare_output(output, sources):
    dst = Path(output).resolve()
    for src in sources:
        src = Path(src).resolve()
        if dst == src or src in dst.parents or dst in src.parents:
            raise ValueError('Output overlaps an input directory')
    if dst.exists() and any(dst.iterdir()):
        raise ValueError('Output directory must be empty')
    dst.mkdir(parents=True, exist_ok=True)


def winner_metadata(top, frame, n_s2):
    'join metadata after global competition, preserving every frozen owner'
    columns = [c for c in ('own', 'seg', 'fold', 'target_address_empty', 'is_s3') if c in frame and c not in top]
    res = top.join(frame.select(*KEYS, *columns), on=KEYS, how='left', maintain_order='left') if columns else top
    if 'is_s3' not in res:
        res = res.with_columns((pl.col('tid') >= n_s2).cast(pl.UInt8).alias('is_s3'))
    if 'seg' not in res or 'target_address_empty' not in res:
        raise ValueError('Winner calibration metadata is incomplete')
    if res.height != top.height or res['tid'].n_unique() != res.height:
        raise ValueError('Winner metadata changed global ownership')
    return res


def source_population(winners):
    if not {'fold', 'qid', 'y', 'own'} <= set(winners.columns):
        raise ValueError('Labeled source metadata missing')
    mask = (winners['fold'].to_numpy() == 0) & (partition(winners['qid'].to_numpy()) == 1)
    return winners.filter(pl.Series(mask)).filter(pl.col('co').is_in(['us', 'india']))


def unlabeled(winners):
    return winners.drop([c for c in ('y', 'own', 'fold') if c in winners])


def blend(winners, posterior, weight):
    p = np.asarray(posterior, dtype=float)
    if p.shape != (winners.height,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid calibrated posterior')
    return winners.with_columns(pl.Series('p', sigmoid((1-weight)*logit(winners['p'].to_numpy()) + weight*logit(p))))


def frozen_accept(winners, policy, baseline):
    'threshold known-country frozen winners and retain authoritative france'
    known = winners.filter(pl.col('co').is_in(['us', 'india']) & (pl.col('p') >= policy['threshold']))
    cols = ['qid', 'tid', 'p'] + (['y'] if 'y' in baseline else [])
    protected = baseline.join(winners.filter(pl.col('co').is_in(['us', 'india'])).select('qid').unique(), on='qid', how='anti')
    accepted = pl.concat([known.select(cols), protected.select(cols)], how='vertical_relaxed')
    if accepted['tid'].n_unique() != accepted.height:
        raise ValueError('Frozen policy assigned multiple owners')
    if known.select(KEYS).join(winners.select(KEYS), on=KEYS, how='anti').height:
        raise ValueError('Frozen policy changed a winner owner')
    return accepted


def guard(candidate, baseline, anchors, min_gain=0., require_country_f=False):
    new, old = anchor_metrics(candidate, anchors), anchor_metrics(baseline, anchors)
    checks = {'strict_macro_gain': new['overall']['macro_f05'] - old['overall']['macro_f05'] > min_gain}
    for region in ['overall', 'us', 'india']:
        a = new['overall'] if region == 'overall' else new['country'].get(region)
        b = old['overall'] if region == 'overall' else old['country'].get(region)
        if a is None or b is None:
            raise ValueError('Evaluation requires both known countries')
        metrics = ['pair_precision', 'pair_recall'] + (['macro_f05'] if require_country_f else [])
        for metric in metrics:
            checks[region+'_'+metric] = a[metric] >= b[metric] - 1e-12
    return {'passed': all(checks.values()), 'checks': checks, 'candidate': new, 'baseline': old}


def constrained_threshold(scored, baseline, anchors, baseline_threshold):
    'exact probability-event search; no check outcomes enter threshold selection'
    old = anchor_metrics(baseline, anchors)
    rows = scored.select('qid', 'p', 'y').join(anchors, on='qid', how='inner').sort(['qid', 'p'], descending=[False, True])
    rows = rows.with_columns(pl.int_range(1, pl.len()+1).over('qid').alias('_n'),
        pl.col('y').cast(pl.Int64).cum_sum().over('qid').alias('_tp'))
    after = pl.when(pl.col('deg') > 0).then(1.25*pl.col('_tp')/(pl.col('_n')+.25*pl.col('deg'))).otherwise(0.)
    rows = rows.with_columns(after.alias('_after')).with_columns(
        pl.col('_after').shift(1).over('qid').fill_null(pl.when(pl.col('deg') == 0).then(1.).otherwise(0.)).alias('_before'))
    expressions = []
    for country in ('us', 'india'):
        mask = pl.col('co') == country
        expressions.extend([pl.when(mask).then(pl.col('_after')-pl.col('_before')).otherwise(0.).sum().alias(country+'_delta'),
            mask.cast(pl.Int64).sum().alias(country+'_n'),
            pl.when(mask).then(pl.col('y')).otherwise(0).sum().alias(country+'_tp')])
    events = rows.group_by('p').agg(expressions)

    cuts = [float(baseline_threshold), float(np.nextafter(1., 2.))]
    extra = pl.DataFrame({'p': cuts, **{c: [0,0] for c in events.columns if c != 'p'}}).cast(events.schema)
    events = pl.concat([events, extra]).group_by('p').agg(pl.all().exclude('p').sum()).sort('p', descending=True)
    metrics = {}
    total_f = np.zeros(events.height)
    total_n = np.zeros(events.height)
    total_tp = np.zeros(events.height)
    truth = 0
    feasible = np.ones(events.height, dtype=bool)
    for country in ('us', 'india'):
        a = anchors.filter(pl.col('co') == country)
        if not a.height:
            raise ValueError('Threshold selection requires both known countries')
        fsum = int((a['deg'] == 0).sum()) + np.cumsum(events[country+'_delta'].to_numpy())
        n = np.cumsum(events[country+'_n'].to_numpy())
        tp = np.cumsum(events[country+'_tp'].to_numpy())
        deg = int(a['deg'].sum())
        f, precision, recall = fsum/a.height, tp/np.maximum(n, 1), tp/max(deg, 1)
        metrics[country] = {'macro_f05': f, 'pair_precision': precision, 'pair_recall': recall}
        for name, values in metrics[country].items():
            feasible &= values >= old['country'][country][name]-1e-12
        total_f += fsum
        total_n += n
        total_tp += tp
        truth += deg
    overall = {'macro_f05': total_f/anchors.height,
        'pair_precision': total_tp/np.maximum(total_n, 1), 'pair_recall': total_tp/max(truth, 1)}
    feasible &= overall['macro_f05'] > old['overall']['macro_f05'] + 1e-12
    for name in ('pair_precision', 'pair_recall'):
        feasible &= overall[name] >= old['overall'][name]-1e-12
    indices = np.flatnonzero(feasible)
    sel = int(indices[np.argmax(overall['macro_f05'][indices])]) if len(indices) else int(np.argmax(overall['macro_f05']))
    return float(events['p'][sel]), {'feasible_cut_found': bool(len(indices)),
        'distinct_cuts': events.height, 'baseline_threshold_included': float(baseline_threshold),
        'best_unconstrained_macro_f05': float(np.max(overall['macro_f05']))}


def lock_policy(winners, posterior, baseline, search, baseline_threshold=.7842838168144226):
    cands, eligible = [], []
    for weight in WEIGHTS:
        scored = blend(winners, posterior, weight)
        cut, threshold_diagnostics = constrained_threshold(scored, baseline, search, baseline_threshold)
        policy = {'rule': 'top1_threshold', 'threshold': cut, 'weight': weight}
        accepted = frozen_accept(scored, policy, baseline)
        review = guard(accepted, baseline, search, require_country_f=True)
        record = {'policy': policy, 'search': review, 'threshold_search': threshold_diagnostics}
        cands.append(record)
        if review['passed']:
            eligible.append(record)
    locked = max(eligible, key=lambda x: x['search']['candidate']['overall']['macro_f05']) if eligible else None
    return locked, cands


def run(data, train, test, incumbent, output):
    from latest_fusion import mixture_calibration as calibration
    from latest_fusion.drop_review import effects
    started = time.monotonic()
    data, train, test, incumbent, output = map(Path, (data, train, test, incumbent, output))
    prepare_output(output, (data, train.parent, test.parent, incumbent))
    prev = json.loads((incumbent/'report.json').read_text())
    source_tsv = incumbent/'output'/'matching_results.tsv'
    if prev['selected'] != METHOD or prev['matching_sha256'] != INCUMBENT_SHA or sha256(source_tsv) != INCUMBENT_SHA:
        raise ValueError('Incumbent provenance mismatch')
    ff = json.loads(train.with_name('features-train.json').read_text())
    if ff != json.loads(test.with_name('features-test.json').read_text()):
        raise ValueError('Feature schemas differ')
    model = lgb.Booster(model_file=str(incumbent/'residual_model.txt'))
    refs = load_refs(str(data), 'train')
    frame = pl.read_parquet(train)
    assert_same_pool(frame, incumbent/'validation_predictions.parquet')
    frame, top, base = replay(frame, model, ff, prev, 'train')
    n_s2 = pl.scan_parquet(data/'train'/'s2.parquet').select(pl.len()).collect().item()
    winners = winner_metadata(top, frame, n_s2)
    _, tuneids = fit_population(refs)
    search, check = split_search_check(refs, tuneids)
    tune = refs.filter(pl.col('rid').is_in(pl.Series(tuneids).implode())).select(pl.col('rid').alias('qid'), 'deg', 'co')
    original = decode.score(base, tune)
    if abs(original['macro_f05'] - prev['methods'][METHOD]['tune']['macro_f05']) > 1e-10:
        raise ValueError('Original tune replay differs')
    structured = split_references(refs)[2].select(pl.col('rid').alias('qid'), 'deg', 'co')
    src = source_population(winners)
    log(f'mixture: fitting densities on {src.height:,} labeled source winners')
    bundle = calibration.fit_source(src, score_col='raw', optional_strata=('is_s3', 'target_address_empty'))
    write_json(output/'calibration_source.json', bundle)
    recipe_train = calibration.adapt(bundle, unlabeled(winners))
    posterior = calibration.predict(bundle, unlabeled(winners), recipe=recipe_train)
    empirical = calibration.predict(bundle, unlabeled(winners))
    cands = []
    for family, values in [('em', posterior), ('empirical', empirical)]:
        _, family_candidates = lock_policy(winners, values, base, search, prev['methods'][METHOD]['tune']['threshold'])
        for record in family_candidates:
            record['family'] = family
            score = record['search']['candidate']['overall']['macro_f05']
            gain = score - record['search']['baseline']['overall']['macro_f05']
            log(f'mixture search {family} weight={record["policy"]["weight"]}: overall macro F0.5={score:.9f}, gain={gain:+.9f}, eligible={record["search"]["passed"]}')
        cands.extend(family_candidates)
    eligible = [record for record in cands if record['search']['passed']]
    locked = max(eligible, key=lambda record: record['search']['candidate']['overall']['macro_f05']) if eligible else None

    selection = {'locked': locked, 'candidates': cands, 'historical_upstream_exposure': WARNING,
                 'source_rows': src.height, 'source_partition': 'fold0 partition1', 'test_labels_used': False}
    write_json(output/'selection.json', selection)
    evaluation, promote = {}, False
    if locked is not None:
        policy = locked['policy']
        selected_posterior = posterior if locked['family'] == 'em' else empirical
        treatment = frozen_accept(blend(winners, selected_posterior, policy['weight']), policy, base)
        control = frozen_accept(blend(winners, empirical, policy['weight']), policy, base)
        for name, anchors in [('search', search), ('owner_check', check), ('structured_check', structured), ('original_tune_descriptive', tune)]:
            evaluation[name] = {'treatment': guard(treatment, base, anchors, 1e-5, True),
                'empirical_control_metrics': anchor_metrics(control, anchors),
                'treatment_effects': effects(treatment, base, anchors),
                'empirical_control_effects': effects(control, base, anchors)}
            review = evaluation[name]['treatment']
            for country in ('overall', 'us', 'india'):
                before = review['baseline']['overall'] if country == 'overall' else review['baseline']['country'][country]
                after = review['candidate']['overall'] if country == 'overall' else review['candidate']['country'][country]
                log(f'mixture {name} {country}: F0.5 {before["macro_f05"]:.9f}->{after["macro_f05"]:.9f}, P {before["pair_precision"]:.9f}->{after["pair_precision"]:.9f}, R {before["pair_recall"]:.9f}->{after["pair_recall"]:.9f}; population checks passed={review["passed"]}')
        promote = all(evaluation[name]['treatment']['passed'] for name in ('owner_check', 'structured_check'))
    selection.update({'promoted': promote, 'evaluation': evaluation,
        'em_vs_empirical_probability_changes': int(np.count_nonzero(np.abs(posterior-empirical) > 1e-7)),
        'attribution': 'EM deployment-prior adaptation' if locked is not None and locked['family'] == 'em' and np.any(np.abs(posterior-empirical) > 1e-7) else 'empirical winner calibration or baseline fallback'})
    write_json(output/'selection.json', selection)
    provenance = {'residual_model_sha256': sha256(incumbent/'residual_model.txt'),
        'features_sha256': sha256(train.with_name('features-train.json')), 'method': METHOD,
        'baseline_threshold': prev['methods'][METHOD]['tune']['threshold'], 'new_neural_training': False}
    del frame, top, src, refs, base, winners, posterior, empirical
    gc.collect()
    log(f'mixture: policy locked; promotion={promote}; replaying test')
    rt, tt = load_refs(str(data), 'test'), load_targets(str(data), 'test')
    frame = pl.read_parquet(test)
    assert_same_pool(frame, incumbent/'test_predictions.parquet')
    frame, top, replay_test = replay(frame, model, ff, prev, 'test')
    base = pl.read_parquet(incumbent/'accepted_test.parquet').select('qid', 'tid', 'p')
    known = rt.filter(pl.col('co').is_in(['us', 'india'])).select(pl.col('rid').alias('qid'))
    assert_same_pairs(replay_test.join(known, on='qid', how='semi'), base.join(known, on='qid', how='semi'), 'Known-country test')
    n_s2 = pl.scan_parquet(data/'test'/'s2.parquet').select(pl.len()).collect().item()
    winners = winner_metadata(top, frame, n_s2)
    test_unlabeled = unlabeled(winners)
    recipe_test = calibration.adapt(bundle, test_unlabeled)
    write_json(output/'mixture_recipes.json', {'train': recipe_train, 'test': recipe_test})
    accepted = base
    if promote:
        policy = locked['policy']
        p = calibration.predict(bundle, test_unlabeled, recipe=recipe_test if locked['family'] == 'em' else None)
        accepted = frozen_accept(blend(winners, p, policy['weight']), policy, base)
    french = rt.filter(pl.col('co') == 'france').select(pl.col('rid').alias('qid'))
    assert_same_pairs(accepted.join(french, on='qid', how='semi'), base.join(french, on='qid', how='semi'), 'Protected France')
    if accepted['tid'].n_unique() != accepted.height or accepted.select(KEYS).join(frame.select(KEYS), on=KEYS, how='anti').height:
        raise ValueError('Export pair ownership/pool violation')
    dest = output/'output'
    dest.mkdir(exist_ok=True)
    export(frame, 'baseline_p', 'mixture_frozen_owners', 0., rt, tt, str(dest), accepted=accepted)
    if not promote:
        shutil.copyfile(source_tsv, dest/'matching_results.tsv')
        if sha256(dest/'matching_results.tsv') != INCUMBENT_SHA:
            raise ValueError('Baseline fallback is not byte-identical')
    from validate_outputs import validate
    val = validate(data, dest)
    accepted.write_parquet(output/'accepted_test.parquet')
    report = {'incumbent_sha256': INCUMBENT_SHA, 'incumbent_leaderboard': .98805,
        'baseline_original_tune': original, 'selection': selection, 'provenance': provenance,
        'changes': changes(accepted, base, rt), 'validation': val,
        'matching_sha256': sha256(dest/'matching_results.tsv'), 'promoted': promote,
        'submission_status': 'candidate_requires_review' if promote else 'exact_baseline_no_demonstrated_improvement',
        'elapsed_minutes': (time.monotonic()-started)/60}
    write_json(output/'report.json', report)
    log(f'mixture finished: promotion={promote}; validator PASS')
    return report
