'fold-1 name-grouped isolation for new heads over prepared candidate features'
import hashlib

import numpy as np
import polars as pl

from .directional_swaps import _records

SEED = 2709
WARNING = ('Historical upstream models and prepared features may already have '
           'exposed these records. This is new-head isolation, not a pristine '
           'end-to-end holdout.')


def _unique(frame, columns, name):
    if frame.select(columns).null_count().sum_horizontal().sum():
        raise ValueError(f'{name} has null keys')
    if frame.select(columns).unique().height != frame.height:
        raise ValueError(f'{name} contains duplicate keys')


def _name_bucket(value):
    digest = hashlib.sha256(f'{SEED}:{value}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % 10


def _orphan_sample(tids, fraction):
    'version-independent target hash; every row of a target samples together'
    x = np.asarray(tids, dtype=np.uint64) ^ np.uint64(SEED)
    with np.errstate(over='ignore'):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x.astype(np.float64) / float(2**64) < fraction


def split_references(refs):
    'Preserve original schema while assigning fold-1 country/name groups'
    _unique(refs, ['rid'], 'References')
    if refs['co'].null_count() or refs['fold'].null_count():
        raise ValueError('References require nonnull country and fold')
    eligible = refs.filter(pl.col('fold') == 1)
    normalized = _records(eligible, 'rid').select('rid', 'co', '_name')
    groups = normalized.select('co', '_name').unique().with_columns(
        pl.struct('co', '_name').struct.json_encode().map_elements(
            _name_bucket, return_dtype=pl.UInt8).alias('_bucket'))
    normalized = normalized.join(groups, on=['co', '_name'], how='left')
    definitions = {'fit': (0, 4), 'calibration': (5, 6), 'check': (7, 9)}
    parts, counts = [], {}
    for name, (low, high) in definitions.items():
        ids = normalized.filter(pl.col('_bucket').is_between(low, high)).select('rid')
        part = eligible.join(ids, on='rid', how='semi', maintain_order='left')
        parts.append(part)
        counts[name] = {'references': part.height,
                        'name_groups': groups.filter(pl.col('_bucket').is_between(low, high)).height}
    return (*parts, {'seed': SEED, 'modulus': 10, 'partition_counts': counts,
                     'fold1_references': eligible.height, 'excluded_other_fold_references': refs.height-eligible.height,
                     'assignment': 'SHA-256(seed2709:JSON(country, normalized full name)), first 64 bits modulo10; fit0..4/cal5..6/check7..9',
                     'historical_upstream_exposure_warning': WARNING})


def _ids(refs, column='qid', dtype=None):
    expression = pl.col('rid')
    if dtype is not None:
        expression = expression.cast(dtype)
    return refs.select(expression.alias(column))


def _class_counts(frame):
    positive = frame.filter(pl.col('y') == 1).height
    orphan = frame.filter(pl.col('own') < 0).height
    return {'pairs': frame.height, 'targets': frame['tid'].n_unique(), 'positive': positive,
            'negative': frame.height-positive, 'wrong_owner_negative': frame.height-positive-orphan,
            'orphan_negative': orphan, 'orphan_targets': frame.filter(pl.col('own') < 0)['tid'].n_unique()}


def _validate(frame, refs, parts):
    _unique(refs, ['rid'], 'References')
    _unique(frame, ['qid', 'tid'], 'Candidate pairs')
    if frame['own'].null_count() or frame['y'].null_count():
        raise ValueError('Candidate labels and ownership must be nonnull')
    if frame.filter(~pl.col('y').is_in([0, 1]) |
                    (pl.col('y') != (pl.col('qid').cast(pl.Int64) == pl.col('own')).cast(pl.Int8))).height:
        raise ValueError('Candidate labels disagree with target ownership')
    if frame.group_by('tid').agg(pl.col('own').n_unique().alias('_owners')).filter(pl.col('_owners') != 1).height:
        raise ValueError('Inconsistent target ownership')
    if frame.select('qid').unique().join(_ids(refs, dtype=frame.schema['qid']), on='qid', how='anti').height:
        raise ValueError('Unknown candidate reference')
    owned = frame.filter(pl.col('own') >= 0).select('own').unique()
    if owned.join(_ids(refs, 'own', frame.schema['own']), on='own', how='anti').height:
        raise ValueError('Unknown true owner')
    used = refs.select('rid').head(0)
    for part in parts:
        _unique(part, ['rid'], 'Partition references')
        if part.select('rid').join(refs.filter(pl.col('fold') == 1).select('rid'), on='rid', how='anti').height:
            raise ValueError('Partition contains references outside fold1')
        if part.select('rid').join(used, on='rid').height:
            raise ValueError('Reference partitions overlap')
        used = pl.concat([used, part.select('rid')])


def training_pool(frame, refs, fitrefs, calrefs, checkrefs):
    'select complete fit-owned targets and deterministic orphan target groups'
    _validate(frame, refs, (fitrefs, calrefs, checkrefs))
    protected = pl.concat([_ids(calrefs, dtype=frame.schema['qid']), _ids(checkrefs, dtype=frame.schema['qid'])])
    held_targets = frame.join(protected, on='qid', how='semi').select('tid').unique()
    available = frame.join(held_targets, on='tid', how='anti', maintain_order='left')
    fraction = fitrefs.height / refs.height if refs.height else 0.
    fit_owned = available.filter(pl.col('own') >= 0).join(_ids(fitrefs, 'own', frame.schema['own']), on='own', how='semi', maintain_order='left')
    orphan = available.filter(pl.col('own') < 0)
    orphan = orphan.filter(pl.Series(_orphan_sample(orphan['tid'].to_numpy(), fraction)))
    allowed = pl.concat([fit_owned.select('qid', 'tid'), orphan.select('qid', 'tid')])
    fitting = frame.join(allowed, on=['qid', 'tid'], how='semi', maintain_order='left')
    target_overlap = fitting.select('tid').unique().join(held_targets, on='tid').height
    candidate_overlap = fitting.join(protected, on='qid', how='semi').height
    true_owner_overlap = fitting.join(protected.select(pl.col('qid').cast(frame.schema['own']).alias('own')), on='own', how='semi').height
    if target_overlap or candidate_overlap or true_owner_overlap:
        raise ValueError('Held-out target/owner isolation failed')
    represented = fitting.filter(pl.col('own') >= 0).select(pl.col('own').cast(refs.schema['rid']).alias('rid')).unique()
    report = {'orphan_fraction': fraction, 'orphan_sampling': 'SplitMix64(tid xor2709) / 2**64 < len(fitrefs)/len(fullrefs)',
              'protected_references': protected.height, 'protected_targets': held_targets.height,
              'excluded_pairs_touching_protected_targets': frame.height-available.height,
              'target_overlap': target_overlap, 'held_candidate_qids_in_fit': candidate_overlap,
              'held_true_owner_ids_in_fit': true_owner_overlap,
              'fit_references_without_owned_training_targets': fitrefs.select('rid').join(represented, on='rid', how='anti').height,
              'fit_reference_scope_note': 'Assigned fit references do not guarantee fitting rows or action-model examples; callers must check the actual training population.',
              'class_counts': {'fit': _class_counts(fitting)},
              'historical_upstream_exposure_warning': WARNING}
    return fitting, report


def prepare_protocol(refs, frame):
    'return fitrefs, clean calrefs, checkrefs, fitframe, report'
    fitrefs, original_calrefs, checkrefs, split_report = split_references(refs)
    fitting, fit_report = training_pool(frame, refs, fitrefs, original_calrefs, checkrefs)
    check_targets = frame.join(_ids(checkrefs, dtype=frame.schema['qid']), on='qid', how='semi').select('tid').unique()
    original_cal_rows = frame.join(_ids(original_calrefs, dtype=frame.schema['qid']), on='qid', how='semi')
    dropped = original_cal_rows.join(check_targets, on='tid', how='semi').select('qid').unique()
    clean_calrefs = original_calrefs.join(dropped.select(pl.col('qid').cast(refs.schema['rid']).alias('rid')), on='rid', how='anti', maintain_order='left')
    cal = frame.join(_ids(clean_calrefs, dtype=frame.schema['qid']), on='qid', how='semi')
    check = frame.join(_ids(checkrefs, dtype=frame.schema['qid']), on='qid', how='semi')
    cal_check_overlap = cal.select('tid').unique().join(check_targets, on='tid').height
    if cal_check_overlap:
        raise ValueError('Calibration/check target isolation failed')
    report = {**split_report, **fit_report,
              'original_calibration_references': original_calrefs.height,
              'clean_calibration_references': clean_calrefs.height,
              'calibration_owners_dropped_for_check_target_overlap': dropped.height,
              'calibration_check_target_overlap': cal_check_overlap,
              'class_counts': {**fit_report['class_counts'], 'calibration': _class_counts(cal), 'check': _class_counts(check)}}
    return fitrefs, clean_calrefs, checkrefs, fitting, report
