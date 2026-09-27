'conditional winner-score mixture calibration from source labels only'
import numpy as np
from sklearn.isotonic import IsotonicRegression

COUNTRIES = ('india', 'us')
EDGES = np.linspace(-12., 12., 49)
MIN_CLASS = 50
MIN_PARENT = 500
MIN_CHILD = 1000
MIN_TARGET = 500
DENSITY_SHRINK = 50.
PRIOR_SHRINK = .5
PRIOR_RATIO = (.5, 2.)
MAX_TV = .08
MIN_SEPARATION = .05


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.-1e-6)
    return np.log(p) - np.log1p(-p)


def _view(frame, score_col, optional):
    'Explicit projection ensures deployment y/own/fold cannot enter adaptation'
    columns = ['co', 'seg', score_col] + list(optional)
    if any(c not in frame for c in columns):
        raise ValueError('Winner metadata missing columns: ' + str(columns))
    values = {c: frame[c].to_numpy() for c in columns}
    score = np.asarray(values[score_col], dtype=float)
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError('Winner scores must be finite in [0,1]')
    if any(frame[c].null_count() for c in ['co', 'seg'] + list(optional)):
        raise ValueError('Winner strata must be nonnull')
    if 'tid' in frame and frame['tid'].n_unique() != frame.height:
        raise ValueError('Winner table must contain one owner per target')
    return values, np.searchsorted(EDGES, _logit(score), side='right')


def _key(values, i, level, optional):
    parts = [str(values['co'][i])]
    if level != 'country':
        parts.append(str(int(values['seg'][i])))
    if level == 'child':
        parts.extend(str(values[c][i]) for c in optional)
    return level + ':' + '|'.join(parts)


def _groups(values, level, optional):


    groups = {}
    columns = ['co'] + (['seg'] if level != 'country' else [])
    if level == 'child':
        columns += list(optional)
    def visit(depth, ids):
        if depth == len(columns):
            groups[_key(values, int(ids[0]), level, optional)] = ids
            return
        column = columns[depth]
        for value in np.unique(values[column][ids]):
            if column == 'co' and str(value) not in COUNTRIES:
                continue
            part = ids[values[column][ids] == value]
            if len(part):
                visit(depth + 1, part)
    visit(0, np.arange(len(values['co']), dtype=np.int64))
    return groups


def _normalise(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError('Invalid nonnegative probability mass')
    return values / values.sum()


def _posterior(density, prior):
    weighted = np.asarray(prior)[:, None] * np.asarray(density)
    return weighted[0] / np.maximum(weighted.sum(axis=0), 1e-15)


def _isotonic(posterior, weights):
    x = np.arange(len(posterior), dtype=float)
    return IsotonicRegression(out_of_bounds='clip').fit_transform(
        x, np.clip(posterior, 0., 1.), sample_weight=np.asarray(weights) + 1.)


def _separation(density):
    density = np.asarray(density)
    if density.shape[0] == 2:
        return float(.5 * np.abs(density[0] - density[1]).sum())
    singular = np.linalg.svd(density.T, compute_uv=False)
    return float(singular[-1] / max(singular[0], 1e-15))


def _density(counts, parent=None):
    counts = np.asarray(counts, dtype=float)
    if parent is None:
        pooled = _normalise(counts.sum(axis=0) + .5)
        parent = np.repeat(pooled[None, :], counts.shape[0], axis=0)
    regularized = counts + DENSITY_SHRINK * np.asarray(parent)
    return regularized / regularized.sum(axis=1, keepdims=True)


def _cell(counts3, parent, minimum):
    totals = counts3.sum(axis=1)
    if totals.sum() < minimum or min(totals[0], totals[1:].sum()) < MIN_CLASS:
        return None
    three = min(totals) >= MIN_CLASS
    parent_density = None
    if parent and parent['class_order'] == ['alias', 'other_owned', 'decoy'] and three:
        parent_density = np.asarray(parent['densities'])
    if three:
        density = _density(counts3, parent_density)
        three = _separation(density) >= MIN_SEPARATION
    if three:
        counts, classes = counts3, ['alias', 'other_owned', 'decoy']
    else:
        counts = np.stack([counts3[0], counts3[1:].sum(axis=0)])
        classes = ['alias', 'nonalias']
        parent_density = None
        if parent:
            pd = np.asarray(parent['densities'])
            pp = np.asarray(parent['source_prior'])
            parent_density = pd if len(pp) == 2 else np.stack([
                pd[0], (pp[1:, None] * pd[1:]).sum(axis=0) / pp[1:].sum()])
        density = _density(counts, parent_density)
    prior = _normalise(counts.sum(axis=1))
    histogram = counts.sum(axis=0)
    separation = _separation(density)
    return {'class_order': classes, 'source_prior': prior.tolist(), 'densities': density.tolist(),
            'source_histogram': histogram.tolist(), 'source_rows': int(histogram.sum()),
            'source_class_counts': counts3.sum(axis=1).astype(int).tolist(),
            'identifiable': separation >= MIN_SEPARATION, 'separation': separation,
            'control_posterior': _isotonic(_posterior(density, prior), histogram).tolist(),
            'negative_model': 'Separate wrong-owner and decoy densities' if three else
                'Binary fallback: insufficient negative-class support or three-class separation'}


def fit_source(source_winners, score_col='raw', optional_strata=('source', 'target_address_empty')):
    'Fit source histograms on caller-supplied fold0/partition1 winners only'
    optional = [c for c in optional_strata if c in source_winners]
    values, bins = _view(source_winners, score_col, optional)
    if any(str(co) not in COUNTRIES for co in values['co']):
        raise ValueError('Source fitting permits India/US labels only')
    if 'fold' in source_winners:
        if source_winners['fold'].null_count() or not (source_winners['fold'] == 0).all():
            raise ValueError('Source calibration must use fold0 only')
        if 'qid' not in source_winners:
            raise ValueError('Source fold validation requires qid')
        from .calibration import partition
        if not (partition(source_winners['qid'].to_numpy()) == 1).all():
            raise ValueError('Source calibration must use partition1 only')
    if 'y' not in source_winners or 'own' not in source_winners:
        raise ValueError('Source labels y/own are required')
    y, own = source_winners['y'].to_numpy(), source_winners['own'].to_numpy()
    if source_winners['y'].null_count() or source_winners['own'].null_count() or not np.isin(y, [0, 1]).all():
        raise ValueError('Source requires complete binary labels and truth ownership')
    if not np.isfinite(own).all() or not np.equal(own, np.floor(own)).all():
        raise ValueError('Source truth ownership must contain finite integer IDs')
    if 'qid' in source_winners and not np.array_equal(y.astype(bool), source_winners['qid'].to_numpy() == own):
        raise ValueError('Source labels disagree with truth ownership')
    labels = np.where(y == 1, 0, np.where(own >= 0, 1, 2))
    cells, support = {}, {}
    for level in ['country', 'parent'] + (['child'] if optional else []):
        for key, ids in _groups(values, level, optional).items():
            counts = np.zeros((3, len(EDGES) + 1), dtype=float)
            np.add.at(counts, (labels[ids], bins[ids]), 1.)
            first = int(ids[0])
            parent_key = _key(values, first, 'country' if level == 'parent' else 'parent', optional)
            minimum = MIN_CHILD if level == 'child' else MIN_PARENT
            cell = _cell(counts, cells.get(parent_key), minimum)
            support[key] = {'source_rows': int(counts.sum()),
                'source_class_counts': counts.sum(axis=1).astype(int).tolist(),
                'available': cell is not None,
                'reason': None if cell is not None else 'sparse_source_backoff'}
            if cell is not None:
                cells[key] = cell
    centres = np.r_[EDGES[0]-1., (EDGES[:-1]+EDGES[1:])/2., EDGES[-1]+1.]
    return {'version': 1, 'score_col': score_col, 'edges': EDGES.tolist(), 'centres': centres.tolist(),
            'optional_strata': optional, 'cells': cells, 'source_support': support,
            'parameters': {'min_class': MIN_CLASS, 'min_parent': MIN_PARENT, 'min_child': MIN_CHILD,
                'min_target': MIN_TARGET, 'density_shrink': DENSITY_SHRINK, 'prior_shrink': PRIOR_SHRINK,
                'prior_ratio': list(PRIOR_RATIO), 'max_mixture_tv': MAX_TV, 'min_separation': MIN_SEPARATION},
            'fit_scope': 'Caller-supplied source labels only; fold0 partition1 checked when fold is supplied',
            'interpretation': 'Conditional label-shift working model; class-conditionals may not transfer'}


def estimate_prior(densities, target_counts, initial, max_iter=200, tolerance=1e-9):
    'unregularized multinomial-mixture em; exposes fit and identifiability'
    density = np.asarray(densities, dtype=float)
    counts = np.asarray(target_counts, dtype=float)
    prior = _normalise(initial)
    if density.ndim != 2 or density.shape != (len(prior), len(counts)) or not np.isfinite(density).all() or (density < 0).any():
        raise ValueError('Invalid class-conditional density matrix')
    if not np.allclose(density.sum(axis=1), 1., atol=1e-8):
        raise ValueError('Class densities must be normalized')
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Invalid target histogram')
    if not counts.sum():
        return {'prior': prior.tolist(), 'passed': False, 'reason': 'empty_target', 'iterations': 0}
    if max_iter <= 0 or tolerance <= 0:
        raise ValueError('EM iterations/tolerance must be positive')
    if _separation(density) < MIN_SEPARATION:
        return {'prior': prior.tolist(), 'passed': False, 'reason': 'unidentifiable_densities', 'iterations': 0}
    observed = _normalise(counts)
    if ((counts > 0) & (density.sum(axis=0) == 0)).any():
        return {'prior': prior.tolist(), 'passed': False, 'reason': 'bad_mixture_fit',
                'iterations': 0, 'mixture_tv': float(.5*np.abs(observed-prior @ density).sum()),
                'unsupported_target_mass': float(observed[density.sum(axis=0) == 0].sum())}
    converged = False
    previous_likelihood = float(np.sum(observed * np.log(np.maximum(prior @ density, 1e-15))))
    criterion = None
    for iteration in range(max_iter):
        mixture = prior @ density
        responsibility = prior[:, None] * density / np.maximum(mixture, 1e-15)
        updated = _normalise((responsibility * counts).sum(axis=1))
        change = float(np.max(np.abs(updated-prior)))
        likelihood = float(np.sum(observed * np.log(np.maximum(updated @ density, 1e-15))))
        likelihood_change = abs(likelihood-previous_likelihood)

        if change < tolerance or (likelihood_change < 1e-12 and change < 1e-6):
            prior, converged = updated, True
            criterion = 'prior_change' if change < tolerance else 'likelihood_and_prior_stability'
            break
        prior = updated
        previous_likelihood = likelihood
    mixture = prior @ density
    tv = float(.5 * np.abs(observed-mixture).sum())
    return {'prior': prior.tolist(), 'passed': bool(converged and tv <= MAX_TV),
            'reason': None if converged and tv <= MAX_TV else 'bad_mixture_fit' if tv > MAX_TV else 'EM_not_converged',
            'iterations': iteration+1, 'converged': converged, 'mixture_tv': tv,
            'convergence_criterion': criterion, 'last_prior_change': change,
            'last_likelihood_change': likelihood_change,
            'log_likelihood_per_row': float(np.sum(observed * np.log(np.maximum(mixture, 1e-15))))}


def _bounded_prior(prior, source):
    'project onto a simplex with explicit per-class source-relative bounds'
    prior, source = np.maximum(np.asarray(prior), 1e-15), np.asarray(source)
    low, high = source * PRIOR_RATIO[0], np.minimum(source * PRIOR_RATIO[1], 1.)
    left, right = 0., float(np.max(high/prior)) + 1.
    for _ in range(100):
        mid = (left+right)/2
        if np.clip(mid*prior, low, high).sum() > 1:
            right = mid
        else:
            left = mid
    bounded = np.clip(((left+right)/2)*prior, low, high)
    return bounded / bounded.sum()


def adapt(bundle, target_winners):
    'Estimate unlabeled target priors; no target y/own/fold access occurs'
    optional = bundle['optional_strata']
    values, bins = _view(target_winners, bundle['score_col'], optional)
    recipe = {'version': 1, 'cells': {}, 'diagnostics': {}, 'target_labels_used': False}
    for level in ['country', 'parent'] + (['child'] if optional else []):
        for key, ids in _groups(values, level, optional).items():
            if key not in bundle['cells']:
                continue
            cell = bundle['cells'][key]
            counts = np.bincount(bins[ids], minlength=len(EDGES)+1).astype(float)
            prior = np.asarray(cell['source_prior'])
            entry = {'prior': prior.tolist(), 'posterior': cell['control_posterior'], 'adapted': False}
            source_hist = np.asarray(cell['source_histogram'])
            if counts.sum() < MIN_TARGET:
                diagnostic = {'passed': False, 'reason': 'sparse_target', 'target_rows': int(counts.sum())}
            elif not cell['identifiable']:
                diagnostic = {'passed': False, 'reason': 'unidentifiable_densities', 'target_rows': int(counts.sum())}
            elif np.allclose(_normalise(counts), _normalise(source_hist), rtol=0, atol=1e-12):
                diagnostic = {'passed': True, 'reason': 'source_target_identity', 'target_rows': int(counts.sum())}
            else:
                diagnostic = estimate_prior(cell['densities'], counts, prior)
                diagnostic['target_rows'] = int(counts.sum())
                if diagnostic['passed']:
                    fitted = _bounded_prior(diagnostic['prior'], prior)
                    shrunk = _normalise(np.exp((1.-PRIOR_SHRINK)*np.log(prior) + PRIOR_SHRINK*np.log(fitted)))
                    shrunk = _bounded_prior(shrunk, prior)
                    density = np.asarray(cell['densities'])
                    mixture_after_shrink = shrunk @ density
                    post = _isotonic(_posterior(density, shrunk), counts)
                    entry = {'prior': shrunk.tolist(), 'posterior': post.tolist(), 'adapted': True}
                    diagnostic['clipped_prior'] = fitted.tolist()
                    diagnostic['shrunk_prior'] = shrunk.tolist()
                    diagnostic['mixture_tv_after_shrink'] = float(.5 * np.abs(_normalise(counts)-mixture_after_shrink).sum())
            recipe['cells'][key], recipe['diagnostics'][key] = entry, diagnostic
            diagnostic.update(source_rows=cell['source_rows'],
                source_class_counts=cell['source_class_counts'], class_order=cell['class_order'],
                identifiable=cell['identifiable'], separation=cell['separation'],
                adapted=entry['adapted'], retained_control=not entry['adapted'])
    recipe['adapted_cells'] = sum(c['adapted'] for c in recipe['cells'].values())
    return recipe


def predict(bundle, frame, recipe=None):
    'Return alias probabilities; unsupported countries/cells retain raw score'
    optional = bundle['optional_strata']
    values, _ = _view(frame, bundle['score_col'], optional)
    scores = np.asarray(values[bundle['score_col']], dtype=float).copy()
    x, centres = _logit(scores), np.asarray(bundle['centres'])
    for level in ['country', 'parent'] + (['child'] if optional else []):
        for key, ids in _groups(values, level, optional).items():
            if key not in bundle['cells']:
                continue
            posterior = bundle['cells'][key]['control_posterior']
            if recipe is not None and key in recipe['cells']:
                posterior = recipe['cells'][key]['posterior']
            scores[ids] = np.interp(x[ids], centres, posterior)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise AssertionError('Invalid calibrated alias probability')
    return scores.astype(np.float32)
