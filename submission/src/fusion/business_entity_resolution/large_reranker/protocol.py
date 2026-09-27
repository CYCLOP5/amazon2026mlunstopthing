'pinned, bounded business-identity inputs for the multilingual 4b reranker'
MODEL = 'Qwen/Qwen3-Reranker-4B'
REVISION = 'e71f49284e452dcb0f5ce1e0cf8332661b323186'
MAX_PARAMETERS = 8_000_000_000
MAX_LENGTH = 384
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements '
          'based on the Query and the Instruct provided. '
          'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION = ('Determine whether these two records identify the same real-world business '
               'at the same location. Account for spelling variations, translations, '
               'abbreviations and legal suffixes. A shared brand or similar address '
               'alone does not establish identity. Missing fields are unknown evidence.')


def clean(value):

    return str(value or '').replace('<|', '‹|').replace('|>', '|›').strip()


def encode_rows(tokenizer, rows, max_length=MAX_LENGTH):
    'budget each record independently, protecting both names and addresses'
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    prefix = enc(PREFIX + '<Instruct>: ' + INSTRUCTION + '\n<Query>: ')
    middle, suffix = enc('\n<Document>: '), enc(SUFFIX)
    labels = [enc('Name: '), enc('\nAddress: '), enc('\nCountry: ')]
    room = max_length - len(prefix) - len(middle) - len(suffix)
    per_side = room // 2
    content = per_side - sum(map(len, labels))
    if content < 24:
        raise ValueError('max_length cannot preserve both business records')
    country_budget = min(8, content // 8)
    name_budget = (content - country_budget) * 2 // 5
    address_budget = content - country_budget - name_budget
    budgets = [name_budget, address_budget, country_budget]
    res = []
    for row in rows:
        def record(side):
            values = [row.get('nm'+side), row.get('ad'+side), row.get('co'+side)]
            tokens = []
            for label, value, budget in zip(labels, values, budgets):
                tokens.extend(label)
                tokens.extend(enc(clean(value) or '[missing]')[:budget])
            return tokens
        ids = prefix + record('1') + middle + record('2') + suffix
        if len(ids) > max_length:
            raise AssertionError('Pair token budget exceeded')
        res.append(ids)
    return res


def answer_tokens(tokenizer):
    yes, no = [tokenizer.encode(t, add_special_tokens=False) for t in ('yes', 'no')]
    if len(yes) != 1 or len(no) != 1 or yes == no:
        raise ValueError('Pinned reranker must have distinct single-token yes/no answers')
    return yes[0], no[0]


def worker_tasks(tasks, rank, workers, shard_index=0, shards=1):
    if min(workers, shards) < 1 or not 0 <= rank < workers or not 0 <= shard_index < shards:
        raise ValueError('Invalid scoring shard or GPU rank')
    return [task for i, task in enumerate(tasks)
            if i % (workers * shards) == shard_index * workers + rank]
