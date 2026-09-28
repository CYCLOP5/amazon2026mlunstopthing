"""recount distinct inference-checkpoint parameters from safetensors headers"""

import argparse
import json
import math
from pathlib import Path as path
import struct


root = path(__file__).resolve().parents[1]


def tensors(file):
    with file.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('incomplete safetensors header')
        length = struct.unpack('<Q', raw)[0]
        if not 0 < length <= min(16 << 20, file.stat().st_size - 8):
            raise ValueError('invalid safetensors header length')
        header = json.loads(stream.read(length))
    parameters = 0
    buffers = 0
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        shape = tensor['shape']
        if not isinstance(shape, list) or any(not isinstance(x, int) or x < 0 for x in shape):
            raise ValueError('invalid tensor shape')
        count = math.prod(shape)
        if tensor['dtype'] in {'I64', 'I32', 'I16', 'I8', 'U8', 'BOOL'} or name.endswith(('position_ids', 'token_type_ids', 'inv_freq', '_float_tensor')):
            buffers += count
        else:
            parameters += count
    return parameters, buffers


def inventory(bundle, config):
    spec = json.loads(config.read_text())
    models = []
    for item in spec['models']:
        parameters = buffers = 0
        for entry in item['files']:
            file = bundle / entry['path']
            if not file.resolve().is_relative_to(bundle.resolve()):
                raise ValueError('unsafe model path')
            p, b = tensors(file)
            if p != entry['parameters']:
                raise ValueError('checkpoint tensor count differs: ' + entry['path'])
            parameters += p
            buffers += b
        if parameters != item['parameters'] or buffers != item['excluded_buffer_elements']:
            raise ValueError('model count differs: ' + item['id'])
        models.append({'id': item['id'], 'model': item['model'], 'parameters': parameters, 'excluded_buffers': buffers})
    total = sum(x['parameters'] for x in models)
    if total != spec['full_neural_parameters']:
        raise ValueError('full neural total differs')
    return {'verified': True, 'distinct_checkpoints': len(models), 'neural_parameters': total,
            'largest_individual_checkpoint': max(x['parameters'] for x in models),
            'per_model_limit': spec['parameter_limit']['limit'], 'models': models}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=path, default=root)
    parser.add_argument('--config', type=path, default=root / 'configs/model-parameters.json')
    args = parser.parse_args()
    print(json.dumps(inventory(args.bundle, args.config), indent=2))
