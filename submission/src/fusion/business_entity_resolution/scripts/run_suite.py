'sequential cpu ablations; choose only by the common heldout tuning subset'
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENT = {'control':'stack_v4_control','v4':'stack_v4','recall':'stack_v4_recall'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('data','scores','lexical','output','work'):
        ap.add_argument('--'+name,type=Path,required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rs = {}
    for variant,name in EXPERIMENT.items():
        print(f'=== START {variant} ===',flush=True)
        cmd = [sys.executable,str(HERE/'run_stack.py'),'--variant',variant,
               '--data',str(args.data),'--scores',str(args.scores),'--lexical',str(args.lexical),
               '--output',str(args.output/'experiments'),'--work',str(args.work/variant)]
        subprocess.run(cmd,check=True)
        experiment = args.output/'experiments'/name
        subprocess.run([sys.executable,str(HERE/'validate_outputs.py'),'--data',str(args.data),
                        '--output',str(experiment/'output')],check=True)
        report = json.loads((experiment/'stack/report.json').read_text())
        method = report['methods'][report['selected']]
        rs[variant] = {'experiment':name,'method':report['selected'],
            'tune_macro_f05':method['tune']['macro_f05'],'audit':method['audit']}
        (args.output/'progress.json').write_text(json.dumps(rs,indent=2))
        print(f'=== FINISHED {variant}: {json.dumps(rs[variant])} ===',flush=True)
    winner = max(rs,key=lambda variant:rs[variant]['tune_macro_f05'])
    src = args.output/'experiments'/EXPERIMENT[winner]
    shutil.copytree(src/'output',args.output/'selected_output',dirs_exist_ok=True)
    (args.output/'selection.json').write_text(json.dumps({
        'selected_variant':winner,'selection_criterion':'heldout fold-0 macro F0.5; audit never selects',
        'selected_bundle':f'experiments/{EXPERIMENT[winner]}/stack/bundle',
        'selected_output':'selected_output','results':rs},indent=2))
    print(f'COMPLETE: selected {winner} on tuning only; outputs in {args.output}/selected_output',flush=True)


if __name__ == '__main__':
    main()
