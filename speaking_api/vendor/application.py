"""Audio/questions to four experimental scores using separately pinned workers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def score_prepared(session_path, bundle, text_artifacts, work_dir, audio_python, text_python,
                   *, local_files_only=False):
    import numpy as np
    from .portable_heads import FeatureScorer
    session_path=Path(session_path).resolve();bundle=Path(bundle).resolve()
    work_dir=Path(work_dir).resolve();work_dir.mkdir(parents=True,exist_ok=True)
    row=json.loads(session_path.read_text())
    manifest=work_dir/'sessions.jsonl'
    manifest.write_text(json.dumps(row,ensure_ascii=False)+'\n')
    here=Path(__file__).resolve().parent
    audio_output=work_dir/'audio_features.npz';text_output=work_dir/'text_features.npz'
    started=time.perf_counter()
    env = dict(os.environ)
    if local_files_only:
        env['HF_HUB_OFFLINE'] = '1'
    text_config=json.loads((Path(text_artifacts)/'compact_config.json').read_text())
    subprocess.run([audio_python,str(here/'audio_encoder_worker.py'),str(manifest),
        str(session_path.parent),str(bundle),str(audio_output)],check=True,env=env)
    subprocess.run([text_python,str(here/'text_encoder_worker.py'),str(manifest),str(text_output),
        '--model',text_config['base_model'],'--revision',text_config['revision']],check=True,env=env)
    scorer=FeatureScorer(text_artifacts,bundle/'decision/normalization.npz')
    sid=row['session_id']
    with np.load(audio_output) as a,np.load(text_output) as t:
        output=scorer.predict(t[sid],a[sid])[0]
    parts=sorted({r['part'] for r in row['responses'] if r.get('part') in (1,2,3)})
    output.update({'session_id':sid,'assessment_scope':'provided_responses',
        'coverage':{'responses':len(row['responses']),'known_parts':parts,
            'unknown_part_responses':sum(r.get('part') is None for r in row['responses']),
            'audio_seconds':row.get('audio_seconds')},
        'input_provenance':row.get('input_provenance',{}),
        'scoring_runtime_seconds':time.perf_counter()-started})
    (work_dir/'scores.json').write_text(json.dumps(output,indent=2,ensure_ascii=False))
    return output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request',help='Input audio/question manifest, or prepared session with --prepared')
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--text-artifacts',required=True)
    parser.add_argument('--work-dir',required=True)
    parser.add_argument('--audio-python',required=True,help='Python with the pinned Unsloth audio stack')
    parser.add_argument('--text-python',required=True,help='Python with the pinned Unsloth decision stack')
    parser.add_argument('--prepared',action='store_true')
    parser.add_argument('--asr-model',default='small.en')
    args=parser.parse_args()
    if args.prepared:session=Path(args.request)
    else:
        from .prepare_input import prepare
        prepared=Path(args.work_dir)/'prepared'
        prepare(args.request,prepared,asr_model=args.asr_model)
        session=prepared/'session.json'
    output=score_prepared(session,args.bundle,args.text_artifacts,args.work_dir,
                          args.audio_python,args.text_python)
    print(json.dumps(output,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
