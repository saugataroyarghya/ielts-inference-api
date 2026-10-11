"""Standalone frozen E4 text encoder. Run in the pinned CUDA/Unsloth environment."""
import argparse
import hashlib
import json
from pathlib import Path
import time

def encode(sessions_path, output_path, revision='23c69c53358a07516b5827588b3fdb12ae78fd65',
           model_name='unsloth/Qwen3.5-0.8B'):
    from unsloth import FastDecisionModel
    from unsloth.models.clef import encode_record
    from unsloth.models.decision import _clef_question, _no_cudnn_attention
    from unsloth.models._decision_fast import pad_length
    import torch
    import numpy as np
    from bitsandbytes.nn import Linear4bit
    try:
        from .rubric_questions import QUESTIONS
        from .contracts import student_turns
    except ImportError:
        from rubric_questions import QUESTIONS
        from contracts import student_turns
    rows=[json.loads(line) for line in Path(sessions_path).read_text().splitlines() if line.strip()]
    if not rows or len({r['session_id'] for r in rows})!=len(rows):
        raise ValueError('Expected nonempty uniquely identified sessions')
    started=time.perf_counter()
    model,tokenizer=FastDecisionModel.from_pretrained(model_name=model_name,
        revision=revision,max_seq_length=8192,load_in_4bit=True)
    FastDecisionModel.freeze_backbone(model)
    for module in model.encoder.modules():
        if isinstance(module,Linear4bit):
            module.compute_dtype=torch.float32
            module.compute_type_is_set=True
    model.eval();model.requires_grad_(False)
    backbone=model._backbone()
    text_model=getattr(backbone,'model',None) or backbone.base_model
    text_model=getattr(text_model,'language_model',text_model)
    tokenizer=getattr(tokenizer,'tokenizer',tokenizer)
    questions={k:_clef_question(v) for k,v in QUESTIONS.items()}
    arrays={};token_counts={};input_hashes={}
    for row in rows:
        state={'responses':student_turns(row)}
        # max_length+1 detects the helper's otherwise silent state truncation.
        record=encode_record(tokenizer,{'state':state,'questions':questions},max_length=8193)
        if len(record.input_ids)>8192:
            raise ValueError(f"Session {row['session_id']} exceeds the trained 8192-token context")
        ids=torch.tensor([record.input_ids],device='cuda',dtype=torch.long)
        inputs=pad_length(model,{'input_ids':ids,'attention_mask':torch.ones_like(ids)})
        with torch.no_grad(),_no_cudnn_attention():
            hidden=text_model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],
                use_cache=False,return_dict=True).last_hidden_state.float()
            mask=inputs['attention_mask'].to(hidden.dtype).unsqueeze(-1)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
        value=pooled[0].cpu().numpy()
        assert np.isfinite(value).all()
        arrays[row['session_id']]=value
        token_counts[row['session_id']]=len(record.input_ids)
        input_hashes[row['session_id']]=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()
        print('TEXT ENCODED',len(arrays),len(rows),flush=True)
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_path,**arrays)
    report={'model':model_name,'revision':revision,'pooling':'masked mean',
        'compute_dtype':'float32','quantized_storage':True,'token_counts':token_counts,
        'input_hashes':input_hashes,'sessions':len(rows),'runtime_seconds':time.perf_counter()-started,
        'peak_memory_bytes':torch.cuda.max_memory_allocated()}
    output_path.with_suffix('.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sessions');parser.add_argument('output')
    parser.add_argument('--model',default='unsloth/Qwen3.5-0.8B')
    parser.add_argument('--revision',default='23c69c53358a07516b5827588b3fdb12ae78fd65')
    args=parser.parse_args()
    print(json.dumps(encode(args.sessions,args.output,args.revision,args.model),indent=2))
