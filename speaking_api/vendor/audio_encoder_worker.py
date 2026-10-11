"""Selected E2 audio encoder and exact 1550-feature training recipe."""
import argparse
import json
from pathlib import Path
import tempfile
import time

def encode(sessions_path, audio_root, bundle_path, output_path):
    from unsloth import FastSentenceTransformer
    import torch
    import numpy as np
    import soundfile as sf
    import librosa
    from sentence_transformers.util import batch_to_device
    bundle=Path(bundle_path);audio_root=Path(audio_root)
    config=json.loads((bundle/'audio/config.json').read_text())
    try:
        from .checkpoint import load_audio_checkpoint
    except ImportError:
        from checkpoint import load_audio_checkpoint
    state,checksum=load_audio_checkpoint(bundle)
    rows=[json.loads(line) for line in Path(sessions_path).read_text().splitlines() if line.strip()]
    if not rows or len({r['session_id'] for r in rows})!=len(rows):
        raise ValueError('Expected nonempty uniquely identified sessions')
    started=time.perf_counter()
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model=FastSentenceTransformer.from_pretrained(model_name=config['model'],revision=config['model_revision'],
        max_seq_length=2048,config_kwargs={'vision_config':None},dtype=dtype,
        load_in_4bit=False,load_in_16bit=True,full_finetuning=False)
    model=FastSentenceTransformer.get_peft_model(model,r=16,lora_alpha=32,lora_dropout=0,bias='none',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        finetune_audio_layers=True,use_gradient_checkpointing=False,random_state=3407,task_type='FEATURE_EXTRACTION')
    head=torch.nn.Sequential(torch.nn.Linear(768,128),torch.nn.GELU(),torch.nn.Dropout(.1),
                            torch.nn.Linear(128,3),torch.nn.Sigmoid()).cuda()
    expected={n for n,p in model.named_parameters() if p.requires_grad}
    assert set(state['adapters'])==expected
    loaded=model.load_state_dict(state['adapters'],strict=False)
    assert not loaded.unexpected_keys and not (set(loaded.missing_keys)&expected)
    head.load_state_dict(state['head'])
    model.eval();model.requires_grad_(False);head.eval();head.requires_grad_(False)
    arrays={}
    with tempfile.TemporaryDirectory(prefix='ielts-audio-') as temp:
        clip_path=Path(temp)/'chunk.wav'
        for row in rows:
            wave,sr=sf.read(audio_root/row['audio_file'],dtype='float32')
            if sr!=16000 or wave.ndim!=1 or not np.isfinite(wave).all():
                raise ValueError('Prepared audio must be finite mono 16 kHz')
            response_waves=[];vectors=[];predictions=[];weights=[]
            for response in row['responses']:
                begin,end=round(response['audio_start']*sr),round(response['audio_end']*sr)
                if not 0<=begin<end<=len(wave):raise ValueError('Response audio boundaries are invalid')
                response_waves.append(wave[begin:end])
                for offset in range(begin,end,20*sr):
                    clip=wave[offset:min(end,offset+20*sr)]
                    if len(clip)<160:continue
                    sf.write(clip_path,clip,sr)
                    inputs=batch_to_device(model.tokenize([{'audio':str(clip_path)}]),'cuda:0')
                    with torch.no_grad():
                        z=model(inputs)['sentence_embedding'].float()
                        vectors.append(z.cpu().numpy()[0]);predictions.append(head(z).cpu().numpy()[0])
                    weights.append(len(clip)/sr)
            if not vectors:raise ValueError('No assessable audio chunks')
            weights=np.array(weights);weights/=weights.sum()
            z,p=np.array(vectors),np.array(predictions)
            zm=(weights[:,None]*z).sum(0);pm=(weights[:,None]*p).sum(0)
            zs=np.sqrt((weights[:,None]*(z-zm)**2).sum(0));ps=np.sqrt((weights[:,None]*(p-pm)**2).sum(0))
            energies=[librosa.feature.rms(y=clip,frame_length=1024,hop_length=256)[0] for clip in response_waves]
            threshold=max(1e-4,float(np.percentile(np.concatenate(energies),90))*.05)
            masks=[energy<threshold for energy in energies]
            pause_lengths=[];pitch_values=[]
            for clip,mask in zip(response_waves,masks):
                changes=np.diff(np.r_[False,mask,False].astype(int))
                lengths=(np.where(changes==-1)[0]-np.where(changes==1)[0])*256/sr
                pause_lengths.extend(lengths[lengths>=.25])
                f0=librosa.yin(clip,fmin=60,fmax=500,sr=sr,frame_length=1024,hop_length=256)
                assert len(f0)==len(mask)
                pitch_values.extend(f0[~mask])
            pauses,voiced=np.asarray(pause_lengths),np.asarray(pitch_values)
            pitch_cv=float(np.std(voiced)/max(np.mean(voiced),1)) if len(voiced) else 0.
            seconds=len(wave)/sr;words=sum(len(r['text'].split()) for r in row['responses'])
            span=max(r['end'] for r in row['responses'])-min(r['start'] for r in row['responses'])
            timing=np.array([np.log1p(seconds),np.log1p(words),words/seconds,seconds/max(span,seconds),
                np.concatenate(masks).mean(),len(pauses)*60/seconds,float(pauses.mean()) if len(pauses) else 0,pitch_cv])
            vector=np.r_[zm,zs,pm,ps,timing].astype(np.float32)
            assert vector.shape==(1550,) and np.isfinite(vector).all()
            arrays[row['session_id']]=vector
            print('AUDIO ENCODED',len(arrays),len(rows),flush=True)
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_path,**arrays)
    report={'sessions':len(rows),'checkpoint_sha256':checksum,'feature_dimension':1550,
        'runtime_seconds':time.perf_counter()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
    output_path.with_suffix('.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sessions');parser.add_argument('audio_root');parser.add_argument('bundle');parser.add_argument('output')
    args=parser.parse_args()
    print(json.dumps(encode(args.sessions,args.audio_root,args.bundle,args.output),indent=2))
