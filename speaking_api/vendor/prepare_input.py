"""Prepare candidate response recordings, preserving within-response timing.

Input JSON: {session_id, responses:[{question, part, audio_path, start?, end?,
transcript?}]}. Each path must contain candidate speech for the chosen interval.
Mixed examiner/candidate recordings need explicit candidate intervals first.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

def prepare(manifest_path, output_dir, asr_model='small.en', asr_device='cpu'):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    request = json.loads(manifest_path.read_text())
    responses = request.get('responses')
    if not isinstance(responses,list) or not responses:
        raise ValueError('At least one candidate response is required')
    output_dir.mkdir(parents=True,exist_ok=True)
    source_rows, blocks, audio, offset = [], [], [], 0.
    recognizer = None
    full_timeline = all('timeline_start' in row for row in responses)
    if any('timeline_start' in row for row in responses) and not full_timeline:
        raise ValueError('Provide timeline_start for every response or omit it for all')
    for index,row in enumerate(responses):
        if not isinstance(row.get('question'),str):
            raise ValueError(f'Response {index}: question must be a string')
        part = row.get('part')
        if isinstance(part,bool) or part not in (None,1,2,3):
            raise ValueError(f'Response {index}: part must be 1, 2, 3 or null')
        path = Path(row['audio_path'])
        if not path.is_absolute():path=manifest_path.parent/path
        wave,sr = sf.read(path,dtype='float32',always_2d=True)
        start,end = float(row.get('start',0)),float(row.get('end',len(wave)/sr))
        if not math.isfinite(start+end) or not 0<=start<end<=len(wave)/sr+1e-3:
            raise ValueError(f'Response {index}: invalid audio interval')
        clip = wave[round(start*sr):round(end*sr)].mean(axis=1)
        if sr!=16000:
            divisor=math.gcd(sr,16000)
            clip=resample_poly(clip,16000//divisor,sr//divisor).astype(np.float32)
        if len(clip)<160 or not np.isfinite(clip).all():
            raise ValueError(f'Response {index}: empty or nonfinite audio')
        duration=len(clip)/16000
        transcript=row.get('transcript')
        words=[]
        if transcript is None:
            if recognizer is None:
                from faster_whisper import WhisperModel
                recognizer=WhisperModel(asr_model,device=asr_device,
                    compute_type='int8' if asr_device=='cpu' else 'float16')
            segments,info=recognizer.transcribe(clip,language='en',beam_size=5,
                word_timestamps=True,vad_filter=False,condition_on_previous_text=False)
            parts=[]
            for segment in segments:
                parts.append(segment.text.strip())
                words.extend({'word':w.word,'start':float(w.start),'end':float(w.end),
                    'probability':float(w.probability)} for w in (segment.words or []))
            transcript=' '.join(parts)
            transcript_source='faster_whisper_unedited'
        else:
            if not isinstance(transcript,str):raise ValueError('transcript must be a string or null')
            transcript_source='caller_supplied'
        timeline_start=float(row['timeline_start']) if full_timeline else offset
        if not math.isfinite(timeline_start) or timeline_start<0:
            raise ValueError('timeline_start must be finite and nonnegative')
        if blocks and timeline_start<blocks[-1]['end']-1e-3:
            raise ValueError('Response timeline intervals overlap or are out of order')
        blocks.append({'question':row['question'],'part':part,'text':transcript,
            'start':timeline_start,'end':timeline_start+duration,
            'audio_start':offset,'audio_end':offset+duration})
        source_rows.append({'response_index':index,'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_interval':[start,end],'transcript_source':transcript_source,
            'words':words,'word_timestamps_basis':'ASR estimates' if words else None})
        audio.append(clip);offset+=duration
    combined=np.concatenate(audio)
    if not any(block['text'].strip() for block in blocks):
        raise ValueError('No candidate speech was transcribed; scoring requires assessable speech')
    sf.write(output_dir/'candidate.flac',combined,16000)
    record={'session_id':str(request.get('session_id','new-recording')),'audio_file':'candidate.flac',
        'responses':blocks,'audio_seconds':offset,'audio_coverage_fraction':1.,
        'input_provenance':{'candidate_intervals':'caller supplied',
            'original_session_timeline_available':full_timeline,
            'candidate_time_share_basis':'original timeline' if full_timeline else 'candidate-only timeline',
            'asr_model':asr_model if recognizer else None,'alignment':'ASR word timing; no forced alignment',
            'disfluency_preservation_verified':False,'responses':source_rows}}
    (output_dir/'session.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
    return record

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest');parser.add_argument('output')
    parser.add_argument('--asr-model',default='small.en')
    parser.add_argument('--asr-device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    result=prepare(args.manifest,args.output,args.asr_model,args.asr_device)
    print(json.dumps({'session_id':result['session_id'],'responses':len(result['responses']),
                      'audio_seconds':result['audio_seconds']}))

if __name__=='__main__':main()
