"""Strict scoring contracts shared by training preparation and inference."""
import json
import math

CRITERIA = ('fluency_and_coherence','lexical_resource',
            'grammatical_range_and_accuracy','pronunciation')

def validate_scores(scores, half_bands=False):
    if any(key not in scores for key in CRITERIA):
        raise ValueError('All four criterion scores are required')
    result={}
    for key in CRITERIA:
        value=scores[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 1<=value<=9:
            raise ValueError('Invalid criterion score: '+key)
        if half_bands and abs(value*2-round(value*2))>1e-6:
            raise ValueError('Expected half-band value: '+key)
        result[key]=float(value)
    return result

def speaking_band(scores):
    values=validate_scores(scores)
    return math.floor(2*sum(values.values())/4+.5)/2

def parse_teacher_output(text):
    decoder=json.JSONDecoder()
    for offset,char in enumerate(text):
        if char!='{': continue
        try:
            value,_=decoder.raw_decode(text[offset:])
            if isinstance(value,dict):
                validate_scores(value,half_bands=True)
                return value
        except (ValueError,TypeError):
            continue
    raise ValueError('No valid four-criterion JSON object in teacher output')

def point_distribution(score):
    if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 1<=score<=9:
        raise ValueError('Band must be in [1,9]')
    lo=math.floor(score);hi=math.ceil(score)
    result=[0.0]*9
    if lo==hi:result[lo-1]=1.0
    else:result[lo-1]=hi-score;result[hi-1]=score-lo
    return result

def assert_group_disjoint(records):
    owners={}
    for row in records:
        group=row['split_group'];split=row['split']
        if group in owners and owners[group]!=split:
            raise ValueError('A speaker/source group crosses splits')
        owners[group]=split

def student_turns(record):
    """Allowlist student inputs: explicitly exclude source scores, titles and teacher evidence."""
    return [{'question':r['question'],'part':r.get('part'),'transcript':r['text']}
            for r in record['responses']]
