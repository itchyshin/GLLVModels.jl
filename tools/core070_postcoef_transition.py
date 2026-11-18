"""Fail-closed structural check; capture qualification is a separate gate."""

from collections import Counter, defaultdict

import hashlib

import json

import math

import re

import copy

TARGET='postfit-policy/POST-COEF-EMPTY'

BOARD_ID='postfit-policy-POST-COEF-EMPTY'

CID='CORE070-POSTFIT-COEF-EMPTY-NATIVE'

NEW='docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/CORE070-POSTFIT-COEF-THREE-FIT-NATIVE.json'

OLD='docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/CORE070-POSTFIT-COEF-EMPTY-NATIVE.json'

MAP='docs/dev-log/core070/true-parity-latest/case-map-postfit.json'

PIN='9539352f66f2db2cc26b1c393e67212a359b60c9'

OLD_TARGET_SHA256='0ada7a1400b5a87eb4756444aba581b4fde4fce797165a6fa5f4ab13c19dcf94'

TIER='numeric: every executable case receipt carries an integer_equality comparison block pinned to P1 (tolerance 0.5, so the integers must be equal), under itchyshin/GLLVModels.jl#684 item 1'

NOTE='Three paired public fits compare coefficient-vector lengths; separate always-empty and extra-coefficient controls are verified. Numerical coefficient values and names are separate claims.'

ORDER=['EVIDENCED','EVIDENCED-BEHAVIOURAL','DISPOSITION-SIGNED','NUMERIC-UNVERIFIED',
       'BEHAVIOURAL-UNVERIFIED','REGISTRATION-ONLY','HELD','PARTIAL','NEEDS-SURFACE',
       'NON-NUMERIC','NON-DISCRIMINATING','FAIL','NOT-MEASURED','NO-TIER','DISPOSITION-UNVERIFIED']

def _equal(a,b):
    if isinstance(a,bool) or isinstance(b,bool): return type(a) is bool and type(b) is bool and a==b
    if isinstance(a,(int,float)) and isinstance(b,(int,float)): return math.isfinite(a) and math.isfinite(b) and a==b
    if isinstance(a,dict) and isinstance(b,dict): return a.keys()==b.keys() and all(_equal(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list): return len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b

def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('duplicate JSON key')
        result[key]=value
    return result

def _reject_constant(value):
    raise ValueError('non-finite JSON number')

def _unsafe(value):
    if isinstance(value,dict):
        if value.get('signed_by') or value.get('disposition') == 'DISPOSITION-SIGNED': return True
        flags=('pass','passed','behaved','harness_pass','all_checks','all_positive_pass','source_unchanged')
        if any(k in value and value[k] is not True for k in flags): return True
        return any(_unsafe(v) for v in value.values())
    if isinstance(value,list): return any(_unsafe(v) for v in value)
    return isinstance(value,str) and value.upper() in {'FAIL','FAILED','FAILURE','NUMERIC_FAIL','NUMERIC-FAIL','ERROR','ERRORED','TIMEOUT','TIMED_OUT','CANCELLED','CANCELED','INTERRUPTED','ABORTED'}

def _recount(doc):
    counts={k:0 for k in doc['counts']}
    for row in doc['rows']:
        tier=row['evidence_tier']
        key='retired_at_p1_not_measured' if row.get('carry_scan_status')=='RETIRED' and tier=='not_measured' else ('numeric_pass' if tier=='numeric' else tier)
        counts[key]=counts.get(key,0)+1
    return counts

def _indexed(doc):
    rows=doc['rows']
    if not isinstance(rows,list) or any(not isinstance(x,dict) or not isinstance(x.get('source_id'),str) for x in rows):
        raise ValueError('bad rows')
    out={x['source_id']:x for x in rows}
    if len(out)!=len(rows) or TARGET not in out: raise ValueError('duplicate or missing row')
    return out

def assembled_transition(before,after,old_map,new_map):
    """Check the assembled map's exact semantic delta; does not qualify evidence."""
    try:
        parse=lambda raw: json.loads(raw,parse_constant=_reject_constant,object_pairs_hook=_unique_object)
        old,new,prior_map,current_map=map(parse,(before,after,old_map,new_map))
        if (json.dumps(new,indent=1)+'\n').encode()!=after: return False
        oldrows,newrows=_indexed(old),_indexed(new)
        prior,current=_indexed(prior_map),_indexed(current_map)
        if not _equal(oldrows[TARGET],prior[TARGET]) or not _equal(newrows[TARGET],current[TARGET]): return False
        expected=copy.deepcopy(old)
        index=next(i for i,row in enumerate(expected['rows']) if row['source_id']==TARGET)
        expected['rows'][index]=copy.deepcopy(current[TARGET])
        matches=[x for x in expected['inputs'] if isinstance(x,dict) and x.get('path')==MAP]
        if len(matches)!=1 or matches[0].get('sha256')!=hashlib.sha256(old_map).hexdigest(): return False
        matches[0]['sha256']=hashlib.sha256(new_map).hexdigest()
        return _equal(expected,new)
    except (KeyError,TypeError,ValueError,StopIteration):
        return False

def _render_totals(counts):
    total=Counter()
    for c in counts.values(): total.update(c)
    present=[s for s in ORDER if total[s]]+sorted(s for s in total if s not in ORDER)
    lines=['| `family` | '+' | '.join(present)+' | total |','|---|'+'---|'*(len(present)+1)]
    for family in sorted(counts):
        c=counts[family]; lines.append(f'| `{family}` | '+' | '.join(str(c[s]) for s in present)+f' | {sum(c.values())} |')
    lines.append('| `all` | '+' | '.join(str(total[s]) for s in present)+f' | {sum(total.values())} |')
    return '\n'.join(lines)

def structural_transition(before,after,old_map,new_map):
    """Compare generated layout only; True never qualifies a capture or promotes a row."""
    try:
        old=json.loads(old_map,parse_constant=_reject_constant,object_pairs_hook=_unique_object); new=json.loads(new_map,parse_constant=_reject_constant,object_pairs_hook=_unique_object); oldrows=_indexed(old); newrows=_indexed(new)
        if list(oldrows)!=list(newrows) or any(not _equal(oldrows[k],newrows[k]) for k in oldrows if k!=TARGET): return False
        if not _equal({k:v for k,v in old.items() if k not in ('rows','counts')}, {k:v for k,v in new.items() if k not in ('rows','counts')}): return False
        a,b=oldrows[TARGET],newrows[TARGET]
        canonical=json.dumps(a,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
        if hashlib.sha256(canonical).hexdigest()!=OLD_TARGET_SHA256: return False
        allowed={'evidence_tier','measured_against','evidence','measured_result','note'}
        if not _equal({k:v for k,v in a.items() if k not in allowed}, {k:v for k,v in b.items() if k not in allowed}): return False
        if _unsafe(a) or _unsafe(b): return False
        protected=('source_id','classification','disposition','signed_by','signed_on','signature_ref','executable_case_ids')
        if any(not _equal(a.get(k),b.get(k)) for k in protected): return False
        if a.get('signed_by') or a.get('disposition') or a.get('receipt_status_exception'): return False
        if a.get('evidence_tier')!='numeric_non_discriminating' or a.get('evidence',{}).get('non_binding_receipts')!=[OLD]: return False
        if set(a.get('evidence',{})) != {'non_binding_receipts','tier'} or not isinstance(a['evidence'].get('tier'),str): return False
        prior_result=a.get('measured_result')
        if not isinstance(prior_result,dict) or set(prior_result)!={'case_verdicts','batch_verifier','discriminating'}: return False
        if not _equal(prior_result,{'case_verdicts':{CID:'PASS'},'batch_verifier':{CID:'PASS'},'discriminating':{CID:False}}): return False
        if b.get('evidence_tier')!='numeric' or b.get('measured_against')!=PIN or b.get('evidence',{}).get('receipt')!=[NEW]: return False
        if set(b.get('evidence',{})) != {'receipt','tier'} or b['evidence'].get('tier') != TIER or b.get('note') != NOTE: return False
        if b.get('executable_case_ids')!=[CID] or b.get('classification')!='required_core': return False
        result=b.get('measured_result',{})
        if not isinstance(result,dict) or set(result)!={'case_verdicts','batch_verifier','row_verdict'}: return False
        if result.get('row_verdict')!='PASS' or result.get('case_verdicts')!={CID:'PASS'} or result.get('batch_verifier')!={CID:'PASS'}: return False
        if not isinstance(old.get('counts'),dict) or not isinstance(new.get('counts'),dict): return False
        if any(type(v) is not int or v<0 for v in [*old['counts'].values(),*new['counts'].values()]): return False
        if not _equal(_recount(old),old['counts']) or not _equal(_recount(new),new['counts']): return False
        counts=dict(old['counts']); counts['numeric_pass']+=1; counts['numeric_non_discriminating']-=1
        if counts['numeric_non_discriminating']<0 or new['counts']!=counts: return False
        text=before.decode('utf-8'); result_text=after.decode('utf-8')
        digest_old=hashlib.sha256(old_map).hexdigest(); digest_new=hashlib.sha256(new_map).hexdigest()
        old_input=f'- `{MAP}` ({len(oldrows)} rows, sha256 `{digest_old}`)'; new_input=f'- `{MAP}` ({len(newrows)} rows, sha256 `{digest_new}`)'
        if text.count(old_input)!=1: return False
        prefix,rest=text.split('## Totals by family\n\n',1); totals,suffix=rest.split('\n\n## P1 twin fixtures present at this head',1)
        counts_by_family=defaultdict(Counter); rows_start=text.split('## Rows\n\n',1)[1]; row_lines=rows_start.splitlines()[2:]
        seen=set(); old_target=None
        for line in row_lines:
            if not line.startswith('| '): return False
            cells=[c.strip() for c in line.split('|')]
            if len(cells)!=7: return False
            identity=cells[1].split(' ',1)[0]; match=re.match(r'^family ([^;]+);',cells[5])
            if not match or identity in seen or cells[3] not in ORDER: return False
            seen.add(identity); counts_by_family[match.group(1)][cells[3]]+=1
            if identity==BOARD_ID: old_target=line
        if old_target is None or totals!=_render_totals(counts_by_family): return False
        exact_old=f'| {BOARD_ID} `{TARGET}` | required_core; cases: {CID} | NON-DISCRIMINATING | not bound; cited: non-binding {OLD} | family postfit; evidence_tier numeric_non_discriminating; measured_against {PIN} |'
        if old_target != exact_old: return False
        counts_by_family['postfit']['NON-DISCRIMINATING']-=1; counts_by_family['postfit']['EVIDENCED']+=1
        new_target=f'| {BOARD_ID} `{TARGET}` | required_core; cases: {CID} | EVIDENCED | {NEW} | family postfit; evidence_tier numeric; measured_against {PIN} |'
        expected=prefix.replace(old_input,new_input)+'## Totals by family\n\n'+_render_totals(counts_by_family)+'\n\n## P1 twin fixtures present at this head'+suffix
        if expected.count(old_target)!=1: return False
        return expected.replace(old_target,new_target,1)==result_text
    except (KeyError,TypeError,ValueError,UnicodeDecodeError):
        return False
