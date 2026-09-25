import re, os, glob
root='/Users/z3437171/local-scratch/gllvm-nb2-finite-20260924/src'
files=sorted(glob.glob(root+'/**/*.jl', recursive=True))
out=[]
for f in files:
    lines=open(f).read().split('\n')
    i=0
    while i < len(lines):
        m=re.match(r'^function (_?fit_[A-Za-z0-9_]+)', lines[i])
        if m:
            name=m.group(1); start=i
            j=i+1
            while j < len(lines) and not re.match(r'^end\b', lines[j]): j+=1
            body='\n'.join(lines[start:j+1])
            tags=[]
            if '_fit_verdict' in body: tags.append('FV')
            if 'Optim.converged' in body: tags.append('OC')
            if 'g_residual' in body: tags.append('GRES')
            if re.search(r'_[a-z]+_verdict\(', body.replace('_fit_verdict','')): tags.append('OWNV:'+','.join(set(re.findall(r'(_[a-z_]+_verdict)\(', body.replace('_fit_verdict','')))))
            if 'Optim.optimize' in body: tags.append('OPT')
            m2=re.findall(r'autodiff\s*=\s*:(\w+)', body)
            if m2: tags.append('AD:'+','.join(sorted(set(m2))))
            ls=re.findall(r'Optim\.(LBFGS|BFGS|NelderMead|Newton|NewtonTrustRegion|ConjugateGradient|GradientDescent|Fminbox|IPNewton)', body)
            if ls: tags.append('ALG:'+','.join(sorted(set(ls))))
            calls=sorted(set(re.findall(r'\b(_?fit_[A-Za-z0-9_]+)\(', body))-{name})
            out.append(f"{os.path.relpath(f,root)}:{start+1}-{j+1} {name} [{' '.join(tags)}] calls={','.join(calls[:6])}")
            i=j
        i+=1
print('\n'.join(out))
