import numpy as np
import torch
from functorch import make_functional
from torch.func import vmap, jacrev
from scipy.optimize import minimize
import scipy.integrate as integrate
import tqdm
from torch.utils.data import Dataset
torch.set_default_dtype(torch.float64)
'''
Let Zk be a dictionary
Zk = {0: zz_0,
      1: zz_1,
      ...
      L-1: zz_{L-1},
      L: zz_{L}}
with L+1 keys, arranged in ascending order, where
zz_l = z_prev(Sk[l+1], Zk[l+1], b_sigma)
and 
Sk = {1: s1_k,
      ...
      L-1: s{L-1}_k,
      L: s{L}_k}

and gamma = {0: gamma0,
             1: gamma1,
             ,...,
             L: gammaL}.
'''


def m0(z, gamma):
    try:
        return (1-gamma-z)/(2*z*gamma) - np.emath.sqrt(((1+np.emath.sqrt(gamma))**2-z))*np.emath.sqrt(((1-np.emath.sqrt(gamma))**2-z))/(2*z*gamma)  
    except:
        print(f'Exception for {z, gamma}')
        return (1-gamma-z)/(2*z*gamma) - np.emath.sqrt(((1+np.emath.sqrt(gamma))**2-z))*np.emath.sqrt(((1-np.emath.sqrt(gamma))**2-z))/(2*z*gamma) 
    
def m0_tilde(z, gamma):
    return m0(z, gamma) - (1/gamma-1)/z

def z_prev(sl, zzl, b_sigma):
    zp = np.array(zzl[:-1], copy=True) + 0.00j
    zp[0] = zp[0] + (1-b_sigma**2) / sl
    zp[-1] = zp[-1] + b_sigma**2 / sl
    return zp

def w_prev(wwl, zzl):
    wl = wwl[-1]; zl = zzl[-1]
    wp = np.array(wwl[:-1], copy=True) + 0.00j
    zp = np.array(zzl[:-1], copy=True) + 0.00j
    wp = wp - (wl/zl)*zp
    return wp


def t(Zk, ww, l, gamma, verbose=False):
    if verbose:
        print(f'l: {l}, ww: {ww}')
    if l >= 1:
        return ww[-1]/Zk[l][-1] + t(Zk, w_prev(ww,Zk[l]), l-1, gamma, verbose=verbose)
    else:
        zz_0 = Zk[0]
        return t0(zz_0, ww, gamma[0], verbose=verbose)
    
def t0(zz_0, ww_0, gamma0, verbose=False):
    if verbose:
        print(f't0: zz_0: {zz_0}, ww_0: {ww_0}: t0: {(zz_0[1]*ww_0[0] - zz_0[0]*ww_0[1])/(zz_0[1]**2) * m0(-zz_0[0]/zz_0[1], gamma0) + ww_0[1] / zz_0[1]}')
    return (zz_0[1]*ww_0[0] - zz_0[0]*ww_0[1])/(zz_0[1]**2) * m0(-zz_0[0]/zz_0[1], gamma0) + ww_0[1] / zz_0[1]
    
def compute_Zk(Sk, zz_L, L, b_sigma):
    range = np.arange(L-1,-1,-1)
    Zk = {}
    Zk[L] = zz_L
    for i in range:
        Zk[i] = z_prev(np.array(Sk[i+1], copy=True), Zk[i+1], b_sigma)
    return Zk

def update_Sk(Zk, gamma, b_sigma):
    L = len(Zk.keys())
    range = np.arange(L-1,0,-1)
    Sk = {}
    for l in range:
        zzl = Zk[l]
        ww_b = np.zeros((l+1)); ww_b[0] = 1 - b_sigma**2; ww_b[-1] = b_sigma**2
        sl = 1 / zzl[-1] + gamma[l] * t(Zk, ww_b, l-1, gamma)
        Sk[l] = sl
    return Sk

def s0(zz_0, gamma_0):
    return 1 / zz_0[1] + gamma_0*t0(zz_0, ww_0=np.array([1,0]),gamma0=gamma_0)

def solve_fp_S(zz_L, gamma, L, b_sigma, its = 100, tol = 1e-5, verbose=False):
    s_imag = 1e-4j

    # Initialise S0
    range = np.arange(L,0,-1)
    Sk = {}
    for l in range:
        Sk[l] = np.random.standard_normal(1) + s_imag

    # Compute Z0
    Zk = compute_Zk(Sk, zz_L, L, b_sigma)

    rc = 0
    # Solve fixed-point equation
    k = 0
    while k < its and rc < its:
        Sprev = Sk
        Sk = update_Sk(Zk, gamma, b_sigma)
        Zk = compute_Zk(Sk, zz_L, L, b_sigma)


        for l in Sk.keys():
            if Sk[l].imag < 1e-14:
                Sk[l] = Sk[l].real + 0j

        err = dif(Sk,Sprev)
        if err < tol:
            imag_neg = False
            for l in Sk.keys():
                if Sk[l].imag < 0:
                    imag_neg = True
                    
            if not imag_neg:                
                if verbose:
                    print('converged')
                    print(f'err: {err:.4}, rc: {rc}, its: {k}')
                break
            else:
                Sprev = Sk
                if (rc + 1) >= its:
                    break
                Sk = {}
                for l in range:
                    Sk[l] = np.random.standard_normal(1) + s_imag

                # Compute Z0
                Zk = compute_Zk(Sk, zz_L, L, b_sigma)
                rc += 1
                k = 0
                if verbose:
                    print(f'converged to negative, times:{rc}, it was {k}, Sk = {Sprev}')
                
        k += 1
    if err > tol:
        if verbose:
            print(f'DNC ::: err: {err:.4}, rc: {rc}, its: {k}')
    else:
        if verbose:
            print(f'err: {err:.4}, rc: {rc}, its: {k}')

    # Compute final Z* - S* and Z* are jointly-defined.
    Zk = compute_Zk(Sk, zz_L, L, b_sigma)

    return Sk, Zk

def dif(S,S_prev):
    diff = 0
    for s,s_prev in zip(S.values(),S_prev.values()):
        diff += abs(s.real-s_prev.real) + abs(s.imag-s_prev.imag)
    return diff

def compute_tL(zz_L, ww_L, gamma, L, b_sigma, S_its = 10000, S_tol = 1e-24, verbose = False):
    S, Z = solve_fp_S(zz_L, gamma, L, b_sigma, its = S_its, tol = S_tol, verbose=False)
    if verbose:
        print(f'S: {S}')
        print(f'Z: {Z}')
    tL = t(Z, ww_L, L, gamma, verbose=verbose)
    return tL

def D(x, zzl, l, gamma, b_sigma, S_its = 100000, S_tol = 1e-18, verbose = False):
    '''
    x: number
    zzl: numpy.array(complex), of length l+2
    l: layer number, 0 <= l <= L
    gamma: list([gamma0,gamma1,...,gammaL])
    '''
    # Create new zzlx array
    zzlx = np.array(zzl, copy=True) + 0j
    zzlx[0] += x

    if l >= 1:
        S, _ = solve_fp_S(zzlx, gamma[:l+1], l, b_sigma, its = S_its, tol = S_tol, verbose=verbose)
        sl = S[l]
    elif l == 0:
        sl = s0(zz_0=zzlx, gamma_0=gamma[0])
        if verbose:
            print(f's0: {sl}')

    if l >= 1:
        z_new = z_prev(sl, zzl, b_sigma)
        if verbose:
            print(f'C{l}: {C(zzl[-1],sl,gamma[l])}, sl: {sl}, zl: {zzl[-1]}')
        return D(x=x, zzl=z_prev(sl, zzl, b_sigma), l=l-1, gamma=gamma[:l], b_sigma=b_sigma,
                 S_its = S_its, S_tol = S_tol, verbose = verbose) + C(zzl[-1],sl,gamma[l])
    elif l == 0:
        if verbose:
            print(f'C{l}: {C(zzl[-1],sl,gamma[l])}')
        if verbose:
            print(f'final addition: {np.log(x + (1/sl + zzl[0]))}, withoutlog: {(x + (1/sl + zzl[0]))}, x: {x}, sl: {sl}, zzl[0]: {zzl[0]}')
        return np.log(1 + (1/sl + zzl[0])/x) + C(zzl[-1],sl,gamma[l])

def C(zl,sl,gammal):
    return np.log(sl) / gammal + 1 / sl / zl / gammal - 1 / gammal + np.log(zl) / gammal

def ntk_constants(L,b_sigma,a_sigma,verbose=False):
    qq = [0]*(L); rr = [0]*(L); r = 0
    for l in range(L):
        qql = (b_sigma**2)**(L-l)
        rrl = a_sigma**(L-l)
        r += rrl - qql
        qq[l] = qql; rr[l] = rrl
    return qq,rr,r

def ntk_log_determinant(lam,gam,L,gamma,b_sigma,a_sigma,verbose):
    qq,rr,r = ntk_constants(L,b_sigma,a_sigma)
    zz_L = np.concatenate((np.array([r]),np.array(qq),np.array([1])))
    logdet = D(x = lam*gam, zzl=zz_L, l=L, gamma=gamma, b_sigma=b_sigma,
              verbose=verbose, S_its=1000, S_tol=1e-12) + np.log(lam*gam)
    return logdet.real

def ntk_stieltjes(z,L,gamma,b_sigma,a_sigma,verbose):
    qq,rr,r = ntk_constants(L,b_sigma,a_sigma)
    zz_L = np.concatenate((np.array([-z + r]),np.array(qq),np.array([1])))
    ww = np.zeros((L+2)); ww[0] = 1
    mntk = compute_tL(zz_L=zz_L, ww_L=ww, gamma=gamma, L=L,
                    b_sigma=b_sigma, verbose=verbose, S_its=1000, S_tol=1e-12)
    return mntk.real

# Conjugate Kernel Attempt
def ck_log_determinant(lam,gam,L,gamma,b_sigma,a_sigma,verbose):
    zz_L = np.zeros(L+2); zz_L[-1] = 1
    logdet = D(x = lam*gam, zzl=zz_L, l=L, gamma=gamma, b_sigma=b_sigma,
              verbose=verbose, S_its=1000, S_tol=1e-12) + np.log(lam*gam)
    return logdet.real

def ck_stieltjes(z,L,gamma,b_sigma,a_sigma,verbose):
    zz_L = np.zeros(L+2); zz_L[-1] = 1; zz_L[0] -= z
    ww = np.zeros((L+2)); ww[0] = 1
    mntk = compute_tL(zz_L=zz_L, ww_L=ww, gamma=gamma, L=L,
                    b_sigma=b_sigma, verbose=verbose, S_its=1000, S_tol=1e-12)
    return mntk.real

def ck_limiting_energy(lam,gam,L,gamma,b_sigma,a_sigma,verbose=False):
    F1 = ck_stieltjes(-lam*gam,L,gamma,b_sigma,a_sigma,verbose) * lam / 2
    F2 = 1/2 * ck_log_determinant(lam,gam,L,gamma,b_sigma,a_sigma,verbose)
    F3 = -1/2*np.log(lam/2/np.pi)
    F = F1 + F2 + F3
    return F

def limiting_energy(kernel,lam,gam,L,gamma,b_sigma,a_sigma,verbose=False):
    if kernel == 'ck':
        F1 = ck_stieltjes(-lam*gam,L,gamma,b_sigma,a_sigma,verbose=verbose) * lam / 2
        if verbose:
            print(f'Stieltjes: {F1}')
        F2 = 1/2 * ck_log_determinant(lam,gam,L,gamma,b_sigma,a_sigma,verbose=verbose)
        if verbose:
            print(f'D: {F2}')
        F3 = -1/2*np.log(lam/2/np.pi)
        F = F1 + F2 + F3
        return F
    elif kernel == 'ntk':
        F1 = ntk_stieltjes(-lam*gam,L,gamma,b_sigma,a_sigma,verbose) * lam / 2
        if verbose:
            print(f'Stieltjes: {F1}')
        F2 = 1/2 * ntk_log_determinant(lam,gam,L,gamma,b_sigma,a_sigma,verbose)
        if verbose:
            print(f'D: {F2}')
        F3 = -1/2*np.log(lam/2/np.pi)
        F = F1 + F2 + F3
        return F
    elif kernel == 'linear':
        return limit_entropy(alpha = 1, beta = 0, gam = gam, lam = lam, c = 1 / gamma)

def limit_entropy(alpha,beta,gam,lam,c):
    z = (beta + gam*lam)/alpha
    if c < 1:
        T = (c-1-c*z+((c*z+c+1)**2-4*c)**0.5)/(2*z)
        D = np.log(1+T/c)-T/(c+T)-c*np.log(T/c)
        term1 = lam/2*((1-c)/(beta+gam*lam))
        term2 = lam/2*1/alpha*T
        term3 = -1/2*np.log(lam/(2*np.pi*alpha))
        term4 = 1/2*D
        term5 = 1/2*(1-c)*np.log(z)
        return term1+term2+term3+term4+term5
    else:
        T = (1-c-c*z+((c*z+c+1)**2-4*c)**0.5)/(2*z)
        D = c*np.log(1+T/c)-c*T/(c+T)-np.log(T)
        term1 = lam/(2*alpha)*T
        term2 = -1/2*np.log(lam/(2*np.pi*alpha))
        term3 = D/2
        return term1+term2+term3
    
def opt_lambda(alpha,beta,gam,c):
    guess = alpha*((c+1)*gam+((c-1)**2+4*c*gam**2)**0.5)
    guess /= c*(1-gam**2)
    obj = lambda x: limit_entropy(alpha,beta,gam,x,c)
    sol = minimize(obj, guess)
    return sol.x[0]
    
class variable_mlp(torch.nn.Module):
    def __init__(self,layer_width,nonlin):
        super().__init__()
        self.layer_width = layer_width
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layer_width[i],layer_width[i+1], bias=False) for i in range(len(self.layer_width)-1)])
        self.lin_out = torch.nn.Linear(self.layer_width[-1],1, bias=False)
        self.nonlin = nonlin

        for lin in self.linear_layers:
            torch.nn.init.normal_(lin.weight, 0, 1)
        torch.nn.init.normal_(self.lin_out.weight, 0, 1)

    # Return full output of nn
    def forward(self,x):
        for i, lin in enumerate(self.linear_layers):
            x = self.nonlin(lin(x)) / (self.layer_width[i]**0.5)
        return self.lin_out(x)

    # Return output of l-th layer
    def layer_l(self,x,l):
        for i in range(l):
            x = self.nonlin(self.linear_layers[i](x)) / (self.layer_width[i+1]**0.5)
        return x

def nonlin(nl='tanh', return_a=False):
    if nl == 'tanh':
        x = np.random.standard_normal(10000000)
        normalising_constant = np.var(np.tanh(x))
        atanh = lambda x : np.tanh(x) / (normalising_constant**0.5)
        dtanh = lambda x : (1 - np.tanh(x)**2) / (normalising_constant**0.5)
        b_sigma = np.mean(dtanh(x))

        if return_a:
            a_sigma = np.mean(dtanh(x)**2)

        nl_func = lambda x : torch.nn.functional.tanh(x) / (normalising_constant**0.5)

    elif nl == 'gelu':
        x = torch.randn(10000000)

        C = 0.39894228 * 1/(2**0.5)

        normalising_constant = torch.nn.functional.gelu(x).var()
        m = torch.distributions.normal.Normal(0,1)

        dgelu  = lambda x : (x*m.log_prob(x).exp() + (1+torch.erf(x/(2**0.5)))/2) / (normalising_constant**0.5)
        b_sigma = dgelu(x).mean()

        if return_a:
            a_sigma = torch.mean(dgelu(x)**2)
        
        nl_func = lambda x : (torch.nn.functional.gelu(x) - C) / (normalising_constant**0.5)

    elif nl == 'silu':
        xn = np.random.standard_normal(10000000)
        sigmoid = lambda x : 1 / (1 + np.exp(-x))
        silu = lambda x : x * sigmoid(x)

        C = 0.20660121656510527

        normalising_constant = np.var(silu(xn))

        silu_bar = lambda x : (silu(x) - C) / (normalising_constant**0.5)

        dsilu_bar = lambda x : sigmoid(x) * (1 + x * (1 - sigmoid(x))) / (normalising_constant**0.5) 

        b_sigma = np.mean(dsilu_bar(xn))

        if return_a:
            a_sigma = np.mean(dsilu_bar(xn)**2)

        nl_func = lambda x : (torch.nn.functional.silu(x) - C) / (normalising_constant**0.5)

    elif nl == 'sigmoid':
        x = np.random.standard_normal(10000000)
        sigmoid = lambda x : 1 / (1 + np.exp(-x))
        normalising_constant = np.var(sigmoid(x))
        C = 0.5

        dsigmoid_bar = lambda x : sigmoid(x)*(1 - sigmoid(x)) / (normalising_constant**0.5) 
        b_sigma = np.mean(dsigmoid_bar(x))

        if return_a:
            a_sigma = np.mean(dsigmoid_bar(x)**2)

        nl_func = lambda x : (torch.nn.functional.sigmoid(x) - C) / (normalising_constant**0.5) 

    if return_a:
        return nl_func,b_sigma,a_sigma
    else:
        return nl_func,b_sigma
    
def custom_data(X0,Y, L,nl_func,ntk_option, epochs, lr, verbose=False):
    n, d0 = X0.shape
    layerwidth = [d0] + [d0 for l in range(L)]
    net = variable_mlp(layerwidth, nl_func)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    loss_fn = torch.nn.MSELoss()
    for i in range(epochs):
        optimizer.zero_grad()
        pred = net(X0)
        loss = loss_fn(pred,Y.reshape(-1,1))
        loss.backward()
        optimizer.step()

    if verbose and epochs > 0:
        print(f'nn loss : {loss.item():.4}')

    layer_outputs = []
    layer_outputs.append(X0.T)
    for l in np.arange(1,L+1):
        layer_outputs.append(net.layer_l(X0,l).T.detach())

    if ntk_option:
        ## Compute jacobian of net, evaluated on training set
        fnet, params = make_functional(net)
        def fnet_single(params, x):
            return fnet(params, x.unsqueeze(0)).squeeze(0)
        
        J = vmap(jacrev(fnet_single), (None, 0))(params, X0)
        J = [j.detach().flatten(1) for j in J]
        J = torch.cat(J,dim=1).detach()

        ntk_kernel = J @ J.T
        return layer_outputs, net, ntk_kernel

    return layer_outputs, net
    
def data(n,gammalist,gam,nl_name='tanh',epochs=0,verbose=False,ntk_option=False,outputs=False):

    layerwidth = [int(n/gammal) for gammal in gammalist]
    nl_func,b_sigma,a_sigma = nonlin(nl_name,return_a=True)
    X0 = torch.randn((layerwidth[0],n),dtype=torch.float64) / (layerwidth[0]**0.5)
    Y = torch.randn((1,n), dtype=torch.float64)*np.sqrt(gam)

    net = variable_mlp(layerwidth, nl_func)
    if verbose:
        print(f'p = {sum(p.numel() for p in net.parameters() if p.requires_grad)}')

    optimizer = torch.optim.SGD(net.parameters(), lr=1, momentum=0.9, weight_decay=0)
    loss_fn = torch.nn.MSELoss()
    for i in range(epochs):
        optimizer.zero_grad()
        pred = net(X0.T)
        loss = loss_fn(pred,Y.T)
        loss.backward()
        optimizer.step()

    if verbose and epochs > 0:
        print(f'nn loss : {loss.item():.4}')

    layer_outputs = []
    layer_outputs.append(X0.detach())
    for l in np.arange(1,len(gammalist)):
        layer_outputs.append(net.layer_l(X0.T,l).T.detach())

    if ntk_option:
        ## Compute jacobian of net, evaluated on training set
        fnet, params = make_functional(net)
        def fnet_single(params, x):
            return fnet(params, x.unsqueeze(0)).squeeze(0)
        
        J = vmap(jacrev(fnet_single), (None, 0))(params, X0.T)
        J = [j.detach().flatten(1) for j in J]
        J = torch.cat(J,dim=1).detach()

        ntk_kernel = J @ J.T

    if epochs > 0:
        if ntk_option:
            return layer_outputs,b_sigma,a_sigma,loss.item(), ntk_kernel
        else:
            return layer_outputs,b_sigma,a_sigma,loss.item()
    else:
        if ntk_option:
            return layer_outputs,b_sigma,a_sigma, ntk_kernel
        else:
            if outputs:
                return layer_outputs,Y
            else:
                return layer_outputs,b_sigma,a_sigma

def empirical_energy_single(n, gammalist, lam, gam, kernel='ntk', nl_name='tanh', epochs=0, verbose=False, ntk_option=False):

    output = data(n,gammalist,gam,nl_name,epochs,verbose,ntk_option)
    layer_outputs = output[0]; b_sigma = output[1]; a_sigma = output[2]
    if epochs > 0:
        loss = output[3]
        if ntk_option:
            K = output[4]
    else:
        if ntk_option:
            K = output[3]

    if kernel == 'ntk':
        if not ntk_option:
            qq,rr,r = ntk_constants(len(gammalist)-1,b_sigma,a_sigma, verbose) # Number of layers L = len(gammalist)-1 as gammalist includes gamma0, i.e. number of data points : data dimension.
            K = r * torch.eye(n)
            for l in range(len(gammalist)-1):
                K += qq[l]*layer_outputs[l].T @ layer_outputs[l]
            K += layer_outputs[-1].T @ layer_outputs[-1]

    elif kernel == 'ck':
        K = layer_outputs[-1].T @ layer_outputs[-1]

    slogdet1 = torch.linalg.slogdet(K + lam*gam*torch.eye(n))[1] / n
    
    e = torch.linalg.eigvalsh(K)
    inverse_trace = 1/n * torch.sum(1 / (e + lam*gam))

    F1 = inverse_trace * lam / 2
    F2 = 1/2 * slogdet1
    F3 = -1/2*np.log(lam/2/np.pi)
    F = F1 + F2 + F3

    if epochs > 0:
        return F.detach().item(), loss
    elif epochs == 0:
        return F.detach().item()
    
def bayes_entropy(gamma, lam, X, y, mean = 0, scale = 1):
    """Computes the negative log marginal likelihood for Bayesian linear regression
    at temperature gamma, and Gaussian distribution with variance lam. 
    
    - gamma, lam > 0
    - X is n x d data matrix of inputs
    - y is n x 1 data vector of labels
    """
    n, d = X.shape
    
    if d > n:
        jac = X @ X.T / d
        P = (gamma * lam) * np.eye(n) +  jac
        PinvY = np.linalg.solve(P, y)
        numer = -lam * np.dot(y, PinvY)/2
        _, denom = np.linalg.slogdet(P)
        denom = 0.5*denom
        const = 0.5 * n * np.log(lam / (2*np.pi))
        result = numer - denom + const
        
    else:
        jac = X.T @ X / d
        P = jac + lam * gamma * np.eye(d)
        PinvY = np.linalg.solve(P, X.T @ y)
        PinvY = np.dot(y, X @ PinvY)
        numer = -1/(2*gamma) * np.dot(y,y) + 1/(2*gamma*d) * PinvY
        _, denom = np.linalg.slogdet(P)
        denom = 0.5*denom
        const = 0.5 * d * np.log(lam) - 0.5 * n * np.log(2*np.pi) + 0.5*(d-n) * np.log(gamma)
        result = numer - denom + const
    
    return -result

def intlogspace(m0, m1, N):
    return [int(m) for m in np.floor(np.logspace(m0,m1,N))]
    
def empirical_energy_custom_welfords(iterations, 
                                     n, 
                                     dims, 
                                     L, 
                                     lam, 
                                     gam, 
                                     b_sigma, 
                                     a_sigma, 
                                     mX=None, 
                                     mY = None, 
                                     epochs=0, 
                                     lr=1, 
                                     kernel='ntk', 
                                     nl_func=None, 
                                     ntk_option=False, 
                                     trace_approx=False, 
                                     network_mean=False, 
                                     log_det_only=False, 
                                     data_fit_only=False,
                                     verbose=False):
    entropies = np.zeros(len(dims)) # array of zeros of size (number of dimensions)
    entropies_ci = np.zeros(len(dims)) # array of zeros of size (number of dimensions)

    ## Welfords online algorithm
    for idx,d in tqdm.tqdm(enumerate(dims)):
        if verbose:
            print(f'p = {d**2 * (L+1)}, n = {n}, d = {d}, gamma_l = {(n/d):.3}')
        entropy = 0
        M2 = 0
        w1 = torch.randn(d)
        for idy in range(iterations): # iterate through the iteration numbers
            if mX is None:
                X = torch.randn(n,d) / (d**0.5)
                y = torch.randn(n)
            elif mX == 'teacher':
                X = torch.randn(n,d) / (d**0.5)
                y = torch.sin(X @ torch.ones(d))
                y = y - y.mean()
                y = y / y.std()
            else:
                total_n, total_d = mX.shape
                indices_dat = np.random.choice(total_n, n, replace=False)
                #indices_dim = np.random.choice(total_d, dim, replace=False)
                X = mX[indices_dat, :]
                X = torch.from_numpy(X[:, :d]) / (d**0.5)
                y = torch.from_numpy(mY[indices_dat])

            Fn = empirical_energy_custom_single(X=X,y=y,L=L,lam=lam[idx],gam=gam,b_sigma=b_sigma,
                                                a_sigma=a_sigma,kernel=kernel,nl_func=nl_func,epochs=epochs,lr=lr,trace_approx=trace_approx,ntk_option=ntk_option,
                                                network_mean=network_mean, log_det_only=log_det_only, data_fit_only=data_fit_only, verbose=verbose)
            delta = Fn - entropy
            entropy += delta / (idy + 1)
            delta2 = Fn - entropy
            M2 += delta * delta2
        entropies[idx] = entropy
        entropies_ci[idx] = 1.96 * (M2 / (iterations - 1) / iterations)**0.5
    return entropies, entropies_ci

def empirical_energy_custom_single(X, 
                                   y, 
                                   L, 
                                   lam, 
                                   gam, 
                                   b_sigma, 
                                   a_sigma, 
                                   kernel='ntk', 
                                   nl_func=None, 
                                   epochs=0, 
                                   lr = 1, 
                                   verbose=False, 
                                   ntk_option=False, 
                                   trace_approx=False, 
                                   network_mean=False, 
                                   log_det_only=False,
                                   data_fit_only=False):
    n, d = X.shape
    if ntk_option:
        layer_outputs, net, ntk = custom_data(X,y,L,nl_func,ntk_option,epochs,lr,verbose)
    else:
        layer_outputs, net = custom_data(X,y,L,nl_func,ntk_option,epochs,lr,verbose)

    if kernel == 'ntk':
        if not ntk_option:
            qq,rr,r = ntk_constants(L,b_sigma,a_sigma, verbose) # Number of layers L = len(gammalist)-1 as gammalist includes gamma0, i.e. number of data points : data dimension.
            K = r * torch.eye(n)
            for l in range(L):
                K += qq[l]*layer_outputs[l].T @ layer_outputs[l]
            K += layer_outputs[-1].T @ layer_outputs[-1]
        else:
            K = ntk

    elif kernel == 'ck':
        K = layer_outputs[-1].T @ layer_outputs[-1]

    P = K.numpy() + lam*gam*np.eye(n)
    slogdet1 = np.linalg.slogdet(P)[1]
    
    if trace_approx:
        e = torch.linalg.eigvalsh(K)
        inverse_trace = torch.sum(1 / (e + lam*gam))

    else:
        if epochs > 0 and network_mean:
            f = net(X).detach().numpy().reshape(-1)
            inverse_trace = np.dot((y-f), np.linalg.solve(P, (y-f)))
        else:
            inverse_trace = np.dot(y, np.linalg.solve(P, y))

    F1 = inverse_trace * lam / 2
    F2 = 1/2 * slogdet1
    F3 = -1/2*np.log(lam/2/np.pi)*n
    if verbose:
        print(f'data-fit: {F1}, log-det: {F2}, constant: {F3}')

    if log_det_only and data_fit_only:
        raise ValueError("Cannot set both log_det_only and data_fit_only to True.")

    if log_det_only:
        F = F2
    else:
        F = F1 + F2 + F3
    if data_fit_only:
        F = F1
    

    return F
    

def empirical_energy(n, gammalist, lam, gam, samples = 1, kernel='ntk', nl_name='tanh', epochs=0, verbose=False, ntk_option=False):
    energy = []
    if epochs > 0:
        loss = []
    if verbose:
        pbar = tqdm.trange(samples)
    else:
        pbar = range(samples)
    for i in pbar:
        if epochs > 0:
            F, l = empirical_energy_single(n, gammalist, lam, gam, kernel, nl_name, epochs, verbose, ntk_option)
            loss.append(l)
        elif epochs == 0:
            F = empirical_energy_single(n, gammalist, lam, gam, kernel, nl_name, epochs, verbose, ntk_option)
        energy.append(F)

    energy = np.array(energy)
    if epochs > 0:
        loss = np.array(loss)
        return energy.mean(), energy.std(), loss.mean(), loss.std()
    elif epochs == 0:
        return energy.mean(), energy.std()
    
def empirical_energy_custom(X, y, L, lam, gam, samples = 1, kernel='ntk', nl_name='tanh', epochs = 0, verbose=False, ntk_option=False):
    energy = []
    if epochs > 0:
        loss = []
    if verbose:
        pbar = tqdm.trange(samples)
    else:
        pbar = range(samples)
    for i in pbar:
        if epochs > 0:
            F, l = empirical_energy_custom_single(X, y, L, lam, gam, kernel, nl_name, epochs, verbose, ntk_option)
            loss.append(l)
        elif epochs == 0:
            F = empirical_energy_custom_single(X, y, L, lam, gam, kernel, nl_name, epochs, verbose, ntk_option)
        energy.append(F)

    energy = np.array(energy)
    if epochs > 0:
        loss = np.array(loss)
        return energy.mean(), energy.std(), loss.mean(), loss.std()
    elif epochs == 0:
        return energy.mean(), energy.std()


def logdet_ntk(lam,gam,L,gamma,b_sigma,a_sigma,verbose=False):

    qq,rr,r = ntk_constants(L,b_sigma,a_sigma)
    zz_L = np.concatenate((np.array([r]),np.array(qq),np.array([1])))
    DL = D(x = lam*gam, zzl=zz_L, l=L, gamma=gamma, b_sigma=b_sigma,
              verbose=verbose, S_its=1000, S_tol=1e-12).real
    i = integrate.quad(lambda z : ntk_stieltjes(-z,L,gamma,b_sigma,a_sigma,verbose),
                                0, lam*gam, limit=50,points=0)[0]
    return DL - i + np.log(lam*gam)

def nn_test(n, layerwidth, nl_name, epochs, lr, lam, gam, verbose=False):
    nl_func,_,_ = nonlin(nl_name,return_a=True)

    X_train = torch.randn((layerwidth[0],n),dtype=torch.float64) / (layerwidth[0]**0.5)

    Y_train = torch.randn((1,n), dtype=torch.float64)*np.sqrt(gam)

    X_test = torch.randn((layerwidth[0],n),dtype=torch.float64) / (layerwidth[0]**0.5)
    Y_test = torch.randn((1,n), dtype=torch.float64)*np.sqrt(gam)

    net = variable_mlp(layerwidth, nl_func)

    # Train network
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=lam)
    loss_fn = torch.nn.MSELoss()
    for i in range(epochs):
        optimizer.zero_grad()
        pred = net(X_train.T)
        loss = loss_fn(pred,Y_train.T)
        loss.backward()
        optimizer.step()
        if verbose:
            print(loss.item())

    train_loss = loss.item()

    # Test network
    test_preds = net(X_test.T)
    test_loss = loss_fn(test_preds, Y_test.T).item()

    return train_loss, test_loss



def ck_limiting_case_small(lam,gam,gamma,b_sigma):
    L = len(gamma)-1
    x = lam*gam

    b_sum = 0
    for i in range(L):
        b_sum += (b_sigma**2)**i
    b_sum *= (1-b_sigma**2)

    F1 = (b_sigma**2)**(-L) * m0(-(x + b_sum)/(b_sigma**2)**L, gamma[0]) * lam / 2
    F1r = (b_sigma**2)**(-L) * 1 / (1 + (x + b_sum)/(b_sigma**2)**L)  * lam / 2
    F1r2 = 1 / (1 + x) * lam / 2
    F2 = 1/2 * (np.log(1 + (b_sum + (b_sigma**2)**L)/x) + np.log(x))
    F2r = 1/2 * (np.log(1 + 1/x) + np.log(x))
    F3 = -1/2*np.log(lam/2/np.pi)
    F = F1 + F2 + F3
    Fr = lam/2 * 1 / (1 + lam*gam) + 1/2*np.log(1 + lam*gam)+F3
    F = 1/2 * ()
    return F


def ck_limiting_case_large(lam,gam,gamma,b_sigma):
    L = len(gamma)-1
    x = lam*gam

    F1 = m0(-x,gamma[0]) * lam / 2
    
    F2 = 1/2 * np.log(x)
    F3 = -1/2*np.log(lam/2/np.pi)
    F = F1 + F2 + F3
    return F

def ck_limiting(type,lam,gam):
    if type=='small':
        x = lam / (1 + lam*gam)
    elif type == 'large':
        x = 1 / gam
    return 1/2 * (x - np.log(x) + np.log(2*np.pi))

def ntk_limiting(type,lam,gam,L,b_sigma,a_sigma):
    qq,rr,r = ntk_constants(L,b_sigma,a_sigma)
    if type=='small':
        x = lam / (sum(rr)+1+lam*gam)
    elif type=='large':
        x = lam / (r + lam*gam)
    return 1/2 * (x - np.log(x) + np.log(2*np.pi))
    







    