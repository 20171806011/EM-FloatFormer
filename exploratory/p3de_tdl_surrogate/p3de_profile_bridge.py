"""Independent CPU TDL-profile OFDM *surrogate*, NOT NVIDIA Sionna runtime.

The normalized tap tables are transcribed from the NVIDIA Sionna v2.2.0
3GPP TR38.901 V19.2 JSON parameter files, Apache-2.0. This implements a
narrow static SISO frequency-domain fractional-delay profile. It does NOT
implement full Sionna PHY/3GPP CDL, Doppler, temporal fading, or 3GPP numerology.
No real radios, emitters, RF attacks, adversarial interference or hardware use.
"""
from dataclasses import dataclass
import math
import torch

N=32
DELTA_F=30000.0
CP_SAMPLES=16
SOURCE_PREFIX='https://raw.githubusercontent.com/NVlabs/sionna/v2.2.0/src/sionna/phy/channel/tr38901/models/v19_2/'
TDL={
 'A': {'delays':[0.0,.3819,.4025,.5868,.4610,.5375,.6708,.5750,.7618,1.5375,1.8978,2.2242,2.1718,2.4942,2.5119,3.0582,4.0810,4.4579,4.5695,4.7966,5.0066,5.3043,9.6586],
       'powers':[-13.4,0,-2.2,-4,-6,-8.2,-9.9,-10.5,-7.5,-15.9,-6.6,-16.7,-12.4,-15.2,-10.8,-11.3,-12.7,-16.2,-18.3,-18.9,-16.6,-19.9,-29.7], 'los':False},
 'C': {'delays':[0,.2099,.2219,.2329,.2176,.6366,.6448,.656,.6584,.7935,.8213,.9336,1.2285,1.3083,2.1704,2.7105,4.2589,4.6003,5.4902,5.6077,6.3065,6.6374,7.0427,8.6523],
       'powers':[-4.4,-1.2,-3.5,-5.2,-2.5,0,-2.2,-3.9,-7.4,-7.1,-10.7,-11.1,-5.1,-6.8,-8.7,-13.2,-13.9,-13.9,-15.8,-17.1,-16,-15.7,-21.6,-22.8], 'los':False},
 'D': {'delays':[0,0,.035,.612,1.363,1.405,1.804,2.596,1.775,4.042,7.937,9.424,9.708,12.525],
       'powers':[-.2,-13.5,-18.8,-21,-22.8,-17.9,-20.1,-21.9,-22.9,-27.8,-23.6,-24.8,-30,-27.7], 'los':True},
}
SCENARIOS={
 'train_A_500ns':('A',.5e-6),
 'train_C_500ns':('C',.5e-6),
 'test_C_1000ns':('C',1e-6),
 'test_D_750ns':('D',.75e-6),
}
TRAIN_SCENARIOS=('train_A_500ns','train_C_500ns')
TEST_SCENARIOS=('train_A_500ns','test_C_1000ns','test_D_750ns')
SNRS=(0,10,20)

def profile_data(name):
    model,delay_spread=SCENARIOS[name]
    raw=TDL[model]
    delays=torch.tensor(raw['delays'],dtype=torch.float64)*delay_spread
    db=torch.tensor(raw['powers'],dtype=torch.float64)
    powers=torch.pow(10.,db/10.)
    powers=powers/powers.sum()
    return model,delays,powers

def covariance(name):
    _,delay,powers=profile_data(name)
    k=torch.arange(N,dtype=torch.float64)-N//2
    F=torch.exp((-2j*math.pi*DELTA_F)*k[:,None]*delay[None,:])
    cov=(F*powers[None,:])@F.conj().T
    return cov.to(torch.complex64)

def rands_complex(shape,generator):
    return torch.complex(torch.randn(shape,generator=generator),torch.randn(shape,generator=generator))/math.sqrt(2)

@dataclass
class Batch:
    yp:torch.Tensor
    yd:torch.Tensor
    h:torch.Tensor
    bits:torch.Tensor
    noise_var:torch.Tensor

def draw(batch,scenario,snr,gen):
    model,delay,powers=profile_data(scenario)
    a=rands_complex((batch,len(powers)),gen)*(powers.float().sqrt())[None,:]
    if TDL[model]['los']:
        # The first TDL-D tap is a specular LoS path with a random frame phase;
        # all other entries are circular Gaussian. This is a static surrogate,
        # not Sionna's time-evolving Rician Doppler implementation.
        phi=torch.rand((batch,),generator=gen)*2*math.pi
        a[:,0]=torch.polar(powers[0].float().sqrt().expand(batch),phi)
    freq=torch.arange(N,dtype=torch.float64)-N//2
    F=torch.exp((-2j*math.pi*DELTA_F)*freq[:,None]*delay[None,:]).to(torch.complex64)
    h=a@F.T
    bits=torch.randint(0,2,(batch,N,2),generator=gen)
    qpsk=torch.complex(2*bits[:,:,0].float()-1,2*bits[:,:,1].float()-1)/math.sqrt(2)
    if isinstance(snr,tuple):
        snr_db=torch.empty(batch,1).uniform_(snr[0],snr[1],generator=gen)
    else:
        snr_db=torch.full((batch,1),float(snr))
    noise_var=10.**(-snr_db/10.)
    std=(noise_var/2).sqrt()
    noise_p=torch.complex(torch.randn((batch,N),generator=gen)*std,torch.randn((batch,N),generator=gen)*std)
    noise_d=torch.complex(torch.randn((batch,N),generator=gen)*std,torch.randn((batch,N),generator=gen)*std)
    return Batch(h+noise_p,h*qpsk+noise_d,h,bits,noise_var)

def baseline_channel(yp,n0,method,train_cov=None,test_cov=None):
    if method=='LS':return yp
    if method.startswith('DFT'):
        taps=int(method[3:]); ht=torch.fft.ifft(yp,dim=-1);ht[:,taps:]=0
        return torch.fft.fft(ht,dim=-1)
    if method=='Threshold':
        ht=torch.fft.ifft(yp,dim=-1)
        t=(n0/N*math.log(N/.05)).sqrt()
        return torch.fft.fft(ht*(ht.abs()>t),dim=-1)
    if method in ('Train-LMMSE','Test-LMMSE'):
        c=(train_cov if method=='Train-LMMSE' else test_cov)
        eye=torch.eye(N,dtype=c.dtype)
        # Observations SISO, known all-carrier Xpilot=1; solve posterior mean.
        m=c[None,:,:]+n0[:,:,None]*eye
        z=torch.linalg.solve(m,yp.unsqueeze(-1)).squeeze(-1)
        return (c@z.T).T
    raise ValueError(method)

def decisions(yd,h,n0):
    z=yd*h.conj()/(h.abs().square()+n0+1e-4)
    return torch.stack((z.real>0,z.imag>0),dim=-1)

def audit_math():
    for name in SCENARIOS:
        model,d,p=profile_data(name)
        assert len(d)==len(p) and abs(float(p.sum())-1)<1e-12
        assert float(d.max())<CP_SAMPLES/(N*DELTA_F),name
        R=covariance(name)
        assert (R-R.conj().T).abs().max()<1e-5
        assert torch.linalg.eigvalsh(R).min()>-2e-5
        assert abs(float(R.diag().real.mean())-1)<1e-5
    # exact integer-delay cyclic transfer vs convolution theorem independent of profile
    g=torch.Generator().manual_seed(20)
    h=rands_complex((4,5),g)
    x=rands_complex((4,N),g)
    y_f=torch.fft.ifft(torch.fft.fft(h,n=N,dim=-1)*torch.fft.fft(x,dim=-1),dim=-1)
    y_t=torch.zeros_like(x)
    for d in range(5):y_t+=h[:,d,None]*torch.roll(x,d,dims=-1)
    assert float((y_f-y_t).abs().max())<1e-5
    return True

if __name__=='__main__':print('Profile bridge math PASS',audit_math())
