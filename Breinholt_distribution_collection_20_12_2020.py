# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 20:43:39 2020

@author: christian Sigvald Breinholt
email: christian.s.breinholt@gmail.com
email2: christian.breinholt@sund.ku.dk
linkedIn: https://dk.linkedin.com/in/christian-sigvald-breinholt

I am always eager to help and answer questions. So reach out if you need help or sparring.

This module contains a range of distributions, here is the list:
    
    Bivariate distributions on the torus:
        BivariateVonMisesSineModel
            BivariateVonMisesSineModel_NNDIST
        SineSkewedBivariateVonMisesSineModel
            SineSkewedBivariateVonMisesSineModel_NNDIST
            SineSkewedBivariateVonMisesSineModel_HYBRID
            
    The ones suffixed by _NNDIST, are identical to their non suffix counterpart, but all parameters
    are given on range zero to one which then gets reparametrized. These distributions are ideal
    for Neural networks and such...
    
    the one suffixed with _HYBRID works like the normal ones except the following parameters should be on scale 0 to 1:
        L1, L2, b_scale, lam_w
        
        and lam_w is used instead of w or lam as in the other distributions

      
The foundation was build in collaboration with Kasper (will add last name after getting accept from him)
and under supervision of Thomas Wim Hamelryck, with guidance from Ahmad Salim Al_Sibahi.


"""
import math
import torch
import pyro
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from pyro.distributions import VonMises, TorchDistribution, Normal, Uniform
from torch.autograd import Variable

pyro.enable_validation(True)
pyro.clear_param_store()

def log_im_multi(order, x): ## x is a parameter, like k1 or k2
    
    # Based on '_log_modified_bessel_fn'
    # Tanabe, A., Fukumizu, K., Oba, S., Takenouchi, T., & Ishii, S. (2007). 
    # Parameter estimation for von Mises–Fisher distributions. Computational Statistics, 22(1), 145-157.
    """ terms to sum over, 10 by 'shape of x' and sums over the first dimension """
    """ vectorized logarithmic Im """
    """ This version is used when needed to initialize several C_inv at once. if parameters are an array instead of single value - Christian Breinholt"""
    
    
    x = x.unsqueeze(-1)
    s = torch.arange(0 , int(5*50+1)).reshape(251, 1).float()
    fs = 2 * s * (x.log() - math.log(2)) - torch.lgamma(s + 1) - torch.lgamma(order + s + 1)
    x= x.squeeze(-1)
    
    return order * (x.log() - math.log(2)) + fs.logsumexp(-2)

def log_im(order, x): ## x is a parameter, like k1 or k2
    
    # Based on '_log_modified_bessel_fn'
    # Tanabe, A., Fukumizu, K., Oba, S., Takenouchi, T., & Ishii, S. (2007). 
    # Parameter estimation for von Mises–Fisher distributions. Computational Statistics, 22(1), 145-157.
    """ terms to sum over, 10 by 'shape of x' and sums over the first dimension """
    """ vectorized logarithmic Im """
    s = torch.arange(0 , int(5*50+1)).reshape(int(5*50+1), 1).float()    
    fs = 2 * s * (x.log() - math.log(2)) - torch.lgamma(s + 1) - torch.lgamma(order + s + 1)
    
    return order * (x.log() - math.log(2)) + fs.logsumexp(-2)




def log_binom(n, k):
    """ Returns of the log of n choose k. """
    
    return torch.lgamma(n+1) - ( torch.lgamma(k+1) + torch.lgamma((n-k)+1))

def logCinv(k1, k2, lam, terms):
    
    # Harshinder Singh, Vladimir Hnizdo, and Eugene Demchuk
    # Probabilistic model for twodependent circular variables.
    # Biometrika, 89(3):719–723, 2002.
    """
    Closed form expression of the normalizing constant
    Vectorized and in log-space
    
    k1, k2 & lam is the parameters from the bivariate von Mises
    
    Since the closed expression is an infinite sum, 'terms' is the number
    of terms, over which the expression is summed over. Estimation by convergence.
    """
    
    # figure out if lambda needs to be with abs or not... I don't think it should
    #lam = abs(lam) + 0.00000001 #gives problems if lam == 0.
    lam += 0.00000001 #gives problems if lam == 0.
    m = torch.arange(0, terms).float()
    
    if k1.shape != (): # this triggers if k1 is NOT instantiated as k1 = torch.tensor(xxxx). In most use cases k1 is instatiatated as such: k1 = torch.tensor([100.]) or as k1 = torch.ones([100,1]).... the difference is the brackets
        logC = log_binom(2*m, m) + m*((2*lam.log()) - (4*k1*k2).log()) + log_im_multi(m, k1) + log_im_multi(m, k2)
        
        return (math.log(4) + 2* math.log(math.pi) + logC.logsumexp(-1)).unsqueeze(-1)
    
    else: # if k1 is instantiated without brackets, ie zero dimensions... k1 = torch.tensor(100.) and NOT k1 = torch.tensor([100.]) or k1 = torch.ones([100,1])
        logC = log_binom(2*m, m) + m*((2*lam.log()) - (4*k1*k2).log()) + log_im(m, k1) + log_im(m, k2)
        return (math.log(4) + 2* math.log(math.pi) + logC.logsumexp(-1)).reshape(1)

def bfind(eig):
    # John  T  Kent,  Asaad  M  Ganeiber,  and  Kanti  V  Mardia.  
    # A new  unified  approach  forthe simulation of a wide class of directional distributions.
    # Journal of Computational andGraphical Statistics, 27(2):291–301, 2018.
    """
    Estimates b0, as being the solution to equation 3.6 in the article mentioned above.
    """
    #print('bfind function')
    q = eig.shape[0]
    if ((eig)**2).sum() == 0.:
        return torch.tensor(q).float()
    else:
        lr = 1e-4
        b = torch.tensor(1., requires_grad=True)
        for _ in range(1000):
            F = torch.abs( 1 - torch.sum(1 / (b + 2 * eig)) )
            F.backward()
            b.data -= lr * b.grad
            b.grad.zero_()
        return b.data

def nsim_check(nsim):
    """nsim_check... deprecated"""
    
    try:
        if nsim == torch.Size([]):
            return(torch.tensor([1]))
    except:
        return nsim
    
def _acg_bound(nsim, k1, k2, lam, mtop = 1000):
    # John  T  Kent,  Asaad  M  Ganeiber,  and  Kanti  V  Mardia.  
    # A new  unified  approach  forthe simulation of a wide class of directional distributions.
    # Journal of Computational andGraphical Statistics, 27(2):291–301, 2018.
    
    """
    Sampling approach used in Kent et al. (2018)
    Samples the cartesian coordinates from bivariate ACG: x, y
    Acceptance criterion:
        - Sample values v, from uniform between 0 and 1
        - If v < fg, accept x, y
    Convert x, y to angles phi using atan2,
    we have now simulated the bessel density.
    """
    ntry = 0; nleft = nsim; mloop = 0
    eig = torch.tensor([0., 0.5 * (k1 - (lam)**2/k2)]); eigmin = 0
    
    if eig[1] < 0:
        eigmin = eig[1]; eig = eig - eigmin

    q = 2; b0 = bfind(eig)
    phi = 1 + 2*eig/b0; den = log_im(0, k2)
    values = torch.empty(nsim, 2); accepted = 0
    
    while nleft > 0 and mloop < mtop:
        x = Normal(0., 1.).sample((nleft*q,)).reshape(nleft, q) *  torch.ones(nleft, 1) * ((1/phi).sqrt()).reshape(1, q)
        r = (x*x).sum(-1).sqrt()
        # Dividing a vector by its norm, gives the unit vector
        # So the ACG samples unit vectors?
        x = x / (r.reshape(nleft, 1) * torch.ones(1, q))
        u = ((x*x) * torch.ones(nleft, 1) * eig.reshape(1, q)).sum(-1)
        v = Uniform(0, 1).sample((nleft, ))
        # eq 7.3 + eq 4.2
        logf = (k1*(x[:,0] - 1) + eigmin) + (log_im(0, torch.sqrt((k2)**2 + (lam)**2 * (x[:,1])**2 )) - den )
        # eq 3.4
        loggi = 0.5 * (q - b0) + q/2 * ((1+2*u/b0).log() + (b0/q).log())
        logfg = logf + loggi

        ind = (v < logfg.exp())
        nacc = ind.sum(); nleft = nleft - nacc; mloop = mloop + 1; ntry=ntry+nleft
        if nacc > 0:
            start = accepted
            accepted += x[ind].shape[0] 
            values[start:accepted,:] = x[ind,:]

    #print("Sampling efficiency:", (nsim - nleft.item())/ntry.item())

    return torch.atan2(values[:,1], values[:,0])

def get_marg_cond(sample_shape, mu, nu, k1, k2, lam):
        # Sampling from the marginal distribution
        marg = _acg_bound(sample_shape, k1, k2, lam)
 
        # Applying the mean angle
        marg = (marg + mu + math.pi) % (2 * math.pi) - math.pi

        # Sampling from the conditional distribution
        alpha = torch.sqrt( (k2)**2 + (lam)**2 * (torch.sin(marg - mu))**2 )
        beta = torch.atan( lam / k2 * torch.sin(marg - mu) )
 
        cond = pyro.sample("psi", VonMises(nu + beta, alpha))
        #print('marg: ', marg)
        return (torch.stack( (marg, cond) )).T


class BivariateVonMisesSineModel(TorchDistribution):
    
    """
    Bivariate von Mises distribution on the torus
    
    Modality:
        If lam^2 / (k1*k2) > 1, the distribution is bimodal, otherwise unimodal.
            - This distribution is only defined for some 'slightly' bimodal cases (alpha < -7)
    
    :param torch.Tensor mu, nu: an angle in radians.
        - mu & nu can be any real number but are interpreted as 2*pi
    :param torch.Tensor k1, k2: concentration parameter
        - This distribution is only defined for k1, k2 > 0
    :param torch.Tensor lam: correlation parameter
        - Can be any real number, but is not defined for very bimodal cases
        - See 'Modality' above
        
    :param torch.Tensor w: reparameterization parameter
        - Has to be between -1 and 1
    """
    
    arg_constraints = {'mu': constraints.real, 'nu': constraints.real,
                       'k1': constraints.positive, 'k2': constraints.positive, 
                       'lam': constraints.real}
    support = constraints.real
    has_rsample = False
    
    def __init__(self, mu, nu, k1, k2, lam=None, w=None, validate_args=None):
        
        
        if (lam is None) == (w is None):
            
            raise ValueError("Either `lam` or `w` must be specified, but not both.")
        
        
        elif w is None:
            self.mu, self.nu, self.k1, self.k2, self.lam = broadcast_all(mu, nu, k1, k2, lam)
            
            #### The following is a check for bimodality:
            
            alpha = (self.k1 - (self.lam)**2 / self.k2) / 2 ## parenthasis around (lam)**2 important, because in python: -3**2 = -9, but (-3)**2 = 9, which is what is correct
            
            if alpha.shape == ():
                if (alpha >= -7) == False: #is triggered if any value in alpha is smaller than -7
                    raise ValueError("Distribution is too bimodal or has too high concentration while being bimodal.")
        
            else: # if alpha has dimension greater than zero. eg when dist is instantiated with parameters of shape 100 for example
                if any( (alpha >= -7) ) == False: #is triggered if any value in alpha is smaller than -7
                    raise ValueError("Distribution is too bimodal or has too high concentration while being bimodal.")
            
        
        elif lam is None:
            self.mu, self.nu, self.k1, self.k2, self.w = broadcast_all(mu, nu, k1, k2, w)
            self.lam = torch.sqrt(self.k1 * self.k2) * self.w
                
        batch_shape = self.mu.shape
        event_shape = torch.Size([2])
        
        self.logC = logCinv(self.k1, self.k2, self.lam, 50)
        
        super().__init__(batch_shape, event_shape, validate_args)
    
    
    
    def log_prob(self, phi_psi):
        # Actual likelihood function
        """ log Joint distribution of phi and psi """
        
        # if the distribution itself is non-dimensional...
        if self.batch_shape == ():
            phi = phi_psi.T[0]
            psi = phi_psi.T[1]
            logProb = (self.k1 * torch.cos(phi - self.mu) + self.k2 * torch.cos(psi - self.nu) + 
                self.lam * torch.sin(phi - self.mu) * torch.sin(psi - self.nu)) - self.logC
            if phi_psi.shape == (2,): # this indicates that both sample_shape and batch_shape is empty and thus log_prob should return log_prob with shape == batch_shape == ()
                assert logProb[0].shape == ()
                return logProb[0] # indexing in because shape must be same as batch_shape
            else: #if batch_shape is non-dimensional, but sample_shape is not 
                
                assert logProb.shape == torch.Size([phi_psi.shape[0]]) # this is sample_shape, because we know batch_shape is empty and event_shape is 2, but will be in location [1]
                return logProb #logProb has shape sample_shape with non dimensioanal batch_shape
        
        else: #if batch_shape has dimensions... eg. if mu = torch.tensor([1]) or mu = torch.ones([100,1])
            phi = phi_psi.T[0].T
            psi = phi_psi.T[1].T
                
            return(self.k1 * torch.cos(phi - self.mu) + self.k2 * torch.cos(psi - self.nu) + 
                    self.lam * torch.sin(phi - self.mu) * torch.sin(psi - self.nu)) - self.logC
    
    

        

    def sample(self, sample_shape=torch.Size()):
        #print('sample function')
        # Harshinder Singh, Vladimir Hnizdo, and Eugene Demchuk
        # Probabilistic model for twodependent circular variables.
        # Biometrika, 89(3):719–723, 2002.
        """
        marg: marginal distribution (using _acg_bound())
        cond: conditional distribution using a modified univariate von Mises
            - as described in Singh et al. (2002)
        """
        
        # sampling for the zero dimensional distribution case:
        if  self.batch_shape == (): # if it's just a zero dimensional distribution instantiated with mu = torch.tensor(1.) instead of mu = torch.tensor([1.]) or mu = torch.ones([100,1])
            
            # Sampling from the marginal distribution
            
            if sample_shape == (): # _acg_bound can't work with an empty nsim
                the_sample = get_marg_cond(1, self.mu, self.nu,  self.k1, self.k2, self.lam)
                return the_sample[0] # indexing in because the sample is for a distribution with a zero dimensional batch_size.... batch_size == ()
        
            else:
                the_sample = get_marg_cond(sample_shape[0], self.mu, self.nu,  self.k1, self.k2, self.lam)
                return the_sample # shape: sample_shape, batch_shape, event_shape
            
        
        # sampling for distribution with dimensions
        
        elif  self.batch_shape == (1,): # if it's just a single distribution instantiated with mu = torch.tensor([1.])
            # Sampling from the marginal distribution
            
            if sample_shape == (): # _acg_bound can't work with an empty nsim
                the_sample = get_marg_cond(1, self.mu, self.nu,  self.k1, self.k2, self.lam)
                return the_sample # do not need to index in as above because there is an extra layer now due to batch_size
            else:
                the_sample = get_marg_cond(sample_shape[0], self.mu, self.nu,  self.k1, self.k2, self.lam)
                return the_sample.reshape(sample_shape[0], 1, 2) # must have shape: sample_shape, batch_shape, event_shape 
        
        
        
        elif len(self.batch_shape) == 2: # if mu = torch.ones([100,1]) and others
            # Christian Thygesen wants to be able to sample with these shapes:
                # shape = [100,1]
                #k1 = torch.tensor(torch.ones(shape)*0.5)
                # k2....
                # mu.. nu..., lam...
                # create dost with those shapes (100,1)
                # dist.sample()
                
                # remember: [sample_shape, batch_shape, eventshape]
                # thus if parameters (mu etc) has shape [100,1] and we want 5 samples, then
                # sample.shape == [5, 100, 1, 2]
        
            assert(self.batch_shape[-1] == 1) ## making sure it's a column tensor... ie shape = [x,1]
            if sample_shape == () or sample_shape == (1,): # empty or a single sample
                
                sample_batch_all = torch.ones([self.batch_shape[0],self.batch_shape[1],2]) # torch ones, this way we can see if there is an issue as samples will be overrepresented with ones
                #sample_batch_all[:,:,:] = np.nan
                
                for dist_nr in range(self.batch_shape[0]):
                    k1 = self.k1[dist_nr]
                    k2 = self.k2[dist_nr]
                    lam = self.lam[dist_nr]
                    mu = self.mu[dist_nr]
                    nu = self.nu[dist_nr]
                    
                    a_sample = get_marg_cond(1, mu, nu,  k1, k2, lam)
                    sample_batch_all[dist_nr] = a_sample[0]
                    
                if sample_shape == (1,): #sample shape of one
                    return sample_batch_all.reshape([1,self.batch_shape[0], self.batch_shape[1], 2])
                else: # non dimensional sample shape... ie no sample
                    return sample_batch_all # shape = [batch_shape, 2], but remember that batch_shape in this case == [x,1], thus the sample shape will be == [x,1,2]
            
            
            else: #if sampleshape is greater than 1... not including 1 because that is dealt with above
                assert(len(sample_shape) == 1) # is not implemented for multidimensional sampling!!!
                
                # remember: [sample_shape, batch_shape, eventshape]
                # thus if parameters (mu etc) has shape [100,1] and we want 5 samples, then
                # sample.shape == [5, 100, 1, 2]
                
                # here we also have sample_shape to take into consideration:
                sample_batch_all = torch.ones([sample_shape[0], self.batch_shape[0],self.batch_shape[1],2]) # torch ones, this way we can see if there is an issue as samples will be overrepresented with ones
                #sample_batch_all[:,:,:] = np.nan
                
                for dist_nr in range(self.batch_shape[0]):
                    k1 = self.k1[dist_nr]
                    k2 = self.k2[dist_nr]
                    lam = self.lam[dist_nr]
                    mu = self.mu[dist_nr]
                    nu = self.nu[dist_nr]

                    a_sample = get_marg_cond(sample_shape[0], mu, nu,  k1, k2, lam)
                    
                    for sample_nr in range(sample_shape[0]):
                        sample_batch_all[sample_nr, dist_nr] = a_sample[sample_nr]
                return sample_batch_all # shape = [batch_shape, 2], but remember that batch_shape in this case == [x,1], thus the sample shape will be == [x,1,2]
        
        
        else:
            print(f'Sampler not implemented for distribution with batch_shape {self.batch_shape}')
            return None

    def expand(self, batch_shape):
        
        try:
            return super().expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            mu = self.mu.expand(batch_shape)
            nu = self.nu.expand(batch_shape)
            k1 = self.k1.expand(batch_shape)
            k2 = self.k2.expand(batch_shape)
            lam = self.lam.expand(batch_shape)
            
            # I have a suspicion we need the C_inv here as well...
            
            return type(self)(mu, nu, k1, k2, lam, validate_args=validate_args)
    




def skew_samples(Y, sample_shape, mu, nu, L1, L2 ):
    
                
    Y1_list = torch.empty((sample_shape))
    Y2_list = torch.empty((sample_shape))
    
    for i in range(sample_shape[0]):
        Y1 = Y[i,0]
        Y2 = Y[i,1]
        
        U = pyro.sample("U", pyro.distributions.Uniform(0., 1.))
        
        # Y1:
        if U <= ( ( 1 + L1 * torch.sin(Y1 - mu) + L2 * torch.sin(Y1 - nu ) ) / 2):
            Y1_new = Y1
        else:
            Y1_new = -Y1 + 2*mu
        
        #Y2:
        if U <= ( ( 1 + L1 * torch.sin(Y2 - mu) + L2 * torch.sin(Y2 - nu ) ) / 2):
            Y2_new = Y2
        else:
            Y2_new = -Y2 + 2*nu
        
        Y1_list[i] = Y1_new
        Y2_list[i] = Y2_new

    

    return (torch.stack( (Y1_list, Y2_list) )).T






class SineSkewedBivariateVonMisesSineModel(BivariateVonMisesSineModel):
    """
    Sine skewed Bivariate von Mises distribution on the torus
    
    This distribution is a child class of Bivariate Von Mises (unimodal) Sine model
    
    Modality:
        If lam^2 / (k1*k2) > 1, the distribution is bimodal, otherwise unimodal.
            - This distribution is only defined for some 'slightly' bimodal cases (alpha < -7)
    
    :param torch.Tensor L1, L2: skew-factors.
        - L1 & L2 has to be real numbers and comply with: |L1| + |L2| <= 1.0

    
    :param torch.Tensor mu, nu: an angle in radians.
        - mu & nu can be any real number but are interpreted as 2*pi

    :param torch.Tensor k1, k2: concentration parameter
        - This distribution is only defined for k1, k2 > 0

    :param torch.Tensor lam: correlation parameter
        - Can be any real number, but is not defined for very bimodal cases
        - See 'Modality' above
        
    :param torch.Tensor w: reparameterization parameter
        - Has to be between -1 and 1
    
    
    This distribution is build from the article 'Sine-skewed toroidal distributions 
        and their application in protein bioinformatics'
    Authors of the article: Jose Ameijeiras-Alonso and  Christophe Ley, from KU leuven and Ghent University.
    
    This distribution was written by Christian Sigvald Breinholt from Copenhagen University, 
        under supervision of Associate Professor Thomas Wim Hamelryck.
    """
    arg_constraints = {'L1': constraints.real, 'L2': constraints.real} 
    ## The other parameter constraints are in the parent class
    
    
    support = constraints.real
    has_rsample = False
        
    
    def __init__(self, L1, L2, mu, nu, k1, k2, lam=None, w=None, validate_args=None):
        
        ## It's important that |L1| + |L2| is <= 1 and that both are on the range -1 to 1. For now the check has been disabled as it cause issues with enumration
        
        if L1.shape == ():
            if ((torch.abs(L1) + torch.abs(L2)) > 1.) == True:
                raise ValueError("|skew1| + |skew2| has to be less than 1")
            
        else:
            if any(  ((torch.abs(L1) + torch.abs(L2)) > 1.)) == True:    
                raise ValueError("|skew1| + |skew2| has to be less than 1")
            
        
        self.L1, self.L2 = broadcast_all(L1, L2)
    
        super().__init__(mu = mu, nu = nu, k1 = k1, k2 = k2, lam = lam, w = w)
        
    
    def log_prob(self, phi_psi): ## 
        # Actual likelihood function
        """ log Joint distribution of phi and psi """
        # if the distribution itself is non-dimensional...
        if self.batch_shape == ():
            phi = phi_psi.T[0]
            psi = phi_psi.T[1]
            logProb = (self.k1 * torch.cos(phi - self.mu) + self.k2 * torch.cos(psi - self.nu) + 
                self.lam * torch.sin(phi - self.mu) * torch.sin(psi - self.nu)) - self.logC + torch.log( (1 + self.L1 * torch.sin(phi - self.mu) + self.L2 * torch.sin( psi - self.nu )))
            if phi_psi.shape == (2,): # this indicates that both sample_shape and batch_shape is empty and thus log_prob should return log_prob with shape == batch_shape == ()
                assert logProb[0].shape == ()
                return logProb[0] # indexing in because shape must be same as batch_shape
            else: #if batch_shape is non-dimensional, but sample_shape is not 
                assert logProb.shape == torch.Size([phi_psi.shape[0]]) # this is sample_shape, because we know batch_shape is empty and event_shape is 2, but will be in location [1]
                return logProb #logProb has shape sample_shape with non dimensioanal batch_shape
        
        else: #if batch_shape has dimensions... eg. if mu = torch.tensor([1]) or mu = torch.ones([100,1])
            phi = phi_psi.T[0].T
            psi = phi_psi.T[1].T
                
            return(self.k1 * torch.cos(phi - self.mu) + self.k2 * torch.cos(psi - self.nu) + 
                    self.lam * torch.sin(phi - self.mu) * torch.sin(psi - self.nu)) - self.logC + torch.log( (1 + self.L1 * torch.sin(phi - self.mu) + self.L2 * torch.sin( psi - self.nu )))
        
        
        
    def sample(self, sample_shape=torch.Size([])):
        
        """
        marg: marginal distribution (using _acg_bound())
        cond: conditional distribution using a modified univariate von Mises
            - as described in Singh et al. (2002)
        then skewed as described in:
        # 'Sine-skewed toroidal distributions and their application in protein bioinformatics'
        # Authors of the article: Jose Ameijeiras-Alonso and  Christophe Ley, from KU leuven and Ghent University.
        # 
        """
        Y = super().sample(sample_shape)
        
        if self.batch_shape == ():
            
            if sample_shape == ():
                Y_skewed = skew_samples(Y.reshape(1,2), torch.Size([1]), self.mu, self.nu, self.L1, self.L2)
                return Y_skewed[0]
            else: # if sample_shape is not empty
                Y_skewed = skew_samples(Y, sample_shape, self.mu, self.nu, self.L1, self.L2)
                return Y_skewed
        
        if self.batch_shape == (1,):
            if sample_shape == ():
                Y_skewed = skew_samples(Y, torch.Size([1]), self.mu, self.nu, self.L1, self.L2)
                return Y_skewed
            else: # if sample_shape is not empty
                # Y.squeeze(1) is to remove the batch dimension
                Y_skewed = skew_samples(Y.squeeze(1), sample_shape, self.mu, self.nu, self.L1, self.L2)
                return Y_skewed.reshape(sample_shape[0],1,2)
        
        elif len(self.batch_shape) == 2: ## batch_shape [100,1]
            assert(self.batch_shape[-1] == 1) ## making sure batch_shape is a column vector
            
            
            if sample_shape == () or sample_shape == (1,): # empty, thus sample shape == [100,1,2]
                if sample_shape == (1,):
                    Y = Y.squeeze(0) # we want Y to have shape batch_shape, event_shape, so by .squeeze(0) we remove the sample_shape of 1
            
                sample_batch_all = torch.ones([self.batch_shape[0],self.batch_shape[1],2]) # torch ones, this way we can see if there is an issue as samples will be overrepresented with ones
                #sample_batch_all[:,:,:] = np.nan
                
                for dist_nr in range(self.batch_shape[0]):
                    mu = self.mu[dist_nr]
                    nu = self.nu[dist_nr]
                    L1 = self.L1[dist_nr]
                    L2 = self.L2[dist_nr]
                    # this is a bit of a hack, but essentially when indexing into Y in the next line, and passing it 
                    # to skew_samples, Y is shape 1,2. our sample Y has this shape from the 
                    #leftover singleton dimension of batch_shape + event-shape. 
                    # But skew_samples takes the leftover singleton as stating its a 
                    # single sample... this is just a note about it. It doesn't matter.
                    
                    if sample_shape == ():
                        a_sample = skew_samples(Y[dist_nr], torch.Size([1]), mu, nu, L1, L2)
                    else:
                        a_sample = skew_samples(Y[dist_nr], sample_shape, mu, nu, L1, L2)
                    sample_batch_all[dist_nr] = a_sample[0]
                    
                if sample_shape == (1,): #sample shape of one
                    return sample_batch_all.reshape([1,self.batch_shape[0], self.batch_shape[1], 2]) # shape == sample_shape, batch_shape, event_shape
                else: # non dimensional sample shape... ie no sample
                    return sample_batch_all # shape = [batch_shape, 2], but remember that batch_shape in this case == [x,1], thus the sample shape will be == [x,1,2]
            
            
            else: #if sampleshape is greater than 1... not including 1 because that is dealt with above
                assert(len(sample_shape) == 1) # is not implemented for multidimensional sampling!!!
                
                # Y.shape = sample_shape(5), batch_shape (100,1), event_shape(2)
                # Y.shape = [5, 100, 1, 2]
                
                # remember: [sample_shape, batch_shape, eventshape]
                # thus if parameters (mu etc) has shape [100,1] and we want 5 samples, then
                # sample.shape == [5, 100, 1, 2]
                
                # here we also have sample_shape to take into consideration:
                sample_batch_all = torch.ones([sample_shape[0], self.batch_shape[0],self.batch_shape[1],2]) # torch ones, this way we can see if there is an issue as samples will be overrepresented with ones
                #sample_batch_all[:,:,:] = np.nan
                
                for dist_nr in range(self.batch_shape[0]):
                    
                    mu = self.mu[dist_nr]
                    nu = self.nu[dist_nr]
                    L1 = self.L1[dist_nr]
                    L2 = self.L2[dist_nr]
                    
                    
                    
                    # indexing into dist_nr and .squeeze(1) removes the now two singleton batch diemnsions
                    a_sample = skew_samples(Y[:,dist_nr,:,:].squeeze(1), sample_shape, mu, nu, L1, L2)
                    # a_sample now has shape [sample_shape, 2], but we know it belongs to distribution nr == dist_nr
                    
                    for sample_nr in range(sample_shape[0]):
                        sample_batch_all[sample_nr, dist_nr] = a_sample[sample_nr]
                
                return sample_batch_all # shape = [batch_shape, 2], but remember that batch_shape in this case == [x,1], thus the sample shape will be == [x,1,2]
                
        
        
        else:
            print('unexplained reason: no samples were skewed because something went wrong')
            
            
        

    
    
    def expand(self, batch_shape):
        
        try:
            return super().expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            mu = self.mu.expand(batch_shape)
            nu = self.nu.expand(batch_shape)
            k1 = self.k1.expand(batch_shape)
            k2 = self.k2.expand(batch_shape)
            lam = self.lam.expand(batch_shape)
            L1 = self.L1.expand(batch_shape)
            L2 = self.L2.expand(batch_shape)
            return type(self)(mu, nu, k1, k2, lam, L1, L2, validate_args=validate_args)
            
        

class SineSkewedBivariateVonMisesSineModel_NNDIST(SineSkewedBivariateVonMisesSineModel):
    """
    Sine skewed Bivariate von Mises distribution on the torus
    
    This distribution is a child class of Bivariate Von Mises (unimodal) Sine model
    
    all parameters must be on the range zero to one. You can read about the actual parameters in the
    actual distribution. This one is just a convenience wrapper
    
    
    This distribution is build from the article 'Sine-skewed toroidal distributions 
        and their application in protein bioinformatics'
    Authors of the article: Jose Ameijeiras-Alonso and  Christophe Ley, from KU leuven and Ghent University.
    
    This distribution was written by Christian Sigvald Breinholt. Who is always available for 
    questions and help. Just reach out.
    """
    arg_constraints = {'L1': constraints.real, 'L2': constraints.real} 
    ## The other parameter constraints are in the parent class
    
    
    support = constraints.real
    has_rsample = False
        
    
    def __init__(self, L1, L2, b_scale, mu, nu, k1, k2, lam_w, mean_max = 100., mean_min = -100., k_max = 1000., k_min = 0.1, validate_args=None):
        
        # all parameters must be on ranmge zero to one
        params = torch.stack((L1, L2, mu, nu, k1,k2,lam_w)).flatten() 
        
        if (any( (params < 0. ) )) == True or (any( (params > 1. ) )) == True: 
                raise ValueError("All paramters must be on range zero to one, both inclusive")
        
        ## It's important that |L1| + |L2| is <= 1 and that both are on the range -1 to 1. For now the check has been disabled as it cause issues with enumration
        # this is taken care of by b_scale
        
        L1 = L1 * b_scale
        L2 = L2 * (1-b_scale)
        
        # set up rest of parameters:
        mu = mu * (mean_max+mean_min) - mean_min 
        nu = nu * (mean_max+mean_min) - mean_min
        
        k1 = k1 * (k_max+k_min) - k_min
        k2 = k2 * (k_max+k_min) - k_min
        
        w = lam_w * 2. - 1. # W is now on range minus one to one
        
        super().__init__(mu = mu, nu = nu, k1 = k1, k2 = k2, L1 = L1, L2 = L2,  lam = None, w = w)
        
    
    def log_prob(self, phi_psi):
        return(super().log_prob(phi_psi))
          
    def sample(self, sample_shape=torch.Size([])):
        return(super().sample(sample_shape))
    
    def expand(self, batch_shape):
        try:
            return super().expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            mu = self.mu.expand(batch_shape)
            nu = self.nu.expand(batch_shape)
            k1 = self.k1.expand(batch_shape)
            k2 = self.k2.expand(batch_shape)
            lam = self.lam.expand(batch_shape)
            L1 = self.L1.expand(batch_shape)
            L2 = self.L2.expand(batch_shape)
            return type(self)(mu, nu, k1, k2, lam, L1, L2, validate_args=validate_args)


class SineSkewedBivariateVonMisesSineModel_HYBRID(SineSkewedBivariateVonMisesSineModel):
    """
    L1, L2, b_scale and lam_w must be on range of zero to one.
    
    k1 and k2 must be above zero and real.
    
    This distribution is a child class of Sine Skewed Bivariate Von Mises (unimodal) Sine model
    
    Modality:
        taken care of with the lam_w parameter
    
    :param torch.Tensor L1, L2: skew-factors.
        - L1 & L2 has to be real numbers on the range zero to one and comply with: |L1| + |L2| <= 1.0 which is taken care of with b_scale
        
    :param torch.Tensor b_scale: refactors the skew-factors to comply with |L1| + |L2| <= 1.0 .
        - mu & nu can be any real number but are interpreted as 2*pi

    
    :param torch.Tensor mu, nu: an angle in radians.
        - mu & nu can be any real number but are interpreted as 2*pi

    :param torch.Tensor k1, k2: concentration parameter
        - This distribution is only defined for k1, k2 > 0

    :param torch.Tensor lam: correlation parameter
        - Can be any real number, but is not defined for very bimodal cases
        - See 'Modality' above
        
    :param torch.Tensor w: reparameterization parameter
        - Has to be between -1 and 1
    
    This distribution is a child class of Sine Skewed Bivariate Von Mises (unimodal) Sine model
    
    all parameters must be on the range zero to one. You can read about the actual parameters in the
    actual distribution. This one is just a convenience wrapper
    
    
    This distribution is build from the article 'Sine-skewed toroidal distributions 
        and their application in protein bioinformatics'
    Authors of the article: Jose Ameijeiras-Alonso and  Christophe Ley, from KU leuven and Ghent University.
    
    This distribution was written by Christian Sigvald Breinholt. Who is always available for 
    questions and help. Just reach out.
    """
    arg_constraints = {'L1': constraints.real, 'L2': constraints.real} 
    ## The other parameter constraints are in the parent class
    
    
    support = constraints.real
    has_rsample = False
        
    
    def __init__(self, L1, L2, b_scale, mu, nu, k1, k2, lam_w, validate_args=None):
        
        # all parameters must be on ranmge zero to one
        params = torch.stack((L1, L2, mu, nu, k1,k2,lam_w, b_scale)).flatten() 
        
        if (any( (params < 0. ) )) == True or (any( (params > 1. ) )) == True: 
                raise ValueError("All paramters must be on range zero to one, both inclusive")
        
        ## It's important that |L1| + |L2| is <= 1 and that both are on the range -1 to 1. For now the check has been disabled as it cause issues with enumration
        # this is taken care of by b_scale
        
        L1 = L1 * b_scale
        L2 = L2 * (1-b_scale)
        
        
        
        w = lam_w * 2. - 1. # W is now on range minus one to one
        
        super().__init__(mu = mu, nu = nu, k1 = k1, k2 = k2, L1 = L1, L2 = L2,  lam = None, w = w)
        
    
    def log_prob(self, phi_psi):
        return(super().log_prob(phi_psi))
          
    def sample(self, sample_shape=torch.Size([])):
        return(super().sample(sample_shape))
    
    def expand(self, batch_shape):
        try:
            return super().expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            mu = self.mu.expand(batch_shape)
            nu = self.nu.expand(batch_shape)
            k1 = self.k1.expand(batch_shape)
            k2 = self.k2.expand(batch_shape)
            lam = self.lam.expand(batch_shape)
            L1 = self.L1.expand(batch_shape)
            L2 = self.L2.expand(batch_shape)
            return type(self)(mu, nu, k1, k2, lam, L1, L2, validate_args=validate_args)



class BivariateVonMisesSineModel_NNDIST(BivariateVonMisesSineModel):
    """
    Bivariate von Mises distribution on the torus
    
    This distribution is a child class of Bivariate Von Mises (unimodal) Sine model
    
    all parameters must be on the range zero to one. You can read about the actual parameters in the
    actual distribution. This one is just a convenience wrapper
    
    
    This distribution was written by Christian Sigvald Breinholt. Who is always available for 
    questions and help. Just reach out.
    """
    
    
    support = constraints.real
    has_rsample = False
        
    
    def __init__(self, mu, nu, k1, k2, lam_w, mean_max = 100., mean_min = -100., k_max = 1000., k_min = 0.1, validate_args=None):
        
        # all parameters must be on ranmge zero to one
        params = torch.stack((mu, nu, k1,k2,lam_w)).flatten() 
        
        if (any( (params < 0. ) )) == True or (any( (params > 1. ) )) == True: 
                raise ValueError("All paramters must be on range zero to one, both inclusive")
        
        # set up rest of parameters:
        mu = mu * (mean_max+mean_min) - mean_min 
        nu = nu * (mean_max+mean_min) - mean_min
        
        k1 = k1 * (k_max+k_min) - k_min
        k2 = k2 * (k_max+k_min) - k_min
        
        w = lam_w * 2. - 1. # W is now on range minus one to one
        
        super().__init__(mu = mu, nu = nu, k1 = k1, k2 = k2,  lam = None, w = w)
        
    
    def log_prob(self, phi_psi):
        return(super().log_prob(phi_psi))
        
        
    def sample(self, sample_shape=torch.Size([])):
        return(super().sample(sample_shape))

    def expand(self, batch_shape):
        
        try:
            return super().expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            mu = self.mu.expand(batch_shape)
            nu = self.nu.expand(batch_shape)
            k1 = self.k1.expand(batch_shape)
            k2 = self.k2.expand(batch_shape)
            lam = self.lam.expand(batch_shape)
            return type(self)(mu, nu, k1, k2, lam, validate_args=validate_args)


































"""
# this section is not needed, but is kept because it can be usefull in the future
def pseudo_log_likelihood(phi, psi, w):
    
    mu = w[0]; nu = w[1]; k1 = w[2]; k2 = w[3]; lam = w[4] 
        
    k1_l = torch.sqrt( (k1)**2 + (lam)**2 * (torch.sin(psi - nu))**2 )
    psi_i = torch.atan( lam / k1 * torch.sin(psi - nu) )
    cond1 = VonMises(mu - psi_i, k1_l).log_prob(phi)
    
    k2_l = torch.sqrt( (k2)**2 + (lam)**2 * (torch.sin(phi - mu))**2 )
    phi_i = torch.atan( lam / k2 * torch.sin(phi - mu) )
    cond2 = VonMises(nu - phi_i, k2_l).log_prob(psi)
    
    return (cond1 + cond2).sum()

def circ_mean(theta):
    
    C = theta.cos().mean(); S = theta.sin().mean()
    return torch.atan2(S, C)

def moments(theta1, theta2, mean1, mean2):
    
    S1 = ((torch.sin(theta1 - mean1))**2).mean()
    S2 = ((torch.sin(theta2 - mean2))**2).mean()
    S12 = (torch.sin(theta1 - mean1) * torch.sin(theta2 - mean2)).mean()
    return torch.tensor([S2, S1, S12]) / (S1*S2 - (S12)**2)

"""
