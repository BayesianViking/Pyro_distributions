# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 21:01:17 2020

@author: christian Sigvald Breinholt
email: christian.s.breinholt@gmail.com
email2: christian.breinholt@sund.ku.dk
linkedIn: https://dk.linkedin.com/in/christian-sigvald-breinholt

I am always eager to help and answer questions. So reach out if you need help or sparring.


This file is here to test the distributions in the module: Breinholt_distributions_collection
the mentioned module contains a range of distributions, here is the list:
    
    Bivariate distributions on the torus:
        BivariateVonMisesSineModel
            BivariateVonMisesSineModel_NNDIST
        SineSkewedBivariateVonMisesSineModel
            SineSkewedBivariateVonMisesSineModel_NNDIST
    
    The ones suffixed by _NNDIST, are identical to their non suffix counterpart, but all parameters
    are given on range zero to one which then gets reparametrized. These distributions are ideal
    for Neural networks and such...



IMPORTANT:
I suggest using the w parameter instead of lam.
w is a reparameritizations parameter on the range -1 to 1, and will calculate a lambda
that is on a range which is within the unimodal case      


"""
import Breinholt_distribution_collection_20_12_2020 as BDC

import torch
import math



####################################################################################
####################################################################################
####################################################################################
# Bivariate Von Mises Sine Model
####################################################################################
# simple test: 1
## Non-dimensional batch_size 

my_dist1 = BDC.BivariateVonMisesSineModel(mu = torch.tensor(1.),
                                 nu = torch.tensor(-1),
                                 k1 = torch.tensor(100.),
                                 k2 = torch.tensor(100.),
                                 lam = torch.tensor(0.1)
                                 )

assert my_dist1.batch_shape == ()
assert my_dist1.event_shape == (2,)

md1a = my_dist1.sample()
md1b = my_dist1.sample(torch.Size([1]))
md1c = my_dist1.sample(torch.Size([5]))
assert md1a.shape == (2,)
assert md1b.shape == (1,2)
assert md1c.shape == (5,2)

assert my_dist1.log_prob(md1a).shape == () # empty sample shape and empty batch_shape
assert my_dist1.log_prob(md1b).shape == (1,) # sample_shape, but empty batch_shape
assert my_dist1.log_prob(md1c).shape == (5,) # sample_shape, but empty batch_shape

####################################################################################################
# simple test: 2
# 1-dimensional batch_size

my_dist2 = BDC.BivariateVonMisesSineModel(mu = torch.tensor([1.]),
                                 nu = torch.tensor([-1]),
                                 k1 = torch.tensor([100.]),
                                 k2 = torch.tensor([100.]),
                                 lam = torch.tensor([0.1])
                                 )
assert my_dist2.batch_shape == (1,)
assert my_dist2.event_shape == (2,)

md2a = my_dist2.sample()
md2b = my_dist2.sample(torch.Size([1]))
md2c = my_dist2.sample(torch.Size([5]))

# test sample_shapes
assert md2a.shape == (1,2)   # == batch_shape + event_shape
assert md2b.shape == (1,1,2) # sample_shape, batch_shape, event_shape
assert md2c.shape == (5,1,2) # sample_shape, batch_shape, event_shape

#test log_prob shapes
assert my_dist2.log_prob(md2a).shape == (1,) # empty sample_shape, batch_shape,
assert my_dist2.log_prob(md2b).shape == (1,1) # sample_shape, batch_shape
assert my_dist2.log_prob(md2c).shape == (5, 1) # sample_shape, batch_shape


###############################################################################
# simple test: 3

shape = [10,1] # Christian T wants shape [100,1], but testing with 10,1 should mean it also works with 100,1


L1 = torch.tensor(torch.ones(shape)*0.5)
L2 = torch.tensor(torch.ones(shape)*0.5)

mu = torch.tensor(torch.ones(shape)*0.1)
nu = torch.tensor(torch.ones(shape)*0.1)

k1 = torch.tensor(torch.ones(shape)*0.1)
k2 = torch.tensor(torch.ones(shape)*0.1)

lam = torch.tensor(torch.ones(shape)*0.1)


my_dist3 = BDC.BivariateVonMisesSineModel(mu = mu,
                                 nu = nu,
                                 k1 = k1,
                                 k2 = k2,
                                 lam = lam
                                 )


assert my_dist3.batch_shape == (10,1)
assert my_dist3.event_shape == (2,)

md3a = my_dist3.sample()
md3b = my_dist3.sample(torch.Size([1]))
md3c = my_dist3.sample(torch.Size([5]))
assert md3a.shape == (10,1,2)
assert md3b.shape == (1,10,1,2)
assert md3c.shape == (5,10,1,2) # sample_shape (5), batch_shape (10,1), event_shape(2)

assert my_dist3.log_prob(md3a).shape == (10,1) # batch_shape (10,1)
assert my_dist3.log_prob(md3b).shape == (1,10,1) # sample_shape(1), batch_shape(10,1)
assert my_dist3.log_prob(md3c).shape == (5,10,1) # sample_shape (5), batch_shape (10,1)

#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#####
#####


# testing the skewed distribution now...

my_dist_skew1 = BDC.SineSkewedBivariateVonMisesSineModel(mu = torch.tensor(1.),
                                 nu = torch.tensor(-1),
                                 k1 = torch.tensor(100.),
                                 k2 = torch.tensor(100.),
                                 L1 = torch.tensor(0.1),
                                 L2 = torch.tensor(0.1),
                                 lam = torch.tensor(0.1)
                                 )

assert my_dist_skew1.batch_shape == ()
assert my_dist_skew1.event_shape == (2,)

mds1a = my_dist_skew1.sample()
mds1b = my_dist_skew1.sample(torch.Size([1]))
mds1c = my_dist_skew1.sample(torch.Size([5]))
assert mds1a.shape == (2,)
assert mds1b.shape == (1,2)
assert mds1c.shape == (5,2)

assert my_dist_skew1.log_prob(mds1a).shape == () # empty sample shape and empty batch_shape
assert my_dist_skew1.log_prob(mds1b).shape == (1,) # sample_shape, but empty batch_shape
assert my_dist_skew1.log_prob(mds1c).shape == (5,) # sample_shape, but empty batch_shape




####################################################################################################
# simple test: 2
# 1-dimensional batch_size

my_dist_skew2 = BDC.SineSkewedBivariateVonMisesSineModel(mu = torch.tensor([1.]),
                                 nu = torch.tensor([-1]),
                                 k1 = torch.tensor([100.]),
                                 k2 = torch.tensor([100.]),
                                 L1 = torch.tensor([0.1]),
                                 L2 = torch.tensor([0.1]),
                                 lam = torch.tensor([0.1])
                                 )
assert my_dist_skew2.batch_shape == (1,)
assert my_dist_skew2.event_shape == (2,)

mds2a = my_dist_skew2.sample()
mds2b = my_dist_skew2.sample(torch.Size([1]))
mds2c = my_dist_skew2.sample(torch.Size([5]))
assert mds2a.shape == (1,2)   # == batch_shape + event_shape
assert mds2b.shape == (1,1,2) # sample_shape, batch_shape, event_shape
assert mds2c.shape == (5,1,2) # sample_shape, batch_shape, event_shape

assert my_dist_skew2.log_prob(mds2a).shape == (1,) # empty sample_shape, batch_shape,
assert my_dist_skew2.log_prob(mds2b).shape == (1,1) # sample_shape, batch_shape
assert my_dist_skew2.log_prob(mds2c).shape == (5, 1) # sample_shape, batch_shape



###############################################################################
# simple test: 3
# now for the case Christian T suggested: shape [100,1], but using 10,1 instead of 100,1 for speed reasons

# now to the test:...

shape = [10,1] # Christian T wants shape [100,1], but testing with 10,1 should mean it also works with 100,1


L1 = torch.tensor(torch.ones(shape)*0.5)
L2 = torch.tensor(torch.ones(shape)*0.5)

mu = torch.tensor(torch.ones(shape)*0.1)
nu = torch.tensor(torch.ones(shape)*0.1)

k1 = torch.tensor(torch.ones(shape)*0.1)
k2 = torch.tensor(torch.ones(shape)*0.1)

lam = torch.tensor(torch.ones(shape)*0.1)


my_dist_skew3 = BDC.SineSkewedBivariateVonMisesSineModel(mu = mu,
                                 nu = nu,
                                 k1 = k1,
                                 k2 = k2,
                                 L1 = L1,
                                 L2 = L2,
                                 lam = lam
                                 )


assert my_dist_skew3.batch_shape == (10,1)
assert my_dist_skew3.event_shape == (2,)

mds3a = my_dist_skew3.sample()
mds3b = my_dist_skew3.sample(torch.Size([1]))
mds3c = my_dist_skew3.sample(torch.Size([5]))
assert mds3a.shape == (10,1,2)
assert mds3b.shape == (1,10,1,2)
assert mds3c.shape == (5,10,1,2) # sample_shape (5), batch_shape (10,1), event_shape(2)

assert my_dist_skew3.log_prob(mds3a).shape == (10,1) # batch_shape (10,1)
assert my_dist_skew3.log_prob(mds3b).shape == (1,10,1) # sample_shape(1), batch_shape(10,1)
assert my_dist_skew3.log_prob(mds3c).shape == (5,10,1) # sample_shape (5), batch_shape (10,1)


####################################################################################
####################################################################################
####################################################################################
#
#    Testing the special versions suffixed _NNDIST
#
#
#       OBS: reusing the names for samples and distributions from above

####################################################################################
####################################################################################
####################################################################################
# Bivariate Von Mises Sine Model
####################################################################################
# simple test: 1
## Non-dimensional batch_size 

my_dist1 = BDC.BivariateVonMisesSineModel_NNDIST(mu = torch.tensor(.5),
                                 nu = torch.tensor(0.5),
                                 k1 = torch.tensor(.5),
                                 k2 = torch.tensor(.5),
                                 lam_w = torch.tensor(0.1)
                                 )

assert my_dist1.batch_shape == ()
assert my_dist1.event_shape == (2,)

md1a = my_dist1.sample()
md1b = my_dist1.sample(torch.Size([1]))
md1c = my_dist1.sample(torch.Size([5]))
assert md1a.shape == (2,)
assert md1b.shape == (1,2)
assert md1c.shape == (5,2)

assert my_dist1.log_prob(md1a).shape == () # empty sample shape and empty batch_shape
assert my_dist1.log_prob(md1b).shape == (1,) # sample_shape, but empty batch_shape
assert my_dist1.log_prob(md1c).shape == (5,) # sample_shape, but empty batch_shape

####################################################################################################
# simple test: 2
# 1-dimensional batch_size

my_dist2 = BDC.BivariateVonMisesSineModel_NNDIST(mu = torch.tensor([.5]),
                                 nu = torch.tensor([0.5]),
                                 k1 = torch.tensor([.5]),
                                 k2 = torch.tensor([.5]),
                                 lam_w = torch.tensor([0.1])
                                 )
assert my_dist2.batch_shape == (1,)
assert my_dist2.event_shape == (2,)

md2a = my_dist2.sample()
md2b = my_dist2.sample(torch.Size([1]))
md2c = my_dist2.sample(torch.Size([5]))

# test sample_shapes
assert md2a.shape == (1,2)   # == batch_shape + event_shape
assert md2b.shape == (1,1,2) # sample_shape, batch_shape, event_shape
assert md2c.shape == (5,1,2) # sample_shape, batch_shape, event_shape

#test log_prob shapes
assert my_dist2.log_prob(md2a).shape == (1,) # empty sample_shape, batch_shape,
assert my_dist2.log_prob(md2b).shape == (1,1) # sample_shape, batch_shape
assert my_dist2.log_prob(md2c).shape == (5, 1) # sample_shape, batch_shape


###############################################################################
# simple test: 3

shape = [10,1] # Christian T wants shape [100,1], but testing with 10,1 should mean it also works with 100,1



mu = torch.tensor(torch.ones(shape)*0.1)
nu = torch.tensor(torch.ones(shape)*0.1)

k1 = torch.tensor(torch.ones(shape)*0.1)
k2 = torch.tensor(torch.ones(shape)*0.1)

lam_w = torch.tensor(torch.ones(shape)*0.1)


my_dist3 = BDC.BivariateVonMisesSineModel_NNDIST(mu = mu,
                                 nu = nu,
                                 k1 = k1,
                                 k2 = k2,
                                 lam_w = lam_w
                                 )


assert my_dist3.batch_shape == (10,1)
assert my_dist3.event_shape == (2,)

md3a = my_dist3.sample()
md3b = my_dist3.sample(torch.Size([1]))
md3c = my_dist3.sample(torch.Size([5]))
assert md3a.shape == (10,1,2)
assert md3b.shape == (1,10,1,2)
assert md3c.shape == (5,10,1,2) # sample_shape (5), batch_shape (10,1), event_shape(2)

assert my_dist3.log_prob(md3a).shape == (10,1) # batch_shape (10,1)
assert my_dist3.log_prob(md3b).shape == (1,10,1) # sample_shape(1), batch_shape(10,1)
assert my_dist3.log_prob(md3c).shape == (5,10,1) # sample_shape (5), batch_shape (10,1)

#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#####
#####


# testing the skewed distribution now...

my_dist_skew1 = BDC.SineSkewedBivariateVonMisesSineModel_NNDIST(mu = torch.tensor(.5),
                                 nu = torch.tensor(0.5),
                                 k1 = torch.tensor(.5),
                                 k2 = torch.tensor(.5),
                                 L1 = torch.tensor(0.8),
                                 L2 = torch.tensor(0.8),
                                 lam_w = torch.tensor(0.1),
                                 b_scale = torch.tensor(0.8)
                                 )

assert my_dist_skew1.batch_shape == ()
assert my_dist_skew1.event_shape == (2,)

mds1a = my_dist_skew1.sample()
mds1b = my_dist_skew1.sample(torch.Size([1]))
mds1c = my_dist_skew1.sample(torch.Size([5]))
assert mds1a.shape == (2,)
assert mds1b.shape == (1,2)
assert mds1c.shape == (5,2)

assert my_dist_skew1.log_prob(mds1a).shape == () # empty sample shape and empty batch_shape
assert my_dist_skew1.log_prob(mds1b).shape == (1,) # sample_shape, but empty batch_shape
assert my_dist_skew1.log_prob(mds1c).shape == (5,) # sample_shape, but empty batch_shape




####################################################################################################
# simple test: 2
# 1-dimensional batch_size

my_dist_skew2 = BDC.SineSkewedBivariateVonMisesSineModel_NNDIST(mu = torch.tensor([.5]),
                                 nu = torch.tensor([0.5]),
                                 k1 = torch.tensor([.5]),
                                 k2 = torch.tensor([.5]),
                                 L1 = torch.tensor([0.1]),
                                 L2 = torch.tensor([0.1]),
                                 lam_w = torch.tensor([0.1]),
                                 b_scale = torch.tensor([0.8])
                                 )
assert my_dist_skew2.batch_shape == (1,)
assert my_dist_skew2.event_shape == (2,)

mds2a = my_dist_skew2.sample()
mds2b = my_dist_skew2.sample(torch.Size([1]))
mds2c = my_dist_skew2.sample(torch.Size([5]))
assert mds2a.shape == (1,2)   # == batch_shape + event_shape
assert mds2b.shape == (1,1,2) # sample_shape, batch_shape, event_shape
assert mds2c.shape == (5,1,2) # sample_shape, batch_shape, event_shape

assert my_dist_skew2.log_prob(mds2a).shape == (1,) # empty sample_shape, batch_shape,
assert my_dist_skew2.log_prob(mds2b).shape == (1,1) # sample_shape, batch_shape
assert my_dist_skew2.log_prob(mds2c).shape == (5, 1) # sample_shape, batch_shape



###############################################################################
# simple test: 3
# now for the case Christian T suggested: shape [100,1], but using 10,1 instead of 100,1 for speed reasons

# now to the test:...

shape = [10,1] # Christian T wants shape [100,1], but testing with 10,1 should mean it also works with 100,1


L1 = torch.tensor(torch.ones(shape)*0.5)
L2 = torch.tensor(torch.ones(shape)*0.5)

mu = torch.tensor(torch.ones(shape)*0.1)
nu = torch.tensor(torch.ones(shape)*0.1)

k1 = torch.tensor(torch.ones(shape)*0.1)
k2 = torch.tensor(torch.ones(shape)*0.1)

lam_w = torch.tensor(torch.ones(shape)*0.1)
b_scale = torch.tensor(torch.ones(shape)*0.8)

my_dist_skew3 = BDC.SineSkewedBivariateVonMisesSineModel_NNDIST(mu = mu,
                                 nu = nu,
                                 k1 = k1,
                                 k2 = k2,
                                 L1 = L1,
                                 L2 = L2,
                                 b_scale = b_scale,
                                 lam_w = lam_w
                                 )


assert my_dist_skew3.batch_shape == (10,1)
assert my_dist_skew3.event_shape == (2,)

mds3a = my_dist_skew3.sample()
mds3b = my_dist_skew3.sample(torch.Size([1]))
mds3c = my_dist_skew3.sample(torch.Size([5]))
assert mds3a.shape == (10,1,2)
assert mds3b.shape == (1,10,1,2)
assert mds3c.shape == (5,10,1,2) # sample_shape (5), batch_shape (10,1), event_shape(2)

assert my_dist_skew3.log_prob(mds3a).shape == (10,1) # batch_shape (10,1)
assert my_dist_skew3.log_prob(mds3b).shape == (1,10,1) # sample_shape(1), batch_shape(10,1)
assert my_dist_skew3.log_prob(mds3c).shape == (5,10,1) # sample_shape (5), batch_shape (10,1)

