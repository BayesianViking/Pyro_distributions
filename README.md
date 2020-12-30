# Pyro_distributions
email: christian.s.breinholt@gmail.com
email2: christian.breinholt@sund.ku.dk
linkedIn: https://dk.linkedin.com/in/christian-sigvald-breinholt

I am always eager to help and answer questions. So reach out if you need help or sparring. it's best to use the contact details provided above.

The script test.py shows how to use the distributions and also checks that they have the appropriate shapes.

The module Breinholt_distribution_collection.py contains a range of distributions, here is the list:
    
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

Papers and their authors are credited in the code within the section their work was used.
