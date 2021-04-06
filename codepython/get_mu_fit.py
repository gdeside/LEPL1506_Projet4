# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:10:12 2020
get_mu_fit returns the coefficient of the power law linking the coefficient of 
and the normal force: mu=k*NF^(n-1)
inputs:
    - mu [-] Coefficient of static friction
    - NF [N] Normal force at which the coefficient of static friction was measured
outputs:
    -k and n, as described above
@author: fschiltz
"""
import numpy as np
def get_mu_fit(mu,NF):
    
    fit_coeff=np.polyfit(np.log(NF),np.log(mu),1)
    k=np.exp(fit_coeff[1])
    n=fit_coeff[0]+1
    
    return k,n



