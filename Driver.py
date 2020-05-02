#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from toolbox import sphereical_transport, error

def main():
    #inputs
    n = 8
    cells = 100
    r = 1.0
    scat_order = 0
    sig_t = 30.0
    sig_a = 0.0
    q_0 = np.ones(cells) #p/cm^3-s
    normalize = False
    psi_incident = np.zeros(int(n/2))
    accelerate = True
    legendre_xs_mom = np.array([30.0])
    #generate angular grid center points (roots of legendre polynomial of degree n) and the
    #corresponding gauss quadrature weights
    ang_cell_centers, w = np.polynomial.legendre.leggauss(n)
    #generate angular cell edge values 
    ang_cell_edges = np.zeros(n+1)
    ang_cell_edges[0] = -1
    for i in range(1,n+1):
        ang_cell_edges[i] = ang_cell_edges[i-1] + w[i-1]
    #generate alpha coefficients
    alpha = np.zeros(n+1)
    for i in range(1,n+1):
        alpha[i] = alpha[i-1] - (2*w[i-1]*ang_cell_centers[i-1])
    alpha[n] = 0
    #generate beta coefficients
    beta = np.zeros(n)
    for i in range(n):
        beta[i] = (ang_cell_centers[i] - ang_cell_edges[i])/(ang_cell_edges[i+1] - ang_cell_edges[i])
    #initialize sphereical transport class
    sph_trans = sphereical_transport(q_0,legendre_xs_mom,cells,n,r,sig_t,sig_a,normalize,psi_incident,scat_order,ang_cell_centers,w,alpha,beta,accelerate)
    #initial guess for source
    ang_flux_0 = np.zeros(n)
    for i in range(cells):
        sph_trans.update_scat_source_mom(i,ang_flux_0)
    #generate initial phi using initial guess for psi
    phi_0 = np.zeros(cells)
    for i in range(cells):
        for ang in range(np.size(ang_flux_0)):
            phi_0[i] += ang_flux_0[ang]*w[ang]
    epsilon = 1
    #begin source iteration
    index = 0
    while epsilon > 1e-4:
        #compute psi at mu = -1 which will be the first angular inflow vector
        ang_inflow, origin_flux = sph_trans.starting_direction()
        #do the full domain sweep
        phi = sph_trans.sweep(ang_inflow,origin_flux)
        if accelerate:
            sph_trans.DSA(phi,phi_0)
        index += 1
        if index > 99:
            print('Iteration limit exceeded')
            break
        epsilon = np.max(error.rel_change(phi,phi_0))
        phi_0 = phi
    print(phi)
    print(index-1)
    #compute balance tables
    abs_rate = 0
    for i in range(cells):
        abs_rate += phi[i]*sig_a*sph_trans.V(i)
    reflection = 0.0
    for ang in range(int(n/2),n):
        reflection += sph_trans.outflow[ang - int(n/2)]*ang_cell_centers[ang]*w[ang]*sph_trans.A(cells)
    sink_rate = abs_rate + reflection
    source_rate = 0
    for i in range(cells):
        source_rate += q_0[i]*sph_trans.V(i)
    for ang in range(0,int(n/2)):
        source_rate += sph_trans.psi_incident[ang]*-ang_cell_centers[ang]*w[ang]*sph_trans.A(cells)
    bal = np.abs((source_rate - sink_rate)/source_rate)
    #print(source_rate)
    #print(abs_rate)
    #print(reflection)
    print(bal)

if __name__ == '__main__':
    main()