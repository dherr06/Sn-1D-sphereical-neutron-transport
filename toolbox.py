import numpy as np
from scipy import special

class sphereical_transport:
    
    def __init__(self,q_0,legendre_xs_mom,cells,n,r,sig_t,sig_a,normalize,psi_incident,scat_order,ang_cell_centers,w,alpha,beta,accelerate):
        #create all member variables
        self.q_0 = q_0
        self.cells = cells
        self.n = n
        self.r = r
        self.sig_t = sig_t
        self.sig_a = sig_a
        self.normalize = normalize
        self.psi_incident = psi_incident
        #if normalize is true go ahead and normalize the incident fluxes
        if normalize == True:
            Jin = 0
            for ang in range(0,int(n/2)):
                Jin += -1*ang_cell_centers[ang]*psi_incident[ang]*w[ang]
            self.psi_incident /= Jin
        self.scat_order = scat_order
        self.ang_cell_centers = ang_cell_centers
        self.w = w
        self.alpha = alpha
        self.beta = beta
        #size a container for the product of the scattering cross sections and flux moments
        self.scat_mom = np.zeros((scat_order+1,cells))
        self.legendre_xs_mom = legendre_xs_mom
        self.accelerate = accelerate
        self.source_mom = np.zeros((scat_order+1,cells))
        #source moments are static so lets compute them and store them now
        for k in range(scat_order+1):
            for i in range(cells):
                self.source_mom[k][i] = self.legendre_inhom_source_mom(k,i)
        #the outer radius of each sphereical cell with the 0'th element being the origin
        self.r_vec = np.linspace(0,r,cells+1)
        
    def legendre_inhom_source_mom(self,degree,i):
        '''
        return the degree'th legendre inhomogenous source moment for cell i
        '''
        q_leg_mom = 0
        for ang in range(self.n):
            q_leg_mom += (self.q_0/2)[i]*self.w[ang]*special.eval_legendre(degree,self.ang_cell_centers[ang])
        return q_leg_mom
    
    def legendre_flux_mom(self,flux,degree):
        '''
        return the degree'th legendre flux moment for corresponding angular flux vector 
        '''
        flux_leg_mom = 0
        for ang in range(self.n):
            flux_leg_mom += flux[ang]*self.w[ang]*special.eval_legendre(degree,self.ang_cell_centers[ang])
        return flux_leg_mom
    
    def return_source(self,i,n):
        '''
        return the current RHS at iterate l if we are solving for iterate l+1 for cell i and direction n
        '''
        Q = 0
        for k in range(self.scat_order+1):
            Q += (((2*k)+1)/2)*(self.scat_mom[k][i] + self.source_mom[k][i])*special.eval_legendre(k,n)
        return Q
    
    def update_scat_source_mom(self,i,ang_flux):
        '''
        update the l'th iterate scattering moments to l+1 using the angular flux we just solved for in cell i
        '''
        self.scat_mom[:,i] = np.zeros(self.scat_order+1)
        for j in range(self.scat_order+1):
            self.scat_mom[j][i] += self.legendre_flux_mom(ang_flux,j)*self.legendre_xs_mom[j]
 
    def pos_spatial_inflow_terms(self,i,ang,spatial_inflow):
        var = (self.ang_cell_centers[ang]*self.A(i+1)*spatial_inflow) + (self.ang_cell_centers[ang]*self.A(i)*spatial_inflow)
        return var
    
    def pos_denom(self,i,ang):
        var1 = (self.A(i+1)-self.A(i))*self.alpha[ang+1]/self.w[ang]/self.beta[ang]/2.0
        var2 = self.ang_cell_centers[ang]*self.A(i+1)*2.0
        var3 = self.sig_t*self.V(i)
        return (var1+var2+var3)
    
    def ang_inflow_terms(self,i,ang,ang_inflow):
        var1 = (self.A(i+1)-self.A(i))*self.alpha[ang+1]*ang_inflow/2.0/self.w[ang]/self.beta[ang]
        var2 = (self.A(i+1)-self.A(i))*self.alpha[ang+1]*ang_inflow/2.0/self.w[ang]
        var3 = (self.A(i+1)-self.A(i))*self.alpha[ang]*ang_inflow/self.w[ang]/2.0
        return (var1 - var2 + var3)
    
    def neg_spatial_inflow_terms(self,i,ang,spatial_inflow):
        var = -self.ang_cell_centers[ang]*spatial_inflow*(self.A(i+1)+self.A(i))
        #var = -(self.ang_cell_centers[ang]*self.A(i+1)*spatial_inflow) - (self.ang_cell_centers[ang]*self.A(i)*spatial_inflow)
        return var
    
    def neg_denom(self,i,ang):
        var1 = (self.A(i+1)-self.A(i))*self.alpha[ang+1]/self.w[ang]/self.beta[ang]/2.0
        var2 = self.ang_cell_centers[ang]*self.A(i)*2.0
        var3 = self.sig_t*self.V(i)
        return (var1-var2+var3)
    
    def V(self,i):
        V = 4.0*np.pi/3.0*(self.r_vec[i+1]**3 - self.r_vec[i]**3)
        return V
        
    def A(self,i):
        A = 4.0*np.pi*self.r_vec[i]**2
        return A
    
    def starting_direction(self):
        '''
        solve the slab transport equation for mu = -1
        '''
        psi = np.zeros(self.cells)
        #if we only have 2 directions the half angle inflow equals S1 inflow
        #otherwise we have to interpolate
        if self.n == 2:
            psi_inflow = self.psi_incident[0]
        else:
            psi_inflow = (self.psi_incident[0]*((-1.0 - self.ang_cell_centers[1])/(self.ang_cell_centers[0] - self.ang_cell_centers[1]))) + (self.psi_incident[1]*((-1.0 - self.ang_cell_centers[0])/(self.ang_cell_centers[1] - self.ang_cell_centers[0])))
        for i in range(self.cells-1,-1,-1):
            delta_x = self.r_vec[i+1] - self.r_vec[i]
            psi[i] = ((self.return_source(i,-1)*delta_x)+(2.0*psi_inflow))/(2.0+(self.sig_t*delta_x))
            psi_inflow = 2.0*psi[i] - psi_inflow
        return psi, psi_inflow
    
    def sweep(self,ang_inflow,origin_flux):
        '''
        full domain sweep
        ang_inflow is a vector of length number of cells
        origin_flux is the angular flux for mu=-1 evaluated at the origin
        '''
        psi = np.zeros((self.n,self.cells))
        self.outflow = np.zeros(int(self.n/2))
        for ang in range(self.n):
            #if negative angle
            if self.ang_cell_centers[ang] < 0:
                spatial_inflow = self.psi_incident[ang]
                for i in range(self.cells-1,-1,-1):
                    psi[ang][i] = (self.return_source(i,self.ang_cell_centers[ang])*self.V(i))+(self.neg_spatial_inflow_terms(i,ang,spatial_inflow))+self.ang_inflow_terms(i,ang,ang_inflow[i])
                    psi[ang][i] /= self.neg_denom(i,ang)
                    #update spatial and angular inflows
                    spatial_inflow = (2.0*psi[ang][i]) - spatial_inflow
                    ang_inflow[i] = (psi[ang][i] - ((1.0-self.beta[ang])*ang_inflow[i]))/self.beta[ang]
            #if positive
            elif self.ang_cell_centers[ang] > 0:
                spatial_inflow = origin_flux
                for i in range(self.cells):
                    psi[ang][i] = (self.return_source(i,self.ang_cell_centers[ang])*self.V(i))+(self.pos_spatial_inflow_terms(i,ang,spatial_inflow))+self.ang_inflow_terms(i,ang,ang_inflow[i])
                    psi[ang][i] /= self.pos_denom(i,ang)
                    #update spatial and angular inflows
                    spatial_inflow = (2.0*psi[ang][i]) - spatial_inflow
                    ang_inflow[i] = (psi[ang][i] - ((1.0-self.beta[ang])*ang_inflow[i]))/self.beta[ang]
                self.outflow[ang - int(self.n/2)] = spatial_inflow
        #update scattering moments
        for i in range(self.cells):    
            self.update_scat_source_mom(i,psi[:,i])
        #if we are accelerating, save psi as a member variable because it is needed for acceleration
        if self.accelerate == True:
            self.psi = psi
        #reuturn phi not psi
        phi = np.zeros(self.cells)
        for i in range(self.cells):
            for ang in range(self.n):
                phi[i] += self.w[ang]*psi[ang][i]
                
        return phi
    
    def DSA(self,phi,phi_0):
        #set up a temporary area average function
        avg_A = lambda i:(self.A(i) + self.A(i+1))/2
        #define sig_1
        if self.scat_order == 0:
            sig_1 = 0
        else:
            sig_1 = self.legendre_xs_mom[1]
        #create diffusion matrix and RHS
        A = np.zeros((self.cells+1,self.cells+1))
        q = np.zeros(self.cells+1)
        #set the first equation
        A[0][0] = (avg_A(0)**2/self.V(0)/3/(self.sig_t - sig_1)) + (self.sig_a*self.V(0)/4)
        A[0][1] = (self.sig_a*self.V(0)/4) - (avg_A(0)**2/self.V(0)/3/(self.sig_t - sig_1))
        q[0] = self.legendre_xs_mom[0]*(phi[0]*self.V(0)/2 - phi_0[0]*self.V(0)/2)
        #set the last equaiton
        A[self.cells][self.cells-1] = (self.sig_a*self.V(self.cells-1)/4) 
        A[self.cells][self.cells-1] -= (avg_A(self.cells-1)**2/self.V(self.cells-1)/3/(self.sig_t - sig_1))
        A[self.cells][self.cells] = (avg_A(self.cells-1)**2/self.V(self.cells-1)/3/(self.sig_t - sig_1))
        A[self.cells][self.cells] += self.sig_a*self.V(self.cells-1)/4
        q[self.cells] = self.legendre_xs_mom[0]*(phi[self.cells-1]*self.V(self.cells-1)/2 - phi_0[self.cells-1]*self.V(self.cells-1)/2)
        summ = 0 
        for ang in range(self.n):
            if self.ang_cell_centers[ang] > 0:
                summ += self.ang_cell_centers[ang]*self.w[ang]
        summ *= self.A(self.cells)
        A[self.cells][self.cells] += summ
        #fill in interior equations
        for i in range(1,self.cells):
            A[i][i-1] = (self.sig_a*self.V(i-1)/4) - (avg_A(i-1)**2/3/self.V(i-1)/(self.sig_t - sig_1))
            A[i][i] = (avg_A(i)**2/self.V(i)/3/(self.sig_t - sig_1))
            A[i][i] += (avg_A(i-1)**2/self.V(i-1)/3/(self.sig_t - sig_1))
            A[i][i] += (self.sig_a*self.V(i)/4) + (self.sig_a*self.V(i-1)/4)
            A[i][i+1] = (self.sig_a*self.V(i)/4) - (avg_A(i)**2/3/self.V(i)/(self.sig_t - sig_1))
            q[i] = self.legendre_xs_mom[0]*(phi[i]*self.V(i)/2 - phi_0[i]*self.V(i)/2)
            q[i] += self.legendre_xs_mom[0]*(phi[i-1]*self.V(i-1)/2 - phi_0[i-1]*self.V(i-1)/2)
        e_phi = np.linalg.solve(A,q)
        #convert edge fluxes to cell centered fluxes 
        #we over write the phi vector that was passed into the function
        for i in range(self.cells):
            phi[i] += 0.5*(e_phi[i]+e_phi[i+1])
        #convert the accelerated scalar fluxes into angular fluxes
        for i in range(self.cells):
            for ang in range(self.n):
                self.psi[ang][i] += ((e_phi[i]+e_phi[i+1])/4.0)
        #update psi outflows
        #we need delta j outflow for this
        delta_J = 0
        for ang in range(int(self.n/2),self.n):
            delta_J += (self.ang_cell_centers[ang]*self.w[ang]*e_phi[self.cells])
        #use delta j to update psi outflows
        for ang in range(int(self.n/2),self.n):
            self.outflow[ang - int(self.n/2)] += ((e_phi[self.cells] + (3.0*delta_J*self.ang_cell_centers[ang]))/2.0)
        #update scattering moments
        for i in range(self.cells):    
            self.update_scat_source_mom(i,self.psi[:,i])

class error:
    
    def rel_change(phi,phi_0):
        return np.abs(phi - phi_0)/phi