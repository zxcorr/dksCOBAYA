"""
darksync.py
============
Friedmann ODE solver for a flat ΛCDM-like cosmology with dark matter,
dark energy, radiation, baryons, and photons.

Definitions
-----------
C_CTE   : Speed of light [km/s]
G_CTE   : Gravitational constant [m^3 kg^-1 s^-2]
pdm     : Dark matter energy density [kg/m^3]
pde     : Dark energy density [kg/m^3]
pr      : Radiation density [kg/m^3]
pb      : Baryon density [kg/m^3]
pg      : Photon density [kg/m^3]
Dc      : Dimensionless comoving distance integral (Dc/dH)
r_t     : Total energy density (pdm + pde + pr + pb)
H(z)    : Hubble parameter at redshift z [km/s/Mpc]
dH      : Hubble distance c/H0 [Mpc]
DM      : Transverse comoving distance [Mpc]
DA      : Angular diameter distance [Mpc]
dL      : Luminosity distance [Mpc]
DV      : Volume distance, classical definition (uses DA)
DVdesi  : Volume distance, DESI definition (uses DM)
rs_drag : Sound horizon at the drag epoch [Mpc]
rs_dec  : Sound horizon at decoupling [Mpc]
R       : Baryon-to-photon momentum ratio (3/4 * pb/pf)
lA      : Acoustic scale parameter
zeq     : Matter-radiation equality redshift
z_drag  : Drag epoch redshift (Eisenstein & Hu 1998)
z_dec   : Decoupling redshift (Hu & Sugiyama 1996)

Omega_m  : Total matter Density Parameter
Omega_cdm : Dark energy Density Parameter
Omega_b  : Baryon Density Parameter

Usage
-----
    model = dks_model(M=..., H0=67.4, Omega_cdm=0.22, Omega_b=0.022)
    Hz, z, failed = model.hubble_z(0.5)
    rs, failed    = model.rs_drag()

Notes
-----
- The ODE is solved once at instantiation using scipy's Radau solver.
- All subsequent calls use cached linear interpolators for speed.
- Methods return a trailing boolean `failed` flag instead of raising exceptions.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import quad as int_quad
from scipy.interpolate import interp1d

# CONSTANTS
C_CTE = 2.99792458e5  # [km/s]
G_CTE = 6.6743e-11    # [m^3 kg^-1 s^-2]
TCMB_CTE = 2.7255 


class darksync:
    def __init__(self,# M, 
                 H0, Omega_cdm, Omega_b):
        """
        Class constructor. Initialises the parameters, solves the ODE once,
        and builds interpolators for all quantities.
        """
        self.parameters = {
            #'M': M,
            'H0': H0,
            'Omega_cdm': Omega_cdm,
            'Omega_b': Omega_b
        }
        self.cosmology_failed = False
        # Interpolator cache (filled in _solve_ode)
        self._interp = None
        # rs_drag cache (filled on demand)
        self._rs_drag_cache = None
        # Solve the ODE ONCE at object creation
        self._solve_ode()

    # ------------------------------------------------------------------ #
    #                    FUNCTIONS AND ODE SYSTEM                        #
    # ------------------------------------------------------------------ #

    def event_negative_density(self, z, y):
        """Stops integration if any individual density becomes negative."""
        M_D = [y[0], y[1], y[2], y[3], y[5]]
        return min(M_D)

    event_negative_density.terminal = True
    event_negative_density.direction = -1

    def event_negative_rt(self, z, y):
        """
           p:  energy density.
           de: dark energy, dm: dark matter, r: radiation, b=baryon, g=photon
           Dc: comovel distance
           rt = pdm + pde + pr + pb
           Function:
           Stops integration if the total density r_t becomes negative.
        """
        pdm, pde, pr, pb, Dc, pg = y
        return pdm + pde + pr + pb  # crosses zero when r_t -> 0

    event_negative_rt.terminal = True
    event_negative_rt.direction = -1

    def ode_system(self, z, y):
        """
        System of Friedmann differential equations.

        Ps.: Only for SNia?        
        """
        #M, H0, Omega_cdm, Omega_b = self.parameters.values()
        H0, Omega_cdm, Omega_b = self.parameters.values()
        pdm, pde, pr, pb, Dc, pg = y

        r_t = pdm + pde + pr + pb

        # Safety check: if r_t <= 0 the cosmology is unphysical.
        # Return zero derivatives to trigger a soft stop.
        if r_t <= 0:
            self.cosmology_failed = True
            return [0., 0., 0., 0., 0., 0.]

        Hi = np.sqrt((8 * np.pi * G_CTE / 3) * r_t)
        E = Hi / H0

        dpdmdz = +3. / (1. + z) * pdm
        dpdedz =  0.#Why 0? LCDM?
        dprdz  = +4. / (1. + z) * pr
        dpbdz  = +3. / (1. + z) * pb
        dDCdz  =  1. / E
        dpgdz  = +4. / (1. + z) * pg

        return [dpdmdz, dpdedz, dprdz, dpbdz, dDCdz, dpgdz]

    # ------------------------------------------------------------------ #
    #           ODE SOLVER (called only once in __init__)                 #
    # ------------------------------------------------------------------ #

    def _solve_ode(self):
        """
        Solves the ODE on a dense grid and stores interpolators.
        After this, any call to sol(), Hz(), luminosity_distance(), etc.
        only evaluates the interpolators — without re-solving the ODE.

        Ps.: Only for SNia?  
        """
        #M, H0, Omega_cdm, Omega_b = self.parameters.values()
        H0, Omega_cdm, Omega_b = self.parameters.values()

        # Initial conditions
        pc0 = 3 * H0**2 / (8 * np.pi * G_CTE) #critical density today
        h   = H0 / 100
        Omega_m  = Omega_cdm + Omega_b
        #zeq = 2.5e4 * Omega_m * h**2 * (2.7255 / 2.7)**(-4) #z at equilibrium rad=matter
        self.zeq = 2.5e4 * Omega_m * h**2 * (TCMB_CTE/ 2.7)**(-4)
        zeq = self.zeq  # keep the local name for the rest of _solve_ode
        #Does this zeq depend on the radiation model?

        pdm0 = Omega_cdm * pc0
        pr0  = (Omega_m / (1 + zeq)) * pc0
        pl0  = (1 - Omega_cdm - Omega_b - Omega_m / (1 + zeq)) * pc0
        pb0  = Omega_b * pc0
        pg0  = (2.47e-5 * h**-2) * pc0
        y0   = [pdm0, pl0, pr0, pb0, 0., pg0] #[<-wt represent this '0'? wt are we setting '0'? Comovel distance?]

        # HYBRID redshift grid:
        # - Dense at low z (0 to 5), where likelihoods request data (DESI, Pantheon+) [<-needs to be defined by the user or somehow by lkl]
        # - Coarser at high z (5 to z_max), where only the sound horizon is needed
        # Total ~100k points, with δz ≈ 0.001 at low z #[<- might be interesting to have the number as input, in order to avoid waste time]
        z_max  = 1. / 0.00001 - 1.
        z_low  = np.linspace(0, 5, 5000)           # δz ≈ 0.001
        z_high = np.linspace(5, z_max, 95000)[1:]  # δz ≈ 1.05, no duplicate at z=5
        zR     = np.concatenate([z_low, z_high])

        # Integration (with two terminal safety events)
        sol = solve_ivp(
            self.ode_system, (0, z_max), y0, t_eval=zR,
            method='Radau', rtol=1e-8, atol=1e-10,
            events=[self.event_negative_density, self.event_negative_rt]
        )

        # Check whether integration stopped due to a terminal event OR
        # whether the flag was set inside ode_system (r_t <= 0)
        if sol.status == 1 or self.cosmology_failed:
            self.cosmology_failed = True
            return

        # Drop index 0 (initial condition at z=0):
        # That point is imposed, not integrated by the ODE.
        # With the hybrid grid the next point is at z ~ 0.001,
        # so low-z interpolation works without extrapolation.
        #
        # return: 
        # the parametrization variable as 't'
        # a list of the arrays for each one of the functions (densities and distance) required as part of 'y'
        z_dense   = sol.t[1:]
        pdm_dense = sol.y[0][1:]
        pde_dense = sol.y[1][1:]
        pr_dense  = sol.y[2][1:]
        pb_dense  = sol.y[3][1:]
        Dc_dense  = sol.y[4][1:]
        pg_dense  = sol.y[5][1:]

        # Check for negative densities
        if (np.any(pdm_dense < 0) or np.any(pb_dense < 0) or
            np.any(pde_dense < 0) or np.any(pr_dense < 0)
           # or np.any(pg_dense  < 0)
           ):
            self.cosmology_failed = True
            return

        # Build interpolators (negligible cost compared to solve_ivp)
        kw = dict(kind='linear', fill_value="extrapolate")
        self._interp = {
            'pdm': interp1d(z_dense, pdm_dense, **kw),
            'pde': interp1d(z_dense, pde_dense, **kw),
            'pr':  interp1d(z_dense, pr_dense,  **kw),
            'pb':  interp1d(z_dense, pb_dense,  **kw),
            'Dc':  interp1d(z_dense, Dc_dense,  **kw),
            'pg':  interp1d(z_dense, pg_dense,  **kw),
        }

    # ------------------------------------------------------------------ #
    #        sol() NOW ONLY EVALUATES INTERPOLATORS (fast!)               #
    # ------------------------------------------------------------------ #

    def sol(self, z_requested):
        """
        Returns cosmological quantities at the requested redshift(s).
        Uses cached interpolators — does NOT re-solve the ODE.
        Output:
           - pdm(z), pde(z), pr(z), [z], pb(z), Dc(z), pg(z), FLAG
        """
        if self.cosmology_failed or self._interp is None:
            return None, None, None, None, None, None, None, True #[<- wt this last True and many None means?Flag?]

        z_arr = np.atleast_1d(z_requested)
        f = self._interp

        pdm_out = f['pdm'](z_arr)
        pde_out = f['pde'](z_arr)
        pr_out  = f['pr'](z_arr)
        pb_out  = f['pb'](z_arr)
        Dc_out  = f['Dc'](z_arr)
        pg_out  = f['pg'](z_arr)

        return pdm_out, pde_out, pr_out, z_arr, pb_out, Dc_out, pg_out, False

    # ------------------------------------------------------------------ #
    #             DERIVED QUANTITIES (no changes to the physics)          #
    # ------------------------------------------------------------------ #

    def hubble_z(self, z):
        """
        Computes the Hubble parameter H(z) in km/s/Mpc.
        Output:
           - H(z), [z], FLAG          
        """
        rho = self.sol(z)
        if rho[-1]:  # cosmology_failed
            return None, None, True #[AM: <- we need to explain somewhere what these/those sequencies mean; 
                                    #    otherwise, if we need to edit it will be a nightmare

        pdm, pde, pr, zS, pb, Dc, _, _ = rho
        r_t = pdm + pde + pr + pb
        # Guard against r_t <= 0 from extrapolation
        if np.any(r_t <= 0):
            return None, None, True
        Hz = np.sqrt((8 * np.pi * G_CTE / 3) * r_t)
        return Hz, zS, False

    #def Dcmodel(self, z): #AM: Only FLAT
    def dimensionless_comoving_distance(self, z): #AM: Only FLAT
        """
        Dimensionless comoving distance Dc/dH.
           Dc(z) = DC(z)/(c/H0) [integral term only: int_0^z dz'/E(z')]
        Output:           
           - Dc(z), [z], FLAG        
        """
        soldc = self.sol(z)
        if soldc[-1]:
            return None, None, True
        return soldc[5], soldc[3], False

    #def dmmodel(self, z):#AM: esses nomes podem gerar cnfusao com modelo de DarkMatter
    def comoving_distance(self, z):
        """
        Radial comoving distance DM in Mpc.
           DM(z) = (c/H0)*Dc(z) [c/H0*int_0^z dz'/E(z')]        
        Output:
          - 
        """
        _vars_ = self.dimensionless_comoving_distance(z)
        if _vars_[2]:
            return None, None, True

        H0 = self.parameters['H0']
        dH = C_CTE / H0  # Mpc
        return dH * _vars_[0], _vars_[1], False

    #def dlmodel(self, z):
    def luminosity_distance(self, z):
        """
        Luminosity distance dL in Mpc.
        Output:
           dL(z), [z], FLAG
        """
        dlI = self.comoving_distance(z)
        if dlI[2]:
            return None, None, True
        return (1 + dlI[1]) * dlI[0], dlI[1], False

    #def dAmodel(self, z):
    def angular_diameter_distance(self, z):        
        """
        Angular diameter distance DA in Mpc.
        output:
        - DA(z), [z], FLAG
        """
        daI = self.comoving_distance(z)
        if daI[2]:
            return None, None, True
        return daI[0] / (1 + daI[1]), daI[1], False

    #def mumodel(self, z):
    def modulus_distance(self, z):        
        """
        Distance modulus.
        mu = 5log10(Dl) + 25
        output:
        - mu(z), [z], FLAG        
        
        """
        dl = self.luminosity_distance(z)
        if dl[2]:
            return None, True
        return 5. * np.log10(dl[0]) + 25, False

    #def DV(self, z):# [AM: Estava definido como funcao e tambem cmo variavel]
    def volume_distance(self, z):        
        """
        Volume distance DV(z) — classical definition using DA.
        DV(z) = [ (c*z/H0) * (DA(1+z))**2 ]**1/3
        output:
          DV(z), [z], FLAG
        """
        DAV = self.angular_diameter_distance(z)
        if DAV[2]:
            return None, None, True

        DA  = DAV[0]
        zDA = DAV[1]
        HI  = self.hubble_z(z)
        H   = HI[0]

        dv3 = ((1. + zDA)**2 * DA**2 * C_CTE * zDA) / H
        DV  = dv3**(1. / 3.)
        return DV, zDA, False

    #def DVdesi(self, z):
    def volume_distance_desi(self, z):        
        """
        Volume distance DV(z) — DESI definition using DM.
        DV(z) = [ (c*z/H0) * DM**2 ]**1/3
        output:
          DVdesi(z), [z], FLAG        
        """
        DMV = self.comoving_distance(z)
        if DMV[2]:
            return None, None, True

        DM  = DMV[0]
        zDM = DMV[1]
        HI  = self.hubble_z(z)
        H   = HI[0]

        dv3 = (DM**2 * C_CTE * zDM) / H
        DVM = dv3**(1. / 3.)
        return DVM, zDM, False

    # ------------------------------------------------------------------ #
    #                         SOUND HORIZON                               #
    # ------------------------------------------------------------------ #

    def drsda(self):
        """
        Derivative of the sound horizon with respect to the scale factor.
        dr_s/da: derived sound horizon in terms of a
        output:
          interpolate dr_s/da
        """
        a_scale0 = np.logspace(np.log10(1.0), np.log10(0.00001), 100000)
        zt  = 1. / a_scale0 - 1.
        rho = self.sol(zt)  # Fast: interpolation only!
        if rho[-1]:
            return None, True

        pdm, pl, pr, zS, pb, Dc, pg, _ = rho
        a_scale1 = 1. / (1. + zS)

        # Guard against slightly negative values from extrapolation
        r_t = np.maximum(pdm + pl + pr + pb, 1e-30)
        H = np.sqrt((8 * np.pi * G_CTE / 3) * r_t)
        R = 3. / 4. * (pb / np.maximum(pg, 1e-30))
        drsda = C_CTE / (a_scale1**2 * H * np.sqrt(3. * (1. + R)))

        drsint = interp1d(a_scale1[::-1], drsda[::-1], kind='linear', fill_value="extrapolate")
        return drsint, False

    def rs_drag(self):
        """
        Computes the sound horizon at the drag epoch (with caching)  [Mpc].
        Eisenstein & Hu fitting formula for z_drag (z at drag epoch)
        Output:
          rs(zdrag), FLAG
        """
        # Return cached value if already computed
        if self._rs_drag_cache is not None:
            return self._rs_drag_cache, False

        #M, Omega_cdm, Omega_b = self.parameters.values() 
        H0, Omega_cdm, Omega_b = self.parameters.values()
        h  = H0 / 100
        Omega_m = Omega_cdm + Omega_b

        # Eisenstein & Hu fitting formula for z_drag
        b_1 = 0.313 * (Omega_m * h**2)**(-0.419) * (1 + 0.607 * (Omega_m * h**2)**0.674)
        b_2 = 0.238 * (Omega_m * h**2)**0.223
        z_drag = (1345 * ((Omega_m * h**2)**0.251 / (1 + 0.659 * (Omega_m * h**2)**0.828))
                  * (1 + b_1 * (Omega_b * h**2)**b_2))
        self.z_drag = z_drag 
        a_drag = 1. / (1. + z_drag)

        drsint_result = self.drsda()
        if drsint_result[1]:  # failed
            return None, True

        drsint = drsint_result[0]
        rs_drag_val, _ = int_quad(drsint, 0.0, a_drag)

        # Store in cache
        self._rs_drag_cache = rs_drag_val
        return rs_drag_val, False

    # ------------------------------------------------------------------ #
    #              ADDITIONAL QUANTITIES (CMB, Omega, etc.)               #
    # ------------------------------------------------------------------ #      
    
    def z_dec(self):
        """
        Redshift at decoupling photon from the matter
        Output:
           z(dec)
        """
        M, H0, Omega_cdm, Omega_b = self.parameters.values()
        h  = H0 / 100
        Omega_m = Omega_cdm + Omega_b
        g1 = (0.0738 * (Omega_b * h**2)**-0.238) / (1 + 39.5 * (Omega_b * h**2)**0.763)
        g2 = 0.560 / (1 + 21.1 * (Omega_b * h**2)**1.81)
        z_dec = 1048 * (1 + 0.00124 * (Omega_b * h**2)**-0.738) * (1 + g1 * (Omega_m * h**2)**g2)
        return z_dec

    def rs_dec(self):
        """
        Sound horizon at decoupling [Mpc].
        output:
            rs(zdec), FLAG
        """
        a_dec = 1. / (1. + self.z_dec())
        drsint_result = self.drsda()
        if drsint_result[1]:
            return None, True
        drsint = drsint_result[0]
        rs_dec_val, _ = int_quad(drsint, 0.0, a_dec)
        return rs_dec_val, False

    def shift(self):
        """
        Shift parameter R.
        R = (H0/c)*sqrt(Omega_m)*Comoving_Distance(zdec)
        Output:
           R, FLAG
        """
        #M, H0, Omega_cdm, Omega_b = self.parameters.values()
        H0, Omega_cdm, Omega_b = self.parameters.values()
        Omega_m = Omega_cdm + Omega_b
        zdec = self.z_dec()
        dm = self.comoving_distance(zdec)
        if dm[2]:
            return None, True
        sh = H0 * np.sqrt(Omega_m) * dm[0] / C_CTE
        return float(np.atleast_1d(sh)[0]), False

    def lA(self):
        """
        Acoustic parameter lA.
        lA= pi * Comoving_Distance(zdec) / rs(zdec)
        Output:
           lA, FLAG        
        """
        zdec = self.z_dec()
        rdec = self.comoving_distance(zdec)
        if rdec[2]:
            return None, True
        rs_val = self.rs_dec()[0]
        lAv = np.pi * rdec[0] / rs_val
        return float(np.atleast_1d(lAv)[0]), False

    #def Omega(self, z):#[AM: EVITAR ESSE NOME PARA FUNCAO]
    def Omega_parameters(self, z):
        """
        Density fractions Ωdm, Ωde, Ωr, Ωb.
        rho_c   = rho_dm+rho_de+rho_rad+rho_b
        Omega_x = rho_x/rho_c
        Output:
           Omega_cdm(z),Omega_de(z),Omega_rad(z),[z],Omega_b(z) 
        """
        pdm, pde, pr, z_sol, pb, Dc, _, failed = self.sol(z)
        if failed:
            return None, None, None, None, None
        pcr = pdm + pde + pr + pb
        return pdm / pcr, pde / pcr, pr / pcr, z_sol, pb / pcr

    def display_info(self):
        """Displays the parameters and current state."""
        print("Parameters:")
        for name, value in self.parameters.items():
            print(f"  {name}: {value}")
        print(f"Cosmology failed: {self.cosmology_failed}")
        if not self.cosmology_failed:
            rs, _ = self.rs_drag()
            print(f"rs_drag = {rs:.4f} Mpc")
    def get_zeq(self):
        """
        Redshift at matter-radiation equality.
        zeq = 2.5e4 * (Omega_cdm+Omega_b) * h^2 * (T_cmb/2.7)^-4
        (Dodelson & Schmidt eq. 2.88 / Kolb & Turner)
        Output:
           zeq, FLAG
        """
        if self.cosmology_failed:
            return None, True
        return self.zeq, False
    def get_zdrag(self):
        """
        Redshift at the drag epoch (baryon decoupling from photons).
        Eisenstein & Hu (1998) fitting formula.
        Output:
           z_drag, FLAG
        """
        if self.cosmology_failed:
            return None, True
            # rs_drag() must be called first to populate self.z_drag
        if not hasattr(self, 'z_drag'):
            _, fail = self.rs_drag()
            if fail:
                return None, True
        return self.z_drag, False              