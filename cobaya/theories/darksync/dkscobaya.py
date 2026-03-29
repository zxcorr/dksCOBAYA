import numpy as np
from cobaya.theory import Theory
from cobaya.log import LoggedError
from .darksync import darksync


# [M]: To highlight part to be modified later
# [AM]: For comments

class dksLCDM(Theory):

    def initialize(self):
        """Called once by Cobaya at startup."""
        self.model = None
        self._rd_cache = None  # rs_drag cache for the current step

    def get_requirements(self):
        """
        Input parameters that Cobaya must supply. List of parameters to be sampled
        Output:
            List of parameters to be sampled
        """
        #return ['M', 'H0', 'Omega_dm', 'Omega_b']# [M]
        return ['H0', 'Omega_dm', 'Omega_b']# [M]

    def get_can_provide(self):
        """
        Function-quantities that this theory provides to the likelihoods.
        Each entry corresponds to a get_*() method below.
        """
        return [
            'luminosity_distance',                 # SN Ia (Pantheon+)
            'angular_diameter_distance',           # BAO (DA)
            'comoving_angular_diameter_distance',  # BAO (DM = (1+z)*DA)
            'Hubble',                              # BAO (H(z))
        ]

    def get_can_provide_params(self):# [M]
        """
        Derived parameters computed by this theory.
        The BAO likelihood retrieves rdrag via get_param("rdrag"),
        which looks in state["derived"] rather than calling get_rdrag().
        """
        return ['rdrag']

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Called at every MCMC step. Instantiates the cosmological model,
        which internally solves the ODE ONCE and builds the interpolators.
        """
        #M        = params_values_dict['M']
        H0       = params_values_dict['H0']
        Omega_dm = params_values_dict['Omega_dm']
        Omega_b  = params_values_dict['Omega_b']

        try:
            # The ODE is solved inside darksync.__init__
            #self.model = darksync(M, H0, Omega_dm, Omega_b)
            self.model = darksync(H0, Omega_dm, Omega_b)#[AM: We need to pay attention to model 
                                                  #this part in order to accept other parameters]

            # Check whether the cosmology failed during integration
            if self.model.cosmology_failed:
                raise LoggedError(
                    self.log,
                    "Cosmology failed: negative density during integration."
                )

            # Pre-compute and cache rs_drag (involves a quad, better done once)
            rd_val, rd_fail = self.model.rs_drag()
            if rd_fail or rd_val is None or not np.isfinite(rd_val) or rd_val <= 0:
                raise LoggedError(
                    self.log,
                    f"Invalid rs_drag: {rd_val}"
                )
            self._rd_cache = rd_val

            # Store rdrag as a derived parameter in the state dict.
            # This is how the BAO likelihood finds it: via get_param("rdrag").
            state["derived"] = {"rdrag": rd_val}

        except LoggedError:
            raise  # Let Cobaya handle it (point is rejected)
        except Exception as e:
            raise LoggedError(self.log, f"Error in model calculation: {e}")

    # ------------------------------------------------------------------ #
    #                   get_* METHODS FOR THE LIKELIHOODS                 #
    # ------------------------------------------------------------------ #

    def get_luminosity_distance(self, z):
        """
        Luminosity distance dL(z) in Mpc.
        Used by Pantheon+ (sn.pantheonplus).
        Output:
           dL(z) :: array
        """
        z = np.atleast_1d(z)
        dl_vals = np.empty_like(z, dtype=float)

        for i, zi in enumerate(z):
            val, _, fail = self.model.luminosity_distance(float(zi))#dL(z), [z]-ODE, FLAG
            if fail:
                raise LoggedError(
                    self.log,
                    f"Failed to compute dL at z={zi}"
                )
            dl_vals[i] = np.atleast_1d(val)[0]

        return dl_vals

    def get_angular_diameter_distance(self, z):
        """
        Angular diameter distance DA(z) in Mpc.
        Output:
           DA(z) :: array        
        """
        z = np.atleast_1d(z)
        da_vals = np.empty_like(z, dtype=float)

        for i, zi in enumerate(z):
            val, _, fail = self.model.angular_diameter_distance(float(zi))#DA(z), [z]-ODE, FLAG
            if fail:
                raise LoggedError(
                    self.log,
                    f"Failed to compute DA at z={zi}"
                )
            da_vals[i] = np.atleast_1d(val)[0]

        return da_vals

    def get_comoving_angular_diameter_distance(self, z):
        """
        Radial comoving distance DM(z) in Mpc.
        DM = (1+z) * DA. Some BAO likelihoods request this directly.
        """
        z = np.atleast_1d(z)
        dm_vals = np.empty_like(z, dtype=float)

        for i, zi in enumerate(z):
            val, _, fail = self.model.comoving_distance(float(zi))#DM(z), [z]-ODE, FLAG
            if fail:
                raise LoggedError(
                    self.log,
                    f"Failed to compute DM at z={zi}"
                )
            dm_vals[i] = np.atleast_1d(val)[0]

        return dm_vals

    def get_Hubble(self, z, units="km/s/Mpc"):
        """
        Hubble parameter H(z).
        The model computes it internally in km/s/Mpc.
        The BAO likelihood may request '1/Mpc' units (H/c),
        so the result is converted according to the 'units' argument.
        """
        # Speed of light in km/s
        _c_km_s = 2.99792458e5

        z = np.atleast_1d(z)
        h_vals = np.empty_like(z, dtype=float)

        for i, zi in enumerate(z):
            val, _, fail = self.model.hubble_z(float(zi))
            if fail:
                raise LoggedError(
                    self.log,
                    f"Failed to compute H(z) at z={zi}"
                )
            h_vals[i] = np.atleast_1d(val)[0]

        # Convert units if needed
        if units == "1/Mpc":
            h_vals = h_vals / _c_km_s

        return h_vals