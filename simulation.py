from core.KineticModel import KineticModel, AirKineticModel
from typing import Callable, List
import numpy as np
from core.bolos import solver
from scipy.optimize import fsolve

class Simulation:
    def __init__(
        self,
        kinet: KineticModel,
        movement_fcn: Callable[[float], float],
        source_voltage: float,
        source_impedance: float,
        dt: float = 1e-3,
        end_time: float = None
    ):
        """Initialize simulation parameters.
        movement_fcn is a function of time t returning a value."""
        self.kinet = kinet
        self.movement_fcn = movement_fcn

        self.k_b = 1.380649e-23 #Boltzmann constant [J.K^-1]
        self.q = 1.60217663e-19 #Elementary charge [C]

        # Initial conditions

        # For now, source impedance is a float and not a complex. 
        # It is because the arc is resistive and we only care about amplitude for now
        self.source_voltage = source_voltage
        self.source_impedance = source_impedance

        #Arc section (supposed constant for now)
        self.arc_section = np.pi*(1e-3)**2 / 4 # 1mm diameter arc

        # Minimal E/N and conductivity to avoid numerical collapse
        self.E_N_min = 1e-2  # Td
        self.sigma_min = 1e-10  # S/m
        self.n_e = 1e20 # Initial electron density [m^-3]
        self.n_0 = 2.5e25 # Initial (fixed for now) air density in atmospheric conditions.

        # Time
        self.t = dt #Do not start at 0 otherwise solver will not converge.
        self.dt = dt
        self.end_time = end_time

    def __iter__(self):
        return self
    
    def __next__(self):
        """Advance the simulation by one step."""
        
        # If the simulation is finished
        if self.end_time is not None and self.t >= self.end_time:
            raise StopIteration

        # Compute arc length using the provided movement function.
        self.arc_length = self.movement_fcn(self.t) 

        def func_E(E):
            """
            Implicit equation for the arc electric field used inside fsolve.

            This function evaluates the residual of the circuit equation
            for a given trial value of the electric field `E`.

            Steps performed:
            1. Compute the reduced electric field E/N (Townsend units) from E.
            A lower bound (self.E_N_min) ensures numerical stability.

            2. Update the kinetic model (EEDF) at this reduced field. This gives
            the correct electron energy distribution for the assumed E/N.

            3. From the updated EEDF, compute the transport parameters
            (mainly the arc conductivity sigma). For the implicit solve,
            densities are not updated — only transport coefficients
            are recomputed.

            4. Compute the arc current I = sigma * E * section.

            5. Return the residual of the circuit equation:
                f(E) = E - (V_source - R_source * I) / arc_length
            The root f(E)=0 corresponds to the consistent value of E
            that balances the plasma column with the external circuit.

            Parameters
            ----------
            E : float
                Trial value of the electric field [V/m].

            Returns
            -------
            residual : float
                Value of the circuit equation residual for this E.
                fsolve seeks the root where residual = 0.
            """

            # 1. Reduced field
            E_N = max(E / (self.n_0 * solver.TOWNSEND), self.E_N_min)

            # 2. Update kinetic model
            self.kinet.variables["EN"] = E_N
            self.kinet.update_eedf()
            self.kinet.update_reaction_rates()

            # 3. Transport parameters (no density update here)
            temp_n = self.kinet.update_densities(
                electronic_only=True,
                update_densities=False
            )

            n_e = temp_n.get("e-", self.kinet.n.get("e-", 0.0))
            sigma = max(n_e * self.kinet.mu_e * self.q, self.sigma_min)
            
            # 4. Compute arc current
            I = sigma * E * self.arc_section

            # 5. Return residual of circuit equation
            return E - (self.V_source - self.R_source * I) / self.arc_length
        
        # Initial guess for E
        E_guess = self.source_voltage / self.arc_length

        # Solve implicit equation
        E_solution, info, ier, mesg = fsolve(func_E, E_guess, full_output=True)
        if ier != 1:
            raise RuntimeError(f"fsolve failed at time {self.t:.6f}s: {mesg}")
        E = E_solution[0]

        # Compute reduced field
        reduced_field = max(E / (self.arc.n_0 * solver.TOWNSEND), self.E_N_min)
        self.reduced_fields[self.step] = reduced_field

        # Update kinetics fully at the converged field
        self.kinet.update_eedf(reduced_field)
        self.kinet.update_reaction_rates()
        self.kinet.update_densities(electronic_only=False, update_densities=True)

        # Get electron density
        n_e = self.kinet.n.get("e-", 0.0)
        self.electron_densities[self.step] = n_e

        # Update conductivity
        self.kinet.conductivity = max(n_e * self.q * self.kinet.mu_e, self.sigma_min)

        # Compute current and voltage
        self.arc_current[self.step] = max(E * self.kinet.conductivity * self.arc_section, 0.0)
        self.arc_voltage[self.step] = E * self.arc_length
        self.arc_conductivity[self.step] = self.kinet.conductivity

        print(f"Time {self.t:.6f}s: E = {E:.3e} V/m, I = {self.arc_current[self.step]:.3e} A")

        # Advance time
        value = self.movement_fcn(self.t)
        self.t += self.dt
        self.step += 1
        return value

    def run(self) -> List[float]:
        results = []
        while self.t < self.end_time:
            results.append(next(self))
        return results

if __name__ == "__main__":
    import math

    kinet = AirKineticModel("air-cross-sections.txt")

    def sinusoidal(t: float) -> float:
        return math.sin(t)

    sim = Simulation(kinet, sinusoidal, 3e3, 100, dt=0.1, end_time=1.0)

    # Boucle pas à pas
    for val in sim:
        print(val)

    # Réinitialiser et calculer tout d’un coup
    sim.t = 0.0
    all_values = sim.run()
    print(all_values)
