import numpy as np
from .bolos import parser, grid, solver
from abc import ABC, abstractmethod

class Reaction:
    """Represents a reaction with fixed formula-based, or BOLOS-computable rate."""

    def __init__(self, reactants, products, rate_expression, is_superelastic = False):
        """
        reactants, products: list of strings
        rate_expression: either a float (fixed) or a callable(variables) -> float
        is_superelastic: is used to flag superelastic reactions, which are computed in post-processing.
        """
        self.reactants = reactants
        self.products = products
        self.rate_expression = rate_expression
        self.is_superelastic = is_superelastic

    def __repr__(self):
        return f"Reaction({self.reactants} -> {self.products}, {self.rate_expression})\n"

class KineticModel:
    """
    Represents a complete kinetic model with reactions and reaction rates.
    Can provide reaction rates at specitic conditions. 
    Also contains the EEDF which can be used to compute transport parameters.
    """

    def __init__(self, cross_sections_fp: str, dt: float = 1e-3, end_time: float = None):
        """
        cross_sections_fp: cross sections file path. BOLSIG+ cross sections files are supported
        and can be directly retrieved from LXCat.
        """

        # Initialize Boltzmann solver and a grid (0 -> 20 eV, 200 cells.) 
        self.solver = solver.BoltzmannSolver(grid.LinearGrid(1e-6, 20., 100)) #Don't start at zero otherwise BOLOS will not converge.

        # Load cross-sections.
        with open(cross_sections_fp) as fp:
            processes = parser.parse(fp)
        self.solver.load_collisions(processes)

        # Add initial conditions here...

        #Gas temperature (will not be constant but it is for now.)
        self.solver.kT = self.variables["Tgas"] * solver.KB / solver.ELECTRONVOLT

        # Time
        self.t = dt #Do not start at 0 otherwise solver will not converge.
        self.dt = dt
        self.end_time = end_time

    def all_species(self):
        species_set = set()
        for r in self.reactions:
            species_set.update(r.species())
        return sorted(species_set)
    
    def update_eedf(self):
        """Recompute and update eedf at current reduced field (must be defined !)."""
        self.solver.EN = self.variables["EN"] * solver.TOWNSEND
        self.solver.init()
        # Iterative method => get a first guess using a Maxwellian distribution at 2eV.
        fMaxwell = self.solver.maxwell(2.0)
        # Solve iteratively
        self.eedf = self.solver.converge(fMaxwell, maxn=100, rtol=1e-5)

    def update_reaction_rates(self):
        """
        Updates reaction rates at current variables.
        Reduced electric field should be set at the right value.
        """

        # TODO : Implement thermionic emission and update with temperature.
        # Before that, temperature must be dynamic and not fixed.
        # The model should define the rates not defined by databases. Processes such as recombination, thermionic emission etc.
        self.reactions_with_rates = []
        self.superelastics = []

        for i, r in enumerate(self.reactions):
            tmp_r = r

            # If process is super-elastic, don't try to compute it using BOLOS for now.
            # Process will not contribute to the EEDF (Beware, can be inaccurate !).
            if r.is_superelastic:
                self.superelastics.append(r)

            elif callable(r.rate_expression):
                # Lambda expression
                tmp_r.rate_expression = float(r.rate_expression())
                self.reactions_with_rates.append(tmp_r)
            elif isinstance(r.rate_expression, str):
                # BOLOS expression
                # Supprime le préfixe "BOLSIG " dans l'équation de la réaction si présent
                tmp_r.rate_expression = tmp_r.rate_expression.removeprefix("BOLSIG ")
                tmp_r.rate_expression = self.solver.rate(self.eedf, tmp_r.rate_expression)
                self.reactions_with_rates.append(tmp_r)
            else:
                # Constant value
                tmp_r.rate_expression = float(r.rate_expression)
                self.reactions_with_rates.append(tmp_r)

        for reaction in self.superelastics:
            print(f"superelastic : {reaction}")

    def update_densities(self, electronic_only = False, update_densities = True):
        """
        Updates densities of species. 
        When electronic_only is set to True, only electronic processes will be recomputed.
        When update_densities is set to False, the densities will not be stored.
        """

        # Electronic processes are the ones computed by BOLOS but there are certain processes that BOLOS doesn't give (e.g. recombination).
        # The safest way is to directly list all the reactions with electrons involved
        electron_loss_processes = []
        electron_gain_processes = []
        other_processes = []

        for r in self.reactions_with_rates:
            if "e" in r.reactants and "e" not in r.products:
                electron_loss_processes.append(r)
            elif "e" in r.products and "e" not in r.reactants:
                electron_gain_processes.append(r)
            else:
                other_processes.append(r)

        # Determine which species to update
        if electronic_only:
            species_to_update = ["e"]
        else:
            species_to_update = self.all_species()

        # Temporary storage for densities if update_densities=False
        temp_n = {}

        for s in species_to_update:
            delta_n = 0.0

            # Electron-specific processes
            if s == "e":
                for r in electron_loss_processes:
                    delta_n -= r.rate_expression * self.dt
                for r in electron_gain_processes:
                    delta_n += r.rate_expression * self.dt

            # Other processes
            if not electronic_only:
                for r in other_processes:
                    if s in r.reactants:
                        delta_n -= r.rate_expression * self.dt
                    if s in r.products:
                        delta_n += r.rate_expression * self.dt

            # Store densities
            if update_densities:
                if s in self.n:
                    self.n[s] += delta_n
                else:
                    self.n[s] = delta_n
            else:
                temp_n[s] = self.n.get(s, 0.0) + delta_n

        # Compute mobility
        self.mu_e = self.solver.mobility(self.eedf)

        if not update_densities:
            # Return temporary densities if needed
            return temp_n
        else:
            self.n = temp_n
            return temp_n

# Define air kinetic model
class AirKineticModel(KineticModel):
    """
    KINETIC FILE FOR NITROGEN-OXYGEN MIXTURES
    Python version, ported from original Fortran code.
    Originally to be used with ZDPlasKin library (http://www.zdplaskin.laplace.univ-tlse.fr)

    Python port
    Python porting from FORTRAN
    August 2025 by Elie HASSID (elie.hassid@protonmail.com)

    version 1.03
    NxOy species are added from [Capitelli2000]: N2O NO2 NO3 N2O5 N2O^+ NO2^+ N2O^- NO2^- NO3^-
    August 2010 by Sergey PANCHESHNYI (sergey.pancheshnyi@laplace.univ-tlse.fr)

    version 1.02
    Minor compatibility modifications
    April 2010 by Sergey PANCHESHNYI (sergey.pancheshnyi@laplace.univ-tlse.fr)

    version 1.01
    The first version based mainly on [Capitelli2000]
    September 2008 by Sergey PANCHESHNYI (sergey.pancheshnyi@laplace.univ-tlse.fr) and Aicha FLITTI (flittiche@yahoo.fr)

    References
    [Capitelli2000] M. Capitelli, C.M. Ferreira, B.F. Gordiets and A.I. Osipov,
      "Plasma Kinetics in Atmospheric Gases" (2000), Springer.
    [Gordiets1995]  B.F. Gordiets, C.M. Ferreira, V.L. Guerra, J.M.A.H. Loureiro,
      J. Nahorny, D. Pagnon, M. Touzeau and M. Vialle,
      IEEE Transactions on Plasma Science vol. 23 (1995) 750-768.
    [Guerra2004]    V. Guerra, P.A. Sa and J. Loureiro,
      Eur. Phys. J. Appl. Phys. vol. 28 (2004) 125-152.
    [Kossyi1992]    I.A. Kossyi, A.Yu. Kostinsky, A.A. Matveyev and V.P. Silkov,
      Plasma Sources Sci. Technol. vol. 1 (1992) 207-220.
    """

    def __init__(self, cross_sections_fp: str):

        # Global variables
        self.variables = {}
        self.reactions = []

        self.variables["Tgas"] = 300.0 #Gas Temperature
        self.variables["EN"] = 100.0 #Reduced field in Td
        self.variables["Te"] = 3000.0 #Electronic temperature

        super().__init__(cross_sections_fp)

        # Initial conditions
        self.solver.target['N2'].density = 0.80
        self.solver.target['O2'].density = 0.20

        # Constants
        kB = 1.3807e-16       # Boltzmann constant
        m_u = 1.6605e-24      # Atomic mass

        #--------------------------------------------------------------------------------
        #
        # ion and ion-neutral effective temperatures
        #
        # mobilities are tacken from McKnight L.G., McAfee K.B. and Sipler D.P.,
        # "Low-ﬁeld drift velocities and reactions of nitrogen ions in nitrogen",
        # Physical Review 164 62-70 (1967).
        #
        #--------------------------------------------------------------------------------

        # Ionic temperatures
        self.variables["dTion"] = lambda: 2.0 / (3.0 * kB) * m_u * (1.0e-17 * self.variables["EN"])**2
        self.variables["TionN"] =  lambda: self.variables["Tgas"] + self.variables["dTion"]() * 14.0 * (8.0e19)**2
        self.variables["TionN2"] = lambda: self.variables["Tgas"] + self.variables["dTion"]() * 28.0 * (4.1e19)**2
        self.variables["TionN3"] = lambda: self.variables["Tgas"] + self.variables["dTion"]() * 42.0 * (6.1e19)**2
        self.variables["TionN4"] = lambda: self.variables["Tgas"] + self.variables["dTion"]() * 56.0 * (7.0e19)**2

        # Effective temperatures
        self.variables["TeffN"] =  lambda: (self.variables["TionN"]()  + 0.5 * self.variables["EN"]) / (1.0 + 0.5)
        self.variables["TeffN2"] = lambda: (self.variables["TionN2"]() + 1.0 * self.variables["EN"]) / (1.0 + 1.0)
        self.variables["TeffN3"] = lambda: (self.variables["TionN3"]() + 1.5 * self.variables["EN"]) / (1.0 + 1.5)
        self.variables["TeffN4"] = lambda: (self.variables["TionN4"]() + 2.0 * self.variables["EN"]) / (1.0 + 2.0)

        #--------------------------------------------------------------------------------
        #
        # rotational excitation and relaxation (fast quenching)
        #
        #--------------------------------------------------------------------------------

        # SPECIES +: N2(rot) O2(rot)
        # e + N2 => e + N2(rot)                             !   BOLSIG N2 -> N2(rot)
        # e + O2 => e + O2(rot)                             !   BOLSIG O2 -> O2(rot)
        # N2(rot) + ANY_NEUTRAL => N2 + ANY_NEUTRAL         !   ...
        # O2(rot) + ANY_NEUTRAL => O2 + ANY_NEUTRAL         !   ...

        #--------------------------------------------------------------------------------
        #
        # vibrational excitation / de-excitation by electron impact [BOLSIG+]
        #
        #--------------------------------------------------------------------------------

        self.reactions.append(Reaction(["e", "N2"], ["e", "N2(v1)"], "BOLSIG N2 -> N2(v1res)"))

        for i in range(1, 9):
            self.reactions.append(Reaction(["e", "N2"], ["e", f"N2(v{i})"], f"BOLSIG N2 -> N2(v{i})"))
            self.reactions.append(Reaction(["e", f"N2(v{i})"], ["e", "N2"], f"BOLSIG N2(v{i}) -> N2", is_superelastic=True))

        self.reactions.append(Reaction(["e", "O2"], ["e", "O2(v1)"], "BOLSIG O2 -> O2(v1res)"))

        for i in range(1, 5):
            self.reactions.append(Reaction(["e", "O2"], ["e", f"O2(v{i})"], f"BOLSIG O2 -> O2(v{i})"))
            self.reactions.append(Reaction(["e", f"O2(v{i})"], ["e", "O2"], f"BOLSIG O2(v{i}) -> O2", is_superelastic=True))

            
        #--------------------------------------------------------------------------------
        #
        # vibrational-translational relaxation [Capitelli2000, page 105]
        #
        #--------------------------------------------------------------------------------

        # Vibrational energies (K)
        self.variables["energy_vibN2"] = 0.290 * 11605.0
        self.variables["energy_vibO2"] = 0.190 * 11605.0

        # Partition functions (recalculables)
        self.variables["QvibN2"] = lambda: np.exp(-self.variables["energy_vibN2"] / self.variables["Tgas"])
        self.variables["QvibO2"] = lambda: np.exp(-self.variables["energy_vibO2"] / self.variables["Tgas"])

        # N2 coefficients
        self.variables["kVT10_N2N2"] = lambda: (7.80e-12 * self.variables["Tgas"] * 
            np.exp(-218.0 / (self.variables["Tgas"])**(1/3) + 690.0 / self.variables["Tgas"]) /
            (1.0 - self.variables["QvibN2"]()))
        


        self.variables["kVT10_N2N"] =  lambda: 4.00e-16 * (self.variables["Tgas"] / 300.0)**0.5
        self.variables["kVT10_N2O"] = lambda: 1.20e-13 * np.exp(-27.6 / self.variables["Tgas"])

        self.variables["kVT01_N2N2"] = lambda: self.variables["kVT10_N2N2"]() * self.variables["QvibN2"]()
        self.variables["kVT01_N2N"] =   lambda: self.variables["kVT10_N2N"]()  * self.variables["QvibN2"]()
        self.variables["kVT01_N2O"] =   lambda: self.variables["kVT10_N2O"]()  * self.variables["QvibN2"]()

        # O2 coefficients
        self.variables["kVT10_O2O2"] = lambda: (1.35e-12 * self.variables["Tgas"] *
            np.exp(-137.9 / self.variables["Tgas"]**(1/3)) /
            (1.0 - self.variables["QvibO2"]())
        )

        self.variables["kVT10_O2O"] =  lambda: 4.50e-15 * self.variables["Tgas"]
        self.variables["kVT01_O2O2"] = lambda: self.variables["kVT10_O2O2"]() * self.variables["QvibO2"]()
        self.variables["kVT01_O2O"] =  lambda: self.variables["kVT10_O2O"]()  * self.variables["QvibO2"]()

        #--------------------------------------------------------------------------------
        # self.reactions
        #--------------------------------------------------------------------------------

        # N2 + N2
        for i in range(1, 9):
            self.reactions.append(
                Reaction(
                    [f"N2(v{i})", "N2"],
                    [f"N2(v{i-1})" if i > 1 else "N2", "N2"],
                    lambda i=i: self.variables["kVT10_N2N2"]() * i
                )
            )
            self.reactions.append(
                Reaction(
                    [f"N2(v{i-1})" if i > 1 else "N2", "N2"],
                    [f"N2(v{i})", "N2"],
                    lambda i=i: self.variables["kVT01_N2N2"]() * i
                )
            )

        # N2 + N
        for i in range(1, 9):
            self.reactions.append(
                Reaction(
                    [f"N2(v{i})", "N"],
                    [f"N2(v{i-1})" if i > 1 else "N2", "N"],
                    lambda i=i: self.variables["kVT10_N2N"]() * i
                )
            )
            self.reactions.append(
                Reaction(
                    [f"N2(v{i-1})" if i > 1 else "N2", "N"],
                    [f"N2(v{i})", "N"],
                    lambda i=i: self.variables["kVT01_N2N"]() * i
                )
            )

        # N2 + O
        for i in range(1, 9):
            self.reactions.append(
                Reaction(
                    [f"N2(v{i})", "O"],
                    [f"N2(v{i-1})" if i > 1 else "N2", "O"],
                    lambda i=i: self.variables["kVT10_N2O"]() * i
                )
            )
            self.reactions.append(
                Reaction(
                    [f"N2(v{i-1})" if i > 1 else "N2", "O"],
                    [f"N2(v{i})", "O"],
                    lambda i=i: self.variables["kVT01_N2O"]() * i
                )
            )

        # O2 + O2
        for i in range(1, 5):
            self.reactions.append(
                Reaction(
                    [f"O2(v{i})", "O2"],
                    [f"O2(v{i-1})" if i > 1 else "O2", "O2"],
                    lambda i=i: self.variables["kVT10_O2O2"]() * i
                )
            )
            self.reactions.append(
                Reaction(
                    [f"O2(v{i-1})" if i > 1 else "O2", "O2"],
                    [f"O2(v{i})", "O2"],
                    lambda i=i: self.variables["kVT01_O2O2"]() * i
                )
            )

        # O2 + O
        for i in range(1, 5):
            self.reactions.append(
                Reaction(
                    [f"O2(v{i})", "O"],
                    [f"O2(v{i-1})" if i > 1 else "O2", "O"],
                    lambda i=i: self.variables["kVT10_O2O"]() * i
                )
            )
            self.reactions.append(
                Reaction(
                    [f"O2(v{i-1})" if i > 1 else "O2", "O"],
                    [f"O2(v{i})", "O"],
                    lambda i=i: self.variables["kVT01_O2O"]() * i
                )
            )


        #--------------------------------------------------------------------------------
        #
        # excitation of electronic levels by electron impact [Bolsig+]
        #
        #--------------------------------------------------------------------------------

        # N2 electronic excitations
        self.reactions.append(Reaction(["e", "N2"], ["e", "N2(A3)"], "BOLSIG N2 -> N2(A3)"))
        for v in ["v5-9", "v10-"]:
            self.reactions.append(Reaction(["e", "N2"], ["e", f"N2(A3,{v})"], f"BOLSIG N2 -> N2(A3,{v})"))

        self.reactions.append(Reaction(["e", "N2"], ["e", "N2(B3)"], "BOLSIG N2 -> N2(B3)"))
        for state in ["W3", "B'3"]:
            self.reactions.append(Reaction(["e", "N2"], ["e", f"N2({state})"], f"BOLSIG N2 -> N2({state})"))

        self.reactions.append(Reaction(["e", "N2"], ["e", "N2(a'1)"], "BOLSIG N2 -> N2(a'1)"))
        for state in ["a1", "w1"]:
            self.reactions.append(Reaction(["e", "N2"], ["e", f"N2({state})"], f"BOLSIG N2 -> N2({state})"))

        self.reactions.append(Reaction(["e", "N2"], ["e", "N2(C3)"], "BOLSIG N2 -> N2(C3)"))
        for state in ["E3", "a''1"]:
            self.reactions.append(Reaction(["e", "N2"], ["e", f"N2({state})"], f"BOLSIG N2 -> N2({state})"))

        self.reactions.append(Reaction(["e", "N2"], ["e", "N", "N(2D)"], "BOLSIG N2 -> N + N(2D)"))

        # O2 electronic excitations
        self.reactions.append(Reaction(["e", "O2"], ["e", "O2(a1)"], "BOLSIG O2 -> O2(a1)"))
        for state in ["b1", "4.5eV"]:
            self.reactions.append(Reaction(["e", "O2"], ["e", f"O2({state})"], f"BOLSIG O2 -> O2({state})"))

        self.reactions.append(Reaction(["e", "O2"], ["e", "O", "O(1D)"], "BOLSIG O2 -> O + O(1D)"))
        self.reactions.append(Reaction(["e", "O2"], ["e", "O", "O(1S)"], "BOLSIG O2 -> O + O(1S)"))

        self.reactions.append(Reaction(["e", "O2(a1)"], ["e", "O", "O"], "BOLSIG O2(a1) -> O + O"))

        self.reactions.append(Reaction(["e", "O"], ["e", "O(1D)"], "BOLSIG O -> O(1D)"))
        self.reactions.append(Reaction(["e", "O"], ["e", "O(1S)"], "BOLSIG O -> O(1S)"))


        #--------------------------------------------------------------------------------
        #
        # de-excitation of electronic levels by electron impact [Bolsig+]
        #
        #--------------------------------------------------------------------------------

        # N2 de-excitation
        self.reactions.append(Reaction(["e", "N2(A3)"], ["e", "N2"], "BOLSIG N2(A3) -> N2", is_superelastic=True))

        # O2 de-excitation
        self.reactions.append(Reaction(["e", "O2(a1)"], ["e", "O2"], "BOLSIG O2(a1) -> O2", is_superelastic=True))


        #--------------------------------------------------------------------------------
        #
        # ionization by electron impact [Bolsig+]
        #
        #--------------------------------------------------------------------------------

        # Atomic ionization
        self.reactions.append(Reaction(["e", "N"], ["e", "e", "N^+"], "BOLSIG N -> N^+"))
        self.reactions.append(Reaction(["e", "O"], ["e", "e", "O^+"], "BOLSIG O -> O^+"))

        # Molecular ionization
        self.reactions.append(Reaction(["e", "N2"], ["e", "e", "N2^+"], "BOLSIG N2 -> N2^+"))
        self.reactions.append(Reaction(["e", "N2(A3)"], ["e", "e", "N2^+"], "BOLSIG N2(A3) -> N2^+"))
        self.reactions.append(Reaction(["e", "O2"], ["e", "e", "O2^+"], "BOLSIG O2 -> O2^+"))
        self.reactions.append(Reaction(["e", "O2(a1)"], ["e", "e", "O2^+"], "BOLSIG O2(a1) -> O2^+"))
        self.reactions.append(Reaction(["e", "NO"], ["e", "e", "NO^+"], "BOLSIG NO -> NO^+"))
        self.reactions.append(Reaction(["e", "N2O"], ["e", "e", "N2O^+"], "BOLSIG N2O -> N2O^+"))

        # Missing cross sections (commented out)
        # self.reactions.append(Reaction(["e", "NO2"], ["e", "e", "NO2^+"], "BOLSIG NO2 -> NO2^+"))
        # self.reactions.append(Reaction(["e", "O3"], ["e", "e", "O3^+"], "BOLSIG O3 -> O3^+"))
        # self.reactions.append(Reaction(["e", "NO3"], ["e", "e", "NO3^+"], "BOLSIG NO3 -> NO3^+"))
        # self.reactions.append(Reaction(["e", "N2O5"], ["e", "e", "N2O5^+"], "BOLSIG N2O5 -> N2O5^+"))


        #--------------------------------------------------------------------------------
        #
        # electron-ion recombination [Capitelli2000, page 141]
        #
        #--------------------------------------------------------------------------------

        # e + N2^+ -> N + {N, N(2D), N(2P)}
        self.reactions.append(Reaction(["e", "N2^+"], ["N", "N"], lambda: 1.8e-7 * (300/self.variables["Te"])**0.39 * 0.50))
        self.reactions.append(Reaction(["e", "N2^+"], ["N", "N(2D)"], lambda: 1.8e-7 * (300/self.variables["Te"])**0.39 * 0.45))
        self.reactions.append(Reaction(["e", "N2^+"], ["N", "N(2P)"], lambda: 1.8e-7 * (300/self.variables["Te"])**0.39 * 0.05))

        # e + O2^+ -> O + {O, O(1D), O(1S)}
        self.reactions.append(Reaction(["e", "O2^+"], ["O", "O"], lambda: 2.7e-7 * (300/self.variables["Te"])**0.7 * 0.55))
        self.reactions.append(Reaction(["e", "O2^+"], ["O", "O(1D)"], lambda: 2.7e-7 * (300/self.variables["Te"])**0.7 * 0.40))
        self.reactions.append(Reaction(["e", "O2^+"], ["O", "O(1S)"], lambda: 2.7e-7 * (300/self.variables["Te"])**0.7 * 0.05))

        # e + NO^+ -> O + {N, N(2D)}
        self.reactions.append(Reaction(["e", "NO^+"], ["O", "N"], lambda: 4.2e-7 * (300/self.variables["Te"])**0.20))
        self.reactions.append(Reaction(["e", "NO^+"], ["O", "N(2D)"], lambda: 4.2e-7 * (300/self.variables["Te"])**0.85))

        # e + polyatomic ions
        self.reactions.append(Reaction(["e", "N3^+"], ["N2", "N"], lambda: 2.0e-7 * (300/self.variables["Te"])**0.5))
        self.reactions.append(Reaction(["e", "N4^+"], ["N2", "N2"], lambda: 2.3e-6 * (300/self.variables["Te"])**0.53))
        self.reactions.append(Reaction(["e", "N2O^+"], ["N2", "O"], lambda: 2.0e-7 * (300/self.variables["Te"])**0.5))
        self.reactions.append(Reaction(["e", "NO2^+"], ["NO", "O"], lambda: 2.0e-7 * (300/self.variables["Te"])**0.5))
        self.reactions.append(Reaction(["e", "O4^+"], ["O2", "O2"], lambda: 1.4e-6 * (300/self.variables["Te"])**0.5))
        self.reactions.append(Reaction(["e", "O2^+N2"], ["O2", "N2"], lambda: 1.3e-6 * (300/self.variables["Te"])**0.5))

        # e + {N^+, O^+} + e -> {N, O} + e
        for ion, prod in zip(["N^+", "O^+"], ["N", "O"]):
            self.reactions.append(Reaction(["e", ion, "e"], [prod, "O", "e"], lambda: 7.0e-20 * (300/self.variables["Te"])**4.5))

        # e + {N^+, O^+} + ANY_NEUTRAL -> {N, O} + ANY_NEUTRAL
        for ion, prod in zip(["N^+", "O^+"], ["N", "O"]):
            self.reactions.append(Reaction(["e", ion, "ANY_NEUTRAL"], [prod, "O", "ANY_NEUTRAL"], lambda: 6.0e-27 * (300/self.variables["Te"])**1.5))


        #--------------------------------------------------------------------------------
        #
        # electron attachment
        #
        #--------------------------------------------------------------------------------

        # Simple attachments
        self.reactions.append(Reaction(["e", "O2"], ["O^-", "O"], "BOLSIG O2 -> O^- + O"))
        self.reactions.append(Reaction(["e", "NO"], ["O^-", "N"], "BOLSIG NO -> O^- + N"))
        self.reactions.append(Reaction(["e", "O3"], ["O^-", "O2"], "BOLSIG O3 -> O^- + O2"))
        self.reactions.append(Reaction(["e", "O3"], ["O2^-", "O"], "BOLSIG O3 -> O2^- + O"))
        self.reactions.append(Reaction(["e", "N2O"], ["NO^-", "N"], "BOLSIG N2O -> NO^- + N"))

        # Three-body attachments
        self.reactions.append(Reaction(["e", "O2", "O2"], ["O2^-", "O2"], "BOLSIG O2 + O2 -> O2^- + O2"))

        # Capitelli2000 values
        self.reactions.append(Reaction(["e", "NO2"], ["O^-", "NO"], 1.0e-11))
        self.reactions.append(Reaction(["e", "O", "O2"], ["O^-", "O2"], 1.0e-31))
        self.reactions.append(Reaction(["e", "O", "O2"], ["O2^-", "O"], 1.0e-31))

        self.reactions.append(Reaction(["e", "O3", "ANY_NEUTRAL"], ["O3^-", "ANY_NEUTRAL"], 1.0e-31))
        self.reactions.append(Reaction(["e", "NO", "ANY_NEUTRAL"], ["NO^-", "ANY_NEUTRAL"], 8.0e-31))
        self.reactions.append(Reaction(["e", "N2O", "ANY_NEUTRAL"], ["N2O^-", "ANY_NEUTRAL"], 6.0e-33))

        # Kossyi1992 formula
        self.variables["k_O2_N2"] = lambda: 1.1e-31 * (300.0 / self.variables["Te"])**2 \
                            * np.exp(-70.0 / self.variables["Tgas"]) \
                            * np.exp(1500.0 * (self.variables["Te"] - self.variables["Tgas"]) \
                                    / (self.variables["Te"] * self.variables["Tgas"]))

        self.reactions.append(Reaction(["e", "O2", "N2"], ["O2^-", "N2"], self.variables["k_O2_N2"]))


        #--------------------------------------------------------------------------------
        #
        # electron detachment [Capitelli2000, page 182]
        #
        #--------------------------------------------------------------------------------

        # O^- detachment reactions
        self.reactions.append(Reaction(["O^-", "O"], ["O2", "e"], 1.4e-10))
        self.reactions.append(Reaction(["O^-", "N"], ["NO", "e"], 2.6e-10))
        self.reactions.append(Reaction(["O^-", "NO"], ["NO2", "e"], 2.6e-10))
        self.reactions.append(Reaction(["O^-", "N2"], ["N2O", "e"], 5.0e-13))
        self.reactions.append(Reaction(["O^-", "O2"], ["O3", "e"], 5.0e-15))
        self.reactions.append(Reaction(["O^-", "O2(a1)"], ["O3", "e"], 3.0e-10))
        self.reactions.append(Reaction(["O^-", "O2(b1)"], ["O", "O2", "e"], 6.9e-10))
        self.reactions.append(Reaction(["O^-", "N2(A3)"], ["O", "N2", "e"], 2.2e-9))
        self.reactions.append(Reaction(["O^-", "N2(B3)"], ["O", "N2", "e"], 1.9e-9))
        self.reactions.append(Reaction(["O^-", "O3"], ["O2", "O2", "e"], 3.0e-10))

        # O2^- detachment reactions
        self.reactions.append(Reaction(["O2^-", "O"], ["O3", "e"], 1.5e-10))
        self.reactions.append(Reaction(["O2^-", "N"], ["NO2", "e"], 5.0e-10))
        self.reactions.append(Reaction(["O2^-", "O2"], ["O2", "O2", "e"], 
                                    lambda: 2.7e-10 * (self.variables["TeffN2"]()/300.0)**0.5 
                                            * np.exp(-5590.0/self.variables["TeffN2"]())))
        self.reactions.append(Reaction(["O2^-", "O2(a1)"], ["O2", "O2", "e"], 2.0e-10))
        self.reactions.append(Reaction(["O2^-", "O2(b1)"], ["O2", "O2", "e"], 3.6e-10))
        self.reactions.append(Reaction(["O2^-", "N2"], ["O2", "N2", "e"], 
                                    lambda: 1.9e-12 * (self.variables["TeffN2"]()/300.0)**0.5 
                                            * np.exp(-4990.0/self.variables["TeffN2"]())))
        self.reactions.append(Reaction(["O2^-", "N2(A3)"], ["O2", "N2", "e"], 2.1e-9))
        self.reactions.append(Reaction(["O2^-", "N2(B3)"], ["O2", "N2", "e"], 2.5e-9))

        # O3^- detachment reactions
        self.reactions.append(Reaction(["O3^-", "O"], ["O2", "O2", "e"], 3.0e-10))

        # Optional / very small reaction (commented out)
        # self.reactions.append(Reaction(["O3^-", "N2"], ["N2O", "O2", "e"], 1.0e-15))


        #----------------------------------------
        #
        # Detachment for O3^- NO^- N2O^- NO2^- NO3^- has to be verified
        #
        #----------------------------------------

        # NO^- + N => N2O + e
        self.reactions.append(Reaction(["NO^-", "N"], ["N2O", "e"], 5.0e-10))

        # O3^-, N2O^-, NO2^-, NO3^- + N => NO, O2, N2, NO, NO2 + e
        reactants_A = ["O3^-", "N2O^-", "NO2^-", "NO3^-"]
        products_B = ["O2", "N2", "NO", "NO2"]
        for A, B in zip(reactants_A, products_B):
            self.reactions.append(Reaction([A, "N"], ["NO", B, "e"], 5.0e-10))

        # NO^- + O => NO2 + e
        self.reactions.append(Reaction(["NO^-", "O"], ["NO2", "e"], 1.5e-10))

        # N2O^-, NO2^-, NO3^- + O => NO, O2, O3 + e
        reactants_A = ["N2O^-", "NO2^-", "NO3^-"]
        products_B = ["NO", "O2", "O3"]
        for A, B in zip(reactants_A, products_B):
            self.reactions.append(Reaction([A, "O"], [B, "NO", "e"], 1.5e-10))

        # @A + N2(A3) => @B + N2 + e
        reactants_A = ["O3^-", "NO^-", "N2O^-", "NO2^-", "NO3^-"]
        products_B  = ["O3", "NO", "N2O", "NO2", "NO3"]
        for A, B in zip(reactants_A, products_B):
            self.reactions.append(Reaction([A, "N2(A3)"], [B, "N2", "e"], 2.1e-9))

        # @A + N2(B3) => @B + N2 + e
        for A, B in zip(reactants_A, products_B):
            self.reactions.append(Reaction([A, "N2(B3)"], [B, "N2", "e"], 2.5e-9))


        #--------------------------------------------------------------------------------
        #
        # optical transitions and predissociation [Capitelli2000, page 157]
        #
        #--------------------------------------------------------------------------------

        # N2 transitions
        self.reactions.append(Reaction(["N2(A3)"], ["N2"], 0.50))
        self.reactions.append(Reaction(["N2(B3)"], ["N2(A3)"], 1.34e5))
        self.reactions.append(Reaction(["N2(a`1)"], ["N2"], 1.0e2))
        self.reactions.append(Reaction(["N2(C3)"], ["N2(B3)"], 2.45e7))

        # O2 transitions
        self.reactions.append(Reaction(["O2(a1)"], ["O2"], 2.6e-4))
        self.reactions.append(Reaction(["O2(b1)"], ["O2(a1)"], 1.5e-3))
        self.reactions.append(Reaction(["O2(b1)"], ["O2"], 8.5e-2))
        self.reactions.append(Reaction(["O2(4.5eV)"], ["O2"], 11.0))


        #--------------------------------------------------------------------------------
        #
        # quenching and excitation of N2 [Capitelli2000, page 159]
        #
        #--------------------------------------------------------------------------------

        # N2(A3) quenching
        self.reactions.append(Reaction(["N2(A3)", "O"], ["NO", "N(2D)"], 7.0e-12))
        self.reactions.append(Reaction(["N2(A3)", "O"], ["N2", "O(1S)"], 2.1e-11))
        self.reactions.append(Reaction(["N2(A3)", "N"], ["N2", "N"], 2.0e-12))
        self.reactions.append(Reaction(["N2(A3)", "N"], ["N2", "N(2P)"], lambda: 4.0e-11 * (300.0 / self.variables["Tgas"])**0.667))
        self.reactions.append(Reaction(["N2(A3)", "O2"], ["N2", "O", "O(1D)"], lambda: 2.1e-12 * (self.variables["Tgas"] / 300.0)**0.55))
        self.reactions.append(Reaction(["N2(A3)", "O2"], ["N2", "O2(a1)"], lambda: 2.0e-13 * (self.variables["Tgas"] / 300.0)**0.55))
        self.reactions.append(Reaction(["N2(A3)", "O2"], ["N2", "O2(b1)"], lambda: 2.0e-13 * (self.variables["Tgas"] / 300.0)**0.55))
        self.reactions.append(Reaction(["N2(A3)", "O2"], ["N2O", "O"], lambda: 2.0e-14 * (self.variables["Tgas"] / 300.0)**0.55))
        self.reactions.append(Reaction(["N2(A3)", "N2"], ["N2", "N2"], 3.0e-16))
        self.reactions.append(Reaction(["N2(A3)", "NO"], ["N2", "NO"], 6.9e-11))
        self.reactions.append(Reaction(["N2(A3)", "N2O"], ["N2", "N", "NO"], 1.0e-11))
        self.reactions.append(Reaction(["N2(A3)", "NO2"], ["N2", "O", "NO"], 1.0e-12))
        self.reactions.append(Reaction(["N2(A3)", "N2(A3)"], ["N2", "N2(B3)"], 3.0e-10))
        self.reactions.append(Reaction(["N2(A3)", "N2(A3)"], ["N2", "N2(C3)"], 1.5e-10))

        # N2(B3) quenching
        self.reactions.append(Reaction(["N2(B3)", "N2"], ["N2(A3)", "N2"], 3.0e-11))
        self.reactions.append(Reaction(["N2(B3)", "N2"], ["N2", "N2"], 2.0e-12))
        self.reactions.append(Reaction(["N2(B3)", "O2"], ["N2", "O", "O"], 3.0e-10))
        self.reactions.append(Reaction(["N2(B3)", "NO"], ["N2(A3)", "NO"], 2.4e-10))

        # N2(C3) quenching
        self.reactions.append(Reaction(["N2(C3)", "N2"], ["N2(a`1)", "N2"], 1.0e-11))
        self.reactions.append(Reaction(["N2(C3)", "O2"], ["N2", "O", "O(1S)"], 3.0e-10))

        # N2(a`1) quenching
        self.reactions.append(Reaction(["N2(a`1)", "N2"], ["N2(B3)", "N2"], 1.9e-13))
        self.reactions.append(Reaction(["N2(a`1)", "O2"], ["N2", "O", "O"], 2.8e-11))
        self.reactions.append(Reaction(["N2(a`1)", "NO"], ["N2", "N", "O"], 3.6e-10))
        self.reactions.append(Reaction(["N2(a`1)", "N2(A3)"], ["N4^+", "e"], 4.0e-12))
        self.reactions.append(Reaction(["N2(a`1)", "N2(a`1)"], ["N4^+", "e"], 1.0e-11))

        # N + N + @M => N2(A3) + @M
        reactants_M = ["N2", "O2", "NO", "N", "O"]
        rates_M     = [1.7e-33, 1.7e-33, 1.7e-33, 1.0e-32, 1.0e-32]
        for M, rate in zip(reactants_M, rates_M):
            self.reactions.append(Reaction(["N", "N", M], ["N2(A3)", M], rate))

        # N + N + @M => N2(B3) + @M
        rates_M_B3  = [2.4e-33, 2.4e-33, 2.4e-33, 1.4e-32, 1.4e-32]
        for M, rate in zip(reactants_M, rates_M_B3):
            self.reactions.append(Reaction(["N", "N", M], ["N2(B3)", M], rate))


        #--------------------------------------------------------------------------------
        #
        # deactivation of N metastables [Capitelli2000, page 161]
        #
        #--------------------------------------------------------------------------------

        # N(2D) reactions
        self.reactions.append(Reaction(["N(2D)", "O"], ["N", "O(1D)"], 4.0e-13))
        self.reactions.append(Reaction(["N(2D)", "O2"], ["NO", "O"], 5.2e-12))
        self.reactions.append(Reaction(["N(2D)", "NO"], ["N2", "O"], 1.8e-10))
        self.reactions.append(Reaction(["N(2D)", "N2O"], ["NO", "N2"], 3.5e-12))
        self.reactions.append(Reaction(["N(2D)", "N2"], ["N", "N2"], lambda: 1.0e-13 * np.exp(-510.0 / self.variables["Tgas"])))

        # N(2P) reactions
        self.reactions.append(Reaction(["N(2P)", "N"], ["N", "N"], 1.8e-12))
        self.reactions.append(Reaction(["N(2P)", "O"], ["N", "O"], 1.0e-12))
        self.reactions.append(Reaction(["N(2P)", "N"], ["N(2D)", "N"], 6.0e-13))
        self.reactions.append(Reaction(["N(2P)", "N2"], ["N", "N2"], 6.0e-14))
        self.reactions.append(Reaction(["N(2P)", "N(2D)"], ["N2^+", "e"], 1.0e-13))
        self.reactions.append(Reaction(["N(2P)", "O2"], ["NO", "O"], 2.6e-12))
        self.reactions.append(Reaction(["N(2P)", "NO"], ["N2(A3)", "O"], 3.0e-11))


        #--------------------------------------------------------------------------------
        #
        # quenching and excitation of O2 [Capitelli2000, page 160]
        #
        #--------------------------------------------------------------------------------

        # O2(a1) reactions
        self.reactions.append(Reaction(["O2(a1)", "O"], ["O2", "O"], 7.0e-16))
        self.reactions.append(Reaction(["O2(a1)", "N"], ["NO", "O"], lambda: 2.0e-14 * np.exp(-600.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O2(a1)", "O2"], ["O2", "O2"], lambda: 3.8e-18 * np.exp(-205.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O2(a1)", "N2"], ["O2", "N2"], 3.0e-21))
        self.reactions.append(Reaction(["O2(a1)", "NO"], ["O2", "NO"], 2.5e-11))
        self.reactions.append(Reaction(["O2(a1)", "O3"], ["O2", "O2", "O(1D)"], lambda: 5.2e-11 * np.exp(-2840.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O2(a1)", "O2(a1)"], ["O2", "O2(b1)"], lambda: 7.0e-28 * self.variables["Tgas"]**3.8 * np.exp(700.0/self.variables["Tgas"])))

        # O + O3 reaction
        self.reactions.append(Reaction(["O", "O3"], ["O2", "O2(a1)"], lambda: 1.0e-11 * np.exp(-2300.0/self.variables["Tgas"])))

        # O2(b1) reactions
        self.reactions.append(Reaction(["O2(b1)", "O"], ["O2(a1)", "O"], 8.1e-14))
        self.reactions.append(Reaction(["O2(b1)", "O"], ["O2", "O(1D)"], lambda: 3.4e-11 * (300.0/self.variables["Tgas"])**0.1 * np.exp(-4200.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O2(b1)", "O2"], ["O2(a1)", "O2"], lambda: 4.3e-22 * self.variables["Tgas"]**2.4 * np.exp(-281.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O2(b1)", "N2"], ["O2(a1)", "N2"], lambda: 1.7e-15 * (self.variables["Tgas"]/300.0)))
        self.reactions.append(Reaction(["O2(b1)", "NO"], ["O2(a1)", "NO"], 6.0e-14))
        self.reactions.append(Reaction(["O2(b1)", "O3"], ["O2", "O2", "O"], 2.2e-11))

        # O2(4.5eV) reactions
        self.reactions.append(Reaction(["O2(4.5eV)", "O"], ["O2", "O(1S)"], 9.0e-12))
        self.reactions.append(Reaction(["O2(4.5eV)", "O2"], ["O2(b1)", "O2(b1)"], 3.0e-13))
        self.reactions.append(Reaction(["O2(4.5eV)", "N2"], ["O2(b1)", "N2"], 9.0e-15))

        #--------------------------------------------------------------------------------
        #
        # deactivation of O metastables [Capitelli2000, page 161]
        #
        #--------------------------------------------------------------------------------

        # O(1D) reactions
        self.reactions.append(Reaction(["O(1D)", "O"], ["O", "O"], 8.0e-12))
        self.reactions.append(Reaction(["O(1D)", "O2"], ["O", "O2"], lambda: 6.4e-12 * np.exp(67.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O(1D)", "O2"], ["O", "O2(a1)"], 1.0e-12))
        self.reactions.append(Reaction(["O(1D)", "O2"], ["O", "O2(b1)"], lambda: 2.6e-11 * np.exp(67.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O(1D)", "N2"], ["O", "N2"], 2.3e-11))
        self.reactions.append(Reaction(["O(1D)", "O3"], ["O2", "O", "O"], 1.2e-10))
        self.reactions.append(Reaction(["O(1D)", "O3"], ["O2", "O2"], 1.2e-10))
        self.reactions.append(Reaction(["O(1D)", "NO"], ["O2", "N"], 1.7e-10))
        self.reactions.append(Reaction(["O(1D)", "N2O"], ["NO", "NO"], 7.2e-11))
        self.reactions.append(Reaction(["O(1D)", "N2O"], ["O2", "N2"], 4.4e-11))

        # O(1S) reactions
        self.reactions.append(Reaction(["O(1S)", "O"], ["O(1D)", "O"], lambda: 5.0e-11 * np.exp(-300.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O(1S)", "N"], ["O", "N"], 1.0e-12))
        self.reactions.append(Reaction(["O(1S)", "O2"], ["O(1D)", "O2"], lambda: 1.3e-12 * np.exp(-850.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O(1S)", "O2"], ["O", "O", "O"], lambda: 3.0e-12 * np.exp(-850.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O(1S)", "N2"], ["O", "N2"], 1.0e-17))
        self.reactions.append(Reaction(["O(1S)", "O2(a1)"], ["O", "O2(4.5eV)"], 1.1e-10))
        self.reactions.append(Reaction(["O(1S)", "O2(a1)"], ["O(1D)", "O2(b1)"], 2.9e-11))
        self.reactions.append(Reaction(["O(1S)", "O2(a1)"], ["O", "O", "O"], 3.2e-11))
        self.reactions.append(Reaction(["O(1S)", "NO"], ["O", "NO"], 2.9e-10))
        self.reactions.append(Reaction(["O(1S)", "NO"], ["O(1D)", "NO"], 5.1e-10))
        self.reactions.append(Reaction(["O(1S)", "O3"], ["O2", "O2"], 2.9e-10))
        self.reactions.append(Reaction(["O(1S)", "O3"], ["O2", "O", "O(1D)"], 2.9e-10))
        self.reactions.append(Reaction(["O(1S)", "N2O"], ["O", "N2O"], 6.3e-12))
        self.reactions.append(Reaction(["O(1S)", "N2O"], ["O(1D)", "N2O"], 3.1e-12))

        #--------------------------------------------------------------------------------
        #
        # bimolecular nitrogen-oxygen reactions [Capitelli2000, page 168]
        #
        #--------------------------------------------------------------------------------

        self.reactions.append(Reaction(["N", "NO"], ["O", "N2"], lambda: 1.8e-11 * (self.variables["Tgas"]/300.0)**0.5))
        self.reactions.append(Reaction(["N", "O2"], ["O", "NO"], lambda: 3.2e-12 * (self.variables["Tgas"]/300.0) * np.exp(-3150.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["N", "NO2"], ["O", "O", "N2"], 9.1e-13))
        self.reactions.append(Reaction(["N", "NO2"], ["O", "N2O"], 3.0e-12))
        self.reactions.append(Reaction(["N", "NO2"], ["N2", "O2"], 7.0e-13))
        self.reactions.append(Reaction(["N", "NO2"], ["NO", "NO"], 2.3e-12))
        # self.reactions.append(Reaction(["N", "O3"], ["NO", "O2"], rate=<2.0e-16>)  # commented

        self.reactions.append(Reaction(["O", "N2"], ["N", "NO"], lambda: 3.0e-10 * np.exp(-38370.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O", "NO"], ["N", "O2"], lambda: 7.5e-12 * (self.variables["Tgas"]/300.0) * np.exp(-19500.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O", "NO"], ["NO2"], lambda: 4.2e-18))
        self.reactions.append(Reaction(["O", "N2O"], ["N2", "O2"], lambda: 8.3e-12 * np.exp(-14000.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O", "N2O"], ["NO", "NO"], lambda: 1.5e-10 * np.exp(-14090.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O", "NO2"], ["NO", "O2"], lambda: 9.1e-12 * (self.variables["Tgas"]/300.0)**0.18))
        self.reactions.append(Reaction(["O", "NO3"], ["O2", "NO2"], lambda: 1.0e-11))
        # self.reactions.append(Reaction(["O", "N2O5"], ["product"], rate=lambda: 3.0e-16)  # commented

        self.reactions.append(Reaction(["N2", "O2"], ["O", "N2O"], lambda: 2.5e-10 * np.exp(-50390.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["NO", "NO"], ["N", "NO2"], lambda: 3.3e-16 * (300.0/self.variables["Tgas"])**0.5 * np.exp(-39200.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["NO", "NO"], ["O", "N2O"], lambda: 2.2e-12 * np.exp(-32100.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["NO", "NO"], ["N2", "O2"], lambda: 5.1e-13 * np.exp(-33660.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["NO", "O2"], ["O", "NO2"], lambda: 2.8e-12 * np.exp(-23400.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["NO", "O3"], ["O2", "NO2"], lambda: 2.5e-13 * np.exp(-765.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["NO", "N2O"], ["N2", "NO2"], lambda: 4.6e-10 * np.exp(-25170.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["NO", "NO3"], ["NO2", "NO2"], lambda: 1.7e-11))
        self.reactions.append(Reaction(["O2", "O2"], ["O", "O3"], lambda: 2.0e-11 * np.exp(-49800.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["O2", "NO2"], ["NO", "O3"], lambda: 2.8e-12 * np.exp(-25400.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["NO2", "NO2"], ["NO", "NO", "O2"], lambda: 3.3e-12 * np.exp(-13500.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["NO2", "NO2"], ["NO", "NO3"], lambda: 4.5e-10 * np.exp(-18500.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["NO2", "O3"], ["O2", "NO3"], lambda: 1.2e-13 * np.exp(-2450.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["NO2", "NO3"], ["NO", "NO2", "O2"], lambda: 2.3e-13 * np.exp(-1600.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["NO3", "O2"], ["NO2", "O3"], lambda: 1.5e-12 * np.exp(-15020.0/self.variables["Tgas"])))
        self.reactions.append(Reaction(["NO3", "NO3"], ["O2", "NO2", "NO2"], lambda: 4.3e-12 * np.exp(-3850.0/self.variables["Tgas"])))

        # Ionization reactions
        self.reactions.append(Reaction(["N", "N"], ["N2^+", "e"], lambda: 2.7e-11 * np.exp(-67400.0 / self.variables["Tgas"])))
        self.reactions.append(Reaction(
            ["N", "O"], ["NO^+", "e"],
            lambda: 1.6e-12 * (self.variables["Tgas"]/300.0)**0.5 *
                    (0.19 + 8.6*self.variables["Tgas"]) *
                    np.exp(-32000.0 / self.variables["Tgas"])
        ))

        #--------------------------------------------------------------------------------
        #
        # dissociation of nitrogen-oxygen molecules [Capitelli2000, page 169]
        #
        #--------------------------------------------------------------------------------

        # N2 dissociation
        M_list_N2 = ["N2", "O2", "NO", "O", "N"]
        R_list_N2 = [1.0, 1.0, 1.0, 6.6, 6.6]
        for M, R in zip(M_list_N2, R_list_N2):
            self.reactions.append(Reaction(
                ["N2", M],
                ["N", "N", M],
                lambda M=M, R=R: 5.4e-8 * (1.0 - np.exp(-3354.0/self.variables["Tgas"])) *
                                np.exp(-113200.0/self.variables["Tgas"]) * R
            ))

        # O2 dissociation
        M_list_O2 = ["N2", "O2", "O", "N", "NO"]
        R_list_O2 = [1.0, 5.9, 21.0, 1.0, 1.0]
        for M, R in zip(M_list_O2, R_list_O2):
            self.reactions.append(Reaction(
                ["O2", M],
                ["O", "O", M],
                lambda M=M, R=R: 6.1e-9 * (1.0 - np.exp(-2240.0/self.variables["Tgas"])) *
                                np.exp(-59380.0/self.variables["Tgas"]) * R
            ))

        # NO dissociation
        M_list_NO = ["N2", "O2", "O", "N", "NO"]
        R_list_NO = [1.0, 1.0, 20.0, 20.0, 20.0]
        for M, R in zip(M_list_NO, R_list_NO):
            self.reactions.append(Reaction(
                ["NO", M],
                ["N", "O", M],
                lambda M=M, R=R: 8.7e-9 * np.exp(-75994.0/self.variables["Tgas"]) * R
            ))

        # O3 dissociation
        M_list_O3 = ["N2", "O2", "N", "O"]
        R_list_O3 = [1.0, 0.38, 6.3, 6.3]  # exp(170/Tgas) remplacé dans lambda
        for M, R in zip(M_list_O3, R_list_O3):
            self.reactions.append(Reaction(
                ["O3", M],
                ["O2", "O", M],
                lambda M=M, R=R: 6.6e-10 * np.exp(-11600.0/self.variables["Tgas"]) *
                                (R if isinstance(R,float) else R*np.exp(170.0/self.variables["Tgas"]))
            ))

        # N2O dissociation
        M_list_N2O = ["N2", "O2", "NO", "N2O"]
        R_list_N2O = [1.0, 1.0, 2.0, 4.0]
        for M, R in zip(M_list_N2O, R_list_N2O):
            self.reactions.append(Reaction(
                ["N2O", M],
                ["N2", "O", M],
                lambda M=M, R=R: 1.2e-8 * (300.0/self.variables["Tgas"]) *
                                np.exp(-29000.0/self.variables["Tgas"]) * R
            ))

        # NO2 dissociation
        M_list_NO2 = ["N2", "O2", "NO", "NO2"]
        R_list_NO2 = [1.0, 0.78, 7.8, 5.9]
        for M, R in zip(M_list_NO2, R_list_NO2):
            self.reactions.append(Reaction(
                ["NO2", M],
                ["NO", "O", M],
                lambda M=M, R=R: 6.8e-6 * (300.0/self.variables["Tgas"])**2 *
                                np.exp(-36180.0/self.variables["Tgas"]) * R
            ))

        # NO3 dissociation to NO2 + O
        M_list_NO3_1 = ["N2", "O2", "NO", "N", "O"]
        R_list_NO3_1 = [1.0, 1.0, 1.0, 10.0, 10.0]
        for M, R in zip(M_list_NO3_1, R_list_NO3_1):
            self.reactions.append(Reaction(
                ["NO3", M],
                ["NO2", "O", M],
                lambda M=M, R=R: 3.1e-5 * (300.0/self.variables["Tgas"])**2 *
                                np.exp(-25000.0/self.variables["Tgas"]) * R
            ))

        # NO3 dissociation to NO + O2
        M_list_NO3_2 = ["N2", "O2", "NO", "N", "O"]
        R_list_NO3_2 = [1.0, 1.0, 1.0, 12.0, 12.0]
        for M, R in zip(M_list_NO3_2, R_list_NO3_2):
            self.reactions.append(Reaction(
                ["NO3", M],
                ["NO", "O2", M],
                lambda M=M, R=R: 6.2e-5 * (300.0/self.variables["Tgas"])**2 *
                                np.exp(-25000.0/self.variables["Tgas"]) * R
            ))

        # N2O5 dissociation (with ANY_NEUTRAL)
        self.reactions.append(Reaction(
            ["N2O5", "ANY_NEUTRAL"],
            ["NO2", "NO3", "ANY_NEUTRAL"],
            lambda: 2.1e-11 * (300.0/self.variables["Tgas"])**4.4 *
                    np.exp(-11080.0/self.variables["Tgas"])
        ))

        #--------------------------------------------------------------------------------
        #
        # recombination of nitrogen-oxygen molecules [Capitelli2000, page 170]
        #
        #--------------------------------------------------------------------------------

        # N + N + N2 -> N2 + N2
        self.reactions.append(Reaction(
            ["N","N","N2"],
            ["N2","N2"],
            lambda: max(8.3e-34 * np.exp(500.0/self.variables["Tgas"]), 1.91e-33)
        ))

        # N + N + @M -> N2 + @M
        M_list_NN = ["O2", "NO", "N", "O"]
        R_list_NN = [1.0, 1.0, 3.0, 3.0]
        for M, R in zip(M_list_NN, R_list_NN):
            self.reactions.append(Reaction(
                ["N","N",M],
                ["N2", M],
                lambda M=M, R=R: 1.8e-33 * np.exp(435.0/self.variables["Tgas"]) * R
            ))

        # O + O + N2 -> O2 + N2
        self.reactions.append(Reaction(
            ["O","O","N2"],
            ["O2","N2"],
            lambda: max(2.8e-34 * np.exp(720.0/self.variables["Tgas"]), 1.0e-33 * (300.0/self.variables["Tgas"])**0.41)
        ))

        # O + O + @M -> O2 + @M
        M_list_OO = ["O2", "N", "O", "NO"]
        R_list_OO = [1.0, 0.8, 3.6, 0.17]
        for M, R in zip(M_list_OO, R_list_OO):
            self.reactions.append(Reaction(
                ["O","O",M],
                ["O2", M],
                lambda M=M, R=R: 4.0e-33 * (300.0/self.variables["Tgas"])**0.41 * R
            ))

        # N + O + @M -> NO + @M
        M_list_NO1 = ["N2","O2"]
        for M in M_list_NO1:
            self.reactions.append(Reaction(
                ["N","O",M],
                ["NO", M],
                lambda: 1.0e-32 * (300.0/self.variables["Tgas"])**0.5
            ))

        M_list_NO2 = ["N","O","NO"]
        for M in M_list_NO2:
            self.reactions.append(Reaction(
                ["N","O",M],
                ["NO", M],
                lambda: 1.8e-31 * (300.0/self.variables["Tgas"])
            ))

        # O + O2 + N2 -> O3 + N2
        self.reactions.append(Reaction(
            ["O","O2","N2"],
            ["O3","N2"],
            lambda: max(5.8e-34 * (300.0/self.variables["Tgas"])**2.8, 5.4e-34 * (300.0/self.variables["Tgas"])**1.9)
        ))

        # O + O2 + @M -> O3 + @M
        M_list_OO2_1 = ["O2","NO"]
        for M in M_list_OO2_1:
            self.reactions.append(Reaction(
                ["O","O2",M],
                ["O3", M],
                lambda: 7.6e-34 * (300.0/self.variables["Tgas"])**1.9
            ))

        M_list_OO2_2 = ["N","O"]
        for M in M_list_OO2_2:
            self.reactions.append(Reaction(
                ["O","O2",M],
                ["O3", M],
                lambda: min(3.9e-33 * (300.0/self.variables["Tgas"])**1.9, 1.1e-34 * np.exp(1060.0/self.variables["Tgas"]))
            ))

        # O + N2 + ANY_NEUTRAL -> N2O + ANY_NEUTRAL
        self.reactions.append(Reaction(
            ["O","N2","ANY_NEUTRAL"],
            ["N2O","ANY_NEUTRAL"],
            lambda: 3.9e-35 * np.exp(-10400.0/self.variables["Tgas"])
        ))

        # O + NO + @M -> NO2 + @M
        M_list_ONO = ["N2","O2","NO"]
        R_list_ONO = [1.0,0.78,0.78]
        for M,R in zip(M_list_ONO,R_list_ONO):
            self.reactions.append(Reaction(
                ["O","NO",M],
                ["NO2",M],
                lambda R=R: 1.2e-31 * (300.0/self.variables["Tgas"])**1.8 * R
            ))

        # O + NO2 + @M -> NO3 + @M
        M_list_ONO2 = ["N2","O2","N","O","NO"]
        R_list_ONO2 = [1.0,1.0,13.0,13.0,2.4]
        for M,R in zip(M_list_ONO2,R_list_ONO2):
            self.reactions.append(Reaction(
                ["O","NO2",M],
                ["NO3",M],
                lambda R=R: 8.9e-32 * (300.0/self.variables["Tgas"])**2 * R
            ))

        # NO2 + NO3 + ANY_NEUTRAL -> N2O5 + ANY_NEUTRAL
        self.reactions.append(Reaction(
            ["NO2","NO3","ANY_NEUTRAL"],
            ["N2O5","ANY_NEUTRAL"],
            lambda: 3.7e-30 * (300.0/self.variables["Tgas"])**4.1
        ))

        #--------------------------------------------------------------------------------
        #
        # positive ion self.reactions [Capitelli2000, page 179]
        #
        #--------------------------------------------------------------------------------

        # N+ self.reactions
        self.reactions.append(Reaction(["N^+","O"], ["N","O^+"], 1.0e-12))
        self.reactions.append(Reaction(["N^+","O2"], ["O2^+","N"], 2.8e-10))
        self.reactions.append(Reaction(["N^+","O2"], ["NO^+","O"], 2.5e-10))
        self.reactions.append(Reaction(["N^+","O2"], ["O^+","NO"], 2.8e-11))
        self.reactions.append(Reaction(["N^+","O3"], ["NO^+","O2"], 5.0e-10))
        self.reactions.append(Reaction(["N^+","NO"], ["NO^+","N"], 8.0e-10))
        self.reactions.append(Reaction(["N^+","NO"], ["N2^+","O"], 3.0e-12))
        self.reactions.append(Reaction(["N^+","NO"], ["O^+","N2"], 1.0e-12))
        self.reactions.append(Reaction(["N^+","N2O"], ["NO^+","N2"], 5.5e-10))

        # O+ self.reactions
        self.variables["rate_O_N2"] = lambda: (1.5 - 2.0e-3*self.variables["TeffN"]() + 9.6e-7*self.variables["TeffN"]()**2) * 1.0e-12
        self.reactions.append(Reaction(["O^+","N2"], ["NO^+","N"], self.variables["rate_O_N2"]()))
        self.reactions.append(Reaction(["O^+","O2"], ["O2^+","O"], lambda: 2.0e-11 * (300.0/self.variables["TeffN"]())**0.5))
        self.reactions.append(Reaction(["O^+","O3"], ["O2^+","O2"], 1.0e-10))
        self.reactions.append(Reaction(["O^+","NO"], ["NO^+","O"], 2.4e-11))
        self.reactions.append(Reaction(["O^+","NO"], ["O2^+","N"], 3.0e-12))
        self.reactions.append(Reaction(["O^+","N(2D)"], ["N^+","O"], 1.3e-10))
        self.reactions.append(Reaction(["O^+","N2O"], ["NO^+","NO"], 2.3e-10))
        self.reactions.append(Reaction(["O^+","N2O"], ["N2O^+","O"], 2.2e-10))
        self.reactions.append(Reaction(["O^+","N2O"], ["O2^+","N2"], 2.0e-11))
        self.reactions.append(Reaction(["O^+","NO2"], ["NO2^+","O"], 1.6e-9))

        # N2+ self.reactions
        self.reactions.append(Reaction(["N2^+","O2"], ["O2^+","N2"], lambda: 6.0e-11 * (300.0/self.variables["TeffN2"]())**0.5))
        self.reactions.append(Reaction(["N2^+","O"], ["NO^+","N"], lambda: 1.3e-10 * (300.0/self.variables["TeffN2"]())**0.5))
        self.reactions.append(Reaction(["N2^+","O3"], ["O2^+","O","N2"], 1.0e-10))
        self.reactions.append(Reaction(["N2^+","N"], ["N^+","N2"], lambda: 7.2e-13 * (self.variables["TeffN2"]()/300.0)))
        self.reactions.append(Reaction(["N2^+","NO"], ["NO^+","N2"], 3.3e-10))
        self.reactions.append(Reaction(["N2^+","N2O"], ["N2O^+","N2"], 5.0e-10))
        self.reactions.append(Reaction(["N2^+","N2O"], ["NO^+","N","N2"], 4.0e-10))

        # O2+ self.reactions
        self.reactions.append(Reaction(["O2^+","N2"], ["NO^+","NO"], 1.0e-17))
        self.reactions.append(Reaction(["O2^+","N"], ["NO^+","O"], 1.2e-10))
        self.reactions.append(Reaction(["O2^+","NO"], ["NO^+","O2"], 6.3e-10))
        self.reactions.append(Reaction(["O2^+","NO2"], ["NO^+","O3"], 1.0e-11))
        self.reactions.append(Reaction(["O2^+","NO2"], ["NO2^+","O2"], 6.6e-10))

        # N3+ self.reactions
        self.reactions.append(Reaction(["N3^+","O2"], ["O2^+","N","N2"], 2.3e-11))
        self.reactions.append(Reaction(["N3^+","O2"], ["NO2^+","N2"], 4.4e-11))
        self.reactions.append(Reaction(["N3^+","N"], ["N2^+","N2"], 6.6e-11))
        self.reactions.append(Reaction(["N3^+","NO"], ["NO^+","N","N2"], 7.0e-11))
        self.reactions.append(Reaction(["N3^+","NO"], ["N2O^+","N2"], 7.0e-11))

        # NO2+ and N2O+ self.reactions
        self.reactions.append(Reaction(["NO2^+","NO"], ["NO^+","NO2"], 2.9e-10))
        self.reactions.append(Reaction(["N2O^+","NO"], ["NO^+","N2O"], 2.9e-10))

        # N4+ self.reactions
        self.variables["rate_N4_N2"] = lambda: min(2.1e-16 * np.exp(self.variables["TeffN4"]()/121.0), 1.0e-10)
        self.reactions.append(Reaction(["N4^+","N2"], ["N2^+","N2","N2"], self.variables["rate_N4_N2"]()))
        self.reactions.append(Reaction(["N4^+","O2"], ["O2^+","N2","N2"], 2.5e-10))
        self.reactions.append(Reaction(["N4^+","O"], ["O^+","N2","N2"], 2.5e-10))
        self.reactions.append(Reaction(["N4^+","N"], ["N^+","N2","N2"], 1.0e-11))
        self.reactions.append(Reaction(["N4^+","NO"], ["NO^+","N2","N2"], 4.0e-10))

        # O4+ self.reactions
        self.reactions.append(Reaction(["O4^+","N2"], ["O2^+N2","O2"], lambda: 4.6e-12 * (self.variables["TeffN4"]()/300.0)**2.5 * np.exp(-2650.0/self.variables["TeffN4"]())))
        self.reactions.append(Reaction(["O4^+","O2"], ["O2^+","O2","O2"], lambda: 3.3e-6  * (300.0/self.variables["TeffN4"]())**4   * np.exp(-5030.0/self.variables["TeffN4"]())))
        self.reactions.append(Reaction(["O4^+","O2(a1)"], ["O2^+","O2","O2"], 1.0e-10))
        self.reactions.append(Reaction(["O4^+","O2(b1)"], ["O2^+","O2","O2"], 1.0e-10))
        self.reactions.append(Reaction(["O4^+","O"], ["O2^+","O3"], 3.0e-10))
        self.reactions.append(Reaction(["O4^+","NO"], ["NO^+","O2","O2"], 1.0e-10))

        # O2^+N2 self.reactions
        self.reactions.append(Reaction(["O2^+N2","N2"], ["O2^+","N2","N2"], lambda: 1.1e-6 * (300.0/self.variables["TeffN4"]())**5.3 * np.exp(-2360.0/self.variables["TeffN4"]())))
        self.reactions.append(Reaction(["O2^+N2","O2"], ["O4^+","N2"], 1.0e-9))

        # Three-body ion formation
        self.reactions.append(Reaction(["N^+","N2","N2"], ["N3^+","N2"], lambda: 1.7e-29 * (300.0/self.variables["TeffN"]())**2.1))
        self.reactions.append(Reaction(["N^+","O","ANY_NEUTRAL"], ["NO^+","ANY_NEUTRAL"], 1.0e-29))
        self.reactions.append(Reaction(["N^+","N","ANY_NEUTRAL"], ["N2^+","ANY_NEUTRAL"], 1.0e-29))

        self.reactions.append(Reaction(["O^+","N2","ANY_NEUTRAL"], ["NO^+","N","ANY_NEUTRAL"], lambda: 6.0e-29 * (300.0/self.variables["TeffN"]())**2))
        self.reactions.append(Reaction(["O^+","O","ANY_NEUTRAL"], ["O2^+","ANY_NEUTRAL"], 1.0e-29))
        self.reactions.append(Reaction(["O^+","N","ANY_NEUTRAL"], ["NO^+","ANY_NEUTRAL"], 1.0e-29))

        self.reactions.append(Reaction(["N2^+","N2","N2"], ["N4^+","N2"], lambda: 5.2e-29 * (300.0/self.variables["TeffN2"]())**2.2))
        self.reactions.append(Reaction(["N2^+","N","N2"], ["N3^+","N2"], lambda: 9.0e-30 * np.exp(400.0/self.variables["TeffN2"]())))

        self.reactions.append(Reaction(["O2^+","O2","O2"], ["O4^+","O2"], lambda: 2.4e-30 * (300.0/self.variables["TeffN2"]())**3.2))
        self.reactions.append(Reaction(["O2^+","N2","N2"], ["O2^+N2","N2"], lambda: 9.0e-31 * (300.0/self.variables["TeffN2"]())**2))

        #--------------------------------------------------------------------------------
        #
        # negative ion self.reactions [Capitelli2000, pages 182-183]
        #
        #--------------------------------------------------------------------------------

        # O^- self.reactions
        self.reactions.append(Reaction(["O^-","O2(a1)"], ["O2^-","O"], 1.0e-10))
        self.reactions.append(Reaction(["O^-","O3"], ["O3^-","O"], 8.0e-10))
        self.reactions.append(Reaction(["O^-","NO2"], ["NO2^-","O"], 1.2e-9))
        self.reactions.append(Reaction(["O^-","N2O"], ["NO^-","NO"], 2.0e-10))
        self.reactions.append(Reaction(["O^-","N2O"], ["N2O^-","O"], 2.0e-12))

        # O2^- self.reactions
        self.reactions.append(Reaction(["O2^-","O"], ["O^-","O2"], 3.3e-10))
        self.reactions.append(Reaction(["O2^-","O3"], ["O3^-","O2"], 3.5e-10))
        self.reactions.append(Reaction(["O2^-","NO2"], ["NO2^-","O2"], 7.0e-10))
        self.reactions.append(Reaction(["O2^-","NO3"], ["NO3^-","O2"], 5.0e-10))

        # O3^- self.reactions
        self.reactions.append(Reaction(["O3^-","O"], ["O2^-","O2"], 1.0e-11))
        self.reactions.append(Reaction(["O3^-","NO"], ["NO3^-","O"], 1.0e-11))
        self.reactions.append(Reaction(["O3^-","NO"], ["NO2^-","O2"], 2.6e-12))
        self.reactions.append(Reaction(["O3^-","NO2"], ["NO2^-","O3"], 7.0e-11))
        self.reactions.append(Reaction(["O3^-","NO2"], ["NO3^-","O2"], 2.0e-11))
        self.reactions.append(Reaction(["O3^-","NO3"], ["NO3^-","O3"], 5.0e-10))

        # NO^- self.reactions
        self.reactions.append(Reaction(["NO^-","O2"], ["O2^-","NO"], 5.0e-10))
        self.reactions.append(Reaction(["NO^-","NO2"], ["NO2^-","NO"], 7.4e-10))
        self.reactions.append(Reaction(["NO^-","N2O"], ["NO2^-","N2"], 2.8e-14))

        # NO2^- self.reactions
        self.reactions.append(Reaction(["NO2^-","O3"], ["NO3^-","O2"], 1.8e-11))
        self.reactions.append(Reaction(["NO2^-","NO2"], ["NO3^-","NO"], 4.0e-12))
        self.reactions.append(Reaction(["NO2^-","NO3"], ["NO3^-","NO2"], 5.0e-10))
        self.reactions.append(Reaction(["NO2^-","N2O5"], ["NO3^-","NO2","NO2"], 7.0e-10))

        # NO3^- self.reactions
        self.reactions.append(Reaction(["NO3^-","NO"], ["NO2^-","NO2"], 3.0e-15))

        # O4^- self.reactions
        for M in ["N2","O2"]:
            self.reactions.append(Reaction(["O4^-",M], ["O2^-","O2",M], lambda: 1.0e-10 * np.exp(-1044.0/self.variables["TeffN4"]())))
        self.reactions.append(Reaction(["O4^-","O"], ["O3^-","O2"], 4.0e-10))
        self.reactions.append(Reaction(["O4^-","O"], ["O^-","O2","O2"], 3.0e-10))
        self.reactions.append(Reaction(["O4^-","O2(a1)"], ["O2^-","O2","O2"], 1.0e-10))
        self.reactions.append(Reaction(["O4^-","O2(b1)"], ["O2^-","O2","O2"], 1.0e-10))
        self.reactions.append(Reaction(["O4^-","NO"], ["NO3^-","O2"], 2.5e-10))

        # Three-body negative ion formation
        self.reactions.append(Reaction(["O^-","O2","ANY_NEUTRAL"], ["O3^-","ANY_NEUTRAL"], lambda: 1.1e-30 * (300.0/self.variables["TeffN"]())))
        self.reactions.append(Reaction(["O^-","NO","ANY_NEUTRAL"], ["NO2^-","ANY_NEUTRAL"], 1.0e-29))
        self.reactions.append(Reaction(["O2^-","O2","ANY_NEUTRAL"], ["O4^-","ANY_NEUTRAL"], lambda: 3.5e-31 * (300.0/self.variables["TeffN2"]())))

        #--------------------------------------------------------------------------------
        #
        # ion-ion recombination [Kossyi1992]
        #
        #--------------------------------------------------------------------------------

        ion_pairs = [
            ("O^-","N^+","N"), ("O^-","N2^+","N2"), ("O^-","O^+","O"), ("O^-","O2^+","O2"),
            ("O^-","NO^+","NO"), ("O^-","N2O^+","N2O"), ("O^-","NO2^+","NO2")
        ]

        for react, prod, result in ion_pairs:
            self.reactions.append(Reaction([react, prod], [result], lambda: 2.0e-7 * (300.0/self.variables["TionN"]())**0.5))

        # idem pour O2^-, O3^-, NO^-, N2O^-, NO2^-, NO3^- avec mêmes cations
        for ion in ["O2^-","O3^-","NO^-","N2O^-","NO2^-","NO3^-"]:
            for react, prod, result in ion_pairs:
                self.reactions.append(Reaction([ion, prod], [result], lambda: 2.0e-7 * (300.0/self.variables["TionN"]())**0.5))

        # deuxième série avec cations plus complexes
        complex_pairs = [
            ("N2^+","N+N"), ("N3^+","N+N2"), ("N4^+","N2+N2"), ("O2^+","O+O"), ("O4^+","O2+O2"),
            ("NO^+","N+O"), ("N2O^+","N2+O"), ("NO2^+","N+O2"), ("O2^+N2","O2+N2")
        ]

        for ion in ["O^-","O2^-","O3^-","NO^-","N2O^-","NO2^-","NO3^-","O4^-"]:
            for react, prod in complex_pairs:
                self.reactions.append(Reaction([ion, react], [prod], 1.0e-7))

        # O4^- + tout cation
        cations_large = ["N^+","N2^+","O^+","O2^+","NO^+","N2O^+","NO2^+","N3^+","N4^+","O4^+","O2^+N2"]
        products_large = ["N","N2","O","O2","NO","N2O","NO2","N2+N","N2+N2","O2+O2","O2+N2"]

        for A,B in zip(cations_large,products_large):
            self.reactions.append(Reaction(["O4^-", A], ["O2","O2",B], 1.0e-7))

        # Trois corps avec ANY_NEUTRAL
        three_body_pairs = [
            ("O^-","N^+","N"), ("O^-","N2^+","N2"), ("O^-","O^+","O"), ("O^-","O2^+","O2"), ("O^-","NO^+","NO"),
            ("O2^-","N^+","N"), ("O2^-","N2^+","N2"), ("O2^-","O^+","O"), ("O2^-","O2^+","O2"), ("O2^-","NO^+","NO")
        ]

        for react, A, B in three_body_pairs:
            self.reactions.append(Reaction([react,A,"ANY_NEUTRAL"], [B,"ANY_NEUTRAL"], lambda: 2.0e-25 * (300.0/self.variables["TionN"]())**2.5))

        # Transformation d’ions en autres espèces neutres avec ANY_NEUTRAL
        neutral_transforms = [
            ("O^-","N^+","NO"), ("O^-","N2^+","N2O"), ("O^-","O^+","O2"), ("O^-","O2^+","O3"), ("O^-","NO^+","NO2"),
            ("O2^-","N^+","NO2"), ("O2^-","O^+","O3"), ("O2^-","NO^+","NO3")
        ]

        for react,A,B in neutral_transforms:
            self.reactions.append(Reaction([react,A,"ANY_NEUTRAL"], [B,"ANY_NEUTRAL"], lambda: 2.0e-25 * (300.0/self.variables["TionN"]())**2.5))

        #--------------------------------------------------------------------------------
        #
        # Three-body recombination of O3^-, NO^-, N2O^-, NO2^-, NO3^- with ANY_NEUTRAL
        #
        #--------------------------------------------------------------------------------

        three_body_pairs_extended = [
            ("O3^-","N^+","N"), ("O3^-","N2^+","N2"), ("O3^-","O^+","O"), ("O3^-","O2^+","O2"),
            ("O3^-","NO^+","NO"), ("O3^-","N2O^+","N2O"), ("O3^-","NO2^+","NO2"),
            ("NO^-","N^+","N"), ("NO^-","N2^+","N2"), ("NO^-","O^+","O"), ("NO^-","O2^+","O2"),
            ("NO^-","NO^+","NO"), ("NO^-","N2O^+","N2O"), ("NO^-","NO2^+","NO2"),
            ("N2O^-","N^+","N"), ("N2O^-","N2^+","N2"), ("N2O^-","O^+","O"), ("N2O^-","O2^+","O2"),
            ("N2O^-","NO^+","NO"), ("N2O^-","N2O^+","N2O"), ("N2O^-","NO2^+","NO2"),
            ("NO2^-","N^+","N"), ("NO2^-","N2^+","N2"), ("NO2^-","O^+","O"), ("NO2^-","O2^+","O2"),
            ("NO2^-","NO^+","NO"), ("NO2^-","N2O^+","N2O"), ("NO2^-","NO2^+","NO2"),
            ("NO3^-","N^+","N"), ("NO3^-","N2^+","N2"), ("NO3^-","O^+","O"), ("NO3^-","O2^+","O2"),
            ("NO3^-","NO^+","NO"), ("NO3^-","N2O^+","N2O"), ("NO3^-","NO2^+","NO2")
        ]

        for react,A,B in three_body_pairs_extended:
            self.reactions.append(Reaction([react,A,"ANY_NEUTRAL"], [B,"ANY_NEUTRAL"], lambda: 2.0e-25 * (300.0/self.variables["TionN2"]())**2.5))

        #=====================================================================================================================================#
        # END FILE                                                                                                                            #
        #=====================================================================================================================================#