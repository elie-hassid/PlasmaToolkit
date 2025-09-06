import numpy as np

class Reaction:
    def __init__(self, reaction_string, rate_expression, is_superelastic = False):
        """
        Parses a reaction from a string "A + B + ... => C + D + ...".
        Remplit :
          - self.reactants : reactants list
          - self.products  : products list
          - self.label     : label simplifié "A -> C" si un seul réactif et produit principal
        """
        if "->" not in reaction_string:
            raise ValueError("Reaction string must contain '->'")
        
        lhs, rhs = reaction_string.split("->")
        
        self.rate_expression = rate_expression
        self.reaction_string = reaction_string
        self.is_superelastic = is_superelastic
        self.reactants = [r.strip() for r in lhs.split("+")]
        self.products  = [p.strip() for p in rhs.split("+")]
    
    def __repr__(self):
        return self.reaction_string

class Air:
    """
    Simplified model of plasma-chemical processes in air.

    This model includes the main pathways for NOx and O3 formation,
    as well as the key reactions that influence the plasma's electrical properties.

    Notes:
    - Rotational processes are neglected.
    - Vibrational excitation of N2 is included, as it contributes to the efficiency
    of the non-thermal Zeldovich mechanism.
    - Reduction in dissociation energy can be modeled using the Fridman-Macheret alpha model.
    - Main species considered: N2, O2, N, O, NO, NO2, O3, e-.
    """
    
    def __init__(self):

        self.reactions = []

        #-------------------------------------------------------------------
        # Section 1 : Processes computed by the Boltzmann solver
        # These are energy-dependent processes (cross sections)
        # Elastic processes are automatically implemented by BOLOS
        #-------------------------------------------------------------------
        # e + N2 -> e + N2
        # e + O2 -> e + O2
        # ...

        #-------------------------------------------------------------------
        # Vibrational excitation by electron impact.
        # The states will be pooled to simplify the model (including v=0).
        # Important for energy transfer in low-temperature plasmas
        #-------------------------------------------------------------------
        self.reactions.append(Reaction("e + N2 -> e + N2(v)"    , "N2 -> N2(v)"))
        self.reactions.append(Reaction("e + N2(v) -> e + N2"    , "N2(v) -> N2", True))
        # self.reactions.append(Reaction("e + N2(v) -> e + N + N" , "N2(v) -> N + N")) # Not yet implemented, cross section to be computed (Fridman-Macheret)
        self.reactions.append(Reaction("e + O2 -> e + O2(v)"    , "O2 -> O2(v)"))
        self.reactions.append(Reaction("e + O2(v) -> e + O2"    , "O2(v) -> O2", True))
        # self.reactions.append(Reaction("e + O2(v) -> e + O + O" , "O2(v) -> O + O")) # Not yet implemented, cross section to be computed (Fridman-Macheret)

        #-------------------------------------------------------------------
        # Electronic excitation and dissociation (dominant pathways)
        #-------------------------------------------------------------------
        self.reactions.append(Reaction("e + N2 -> e + N2*"      , "N2 -> N2*"))
        self.reactions.append(Reaction("e + N2* -> e + N2"      , "N2* -> N2",True))
        # self.reactions.append(Reaction("e + N2* -> e + N + N"   , "N2* -> N + N")) # Not yet implemented, cross section to be computed (Fridman-Macheret)
        self.reactions.append(Reaction("e + O2 -> e + O2*"      , "O2 -> O2*"))
        self.reactions.append(Reaction("e + O2* -> e + O2"      , "O2* -> O2", True))
        # self.reactions.append(Reaction("e + O2* -> e + O + O"   , "O2* -> O + O")) # Not yet implemented, cross section to be computed

        #-------------------------------------------------------------------
        # Ionization by direct electron impact (+ is the split symbol => N2plus = N2^+)
        #-------------------------------------------------------------------
        self.reactions.append(Reaction("e + N2 -> e + e + N2plus"      , "N2 -> N2^+"))
        self.reactions.append(Reaction("e + O2 -> e + e + O2plus"      , "O2 -> O2^+"))

        #-------------------------------------------------------------------
        # Electron attachment (Only O2 is concerned here)
        #-------------------------------------------------------------------
        self.reactions.append(Reaction("e + O2 -> O2minus"      , "O2 -> O2^-"))
        self.reactions.append(Reaction("e + O2 -> Ominus + O"  , "O2 -> O^- + O"))

        #-------------------------------------------------------------------
        # Section 2 : Processes with dynamic rate (Tgas, Te dependent)
        #-------------------------------------------------------------------
        # Reactions dependent on electron temperature or gas temperature
        #-------------------------------------------------------------------
        # N + O2 -> NO + O
        # N + O -> NO
        # O + O2 + M -> O3 + M  # three-body recombination for ozone

        #--------------------------------------------------------------------------------
        # vibrational-translational relaxation [Capitelli2000, page 105]
        #--------------------------------------------------------------------------------

        #-------------------------------------------------------------------
        # Section 3 : Processes with fixed rate
        # Constant-rate reactions (approximated or not Te-dependent)
        #-------------------------------------------------------------------
        # Optional: minor recombination, secondary reactions
        # NO + O -> NO2  (if NOx chemistry is needed)
        # e + N2+ -> N2  (recombination)
        # e + O2+ -> O2  (recombination)

if __name__ == "__main__":
    from core.bolos import solver, grid, parser
    air = Air()
    boltz = solver.BoltzmannSolver(grid.LinearGrid(0.01, 20., 100))

    # Load cross-sections.
    with open("pooled-cross-sections.txt") as fp:
        processes = parser.parse(fp)
    boltz.load_collisions(processes)

    # Add initial conditions here...

    #Gas temperature (will not be constant but it is for now.)
    boltz.kT = 300 * solver.KB / solver.ELECTRONVOLT
    boltz.EN = 100 * solver.TOWNSEND

    # Initial conditions
    boltz.target['N2'].density = 0.80
    boltz.target['O2'].density = 0.20

    boltz.init()

    # Iterative method => get a first guess using a Maxwellian distribution at 2eV.
    fMaxwell = boltz.maxwell(0.5)

    # Solve iteratively
    eedf = boltz.converge(fMaxwell, maxn=200, rtol=1e-5)

    for reaction in air.reactions:
        rate = 0
        if reaction.is_superelastic == False:
            rate = boltz.rate(eedf, reaction.rate_expression)
            print(f"{reaction} : {rate}")
        else:
            print(f"*superelastic* {reaction} : {rate}")


