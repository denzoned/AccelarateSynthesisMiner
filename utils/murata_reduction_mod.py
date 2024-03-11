from pm4py.util.lp import solver
from pm4py.objects.petri_net.utils.petri_utils import remove_place
from copy import copy
from typing import Tuple
from pm4py.objects.petri_net.obj import PetriNet, Marking


def apply_reduction_mod(net: PetriNet, im: Marking, fm: Marking) -> Tuple[PetriNet, Marking, Marking]:
    """
    Apply the Murata reduction to an accepting Petri net, removing the structurally redundant places, but not source
    places, to keep the initial marking for a WF-net.

    The implementation follows the Berthelot algorithm as in:
    https://svn.win.tue.nl/repos/prom/Packages/Murata/Trunk/src/org/processmining/algorithms/BerthelotAlgorithm.java

    Parameters
    ---------------
    net
        Petri net
    im
        Initial marking
    fm
        Final marking

    Returns
    --------------
    net
        Petri net
    im
        Initial marking
    fm
        Final marking
    """
    Aeq = []
    Aub = []
    beq = []
    bub = []
    places = sorted(list(net.places), key=lambda x: x.name)
    redundant = set()
    for place in places:
        # first constraint
        constraint = [0] * (len(net.places) + 1)
        for p2 in im:
            if p2 not in redundant:
                if p2 == place:
                    constraint[places.index(p2)] = im[p2]
                else:
                    constraint[places.index(p2)] = -im[p2]
        constraint[-1] = -1
        Aeq.append(constraint)
        beq.append(0)

        # second constraints
        for trans in net.transitions:
            constraint = [0] * (len(net.places) + 1)

            for arc in trans.in_arcs:
                p2 = arc.source
                if p2 not in redundant:
                    if p2 == place:
                        constraint[places.index(p2)] = arc.weight
                    else:
                        constraint[places.index(p2)] = -arc.weight
            constraint[-1] = -1
            Aub.append(constraint)
            bub.append(0)

        # third constraints
        for trans in net.transitions:
            constraint = [0] * (len(net.places) + 1)

            for arc in trans.out_arcs:
                p2 = arc.target
                if p2 not in redundant:
                    if p2 == place:
                        constraint[places.index(p2)] = -arc.weight
                    else:
                        constraint[places.index(p2)] = arc.weight
            Aub.append(constraint)
            bub.append(0)

        # fourth constraint
        for p2 in net.places:
            if p2 not in redundant:
                constraint = [0] * (len(net.places) + 1)

                constraint[places.index(p2)] = -1

                Aub.append(constraint)
                if p2 == place:
                    bub.append(-1)
                else:
                    bub.append(0)

        # fifth constraint
        constraint = [0] * (len(net.places) + 1)
        constraint[-1] = -1
        Aub.append(constraint)
        bub.append(0)

        c = [1] * (len(net.places) + 1)
        integrality = [1] * (len(net.places) + 1)

        xx = solver.apply(c, Aub, bub, Aeq, beq, variant=solver.SCIPY, parameters={"integrality": integrality})
        if xx.success:
            redundant.add(place)

    for place in redundant:
        if place not in fm and place not in im:
            net = remove_place(net, place)
            # if place in im:
            #     im = copy(im)
            #     del im[place]

    return net, im, fm