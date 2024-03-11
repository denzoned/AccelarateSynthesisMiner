import math
from typing import Dict, List, Tuple, Set

from pm4py import ProcessTree
from pm4py.objects.log.obj import Trace, Event
from pm4py.objects.petri_net.utils.align_utils import SKIP
from pm4py.objects.process_tree.obj import Operator
from pulp import lpSum, LpVariable, LpProblem, LpMinimize

from utils.alignement_approximation.utilities import concatenate_traces, trace_to_list_of_str, get_process_tree_height, \
    get_costs_from_alignment, __calculate_optimal_alignment

from pm4py.util.xes_constants import DEFAULT_NAME_KEY
import logging, sys

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def approximate_alignment(pt: ProcessTree, a_sets, sa_sets, ea_sets, tau_flags, trace: Trace,
                          trace_length_abortion_criteria=2,
                          process_tree_height_abortion_criteria=2, ):
    if len(trace) <= trace_length_abortion_criteria or get_process_tree_height(
            pt) <= process_tree_height_abortion_criteria:
        logging.debug(
            "\nCalculate opt. alignment for trace:" + str(trace_to_list_of_str(trace)) + " on process tree:" + str(
                pt) + "\n")
        optimal_align = __calculate_optimal_alignment(pt, trace)["alignment"]
        logging.debug(optimal_align)
        logging.debug(get_costs_from_alignment(optimal_align))
        return optimal_align
    else:
        logging.debug(
            "\nApproximate alignment for trace:" + str(trace_to_list_of_str(trace)) + " on process tree:" + str(
                pt) + "\n")
        if pt.operator == Operator.SEQUENCE:
            return __approximate_alignment_on_sequence(pt, trace, a_sets, sa_sets, ea_sets, tau_flags,
                                                       trace_length_abortion_criteria,
                                                       process_tree_height_abortion_criteria)
        elif pt.operator == Operator.LOOP:
            return __approximate_alignment_on_loop(pt, trace, a_sets, sa_sets, ea_sets, tau_flags,
                                                   trace_length_abortion_criteria,
                                                   process_tree_height_abortion_criteria)
        elif pt.operator == Operator.XOR:
            return __approximate_alignment_on_choice(pt, trace, a_sets, sa_sets, ea_sets, tau_flags,
                                                     trace_length_abortion_criteria,
                                                     process_tree_height_abortion_criteria)
        elif pt.operator == Operator.PARALLEL:
            return __approximate_alignment_on_parallel(pt, trace, a_sets, sa_sets, ea_sets, tau_flags,
                                                       trace_length_abortion_criteria,
                                                       process_tree_height_abortion_criteria)


def __approximate_alignment_on_choice(pt: ProcessTree, trace: Trace, pt_a_set: Dict[ProcessTree, Set[str]],
                                      pt_sa_set: Dict[ProcessTree, Set[str]], pt_ea_set: Dict[ProcessTree, Set[str]],
                                      pt_tau_flag: Dict[ProcessTree, bool], tl: int, th: int):
    assert pt.operator == Operator.XOR
    assert len(trace) > 0

    best_suited_subtree = None
    lowest_mismatches = math.inf
    for subtree in pt.children:
        mismatches = 0
        if len(trace) > 0:
            if trace[0]["concept:name"] not in pt_sa_set[subtree]:
                mismatches += 1
            if trace[-1]["concept:name"] not in pt_ea_set[subtree]:
                mismatches += 1
            if len(trace) > 2:
                for a in trace[1:-1]:
                    if a["concept:name"] not in pt_a_set[subtree]:
                        mismatches += 1
        else:
            if not pt_tau_flag[subtree] and len(pt_sa_set[subtree].intersection(pt_ea_set[subtree])) != 0:
                mismatches += 1
            elif not pt_tau_flag[subtree] and len(pt_sa_set[subtree].intersection(pt_ea_set[subtree])) == 0:
                mismatches += 2

        if mismatches < lowest_mismatches:
            best_suited_subtree = subtree
            lowest_mismatches = mismatches
    return approximate_alignment(best_suited_subtree, pt_a_set, pt_sa_set, pt_ea_set, pt_tau_flag, trace, tl, th)


def __approximate_alignment_on_loop(pt: ProcessTree, trace: Trace, pt_a_set: Dict[ProcessTree, Set[str]],
                                    pt_sa_set: Dict[ProcessTree, Set[str]], pt_ea_set: Dict[ProcessTree, Set[str]],
                                    pt_tau_flag: Dict[ProcessTree, bool], tl: int, th: int):
    assert pt.operator == Operator.LOOP
    assert len(pt.children) == 2
    assert len(trace) > 0

    ilp = LpProblem("splitting_trace_and_assign_to_subtrees", LpMinimize)
    # ilp.solve(solvers.GUROBI_CMD())

    # x_i_j = 1 <=> assigns activity i to subtree j
    x_variables: Dict[int, Dict[int, LpVariable]] = {}

    # t_i_j = 1 <=> inserts a tau at position i and assigns it to subtree j
    t_variables: Dict[int, Dict[int, LpVariable]] = {}

    # s_i_j = 1 <=> activity i is a start activity in the current sub-trace assigned to subtree j
    s_variables: Dict[int, Dict[int, LpVariable]] = {}

    # e_i_j = 1 <=> activity i is an end activity in the current sub-trace assigned to subtree j
    e_variables: Dict[int, Dict[int, LpVariable]] = {}

    # v_i_j = 1 <=> activity i is neither a start nor end-activity in the current sub-trace assigned to subtree j
    v_variables: Dict[int, Dict[int, LpVariable]] = {}

    # auxiliary variables
    # p_i_j = 1 <=> previous activity i-1 is assigned to the other subtree or t_1_other-subtree is 1
    p_variables: Dict[int, Dict[int, LpVariable]] = {}

    # n_i_j = 1 <=> next activity i+1 is assigned to the other subtree or t_1_other-subtree is 1
    n_variables: Dict[int, Dict[int, LpVariable]] = {}

    t_costs = {}
    s_costs = {}
    e_costs = {}
    v_costs = {}

    # trace <a_0,...,a_n>
    for i, a in enumerate(trace):
        x_variables[i] = {}
        s_variables[i] = {}
        s_costs[i] = {}
        e_variables[i] = {}
        e_costs[i] = {}
        v_variables[i] = {}
        v_costs[i] = {}
        p_variables[i] = {}
        n_variables[i] = {}
        for j, subtree in enumerate(pt.children):
            x_variables[i][j] = LpVariable('x_' + str(i) + '_' + str(j), cat='Binary')

            s_variables[i][j] = LpVariable('s_' + str(i) + '_' + str(j), cat='Binary')
            s_costs[i][j] = 0 if a[DEFAULT_NAME_KEY] in pt_sa_set[subtree] else 1

            e_variables[i][j] = LpVariable('e_' + str(i) + '_' + str(j), cat='Binary')
            e_costs[i][j] = 0 if a[DEFAULT_NAME_KEY] in pt_ea_set[subtree] else 1

            v_variables[i][j] = LpVariable('v_' + str(i) + '_' + str(j), cat='Binary')
            v_costs[i][j] = 0 if a[DEFAULT_NAME_KEY] in pt_a_set[subtree] else 1

            p_variables[i][j] = LpVariable('p_' + str(i) + '_' + str(j), cat='Binary')
            n_variables[i][j] = LpVariable('n_' + str(i) + '_' + str(j), cat='Binary')

    for i in range(len(trace) + 1):
        t_variables[i] = {}
        t_costs[i] = {}
        for j, subtree in enumerate(pt.children):
            t_variables[i][j] = LpVariable('t_' + str(i) + '_' + str(j), cat='Binary')
            if pt_tau_flag[subtree]:
                t_costs[i][j] = 0
            else:
                if len(pt_sa_set[subtree].intersection(pt_ea_set[subtree])) != 0:
                    t_costs[i][j] = 1
                else:
                    t_costs[i][j] = 2

    # objective function
    ilp += lpSum(
        [s_variables[i][j] * s_costs[i][j] for i in range(len(trace)) for j in range(len(pt.children))] +
        [e_variables[i][j] * e_costs[i][j] for i in range(len(trace)) for j in range(len(pt.children))] +
        [v_variables[i][j] * v_costs[i][j] for i in range(len(trace)) for j in range(len(pt.children))] +
        [t_variables[i][j] * t_costs[i][j] for i in range(len(trace) + 1) for j in
         range(len(pt.children))]), "objective_function"

    # constraints

    # universe j                        {0,1}
    # universe i for t_i_j variables    {0,...,len(trace)}
    # universe i else                   {0,...,len(trace)-1}

    # first tau can never be assigned to the 2nd subtree
    ilp += t_variables[0][1] == 0

    # last tau can never be assigned to the 2nd subtree
    ilp += t_variables[len(trace)][1] == 0

    # if first/last tau is not used --> first/last activity is assigned to 1st subtree
    ilp += 1 - t_variables[0][0] <= x_variables[0][0]
    ilp += 1 - t_variables[len(trace)][0] <= x_variables[len(trace) - 1][0]

    for i in range(len(trace)):
        # every activity is assigned to one subtree
        ilp += lpSum([x_variables[i][j] * 1 for j in range(len(pt.children))]) == 1

        # start/end/intermediate-activity at position i can only be assigned to one subtree
        ilp += lpSum([s_variables[i][j] * 1 for j in range(len(pt.children))]) <= 1
        ilp += lpSum([e_variables[i][j] * 1 for j in range(len(pt.children))]) <= 1
        ilp += lpSum([v_variables[i][j] * 1 for j in range(len(pt.children))]) <= 1

    for i in range(len(trace) + 1):
        # max one tau is used per index
        ilp += lpSum([t_variables[i][j] for j in range(2)]) <= 1

    # if tau is used and hence, assigned to a subtree, the surrounding activities are assigned to the other subtree
    for i in range(1, len(trace)):
        # if tau at position i is assigned to 1st subtree, the previous activity is assigned to 2nd subtree
        ilp += t_variables[i][0] <= x_variables[i - 1][1]
        # if tau at position i is assigned to 1st subtree, the previous activity is assigned to 2nd subtree
        ilp += t_variables[i][1] <= x_variables[i - 1][0]
    for i in range(len(trace)):
        # if tau at position i is assigned to 1st subtree, the next activity is assigned to 2nd subtree
        ilp += t_variables[i][0] <= x_variables[i][1]
        # if tau at position i is assigned to 2nd subtree, the next activity is assigned to 1st subtree
        ilp += t_variables[i][1] <= x_variables[i][0]
    # if last tau is used and assigned to 1st subtree (assigning it to the 2nd subtree is already forbidden by another
    # constraint) --> last activity must be assigned to 2nd subtree
    ilp += t_variables[len(trace)][0] <= x_variables[len(trace) - 1][1]

    # define auxiliary variables n: n_i_1 = 1 <=> next activity i+1 is assigned to 2nd subtree or t_i+1_2 = 1
    for i in range(len(trace) - 1):
        ilp += n_variables[i][0] <= x_variables[i + 1][1] + t_variables[i + 1][1]
        ilp += n_variables[i][0] >= x_variables[i + 1][1]
        ilp += n_variables[i][0] >= t_variables[i + 1][1]

        ilp += n_variables[i][1] <= x_variables[i + 1][0] + t_variables[i + 1][0]
        ilp += n_variables[i][1] >= x_variables[i + 1][0]
        ilp += n_variables[i][1] >= t_variables[i + 1][0]

    ilp += t_variables[len(trace)][1] <= n_variables[len(trace) - 1][0]
    ilp += t_variables[len(trace)][0] <= n_variables[len(trace) - 1][1]

    # define e_i_j variables
    for i in range(len(trace)):
        for j in range(2):
            ilp += e_variables[i][j] <= n_variables[i][j]
            ilp += e_variables[i][j] <= x_variables[i][j]
            ilp += e_variables[i][j] >= n_variables[i][j] + x_variables[i][j] - 1

    # define auxiliary variables p: p_i_1 = 1 <=> previous activity i-1 is assigned to 2nd subtree or t_i-1_2 = 1
    ilp += t_variables[0][1] <= p_variables[0][0]
    ilp += p_variables[0][1] <= t_variables[0][0]

    for i in range(1, len(trace)):
        ilp += p_variables[i][0] <= t_variables[i][1] + x_variables[i - 1][1]
        ilp += p_variables[i][0] >= t_variables[i][1]
        ilp += p_variables[i][0] >= x_variables[i - 1][1]

        ilp += p_variables[i][1] <= t_variables[i][0] + x_variables[i - 1][0]
        ilp += p_variables[i][1] >= t_variables[i][0]
        ilp += p_variables[i][1] >= x_variables[i - 1][0]

    # define s_i_j variables
    for i in range(len(trace)):
        for j in range(2):
            ilp += s_variables[i][j] >= p_variables[i][j] + x_variables[i][j] - 1
            ilp += s_variables[i][j] <= p_variables[i][j]
            ilp += s_variables[i][j] <= p_variables[i][j]
    ilp += 1 - t_variables[0][0] <= s_variables[0][0]

    # define v_i_j variables
    for i in range(len(trace)):
        for j in range(2):
            ilp += v_variables[i][j] >= 1 - s_variables[i][j] + 1 - e_variables[i][j] + x_variables[i][j] - 2
            ilp += v_variables[i][j] <= x_variables[i][j]
            ilp += v_variables[i][j] <= 1 - e_variables[i][j]
            ilp += v_variables[i][j] <= 1 - s_variables[i][j]

    status = ilp.solve()
    logging.debug("LP status: " + str(status))
    assert status == 1
    # LpStatusOptimal    “Optimal”    1
    # LpStatusNotSolved  “Not Solved” 0
    # LpStatusInfeasible “Infeasible” -1
    # LpStatusUnbounded  “Unbounded”  -2
    # LpStatusUndefined  “Undefined”  -3

    # DEBUG code
    # logging.debug('Trace length: ' + str(len(trace)))
    # trace_str = "\t\t\t"
    # for e in trace:
    #     trace_str += e['concept:name'] + "\t\t\t\t\t"
    # x_str_j_1 = "\t\t\t"
    # x_str_j_2 = "\t\t\t"
    #
    # s_str_j_1 = "\t\t\t"
    # s_str_j_2 = "\t\t\t"
    #
    # e_str_j_1 = "\t\t\t"
    # e_str_j_2 = "\t\t\t"
    #
    # n_str_j_1 = "\t\t\t"
    # n_str_j_2 = "\t\t\t"
    #
    # p_str_j_1 = "\t\t\t"
    # p_str_j_2 = "\t\t\t"
    #
    # v_str_j_1 = "\t\t\t"
    # v_str_j_2 = "\t\t\t"
    #
    # t_str_j_1 = ""
    # t_str_j_2 = ""
    #
    # for i in range(len(trace)):
    #     x_str_j_1 += "x_" + str(i) + "_" + "0" + ": " + str(int(x_variables[i][0].varValue)) + "\t\t\t"
    #     x_str_j_2 += "x_" + str(i) + "_" + "1" + ": " + str(int(x_variables[i][1].varValue)) + "\t\t\t"
    #     s_str_j_1 += "s_" + str(i) + "_" + "0" + ": " + str(int(s_variables[i][0].varValue)) + "\t\t\t"
    #     s_str_j_2 += "s_" + str(i) + "_" + "1" + ": " + str(int(s_variables[i][1].varValue)) + "\t\t\t"
    #     e_str_j_1 += "e_" + str(i) + "_" + "0" + ": " + str(int(e_variables[i][0].varValue)) + "\t\t\t"
    #     e_str_j_2 += "e_" + str(i) + "_" + "1" + ": " + str(int(e_variables[i][1].varValue)) + "\t\t\t"
    #     n_str_j_1 += "n_" + str(i) + "_" + "0" + ": " + str(int(n_variables[i][0].varValue)) + "\t\t\t"
    #     n_str_j_2 += "n_" + str(i) + "_" + "1" + ": " + str(int(n_variables[i][1].varValue)) + "\t\t\t"
    #     p_str_j_1 += "p_" + str(i) + "_" + "0" + ": " + str(int(p_variables[i][0].varValue)) + "\t\t\t"
    #     p_str_j_2 += "p_" + str(i) + "_" + "1" + ": " + str(int(p_variables[i][1].varValue)) + "\t\t\t"
    #     v_str_j_1 += "v_" + str(i) + "_" + "0" + ": " + str(int(v_variables[i][0].varValue)) + "\t\t\t"
    #     v_str_j_2 += "v_" + str(i) + "_" + "1" + ": " + str(int(v_variables[i][1].varValue)) + "\t\t\t"
    # for i in range(len(trace) + 1):
    #     t_str_j_1 += "t_" + str(i) + "_" + "0" + ": " + str(int(t_variables[i][0].varValue)) + "\t\t\t"
    #     t_str_j_2 += "t_" + str(i) + "_" + "1" + ": " + str(int(t_variables[i][1].varValue)) + "\t\t\t"
    #
    # logging.debug(trace_str)
    # logging.debug(t_str_j_1)
    # logging.debug(t_str_j_2)
    # logging.debug(x_str_j_1)
    # logging.debug(x_str_j_2 + "\n")
    # logging.debug(s_str_j_1)
    # logging.debug(s_str_j_2 + "\n")
    # logging.debug(e_str_j_1)
    # logging.debug(e_str_j_2 + "\n")
    # logging.debug(n_str_j_1)
    # logging.debug(n_str_j_2 + "\n")
    # logging.debug(p_str_j_1)
    # logging.debug(p_str_j_2 + "\n")
    # logging.debug(v_str_j_1)
    # logging.debug(v_str_j_2 + "\n")

    alignments_to_calculate: List[Tuple[ProcessTree, Trace]] = []
    sub_trace = Trace()
    current_subtree_idx = 0
    for i in range(len(trace)):
        for j in range(2):
            if t_variables[i][j].varValue:
                if i == 0:
                    # first tau can be only assigned to first subtree
                    assert j == 0
                    alignments_to_calculate.append((pt.children[j], Trace()))
                    current_subtree_idx = 1
                else:
                    alignments_to_calculate.append((pt.children[current_subtree_idx], sub_trace))
                    alignments_to_calculate.append((pt.children[j], Trace()))
                    sub_trace = Trace()
        for j in range(2):
            if x_variables[i][j].varValue:
                if j == current_subtree_idx:
                    sub_trace.append(trace[i])
                else:
                    alignments_to_calculate.append((pt.children[current_subtree_idx], sub_trace))
                    sub_trace = Trace()
                    sub_trace.append(trace[i])
                    current_subtree_idx = j
    if len(sub_trace) > 0:
        alignments_to_calculate.append((pt.children[current_subtree_idx], sub_trace))
    if t_variables[len(trace)][0].varValue:
        alignments_to_calculate.append((pt.children[0], Trace()))

    res = []
    for subtree, sub_trace in alignments_to_calculate:
        res.extend(approximate_alignment(subtree, pt_a_set, pt_sa_set, pt_ea_set, pt_tau_flag, sub_trace, tl, th))
    return res


def __approximate_alignment_on_parallel(pt: ProcessTree, trace: Trace, pt_a_set: Dict[ProcessTree, Set[str]],
                                        pt_sa_set: Dict[ProcessTree, Set[str]], pt_ea_set: Dict[ProcessTree, Set[str]],
                                        pt_tau_flag: Dict[ProcessTree, bool],
                                        tl: int, th: int):
    assert pt.operator == Operator.PARALLEL
    assert len(pt.children) > 0
    assert len(trace) > 0

    ilp = LpProblem("splitting_trace_and_assign_to_subtrees", LpMinimize)
    # ilp.solve(solvers.GUROBI_CMD())

    # x_i_j = 1 <=> assigns activity i to subtree j
    x_variables: Dict[int, Dict[int, LpVariable]] = {}

    # s_i_j = 1 <=> activity i is a start activity in the current sub-trace assigned to subtree j
    s_variables: Dict[int, Dict[int, LpVariable]] = {}

    # e_i_j = 1 <=> activity i is an end activity in the current sub-trace assigned to subtree j
    e_variables: Dict[int, Dict[int, LpVariable]] = {}

    # auxiliary u_j <=> u_j=1 if an activity is assigned to subtree j
    u_variables: Dict[int, LpVariable] = {}

    # v_i_j = 1 <=> activity i is neither a start nor end-activity in the current sub-trace assigned to subtree j
    v_variables: Dict[int, Dict[int, LpVariable]] = {}

    s_costs = {}
    e_costs = {}
    u_costs = {}
    v_costs = {}

    # trace <a_0,...,a_n>
    for i, a in enumerate(trace):
        x_variables[i] = {}
        s_variables[i] = {}
        s_costs[i] = {}
        e_variables[i] = {}
        e_costs[i] = {}
        v_variables[i] = {}
        v_costs[i] = {}

        for j, subtree in enumerate(pt.children):
            x_variables[i][j] = LpVariable('x_' + str(i) + '_' + str(j), cat='Binary')

            s_variables[i][j] = LpVariable('s_' + str(i) + '_' + str(j), cat='Binary')
            s_costs[i][j] = 0 if a[DEFAULT_NAME_KEY] in pt_sa_set[subtree] else 1

            e_variables[i][j] = LpVariable('e_' + str(i) + '_' + str(j), cat='Binary')
            e_costs[i][j] = 0 if a[DEFAULT_NAME_KEY] in pt_ea_set[subtree] else 1

            v_variables[i][j] = LpVariable('v_' + str(i) + '_' + str(j), cat='Binary')
            v_costs[i][j] = 0 if a[DEFAULT_NAME_KEY] in pt_a_set[subtree] else 1

    for j in range(len(pt.children)):
        u_variables[j] = LpVariable('u_' + str(j), cat='Binary')
        # define costs to not assign anything to subtree j
        if pt_tau_flag[pt.children[j]]:
            u_costs[j] = 0
        elif pt_sa_set[pt.children[j]] & pt_ea_set[pt.children[j]]:
            # intersection of start-activities and end-activities is not empty
            u_costs[j] = 1
        else:
            # intersection of start-activities and end-activities is empty
            u_costs[j] = 2

    # objective function
    ilp += lpSum(
        [v_variables[i][j] * v_costs[i][j] for i in range(len(trace)) for j in range(len(pt.children))] +
        [s_variables[i][j] * s_costs[i][j] for i in range(len(trace)) for j in range(len(pt.children))] +
        [e_variables[i][j] * e_costs[i][j] for i in range(len(trace)) for j in range(len(pt.children))] +
        [1 - u_variables[j] * u_costs[j] for j in range(len(pt.children))]), "objective_function"

    # constraints

    # universe j                        {0,1}
    # universe i                        {0,...,len(trace)-1}

    for i in range(len(trace)):
        # every activity is assigned to one subtree
        ilp += lpSum([x_variables[i][j] * 1 for j in range(len(pt.children))]) == 1

    for j in range(len(pt.children)):
        # first activity is a start activity
        ilp += x_variables[0][j] <= s_variables[0][j]
        # last activity is an end-activity
        ilp += x_variables[len(trace) - 1][j] <= e_variables[len(trace) - 1][j]

    # define s_i_j variables
    for i in range(len(trace)):
        for j in range(len(pt.children)):
            ilp += s_variables[i][j] <= x_variables[i][j]
            for k in range(i):
                ilp += s_variables[i][j] <= 1 - x_variables[k][j]
        # activity can be only a start-activity for one subtree
        ilp += lpSum(s_variables[i][j] for j in range(len(pt.children))) <= 1

    # define e_i_j variables
    for i in range(len(trace)):
        for j in range(len(pt.children)):
            ilp += e_variables[i][j] <= x_variables[i][j]
            for k in range(i + 1, len(trace)):
                ilp += e_variables[i][j] <= 1 - x_variables[k][j]
        # activity can be only an end-activity for one subtree
        ilp += lpSum(e_variables[i][j] for j in range(len(pt.children))) <= 1

    for j in range(len(pt.children)):
        for i in range(len(trace)):
            # define u_j variables
            ilp += u_variables[j] >= x_variables[i][j]
        # if u_j variable = 1 ==> a start activity must exist
        ilp += u_variables[j] <= lpSum(s_variables[i][j] for i in range(len(trace)))
        # if u_j variable = 1 ==> an end activity must exist
        ilp += u_variables[j] <= lpSum(e_variables[i][j] for i in range(len(trace)))

    # define v_i_j variables
    for i in range(len(trace)):
        for j in range(2):
            ilp += v_variables[i][j] >= 1 - s_variables[i][j] + 1 - e_variables[i][j] + x_variables[i][j] - 2
            ilp += v_variables[i][j] <= x_variables[i][j]
            ilp += v_variables[i][j] <= 1 - e_variables[i][j]
            ilp += v_variables[i][j] <= 1 - s_variables[i][j]

    status = ilp.solve()
    logging.debug("LP status: " + str(status))
    assert status == 1
    # LpStatusOptimal    “Optimal”    1
    # LpStatusNotSolved  “Not Solved” 0
    # LpStatusInfeasible “Infeasible” -1
    # LpStatusUnbounded  “Unbounded”  -2
    # LpStatusUndefined  “Undefined”  -3

    # DEBUG code --- start
    # logging.debug('Trace length: ' + str(len(trace)))
    # trace_str = "\t\t\t"
    # for e in trace:
    #     trace_str += e['concept:name'] + "\t\t\t\t\t"
    #
    # x_str_j = []
    # s_str_j = []
    # e_str_j = []
    # for j in range(len(pt.children)):
    #     x_str_j.append("\t\t\t")
    #     s_str_j.append("\t\t\t")
    #     e_str_j.append("\t\t\t")
    #
    # for j in range(len(pt.children)):
    #     for i in range(len(trace)):
    #         x_str_j[j] += "x_" + str(i) + "_" + str(j) + ": " + str(int(x_variables[i][j].varValue)) + "\t\t\t"
    #         s_str_j[j] += "s_" + str(i) + "_" + str(j) + ": " + str(int(s_variables[i][j].varValue)) + "\t\t\t"
    #         e_str_j[j] += "e_" + str(i) + "_" + str(j) + ": " + str(int(e_variables[i][j].varValue)) + "\t\t\t"
    #
    # logging.debug(trace_str)
    #
    # for j in range(len(pt.children)):
    #     logging.debug(x_str_j[j])
    # logging.debug("\n")
    # for j in range(len(pt.children)):
    #     logging.debug(s_str_j[j])
    # logging.debug("\n")
    # for j in range(len(pt.children)):
    #     logging.debug(e_str_j[j])
    # DEBUG code --- end

    # trace_parts list contains trace parts mapped onto the determined subtree
    trace_parts: List[Tuple[ProcessTree, Trace]] = []
    last_subtree: ProcessTree = None
    for i in range(len(trace)):
        for j in range(len(pt.children)):
            subtree = pt.children[j]
            if x_variables[i][j].varValue == 1:
                if last_subtree and subtree == last_subtree:
                    trace_parts[-1][1].append(trace[i])
                else:
                    assert last_subtree is None or subtree != last_subtree
                    t = Trace()
                    t.append(trace[i])
                    trace_parts.append((subtree, t))
                    last_subtree = subtree
                continue

    # calculate an alignment for each subtree
    alignments_per_subtree: Dict[ProcessTree] = {}
    for j in range(len(pt.children)):
        subtree: ProcessTree = pt.children[j]
        sub_trace = Trace()
        for trace_part in trace_parts:
            if subtree == trace_part[0]:
                sub_trace = concatenate_traces(sub_trace, trace_part[1])
        alignments_per_subtree[subtree] = approximate_alignment(subtree, pt_a_set, pt_sa_set, pt_ea_set,
                                                                pt_tau_flag, sub_trace, tl, th)
    # compose alignments from subtree alignments
    res = []
    for trace_part in trace_parts:
        activities_to_cover = trace_to_list_of_str(trace_part[1])
        activities_covered_so_far = []
        alignment = alignments_per_subtree[trace_part[0]]
        while activities_to_cover != activities_covered_so_far:
            move = alignment.pop(0)
            res.append(move)
            # if the alignment move is NOT a model move add activity to activities_covered_so_far
            if move[0] != SKIP:
                activities_covered_so_far.append(move[0])
    # add possible remaining alignment moves to resulting alignment, the order does not matter (parallel operator)
    for subtree in alignments_per_subtree:
        if len(alignments_per_subtree[subtree]) > 0:
            res.extend(alignments_per_subtree[subtree])
    return res


def __approximate_alignment_on_sequence(pt: ProcessTree, trace: Trace, pt_a_set: Dict[ProcessTree, Set[str]],
                                        pt_sa_set: Dict[ProcessTree, Set[str]],
                                        pt_ea_set: Dict[ProcessTree, Set[str]],
                                        pt_tau_flag: Dict[ProcessTree, bool],
                                        tl: int,
                                        th: int):
    assert pt.operator == Operator.SEQUENCE
    assert len(pt.children) > 0
    assert len(trace) > 0

    ilp = LpProblem("splitting_trace_and_assign_to_subtrees", LpMinimize)
    # ilp.solve(solvers.GUROBI_CMD())

    # x_i_j = 1 <=> assigns activity i to subtree j
    x_variables: Dict[int, Dict[int, LpVariable]] = {}

    # s_i_j = 1 <=> activity i is a start activity in the current sub-trace assigned to subtree j
    s_variables: Dict[int, Dict[int, LpVariable]] = {}

    # e_i_j = 1 <=> activity i is an end activity in the current sub-trace assigned to subtree j
    e_variables: Dict[int, Dict[int, LpVariable]] = {}

    # auxiliary u_j <=> u_j=1 if an activity is assigned to subtree j
    u_variables: Dict[int, LpVariable] = {}

    # v_i_j = 1 <=> activity i is neither a start nor end-activity in the current sub-trace assigned to subtree j
    v_variables: Dict[int, Dict[int, LpVariable]] = {}

    s_costs = {}
    e_costs = {}
    u_costs = {}
    v_costs = {}

    # trace <a_0,...,a_n>
    for i, a in enumerate(trace):
        x_variables[i] = {}
        s_variables[i] = {}
        s_costs[i] = {}
        e_variables[i] = {}
        e_costs[i] = {}
        v_variables[i] = {}
        v_costs[i] = {}

        for j, subtree in enumerate(pt.children):
            x_variables[i][j] = LpVariable('x_' + str(i) + '_' + str(j), cat='Binary')

            s_variables[i][j] = LpVariable('s_' + str(i) + '_' + str(j), cat='Binary')
            s_costs[i][j] = 0 if a[DEFAULT_NAME_KEY] in pt_sa_set[subtree] else 1

            e_variables[i][j] = LpVariable('e_' + str(i) + '_' + str(j), cat='Binary')
            e_costs[i][j] = 0 if a[DEFAULT_NAME_KEY] in pt_ea_set[subtree] else 1

            v_variables[i][j] = LpVariable('v_' + str(i) + '_' + str(j), cat='Binary')
            v_costs[i][j] = 0 if a[DEFAULT_NAME_KEY] in pt_a_set[subtree] else 1

    for j in range(len(pt.children)):
        u_variables[j] = LpVariable('u_' + str(j), cat='Binary')
        # define costs to not assign anything to subtree j
        if pt_tau_flag[pt.children[j]]:
            u_costs[j] = 0
        elif pt_sa_set[pt.children[j]] & pt_ea_set[pt.children[j]]:
            # intersection of start-activities and end-activities is not empty
            u_costs[j] = 1
        else:
            # intersection of start-activities and end-activities is empty
            u_costs[j] = 2

    # objective function
    ilp += lpSum(
        [v_variables[i][j] * v_costs[i][j] for i in range(len(trace)) for j in range(len(pt.children))] +
        [s_variables[i][j] * s_costs[i][j] for i in range(len(trace)) for j in range(len(pt.children))] +
        [e_variables[i][j] * e_costs[i][j] for i in range(len(trace)) for j in range(len(pt.children))] +
        [1 - u_variables[j] * u_costs[j] for j in range(len(pt.children))]), "objective_function"

    # constraints

    # universe j                        {0,1}
    # universe i                        {0,...,len(trace)-1}

    for i in range(len(trace)):
        # every activity is assigned to one subtree
        ilp += lpSum([x_variables[i][j] * 1 for j in range(len(pt.children))]) == 1

    for j in range(len(pt.children)):
        # first activity is start activity
        ilp += x_variables[0][j] <= s_variables[0][j]
        # last activity is end-activity
        ilp += x_variables[len(trace) - 1][j] <= e_variables[len(trace) - 1][j]

    # define s_i_j variables
    for i in range(1, len(trace)):
        for j in range(len(pt.children)):
            ilp += s_variables[i][j] >= x_variables[i][j] + 1 - x_variables[i - 1][j] - 1
            ilp += s_variables[i][j] <= x_variables[i][j]
            ilp += s_variables[i][j] <= 1 - x_variables[i - 1][j]
    for i in range(len(trace)):
        # activity can be only for one subtree a start-activity
        ilp += lpSum(s_variables[i][j] for j in range(len(pt.children))) <= 1

    # define e_i_j variables
    for i in range(len(trace) - 1):
        for j in range(len(pt.children)):
            ilp += e_variables[i][j] >= x_variables[i][j] + 1 - x_variables[i + 1][j] - 1
            ilp += e_variables[i][j] <= x_variables[i][j]
            ilp += e_variables[i][j] <= 1 - x_variables[i + 1][j]
    for i in range(len(trace)):
        # activity can be only for one subtree an end-activity
        ilp += lpSum(e_variables[i][j] for j in range(len(pt.children))) <= 1

    # constraint - preserving sequence when assigning activities to subtrees
    for i in range(len(trace) - 1):
        for j in range(len(pt.children)):
            ilp += lpSum(x_variables[i + 1][k] for k in range(j, len(pt.children))) >= x_variables[i][j]

    for j in range(len(pt.children)):
        for i in range(len(trace)):
            # define u_j variables
            ilp += u_variables[j] >= x_variables[i][j]

        # if u_j variable = 1 ==> a start activity must exist
        ilp += u_variables[j] <= lpSum(s_variables[i][j] for i in range(len(trace)))
        # if u_j variable = 1 ==> an end activity must exist
        ilp += u_variables[j] <= lpSum(e_variables[i][j] for i in range(len(trace)))

    # define v_i_j variables
    for i in range(len(trace)):
        for j in range(2):
            ilp += v_variables[i][j] >= 1 - s_variables[i][j] + 1 - e_variables[i][j] + x_variables[i][j] - 2
            ilp += v_variables[i][j] <= x_variables[i][j]
            ilp += v_variables[i][j] <= 1 - e_variables[i][j]
            ilp += v_variables[i][j] <= 1 - s_variables[i][j]

    status = ilp.solve()
    logging.debug("LP status: " + str(status))
    assert status == 1
    # LpStatusOptimal    “Optimal”    1
    # LpStatusNotSolved  “Not Solved” 0
    # LpStatusInfeasible “Infeasible” -1
    # LpStatusUnbounded  “Unbounded”  -2
    # LpStatusUndefined  “Undefined”  -3

    # DEBUG code --- start
    # logging.debug('Trace length: ' + str(len(trace)))
    # trace_str = "\t\t\t"
    # for e in trace:
    #     trace_str += e['concept:name'] + "\t\t\t\t\t"
    #
    # x_str_j = []
    # s_str_j = []
    # e_str_j = []
    # v_str_j = []
    # for j in range(len(pt.children)):
    #     x_str_j.append("\t\t\t")
    #     s_str_j.append("\t\t\t")
    #     e_str_j.append("\t\t\t")
    #     v_str_j.append("\t\t\t")
    #
    # for j in range(len(pt.children)):
    #     for i in range(len(trace)):
    #         x_str_j[j] += "x_" + str(i) + "_" + str(j) + ": " + str(int(x_variables[i][j].varValue)) + "\t\t\t"
    #         s_str_j[j] += "s_" + str(i) + "_" + str(j) + ": " + str(int(s_variables[i][j].varValue)) + "\t\t\t"
    #         e_str_j[j] += "e_" + str(i) + "_" + str(j) + ": " + str(int(e_variables[i][j].varValue)) + "\t\t\t"
    #         v_str_j[j] += "v_" + str(i) + "_" + str(j) + ": " + str(int(v_variables[i][j].varValue)) + "\t\t\t"
    #
    #
    # logging.debug(trace_str)
    #
    # for j in range(len(pt.children)):
    #     logging.debug(x_str_j[j])
    # logging.debug("\n")
    # for j in range(len(pt.children)):
    #     logging.debug(s_str_j[j])
    # logging.debug("\n")
    # for j in range(len(pt.children)):
    #     logging.debug(e_str_j[j])
    # logging.debug("\n")
    # for j in range(len(pt.children)):
    #     logging.debug(v_str_j[j])

    # DEBUG code --- end

    alignments_to_calculate: List[Tuple[ProcessTree, Trace]] = []
    for j in range(len(pt.children)):
        sub_trace = Trace()
        for i in range(len(trace)):
            if x_variables[i][j].varValue == 1:
                sub_trace.append(trace[i])
        alignments_to_calculate.append((pt.children[j], sub_trace))
    # calculate and compose alignments
    res = []
    for subtree, sub_trace in alignments_to_calculate:
        res.extend(approximate_alignment(subtree, pt_a_set, pt_sa_set, pt_ea_set, pt_tau_flag, sub_trace, tl, th))
    return res


def create_trace(labels: List[str]) -> Trace:
    trace = Trace()
    for label in labels:
        e = Event()
        e["concept:name"] = label
        trace.append(e)
    return trace
