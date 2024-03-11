import copy
import os
import sys
import time
import numpy as np
import pm4py
from pm4py import Marking
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.utils import check_soundness, incidence_matrix
from pm4py.objects.petri_net.utils.check_soundness import check_wfnet
from pm4py.objects.petri_net.utils.petri_utils import remove_arc, add_arc_from_to
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator

from utils.subtree_utils import reduce_net, hybrid_tree_w_subnet, change_info_to_placeholders, rename_placeholder_transition, \
    sync_transition_names_with_labels

from synthesis_miner import create_pn_from_incidence_mat, remove_tran_by_name
from external_utils.approx.original import apply as approx_log_to_tree


def find_im_fm(net):
    m = incidence_matrix.construct(net)
    incidence_mat = np.array(m.a_matrix)
    source_ind = np.where(np.all(incidence_mat <= 0, axis=1))[0][0]
    sink_ind = np.where(np.all(incidence_mat >= 0, axis=1))[0][0]
    places_dict = {k.name: int(v) for k, v in m.places.items()}
    trans_dict = {k.name: int(v) for k, v in m.transitions.items()}
    source_name = [k for k, v in places_dict.items() if v == source_ind][0]
    sink_name = [k for k, v in places_dict.items() if v == sink_ind][0]
    for p in net.places:
        if p.name == source_name:
            p_source = p
            im = Marking({p_source: 1})
        elif p.name == sink_name:
            p_sink = p
            fm = Marking({p_sink: 1})
    return im, fm


def analyse_new_nets(new_nets, t_add, restructured_pn, restructured_pt, log, subtree, extended_pt, use_approx=False):
    # get the best net from the new nets, operate similar to what is done in extend_petri_net_w_ld

    # pm4py.write_xes(log, os.path.join('results', 'nets', 'temp', 'log.xes'))
    count_not_sound = 0
    best_f1_appr = 0
    best_f1 = 0
    best_precision = 0
    best_precision_appr = 0
    best_fitness = 0
    best_fitness_appr = 0
    best_net_appr = None
    best_net_appr_sub = None
    best_net = None
    best_net_sub = None

    print('Subtree: ', subtree)

    if use_approx:
        if subtree.operator is None and 'subnet' not in subtree._properties:
            n_log = pm4py.filter_event_attribute_values(log, "concept:name", [t_add, subtree.label], level="event", retain=True)
        else:
            # TODO approx adjust to properties
            new_filtered_log, extended_pt = approx_log_to_tree(log, extended_pt)
            if (new_filtered_log == []) or (new_filtered_log is None):
                print('No traces left after approx')
                # save extended_pt and subtree
                pm4py.save_vis_process_tree(extended_pt, os.path.join('results', 'trees', 'temp', 'no_log_extended_pt.png'))
                pm4py.save_vis_process_tree(subtree, os.path.join('results', 'trees', 'temp', 'no_log_subtree.png'))
                sys.exit(0)
            n_log = EventLog()
            for trace in new_filtered_log:
                n_log.append(trace)
    else:
        n_log = log

    pm4py.write_xes(n_log, os.path.join('results', 'nets', 'temp', 'sublog.xes'))

    i = 0
    approx_score = {'f1_approx': [], 'f1_complete': [], 'precision_approx': [], 'precision_complete': [],
                    'fitness_approx': [], 'fitness_complete': []}

    # check time for all nets, but divide in normal and subtree nets
    t_all = 0
    t_subtree = 0




    for net_dict in new_nets:
        i += 1
        new_petri_dict = create_pn_from_incidence_mat(incidence_mat=net_dict['incidence_mat'],
                                                      places_dict=net_dict['places_dict'],
                                                      trans_dict=net_dict['trans_dict'],
                                                      return_net_dict=True)
        new_petri_temp = new_petri_dict['petri']
        new_petri = copy.deepcopy(new_petri_temp)
        new_petri = remove_tran_by_name(new_petri, trans_name='short_circuited_transition')
        new_petri_copy = copy.deepcopy(new_petri)
        restructured_pn_copy = copy.deepcopy(restructured_pn)
        complete_net = reinsert_sub_net(restructured_pn_copy, new_petri_copy)

        if check_wfnet(complete_net):
            im_c, fm_c = find_im_fm(complete_net)
            # pm4py.view_petri_net(net2, im2, fm2, format='svg')
            # pm4py.view_petri_net(new_petri, format='svg')
            pm4py.save_vis_petri_net(complete_net, im_c, fm_c, os.path.join('results', 'nets', 'temp', 'net_' + str(i) + 'subtreenet_.png'))
            pm4py.save_vis_petri_net(new_petri, im_c, fm_c, os.path.join('results', 'nets', 'temp', 'net_' + str(i) + 'subtreenet_reinserted.png'))
            if not check_soundness.check_easy_soundness_net_in_fin_marking(complete_net, im_c, fm_c):
                if not check_soundness.check_easy_soundness_of_wfnet(new_petri):
                    print('Subnet not sound')
                    print(net_dict['rule'])
                print('Net not sound')
                count_not_sound += 1
            else:
                # checking fitness based on the complete net
                t_alL_start = time.time()
                fitness_complete = replay_fitness_evaluator.apply(log, complete_net, im_c, fm_c)
                fitness= fitness_complete['log_fitness']
                precision_complete = pm4py.precision_alignments(log, complete_net, im_c, fm_c)
                if fitness != 0:
                    f1_complete = 2 * ((precision_complete * fitness) / (precision_complete + fitness))
                else:
                    f1_complete = 0
                print(precision_complete, fitness, f1_complete)
                if f1_complete > best_f1:
                    best_f1 = f1_complete
                    best_net = complete_net
                    best_net_sub = new_petri
                    best_precision = precision_complete
                    best_fitness = fitness

                t_all_end = time.time()
                t_all_temp = t_all_end - t_alL_start
                t_all += t_all_temp

                if use_approx:
                    t_subtree_start = time.time()
                    # approximated fitness on the subnet
                    nim, nfm = find_im_fm(new_petri)
                    precision_sub = pm4py.precision_alignments(n_log, new_petri, nim, nfm)

                    fitness_dict = replay_fitness_evaluator.apply(n_log, new_petri, nim, nfm,
                                                                  variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
                    # recall = fitness_dict['percentage_of_fitting_traces'] / 100
                    fitness_sub = fitness_dict['log_fitness']
                    if fitness_sub != 0:
                        f1_appr = 2 * ((precision_sub * fitness_sub) / (precision_sub + fitness_sub))
                    else:
                        f1_appr = 0

                    if f1_appr > best_f1_appr:
                        best_f1_appr = f1_appr
                        best_fitness_appr = fitness_sub
                        best_precision_appr = precision_sub
                        best_net_appr = complete_net
                        best_net_appr_sub = new_petri
                        best_f1_appr_whole = f1_complete
                        best_fitness_appr_whole = fitness
                        best_precision_appr_whole = precision_complete
                    t_subtree_end = time.time()
                    t_subtree_temp = t_subtree_end - t_subtree_start
                    t_subtree += t_subtree_temp
                else:
                    f1_appr = 0
                    precision_sub = 0
                    fitness_sub = 0
                approx_score['f1_approx'].append(f1_appr)
                approx_score['f1_complete'].append(f1_complete)
                approx_score['precision_approx'].append(precision_sub)
                approx_score['precision_complete'].append(precision_complete)
                approx_score['fitness_approx'].append(fitness_sub)
                approx_score['fitness_complete'].append(fitness)

        else:
            print('Net not wfnet')
            path = os.path.join('results', 'nets','not_wf_net_' + str(i) + '.png')
            tim = Marking({})
            tfm = Marking({})
            pm4py.save_vis_petri_net(new_petri, tim, tfm, path)
            path = os.path.join('results', 'nets','not_wf_net_' + str(i) + '_reinserted.png')
            pm4py.save_vis_petri_net(complete_net, tim, tfm, path)
            path = os.path.join('results', 'nets','not_wf_net_' + str(i) + '_restructured_pn.png')
            rim, rfm = find_im_fm(restructured_pn)
            pm4py.save_vis_petri_net(restructured_pn, rim, rfm, path)

    bestnet_hybrid = None
    pt = None
    if best_net is not None:
        bestnet_n, bestnet_im, bestnet_fm = reduce_net(best_net)
        best_sub_net, bsn_im, bsn_fm = reduce_net(best_net_sub)

        # either include subnet as subtree or as a leaf with subnet
        pt, bestnet_hybrid = hybrid_tree_w_subnet(bestnet_n, bestnet_im, bestnet_fm, best_net_sub, restructured_pt)
        try:
            pm4py.convert_to_process_tree(best_sub_net, bsn_im, bsn_fm)
            pt = pm4py.convert_to_process_tree(best_sub_net, bsn_im, bsn_fm)
            print('subtree: ', pt)
            restructured_pn = None
        except:
            print('Net is non-blocking')
            best_sub_net = sync_transition_names_with_labels(best_sub_net)
            change_info_to_placeholders(restructured_pn, subnet=best_sub_net)
            restructured_pn = rename_placeholder_transition(restructured_pn)

        print('hybrid tree: ', pt)

    return bestnet_hybrid, best_net_sub, restructured_pn, pt, best_f1, best_precision, best_fitness, best_f1_appr_whole, best_precision_appr_whole, best_fitness_appr_whole, best_net_appr, best_net_appr_sub, approx_score, t_all, t_subtree


def reinsert_sub_net(net, sub_net_in, placeholder_transition_label='placeholder_minimal_subtree'):
    # Find the placeholder transition in the net
    # print(sub_net_in)
    placeholder_transition = None
    for t in net.transitions:
        if t.label == placeholder_transition_label:
            placeholder_transition = t
            break
    sub_net = copy.deepcopy(sub_net_in)
    # Find the subnet's start and end transitions
    subnet_start_transitions = []
    subnet_end_transitions = []
    for t in sub_net.transitions:
        for arc in list(t.in_arcs):
            if arc.source.name == 'source':
                subnet_start_transitions.append(t)
                # remove the arc from the subnet and from the transition
                remove_arc(sub_net, arc)
        for arc in list(t.out_arcs):
            if arc.target.name == 'sink':
                subnet_end_transitions.append(t)
                # remove the arc from the subnet and from the transition
                remove_arc(sub_net, arc)

    # Identify the placeholder transition's immediate predecessor and successor places
    placeholder_predecessors = [arc.source for arc in placeholder_transition.in_arcs]
    placeholder_successors = [arc.target for arc in placeholder_transition.out_arcs]

    # Replace the placeholder transition with the subnet
    # remove the arcs from placeholder transition
    for arc in list(placeholder_transition.in_arcs):
        remove_arc(net, arc)
    for arc in list(placeholder_transition.out_arcs):
        remove_arc(net, arc)
    net.transitions.remove(placeholder_transition)


    # 2. Add all the transitions, places (except sink and source), and arcs of the subnet to the net
    net.transitions.update(sub_net.transitions)
    for p in sub_net.places:
        if p.name not in ['source', 'sink']:
            net.places.add(p)
    # add arcs that do not include source and sink
    for arc in sub_net.arcs:
        if arc.source.name not in ['source', 'sink'] and arc.target.name not in ['source', 'sink']:
            net.arcs.add(arc)

    # 3. Connect the subnet's start transitions to the placeholder transition's predecessor places
    for t in subnet_start_transitions:
        for p in placeholder_predecessors:
            add_arc_from_to(p, t, net)

    # 4. Connect the subnet's end transitions to the placeholder transition's successor places
    for t in subnet_end_transitions:
        for p in placeholder_successors:
            add_arc_from_to(t, p, net)

    return net