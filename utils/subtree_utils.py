import copy
from enum import Enum

import numpy as np
import pm4py
from pm4py import reduce_petri_net_implicit_places, reduce_petri_net_invisibles
from pm4py import PetriNet, Marking
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.utils import incidence_matrix
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to, remove_arc
from pm4py.objects.process_tree.obj import Operator, ProcessTree
from pm4py.util import constants
from pm4py.objects.petri_net.utils.reduction import apply_simple_reduction

from synthesis_miner import get_nodes_on_the_path

class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    TOKEN_REPLAY_VARIANT = "token_replay_variant"
    CLEANING_TOKEN_FLOOD = "cleaning_token_flood"
    SHOW_PROGRESS_BAR = "show_progress_bar"
    MULTIPROCESSING = "multiprocessing"
    CORES = "cores"


def sync_transition_names_with_labels(net):
    if net is None:
        return net
    for transition in net.transitions:
        if transition.label is not None:
            transition.name = transition.label
    return net


def change_info_to_placeholders(net: PetriNet, tran='placeholder_minimal_subtree', activities=None, subnet=None):
    """
    This function changes the information of a specific transition in a given Petri net to placeholders.
    It specifically modifies the 'activities' and 'subnet' properties of the transition.
    :param net: Petri net
    :param tran: name of the transition to be modified
    :param activities: list of activities to be added to the transition
    :param subnet: list of transitions to be added to the transition
    :return: None
    """
    sync_transition_names_with_labels(net)
    placeholder_transition = None
    for t in net.transitions:
        if t.label == tran:
            placeholder_transition = t
            break

    if placeholder_transition is None:
        raise ValueError("Placeholder transition not found")

    if activities is not None:
        placeholder_transition.properties['activities'] = activities
    if subnet is not None:
        placeholder_transition.properties['subnet'] = subnet
        # if activities is None:
        placeholder_transition.properties['activities'] = [transition.name for transition in subnet.transitions]


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


def get_all_transitions(tree):
    transitions = set()
    # Create a list (acting as a stack for DFS)
    stack = [tree]
    # Depth-first search traversal of the tree
    while stack:
        node = stack.pop()
        # If the node has no children, add its label to the transitions
        if not node.children:
            transitions.add(node.label)
        else:
            # If the node has children, add them to the stack
            stack.extend(node.children)
    return transitions


def get_all_properties(tree):
    properties_dict = {}
    # Create a list (acting as a stack for DFS)
    stack = [tree]
    # Depth-first search traversal of the tree
    while stack:
        node = stack.pop()
        # If the node has a label and properties, add them to the dictionary
        if node.label is not None and node._properties is not None:
            properties_dict[node.label] = node._properties
        # If the node has children, add them to the stack
        if node.children:
            stack.extend(node.children)
    return properties_dict


def copy_process_tree_with_properties(node, properties_dict):
    """
    Function to create a copy of a ProcessTree node and apply properties based on a dictionary.

    Parameters
    ------------
    node : ProcessTree
        The ProcessTree node to be copied.
    properties_dict : dict
        A dictionary that maps labels to properties.

    Returns
    ------------
    new_node : ProcessTree
        The copied ProcessTree node.
    """
    # Create a new ProcessTree node with the same operator and label
    new_node = ProcessTree(operator=node.operator, label=node.label)

    # Apply the properties from the dictionary
    if new_node.label in properties_dict:
        new_node._properties = properties_dict[new_node.label]

    # Recursively copy the children
    new_node.children = [copy_process_tree_with_properties(child, properties_dict) for child in node.children]

    return new_node


def find_minimal_subtree(process_tree: ProcessTree, transitions):
    """
    Function to find a minimal subtree from a process tree, based on transitions and log.

    Parameters
    ------------
    process_tree: ProcessTree
        The process tree from which to find the minimal subtree.
    transitions
        The transitions to be considered while finding the minimal subtree.
    log: EventLog
        The event log used in the process tree.

    Returns
    ------------
    minimal_subtree_copy: ProcessTree
        The minimal subtree that has been found.
    process_tree: ProcessTree
        The original process tree.
    sublogs
        The sublogs associated with the minimal subtree.
    """
    # properties_dict = get_all_properties(process_tree)
    minimal_subtree = process_tree
    min_subtree_size = float('inf')
    # sublogs = None

    def all_transitions_present(tree, transitions):
        tree_transitions = get_all_transitions(tree)
        # Check if tree has subnet and adjust transitions accordingly
        if 'subnet' in tree._properties:
            tree_transitions.update(tree._properties['activities'])
        return transitions.issubset(tree_transitions)

    def subtree_size(tree: ProcessTree):
        if tree.children:
            return 1 + sum(subtree_size(child) for child in tree.children)
        else:
            return 1

    def contains_any_transition(tree, transitions):
        tree_transitions = get_all_transitions(tree)
        # Check if tree has subnet and adjust transitions accordingly
        if 'subnet' in tree._properties:
            tree_transitions.update(tree._properties['activities'])
        return not transitions.isdisjoint(tree_transitions)

    def restructure(tree: ProcessTree):
        """
        Function to restructure a process tree.

        Parameters
        ------------
        tree: ProcessTree
            The process tree that is to be restructured.

        Returns
        ------------
        bool
            Returns True if the tree was restructured, False otherwise.
        """
        restructured = False
        if tree.operator == Operator.SEQUENCE:
            # Check if the subtree already starts and ends with a transition
            if contains_any_transition(tree.children[0], transitions) and contains_any_transition(tree.children[-1],
                                                                                                  transitions):
                # If it does, then skip restructuring
                return False

            first_node_with_transitions = None
            last_node_with_transitions = None
            for i, child in enumerate(tree.children):
                if contains_any_transition(child, transitions):
                    if first_node_with_transitions is None:
                        first_node_with_transitions = i
                    last_node_with_transitions = i

            # If there are transitions, create a new node to replace the current sequence
            if first_node_with_transitions is not None and last_node_with_transitions is not None:
                new_node = ProcessTree(operator=Operator.SEQUENCE)
                new_node.children = tree.children[first_node_with_transitions:last_node_with_transitions + 1]

                # Only restructure the sequence if there's more than one node with transitions
                if len(new_node.children) > 1:
                    tree.children = tree.children[:first_node_with_transitions] + [new_node] + tree.children[
                                                                                               last_node_with_transitions + 1:]
                    restructured = True

        elif tree.operator in [Operator.XOR, Operator.PARALLEL]:
            print('XOR/PAR restructuring')
            if any(not contains_any_transition(child, transitions) for child in tree.children):
                new_node = ProcessTree(operator=tree.operator)
                new_node.children = [child for child in tree.children if contains_any_transition(child, transitions)]
                tree.children = [child for child in tree.children if
                                 not contains_any_transition(child, transitions)] + [
                                    new_node]
                restructured = True

        return restructured


    def replace_node_in_parent(node, new_node_label, root):
        """
        Function to replace a node in a tree.

        Parameters
        ------------
        node
            The node to be replaced.
        new_node_label
            The label of the new node that will replace the old node.
        root
            The root of the tree where the node will be replaced.
        """
        for child in node.children:
            if child == minimal_subtree:
                child.label = new_node_label
                child.operator = None
                child.children = []
            else:
                replace_node_in_parent(child, new_node_label, root)

    def dfs(root):
        """
        Function to perform Depth-First Search (DFS) in the tree.

        Parameters
        ------------
        root
            The node from which DFS starts.
        """
        nonlocal minimal_subtree, min_subtree_size
        stack = [root]

        while stack:
            node = stack.pop()

            if all_transitions_present(node, transitions):
                current_subtree_size = subtree_size(node)
                if current_subtree_size < min_subtree_size:
                    minimal_subtree = node
                    min_subtree_size = current_subtree_size

            # Add children to stack (if any)
            if node.children:
                stack.extend(node.children)
    counter = 0
    while True:
        dfs(process_tree)
        tree_restructured = restructure(minimal_subtree)
        if not tree_restructured:
            # approx_log_to_tree(importlog, tree_restructured)
            break
        counter += 1
        if counter > 10:
            break

    minimal_subtree_copy = copy.deepcopy(minimal_subtree)
    # minimal_subtree_copy = copy_process_tree_with_properties(minimal_subtree, properties_dict)
    print('Replacing minimal subtree in process tree')
    if minimal_subtree == process_tree:
        # The entire process tree is the minimal subtree, replace the root of the process tree
        process_tree.label = 'placeholder_minimal_subtree'
        process_tree.operator = None
        process_tree.children = []
    else:
        # The minimal subtree is a part of the process tree, find its parent and replace it
        replace_node_in_parent(process_tree, 'placeholder_minimal_subtree', process_tree)
    return minimal_subtree_copy, process_tree


def get_activity_names(tree: ProcessTree):
    if tree.children:  # if the node is not a leaf
        # recursively call get_activity_names for each child
        return [activity for child in tree.children for activity in get_activity_names(child)]
    else:  # if the node is a leaf
        if tree.label != 'tau':  # if the leaf is not a silent transition
            # check if the leaf node has a 'subnet'
            if 'subnet' in tree._properties:
                # if it does, return its 'activities' as the activities
                return tree._properties['activities']
            else:
                # if it doesn't, return its label as an activity
                return [tree.label]
        else:
            return []  # return an empty list for silent transitions


def replace_subtree_by_label(root: ProcessTree, label: str, new_tree: ProcessTree):
    """
    Replace a certain child node in a process tree based on its label with another process tree.

    Parameters
    ------------
    root : ProcessTree
        The root of the process tree where the replacement should take place.
    label : str
        The label of the child node that should be replaced.
    new_tree : ProcessTree
        The new process tree that should replace the child node.

    Returns
    ------------
    root : ProcessTree
        The root of the updated process tree.
    """
    new_tree._properties['min_tree'] = True
    if root.label == label:
        root = new_tree
        root._properties['min_tree'] = True
        return root


    for i, child in enumerate(root.children):
        if child.label == label:
            root.children[i] = new_tree
            break
        else:
            root.children[i] = replace_subtree_by_label(child, label, new_tree)
    return root


def add_start_end_tau_to_process_tree(process_tree):
    # Create 'start_tau' and 'end_tau' nodes
    start_tau = ProcessTree(operator=None, label='start_tau')
    end_tau = ProcessTree(operator=None, label='end_tau')

    # Create a new root with a sequential operator
    new_root = ProcessTree(operator=Operator.SEQUENCE)

    # Add the new nodes and the old root as children of the new root
    new_root.children = [start_tau, process_tree, end_tau]

    # Return the new root
    return new_root


def transfer_properties_from_tree_to_net(process_tree, hybrid_net):
    """
    This function transfers the properties 'activities' and 'subnet' from the nodes of a process tree to the transitions
    :param process_tree:
    :param hybrid_net:
    :return:
    """
    def transfer_properties_from_tree_to_transition(node, transition):
        """
        Helper function that transfers the properties from a node to a transition
        :param node:
        :param transition:
        :return:
        """
        if node.label == transition.label:
            if 'activities' in node._properties:
                transition.properties['activities'] = node._properties['activities']
            if 'subnet' in node._properties:
                transition.properties['subnet'] = node._properties['subnet']

        for child in node.children:
            transfer_properties_from_tree_to_transition(child, transition)

    # For each transition in the hybrid net, apply the helper function
    for t in hybrid_net.transitions:
        transfer_properties_from_tree_to_transition(process_tree, t)


def transform_hybrid_to_normal_petri_net(original_net):
    # iterate over all transitions in the Petri net
    for t in list(original_net.transitions):
        # check if the transition has a 'subnet' property
        if 'subnet' in t.properties:
            # if it does, reinsert the subnet into the original Petri net
            # print(t.properties['subnet'])
            original_net = reinsert_sub_net(original_net, t.properties['subnet'], t.label)

    return original_net


def reduce_net(net):
    im, fm = find_im_fm(net)
    net, im, fm = reduce_petri_net_implicit_places(net, im, fm)
    net = remove_start_end_trans(net, im, fm)
    net = apply_simple_reduction(net)
    net = reduce_petri_net_invisibles(net)
    im, fm = find_im_fm(net)

    remove_implicit_place(net, im, fm)
    return net, im, fm


def find_im_fm(net):
    m = incidence_matrix.construct(net)
    incidence_mat = np.array(m.a_matrix)
    source_ind = np.where(np.all(incidence_mat <= 0, axis=1))[0][0]
    sink_ind = np.where(np.all(incidence_mat >= 0, axis=1))[0][0]
    places_dict = {k.name: int(v) for k, v in m.places.items()}
    # trans_dict = {k.name: int(v) for k, v in m.transitions.items()}
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


def remove_start_end_trans(net_in, im, fm):
    net = copy.deepcopy(net_in)
    im, fm = find_im_fm(net)
    source_place_name = list(im.keys())[0].name
    sink_place_name = list(fm.keys())[0].name

    for place in net.places:
        if place.name == source_place_name:
            source = place
        elif place.name == sink_place_name:
            sink = place

    for transition in list(net.transitions):
        # check for source -> transition -> place and arc is one on each side
        if (transition.label is None or transition.label == '') and (transition.name == 'start_tau' or transition.name == 'end_tau'):
            for arc in list(transition.in_arcs):
                if arc.source is source and len(transition.in_arcs) == 1 and len(transition.out_arcs) == 1 and len(source.out_arcs) == 1:
                    out_arc = next(iter(transition.out_arcs))
                    target_place = out_arc.target
                    # check if the only target of the one outarc also only has one outarc
                    if len(target_place.in_arcs) == 1:
                        for out_arc2 in list(target_place.out_arcs):
                            new_target_transition = out_arc2.target
                            add_arc_from_to(source, new_target_transition, net)
                            net.arcs.remove(out_arc2)
                            target_place.out_arcs.remove(out_arc2)
                            new_target_transition.in_arcs.remove(out_arc2)
                        remove_arc(net, arc)
                        remove_arc(net, out_arc)
                        net.transitions.remove(transition)
                        net.places.remove(target_place)
                    break

            for arc in transition.out_arcs:
                if arc.target is sink and len(transition.in_arcs) == 1 and len(transition.out_arcs) == 1 and len(sink.in_arcs) == 1:
                    in_arc = next(iter(transition.in_arcs))
                    source_place = in_arc.source
                    if len(source_place.out_arcs) == 1:
                        for in_arc2 in list(source_place.in_arcs):
                            new_source_transition = in_arc2.source
                            add_arc_from_to(new_source_transition, sink, net)
                            net.arcs.remove(in_arc2)
                            source_place.in_arcs.remove(in_arc2)
                            new_source_transition.out_arcs.remove(in_arc2)

                        remove_arc(net, arc)
                        remove_arc(net, in_arc)
                        net.transitions.remove(transition)
                        net.places.remove(source_place)
                    break
    return net


def remove_implicit_place(net, im, fm):
    for place in list(net.places):
        if len(place.in_arcs) == 1 and len(place.out_arcs) == 1:
            in_trans = next(iter(place.in_arcs)).source
            out_trans = next(iter(place.out_arcs)).target
            if len(in_trans.out_arcs) > 1 and len(out_trans.in_arcs) > 1:
                # print(in_trans.name, out_trans.name)
                # print(net)
                paths = get_nodes_on_the_path(net, {in_trans.name}, {out_trans.name})
                if len(paths['places']) > 1:
                    # remove place and arcs
                    # remove for in_trans the arc to place, for out_trans the arc from place
                    in_arc = next(iter(place.in_arcs))
                    out_arc = next(iter(place.out_arcs))

                    in_trans.out_arcs.remove(in_arc)
                    out_trans.in_arcs.remove(out_arc)

                    # remove the two arcs from the net and remove the place from the net
                    net.arcs.remove(in_arc)
                    net.arcs.remove(out_arc)
                    net.places.remove(place)
    return net, im, fm


def hybrid_tree_w_subnet(net: PetriNet, im, fm, best_sub_net, tree):
    sync_transition_names_with_labels(best_sub_net)
    sync_transition_names_with_labels(net)
    im, fm = find_im_fm(net)
    try:
        pm4py.convert_to_process_tree(net, im, fm)
        print('Converting to process tree: ', net)
        new_tree = pm4py.convert_to_process_tree(net, im, fm)

    except ValueError:
        print('Best net is non-blocking')

        # save net under figures temp as nonblocknet
        filename = 'figures/temp/nonblocknet.svg'
        pm4py.save_vis_petri_net(net, im, fm, filename)

        # 1. check for node names called 'petri_net_xx'
        # 2. starting from 0, add the new node
        all_nodes = get_all_nodes(tree)

        # get existing petri_net_xx nodes
        petri_net_nodes = [node for node in all_nodes if node.label is not None and node.label.startswith('petri_net_')]

        # find the highest existing index
        if petri_net_nodes:
            highest_index = max(int(node.label.split('_')[2]) for node in petri_net_nodes)
        else:
            highest_index = 0

        # assign new node name
        node_name = 'petri_net_' + str(highest_index + 1)

        # 1. find transition nodes that belong to best_sub_net
        # 2. find those in net and add attribute 'subtree' with the belonging subtree (node_name) as a value
        t_labels = [t.name for t in best_sub_net.transitions]
        for t in net.transitions:
            if t.label in t_labels:
                t.properties['node'] = node_name

        new_tree = replace_node_and_insert_petri(tree, 'placeholder_minimal_subtree', node_name, best_sub_net)

    return new_tree, net


def get_all_nodes(tree: ProcessTree):
    return {tree}.union({n for nn in tree.children for n in get_all_nodes(nn)})


def replace_node_and_insert_petri(node, old_label, new_node_label, net, root=None):
    if root is None:
        root = node._get_root()

    if node.label == old_label:
        # If the node is found, replace its label
        node.label = new_node_label
        node.operator = None
        node.children = []
        # add the Petri net to the node properties
        node._properties['net'] = net
    else:
        for child in node.children:
            replace_node_and_insert_petri(child, old_label, new_node_label, net, root)
    return node


def rename_placeholder_transition(net):
    """
    Rename placeholder_minimal_subtree transition to petri_net_<index>.
    """
    # Get the placeholder transition
    placeholder_transition = next((transition for transition in net.transitions if transition.label == 'placeholder_minimal_subtree'), None)

    if placeholder_transition is not None:
        # Get next index
        next_index = get_next_petri_net_index(net)

        # Rename the transition
        placeholder_transition.label = 'petri_net_' + str(next_index)
        placeholder_transition.name = 'petri_net_' + str(next_index)

    return net


def get_next_petri_net_index(net):
    """
    Get the next index for a petri_net transition.
    """
    # Get existing petri_net transitions
    petri_net_transitions = [transition for transition in net.transitions if transition.label is not None and transition.label.startswith('petri_net_')]

    # Find the highest existing index
    if petri_net_transitions:
        highest_index = max(int(transition.label.split('_')[2]) for transition in petri_net_transitions)
    else:
        highest_index = 0

    return highest_index + 1


def transfer_properties_from_net_to_tree(hybrid_net, process_tree):
    """
    Transfer the properties from the hybrid net to the process tree
    :param hybrid_net:
    :param process_tree:
    :return:
    """

    def transfer_properties_from_transition_to_tree(transition, node):
        """
        Helper function for transfer_properties_from_net_to_tree. Recursively traverses the process tree
        :param transition:
        :param node:
        :return:
        """
        # If the labels match, copy the properties
        if transition.label == node.label:
            if 'activities' in transition.properties:
                node._properties['activities'] = transition.properties['activities']
            if 'subnet' in transition.properties:
                node._properties['subnet'] = transition.properties['subnet']

        # Repeat for all child nodes
        for child in node.children:
            transfer_properties_from_transition_to_tree(transition, child)

    # For each transition in the hybrid net, apply the helper function
    for t in hybrid_net.transitions:
        transfer_properties_from_transition_to_tree(t, process_tree)
