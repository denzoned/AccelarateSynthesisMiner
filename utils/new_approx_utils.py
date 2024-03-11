from pm4py import ProcessTree
from pm4py.objects.log.obj import Trace


def get_node_with_min_tree(pt: ProcessTree):
    """
    Search for a node in a process tree where the property 'min_tree' is True.

    Parameters
    ------------
    pt: ProcessTree
        The root of the process tree where the search should take place.

    Returns
    ------------
    traces : list
        The 'traces' property of the node where 'min_tree' is True, or None if such a node is not found.
    """
    traces = None
    if 'min_tree' in pt._properties:
        if pt._properties['min_tree']:
            if 'traces' not in pt._properties:
                print('Error: min_tree is True but no traces property is found')
                print(pt)
                return None
            return pt._properties['traces']

    for child in pt.children:
        traces = get_node_with_min_tree(child)
        if traces is not None:
            return traces

    return traces


def add_trace_to_subtree(pt: ProcessTree, trace: Trace):
    # print(str(trace_to_list_of_str(trace)) + ' added to ' + str(pt))
    if 'traces' in pt._properties:
        pt._properties['traces'] = []
        pt._properties['traces'].append(trace)

    else:
        pt._properties['traces'] = []
        pt._properties['traces'].append(trace)
