import itertools

try:
    import graphviz
except ImportError as e:
    graphviz = None


def visualize_expression(expr, comment='Matchpy Expression', with_attrs=True):
    if graphviz is None:
        raise ImportError('The graphviz package is required to draw expressions')

    from .expressions import Operation

    dot = graphviz.Digraph(comment=comment)
    counter = itertools.count()
    default_node_attr = dict(color='black', fillcolor='white', fontcolor='black')

    def _label_node(dot, expr):
        unique_id = str(next(counter))

        if hasattr(expr, '_repr_gviz_node_'):
            node_description, node_attr = expr._repr_gviz_node_()
        else:
            raise ValueError(f'matchpy expression does not have _repr_gviz_node_: "{repr(self)}"')

        if with_attrs:
            dot.attr('node', **{**default_node_attr, **node_attr})
        dot.node(unique_id, node_description)
        return unique_id

    def _visualize_node(dot, expr):
        expr_id = _label_node(dot, expr)

        if isinstance(expr, Operation):
            for sub_expr in expr:
                sub_expr_id = _visualize_node(dot, sub_expr)
                dot.edge(expr_id, sub_expr_id)
        return expr_id

    _visualize_node(dot, expr)
    return dot
