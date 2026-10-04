"""Module for exporting data to different file formats."""

from typing import TYPE_CHECKING

import networkx as nx
import polars as pl
import xlsxwriter

from .node import Node

if TYPE_CHECKING:
    from .zhao.config import ExportConfig


def write_excel(
    graph: nx.DiGraph | nx.MultiDiGraph,
    path: str,
    config: "ExportConfig",
) -> None:
    """Create a list of all unique nodes in the graph and their attributes.

    Parameters
    ----------
    graph : nx.DiGraph | nx.MultiDiGraph
        Graph to create node list from
    path : str
        Path to Excel file
    config : ExportConfig
        Configuration object containing export settings, such as node and edge keys,
        column widths, etc.

    """
    # Create lists to store dictionaries of node and edge attributes
    node_df_rows = []
    edge_df_rows = []

    # Create node and edge dictionaries from the graph and store them in lists
    for node, data in graph.nodes(data=True):
        node_attr = {"id": node.id}
        node_attr.update({key: node.__dict__[key] for key in config.node_keys})
        node_attr.update(data)
        node_df_rows.append(node_attr)

    for origin, target, data in graph.edges(data=True):
        edge_attr = {
            "origin": origin.id,
            "target": target.id,
        }
        edge_attr.update({key: data[key] for key in config.edge_keys})
        edge_df_rows.append(edge_attr)

    # Create DataFrames from dictionaries
    node_df = pl.from_dicts(node_df_rows)
    edge_df = pl.from_dicts(edge_df_rows)

    # Write DataFrames to Excel file
    with xlsxwriter.Workbook(path) as workbook:
        node_df.write_excel(
            workbook=workbook,
            worksheet="Nodes",
            table_name="nodes",
            column_widths=config.node_column_widths,
        )
        edge_df.write_excel(
            workbook=workbook,
            worksheet="Edges",
            table_name="edges",
            column_widths=config.edge_column_widths,
        )


def export_for_gephi(
    graph: nx.Graph,
    id_firm_map: dict,
    path: str,
) -> None:
    """Export a networkx graph to a GEXF file.

    This function creates a clean copy of the graph, excluding all edge
    attributes and all node attributes except for the one specified in
    `node_attribute_to_keep`. The resulting GEXF file is ideal for
    direct import into Gephi.

    Args:
        graph (nx.Graph): The input networkx graph.
        path (str): The path to save the output GEXF file.
        node_attribute_to_keep (str): The single node attribute to retain.

    """
    # Create a new graph to hold the filtered data
    clean_graph = nx.DiGraph()

    # Iterate through nodes and their attributes in the original graph
    for node in graph.nodes:
        # Add the node to the new graph
        clean_graph.add_node(node)

        if id_firm_map[node].nation_id is not None:
            # Add only the specified node attribute to the new graph
            clean_graph.nodes[node]["nation_id"] = id_firm_map[node].nation_id

    # Add all edges from the original graph without their attributes
    clean_graph.add_edges_from(graph.edges())

    # Export the clean graph to a GEXF file
    nx.write_gexf(clean_graph, path)
