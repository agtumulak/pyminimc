#!/usr/bin/env python
"""
Functions for interacting with minimc
"""
import xml.etree.ElementTree as ET


def update_tsl(
    input_xml_path: str,
    output_xml_path: str,
    beta_partitions: list[dict[str, str]],
    alpha_partitions: list[dict[str, str]],
):
    """
    Overwrites `partition` nodes from `beta_partitions` and `alpha_partitions`

    Parameters
    ----------
    input_xml_path:
        Absolute path to input file whose `beta_partitions` and
        `alpha_partitions` nodes will be overwritten
    output_xml_path:
        Absolute path to input file with overwritten `beta_partitions` and
        `alpha_partitions` nodes
    beta_partitions:
        A list of dicts. Each dict corresponds to a partition. For each dict,
        the key is the attribute name and the value is a path to an HDF5
        DataFrame. The order each dict appears in the list is the order each
        partition is added to the partitions node.
    alpha_partitions:
        A list of dicts. Each dict corresponds to a partition. For each dict,
        the key is the attribute name and the value is a path to an HDF5
        DataFrame. The order each dict appears in the list is the order each
        partition is added to the partitions node.
    """

    def append_partitions(tree, partitions_path, partitions):
        if (partitions_node := tree.find(partitions_path)) is None:
            raise RuntimeError(f"node not found: {partitions_path}")
        # remove any existing partition nodes
        for partition_node in partitions_node.findall("partition"):
            partitions_node.remove(partition_node)
        # add partitions
        for partition in partitions:
            partition_node = ET.Element("partition")
            for tag, path in partition.items():
                partition_node.set(tag, path)
            partitions_node.append(partition_node)

    tree = ET.parse(input_xml_path)
    tsl_path = "nuclides/continuous/nuclide/neutron/scatter/tsl"

    append_partitions(tree, tsl_path + "/beta_partitions", beta_partitions)
    append_partitions(tree, tsl_path + "/alpha_partitions", alpha_partitions)

    tree.write(output_xml_path)


if __name__ == "__main__":
    input_xml_path = (
        "/Users/atumulak/Developer/minimc/benchmarks/broomstick.xml"
    )
    output_xml_path = "out.xml"
    beta_partitions = [
        {
            "CDF": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/beta_endfb8_0_CDF_coeffs.hdf5",
            "S": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/beta_endfb8_0_S_coeffs.hdf5",
            "E_T": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/beta_endfb8_0_E_T_coeffs.hdf5",
        },
        {
            "CDF": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/beta_endfb8_1_CDF_coeffs.hdf5",
            "S": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/beta_endfb8_1_S_coeffs.hdf5",
            "E_T": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/beta_endfb8_1_E_T_coeffs.hdf5",
        },
        {
            "CDF": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/beta_endfb8_2_CDF_coeffs.hdf5",
            "S": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/beta_endfb8_2_S_coeffs.hdf5",
            "E_T": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/beta_endfb8_2_E_T_coeffs.hdf5",
        },
        {
            "CDF": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/beta_endfb8_3_CDF_coeffs.hdf5",
            "S": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/beta_endfb8_3_S_coeffs.hdf5",
            "E_T": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/beta_endfb8_3_E_T_coeffs.hdf5",
        },
    ]
    alpha_partitions = [
        {
            "CDF": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/alpha_endfb8_0_CDF_coeffs.hdf5",
            "S": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/alpha_endfb8_0_S_coeffs.hdf5",
            "beta_T": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/alpha_endfb8_0_beta_T_coeffs.hdf5",
        },
        {
            "CDF": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/alpha_endfb8_1_CDF_coeffs.hdf5",
            "S": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/alpha_endfb8_1_S_coeffs.hdf5",
            "beta_T": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/alpha_endfb8_1_beta_T_coeffs.hdf5",
        },
        {
            "CDF": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/alpha_endfb8_2_CDF_coeffs.hdf5",
            "S": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/alpha_endfb8_2_S_coeffs.hdf5",
            "beta_T": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/alpha_endfb8_2_beta_T_coeffs.hdf5",
        },
        {
            "CDF": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/alpha_endfb8_3_CDF_coeffs.hdf5",
            "S": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/alpha_endfb8_3_S_coeffs.hdf5",
            "beta_T": "/Users/atumulak/Developer/minimc/data/tsl/endfb8-adaptive-coarse/alpha_endfb8_3_beta_T_coeffs.hdf5",
        },
    ]
    update_tsl(
        input_xml_path, output_xml_path, beta_partitions, alpha_partitions
    )
