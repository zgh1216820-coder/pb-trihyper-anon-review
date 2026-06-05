from src.client.trimoe_nofish_nofuzzy import (
    EmbedNetwork,
    HierarchicalFusion,
    TaskEmbedding,
    TriBranchHyperNet,
    trimoe_nofish_nofuzzyClient,
)


class pb_trihyperClient(trimoe_nofish_nofuzzyClient):
    """Anonymous PB-TriHyper client alias.

    The class inherits the paper implementation without changing algorithm
    behavior. It exists so the FL-bench dynamic loader can use method:
    pb_trihyper in anonymous review configs.
    """

    pass
